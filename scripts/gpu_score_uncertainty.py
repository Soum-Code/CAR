"""Attach per-step uncertainty features to an existing corpus. [NEEDS GPU]

The gate needs an uncertainty score per step. Generating a corpus *with*
features costs a full sampling run; recovering them afterwards costs one
teacher-forced forward pass per solution, because token entropy, surprisal and
log-probability are all functions of the model's predictive distribution over
text it is merely reading. 500 solutions is minutes, not hours.

    generation run   8,292 rollouts, 7h40m   (already paid, runs/generated_*.jsonl)
    this script      500 forward passes, ~5 min

What is recovered, and what is not
----------------------------------
  token_entropy    mean predictive entropy over the step's tokens
  max_surprisal    the single most surprising token in the step
  mean_logprob     mean log p(token), the model's own fluency signal

  semantic_divergence CANNOT be recovered this way. It is defined over
  independently sampled continuations, and no amount of re-reading one sample
  produces the others. It is left at 0.0 and the config weight must be zeroed
  with it -- a feature silently pinned to a constant is worse than an absent
  one, because a fitted combiner will happily assign it a weight.

A note on faithfulness. The corpus text has been whitespace-normalised and the
final step carries an appended "The answer is: N", so what is scored is not
byte-identical to what was sampled. The measured quantity is therefore "how
surprising does the model find this step", which is what the gate consumes
anyway, rather than a replay of the original sampling distribution.

RUNNING

    python scripts/gpu_score_uncertainty.py \
        --corpus runs/generated_qwen25_7b.jsonl \
        --out runs/uncertainty_qwen25_7b.jsonl

Writes one JSON record per corpus record, in the same order, with a `steps`
array of the same length. `car.generation.replay.load_corpus` asserts both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data.generated import ANSWER_PREFIX, build_fewshot, generation_prompt  # noqa: E402
from car.data.math_shepherd import load_solutions  # noqa: E402

TRAIN = Path("data/raw/gsm8k/train.jsonl")
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def strip_answer_tail(text: str) -> str:
    """Drop the `The answer is: N` the serialiser appends to the final step.

    It is an artifact of the Math-Shepherd format, not something the model
    wrote, and scoring it would put the same tokens in every corpus's last
    step and bias that position's features.
    """
    i = text.find(ANSWER_PREFIX)
    return text[:i].strip() if i != -1 else text


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not tok.is_fast:
        raise RuntimeError(
            "a fast tokenizer is required: step spans are located by character "
            "offset mapping, which the slow tokenizer does not provide"
        )
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        dtype = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto"
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        ).eval()
    print(f"  loaded {model_id}, dtype={dtype}", flush=True)
    return model, tok


def step_features(model, tok, prompt, steps, max_length=2048):
    """Teacher-force prompt+steps and return per-step feature dicts."""
    import torch

    body_parts, spans, cursor = [], [], 0
    for s in steps:
        if body_parts:
            cursor += 1  # the newline joining the previous step
        body_parts.append(s)
        spans.append((cursor, cursor + len(s)))
        cursor += len(s)
    body = "\n".join(body_parts)

    enc = tok(prompt, add_special_tokens=True, return_tensors=None)
    n_prompt = len(enc["input_ids"])
    benc = tok(body, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"] + benc["input_ids"]
    offsets = benc["offset_mapping"]

    if len(ids) > max_length:  # keep the tail: the body is what gets scored
        drop = len(ids) - max_length
        ids = ids[drop:]
        n_prompt -= drop
        if n_prompt < 1:
            return None

    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(x).logits[0].float()

    # logits[t] predicts token t+1, so the distribution for body token at
    # absolute index i lives at row i-1. Getting this off by one silently
    # shifts every feature by one token.
    logp = torch.log_softmax(logits[:-1], dim=-1)
    targets = x[0, 1:]
    tok_logp = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    entropy = -(logp.exp() * logp).sum(dim=-1)

    out = []
    for lo, hi in spans:
        rows = [
            n_prompt + j - 1
            for j, (a, b) in enumerate(offsets)
            if a < hi and b > lo and 0 <= n_prompt + j - 1 < tok_logp.shape[0]
        ]
        if not rows:
            out.append({"token_entropy": 0.0, "max_surprisal": 0.0,
                        "mean_logprob": 0.0, "semantic_divergence": 0.0,
                        "n_tokens": 0})
            continue
        idx = torch.tensor(rows, device=tok_logp.device)
        lp = tok_logp[idx]
        out.append({
            "token_entropy": float(entropy[idx].mean()),
            "max_surprisal": float((-lp).max()),
            "mean_logprob": float(lp.mean()),
            "semantic_divergence": 0.0,
            "n_tokens": len(rows),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("runs/generated_qwen25_7b.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/uncertainty_qwen25_7b.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"missing {args.corpus}")
        return 1
    if not TRAIN.exists():
        print(f"missing {TRAIN}; run python scripts/download_data.py gsm8k")
        return 1

    train = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    # Same shots and seed as the generation run, so the prompt the model is
    # conditioned on here is the prompt it actually generated under.
    fewshot = build_fewshot(train, k=args.shots, seed=args.seed)

    sols = load_solutions(args.corpus)
    if args.limit:
        sols = sols[: args.limit]
    print(f"{len(sols)} solutions, {sum(len(s.steps) for s in sols)} steps", flush=True)

    model, tok = load_model(args.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, sol in enumerate(sols):
            texts = [s.text for s in sol.steps]
            texts[-1] = strip_answer_tail(texts[-1]) or texts[-1]
            prompt = generation_prompt(fewshot, sol.question)
            feats = step_features(model, tok, prompt, texts)
            if feats is None:
                feats = [{"token_entropy": 0.0, "max_surprisal": 0.0,
                          "mean_logprob": 0.0, "semantic_divergence": 0.0,
                          "n_tokens": 0} for _ in texts]
                n_empty += 1
            fh.write(json.dumps({"n_steps": len(texts), "steps": feats}) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(sols)}", flush=True)

    print(f"wrote {args.out}", flush=True)
    if n_empty:
        print(f"  WARNING: {n_empty} solutions were truncated away entirely", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
