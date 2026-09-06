"""Measure semantic divergence per step. [NEEDS GPU]

The one uncertainty signal the original spec weighted most heavily, and the one
the end-to-end run could not test: `gpu_score_uncertainty.py` recovers entropy,
surprisal and log-probability by teacher-forcing, but semantic divergence is
defined over *independently sampled continuations* and no amount of re-reading
one sample produces the others. It has to be resampled.

    for each step t of each solution:
        prompt the model with the prefix ending at t-1
        sample K continuations of the NEXT step only
        cluster them into meaning classes
        divergence = normalised entropy over the cluster distribution

Farquhar et al. (Nature 2024) do this over whole answers with bidirectional
entailment. Here it is one intermediate step, which is precisely the regime
arXiv:2602.02427 reports sampling-agreement methods are weaker in -- so this
run tests a contested claim rather than assuming it.

EQUIVALENCE

`numeric_equivalence`: two arithmetic steps agree if they assert the same
number, whatever the phrasing. Not a cheap substitute for entailment so much as
a closer fit to what an arithmetic step actually asserts. `--equivalence exact`
runs the string-equality floor instead, which is the ablation showing how much
of any measured divergence is really notational variety -- Qwen writes the same
arithmetic as `<<48/2=24>>`, `\\( 48/2 = 24 \\)` and `48 / 2 = 24`, and exact
match calls those three different meanings.

COST

K samples per step, ~48 new tokens each, against the 256 the rollout run
needed. 2,573 steps at K=5 is ~12.9k short generations.

RUNNING

    python scripts/gpu_semantic_divergence.py \\
        --corpus runs/generated_qwen25_7b.jsonl \\
        --features runs/uncertainty_qwen25_7b.jsonl \\
        --out runs/uncertainty_qwen25_7b_sem.jsonl

Reads the existing feature file and writes an augmented copy with
`semantic_divergence` filled in, so alignment with the corpus is inherited
rather than re-derived.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data.generated import ANSWER_PREFIX, build_fewshot, generation_prompt  # noqa: E402
from car.data.math_shepherd import load_solutions  # noqa: E402
from car.uncertainty.semantic import (  # noqa: E402
    cluster_by_equivalence,
    exact_match_equivalence,
    normalised_semantic_divergence,
    numeric_equivalence,
)

TRAIN = Path("data/raw/gsm8k/train.jsonl")
MODEL = "Qwen/Qwen2.5-7B-Instruct"

EQUIVALENCE = {"numeric": numeric_equivalence, "exact": exact_match_equivalence}


def strip_answer_tail(text: str) -> str:
    i = text.find(ANSWER_PREFIX)
    return text[:i].strip() if i != -1 else text


def first_line(text: str) -> str:
    """The next step only. A sample that runs on is not evidence about step t."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LEFT padding: batched decoder-only generation puts pad tokens between
    # prompt and continuation otherwise, and the model generates from padding.
    tok.padding_side = "left"

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


def sample_batches(model, tok, prompts, *, k, temperature, max_new_tokens,
                   batch_size, max_length=1536):
    """K samples for each prompt, returned as a list of K-length lists."""
    import torch

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[list[str]] = [[] for _ in prompts]
    t0 = time.time()

    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start : start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=k,
                pad_token_id=tok.pad_token_id,
            )
        cut = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            rows = gen[j * k : (j + 1) * k]
            out[i] = [
                first_line(tok.decode(r[cut:], skip_special_tokens=True)) for r in rows
            ]
        if b % 20 == 0:
            done = start + len(idx)
            rate = done / max(1e-9, time.time() - t0)
            eta = (len(prompts) - done) / max(1e-9, rate) / 60
            print(f"  {done}/{len(prompts)}  {rate:.2f} prompts/s  eta {eta:.0f}m",
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("runs/generated_qwen25_7b.jsonl"))
    ap.add_argument("--features", type=Path,
                    default=Path("runs/uncertainty_qwen25_7b.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("runs/uncertainty_qwen25_7b_sem.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--equivalence", choices=list(EQUIVALENCE), default="numeric")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for p in (args.corpus, args.features, TRAIN):
        if not p.exists():
            print(f"missing {p}")
            return 1

    train = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    fewshot = build_fewshot(train, k=args.shots, seed=args.seed)

    sols = load_solutions(args.corpus)
    feats = [json.loads(x) for x in
             args.features.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(feats) != len(sols):
        print(f"feature file has {len(feats)} records, corpus has {len(sols)}")
        return 1
    if args.limit:
        sols, feats = sols[: args.limit], feats[: args.limit]

    # One prompt per step: the prefix ENDING BEFORE that step, so the samples
    # are alternative continuations of the same context the real step came from.
    prompts, index = [], []
    for si, sol in enumerate(sols):
        texts = [s.text for s in sol.steps]
        texts[-1] = strip_answer_tail(texts[-1]) or texts[-1]
        for ti in range(len(texts)):
            prompts.append(generation_prompt(fewshot, sol.question, texts[:ti]))
            index.append((si, ti))

    print(f"{len(sols)} solutions, {len(prompts)} steps, k={args.k} "
          f"-> {len(prompts) * args.k:,} generations", flush=True)

    model, tok = load_model(args.model)
    samples = sample_batches(
        model, tok, prompts, k=args.k, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )

    eq = EQUIVALENCE[args.equivalence]
    n_singleton = n_unanimous = 0
    for (si, ti), texts in zip(index, samples, strict=True):
        texts = [t for t in texts if t]
        if len(texts) < 2:
            div = 0.0
            n_singleton += 1
        else:
            ids = cluster_by_equivalence(texts, eq)
            div = normalised_semantic_divergence(ids)
            if len(set(ids)) == 1:
                n_unanimous += 1
        feats[si]["steps"][ti]["semantic_divergence"] = float(div)
        feats[si]["steps"][ti]["n_samples"] = len(texts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in feats:
            fh.write(json.dumps(row) + "\n")

    vals = [f["semantic_divergence"] for r in feats for f in r["steps"]]
    print(f"\nwrote {args.out}", flush=True)
    print(f"  equivalence          {args.equivalence}", flush=True)
    print(f"  steps with <2 usable samples  {n_singleton}", flush=True)
    print(f"  unanimous (divergence 0)      {n_unanimous}", flush=True)
    print(f"  divergence min/mean/max       {min(vals):.4f} / "
          f"{sum(vals) / len(vals):.4f} / {max(vals):.4f}", flush=True)
    if min(vals) == max(vals):
        print("  WARNING: divergence is constant; it carries no information and "
              "its weight must stay zero", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
