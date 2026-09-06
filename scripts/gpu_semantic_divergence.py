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
                   token_budget=30_000, max_length=1536, checkpoint=None,
                   done=None):
    """K samples for each prompt, returned as a list of K-length lists.

    Batched by TOKEN BUDGET, not by a fixed count. `num_return_sequences=k`
    multiplies the batch, so a nominal batch of 8 with k=5 is 40 sequences in
    one prefill; at ~900 prompt tokens each that is 36k tokens through the MLP
    at once, and a first version OOMed on it. Because prompts are sorted by
    length for padding efficiency, the longest ones come LAST -- so the failure
    arrived two hours in, after all the easy work was done and thrown away.

    Hence both fixes here: the batch shrinks as prompts get longer, and results
    are checkpointed as they are produced so a crash costs minutes, not a
    session.
    """
    import torch

    lengths = [len(tok(p, add_special_tokens=True)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    done = done or {}
    order = [i for i in order if i not in done]
    out: list[list[str]] = [done.get(i, []) for i in range(len(prompts))]
    t0, n_done, n_oom = time.time(), 0, 0

    fh = checkpoint.open("a", encoding="utf-8") if checkpoint else None
    try:
        pos = 0
        while pos < len(order):
            span = min(lengths[order[pos]] + 32, max_length)
            size = max(1, token_budget // max(1, k * span))
            idx = order[pos : pos + size]

            while True:
                try:
                    enc = tok([prompts[i] for i in idx], return_tensors="pt",
                              padding=True, truncation=True,
                              max_length=max_length).to(model.device)
                    with torch.no_grad():
                        gen = model.generate(
                            **enc, max_new_tokens=max_new_tokens, do_sample=True,
                            temperature=temperature, top_p=0.95,
                            num_return_sequences=k, pad_token_id=tok.pad_token_id,
                        )
                    break
                except torch.OutOfMemoryError:
                    # Halve and retry rather than losing the run. Recorded so a
                    # log that is full of these says the budget is set wrong.
                    n_oom += 1
                    torch.cuda.empty_cache()
                    if len(idx) == 1:
                        raise
                    idx = idx[: max(1, len(idx) // 2)]
                    print(f"  OOM -> retrying at batch {len(idx)}", flush=True)

            cut = enc["input_ids"].shape[1]
            for j, i in enumerate(idx):
                rows = gen[j * k : (j + 1) * k]
                out[i] = [
                    first_line(tok.decode(r[cut:], skip_special_tokens=True))
                    for r in rows
                ]
                if fh is not None:
                    fh.write(json.dumps({"i": i, "samples": out[i]}) + "\n")
            if fh is not None:
                fh.flush()

            pos += len(idx)
            n_done += len(idx)
            if n_done % 200 < len(idx):
                rate = n_done / max(1e-9, time.time() - t0)
                eta = (len(order) - n_done) / max(1e-9, rate) / 60
                print(f"  {n_done}/{len(order)}  batch {len(idx)}  "
                      f"{rate:.2f} prompts/s  eta {eta:.0f}m", flush=True)
    finally:
        if fh is not None:
            fh.close()

    if n_oom:
        print(f"  recovered from {n_oom} OOM(s) by shrinking the batch", flush=True)
    return out


def load_checkpoint(path) -> dict:
    """Resume: prompt index -> samples already produced."""
    if path is None or not Path(path).exists():
        return {}
    done = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line after a hard kill
            done[int(row["i"])] = row["samples"]
    return done


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
    ap.add_argument("--token-budget", type=int, default=30_000,
                    help="prompt tokens x k per forward pass; the batch size "
                         "is derived from it so long prompts get small batches")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="append samples here as they are produced, and resume "
                         "from it on restart")
    ap.add_argument("--equivalence", choices=list(EQUIVALENCE), default="numeric")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cluster-only", action="store_true",
                    help="re-cluster samples already in --checkpoint and skip "
                         "the model entirely; how the equivalence ablation runs "
                         "for free over the whole corpus instead of a subset")
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

    ckpt = args.checkpoint
    already = load_checkpoint(ckpt)
    if already:
        print(f"{len(already)}/{len(prompts)} prompts already in the checkpoint",
              flush=True)

    if args.cluster_only:
        missing = len(prompts) - len(already)
        if missing:
            print(f"cluster-only, but {missing} prompts have no samples")
            return 1
        samples = [already[i] for i in range(len(prompts))]
    else:
        model, tok = load_model(args.model)
        samples = sample_batches(
            model, tok, prompts, k=args.k, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, token_budget=args.token_budget,
            checkpoint=ckpt, done=already,
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
