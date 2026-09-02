"""Can a SEMANTIC verifier see the 80% that arithmetic cannot? [NEEDS GPU]

scripts/exp_verifier_scope.py established, on 35,535 real steps, that an
arithmetic verifier saturates at scope 0.1999 no matter how much lookback it
gets. The other 80.0% of inherited corruption has no upstream arithmetic error
at all -- the mistake is in the setup, not the calculation.

That defines the remaining question precisely:

    On the 28,433 steps arithmetic provably cannot see, what scope does a
    semantic verifier achieve?

Three outcomes, all publishable:
  high scope  -> semantic verification closes the gap; reach is about SEMANTICS,
                 not window size, and that is the design rule
  low scope   -> step-level verification cannot close the gap at all, and the
                 intervention has to move to the reasoning structure. This is
                 the most interesting outcome.
  mid scope   -> a scope estimate per verifier type, which is what ch. 5 needs
                 either way

RUNNING THIS
------------
Colab / Kaggle / RunPod, ~24 GB VRAM for the 7B PRM:

    git clone https://github.com/Soum-Code/CAR && cd CAR
    pip install -e ".[model]"
    python scripts/download_data.py math-shepherd
    python scripts/gpu_semantic_scope.py --verifier prm --limit 4000

Writes runs/semantic_scope_<verifier>.json. Copy that back and analyse on CPU:

    python scripts/gpu_semantic_scope.py --analyse runs/semantic_scope_prm.json

A note on precision: the README warns against 4-bit because quantisation
distorts the logit distribution this project measures. That warning is about
the GENERATOR whose uncertainty is being studied. Here the model is a JUDGE and
only its verdict is used, so --load-4bit is acceptable if VRAM is tight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path("data/raw/mathshepherd/strided.jsonl")
OUT = Path("runs")

PRM_MODEL = "peiyi9979/math-shepherd-mistral-7b-prm"
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

JUDGE_PROMPT = """You are checking one step of a math solution.

Problem: {question}

Steps so far:
{prefix}

Step under review: {step}

The arithmetic in this step is correct. Your job is to decide whether the step
is nonetheless WRONG because it rests on a mistaken earlier step, uses the
wrong quantity, or applies the wrong operation for this problem.

Answer with exactly one word: SOUND or UNSOUND."""


# ---- population ------------------------------------------------------


def build_population(limit=None, only_arithmetic_blind=True):
    """Steps that are locally valid, globally wrong, and (by default) invisible
    to arithmetic checking at ANY lookback -- the 80% ceiling population."""
    from car.data.math_shepherd import load_solutions
    from car.verification.lookback import (
        LookbackVerifier,
        corruption_distance,
        inherited_steps,
    )

    sols = load_solutions(DATA)
    v = LookbackVerifier(k=None)
    pop = []
    for sol in sols:
        for s in inherited_steps(sol):
            if only_arithmetic_blind and v.verify_step(sol, s.index).detected:
                continue  # arithmetic already catches this one
            prefix = [x for x in sol.steps if x.index < s.index]
            pop.append(
                {
                    "question": sol.question,
                    "prefix": "\n".join(f"Step {x.index + 1}: {x.text}" for x in prefix),
                    "step": s.text,
                    "step_index": s.index,
                    "distance": corruption_distance(sol, s.index),
                    "global_ok": s.global_ok,
                }
            )
            if limit and len(pop) >= limit:
                return pop
    return pop


def build_control(limit=None):
    """Locally valid AND globally correct steps.

    Without this the experiment cannot distinguish a verifier with real scope
    from one that simply says UNSOUND to everything. Detection rate alone is
    meaningless; what matters is detection minus false-alarm rate.
    """
    from car.data.math_shepherd import load_solutions

    sols = load_solutions(DATA)
    out = []
    for sol in sols:
        if not sol.final_correct:
            continue
        for s in sol.steps:
            if s.local_ok is True and s.global_ok:
                prefix = [x for x in sol.steps if x.index < s.index]
                out.append(
                    {
                        "question": sol.question,
                        "prefix": "\n".join(
                            f"Step {x.index + 1}: {x.text}" for x in prefix
                        ),
                        "step": s.text,
                        "step_index": s.index,
                        "distance": None,
                        "global_ok": True,
                    }
                )
                if limit and len(out) >= limit:
                    return out
    return out


# ---- verifiers -------------------------------------------------------


def run_prm(items, batch_size=8, load_4bit=False):
    """Math-Shepherd's own PRM. The baseline a reviewer will ask for.

    Scores each step; a low score means "does not lead to a correct answer",
    which is exactly the global-correctness signal being tested for.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PRM_MODEL)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_4bit:
        kwargs |= {"load_in_4bit": True}
    model = AutoModelForCausalLM.from_pretrained(PRM_MODEL, **kwargs).eval()

    good_id = tok.encode("+")[-1]
    bad_id = tok.encode("-")[-1]
    step_tag_id = tok.encode("ки")[-1]

    scores = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        texts = [
            f"{c['question']} {c['prefix']}\nStep {c['step_index'] + 1}: {c['step']} ки"
            for c in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, :, [bad_id, good_id]]
        probs = logits.softmax(dim=-1)[:, :, 1]
        for j, c in enumerate(chunk):
            mask = enc["input_ids"][j] == step_tag_id
            s = probs[j][mask]
            scores.append(float(s[-1]) if len(s) else float("nan"))
        if i % (batch_size * 20) == 0:
            print(f"  {i}/{len(items)}", flush=True)
    return scores


def run_judge(items, batch_size=8, load_4bit=False):
    """A general instruct model asked directly whether the step is sound."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_4bit:
        kwargs |= {"load_in_4bit": True}
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, **kwargs).eval()

    scores = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": JUDGE_PROMPT.format(**c)}],
                tokenize=False, add_generation_prompt=True,
            )
            for c in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            # 1.0 = flagged as unsound, so higher means "detected", matching
            # the PRM orientation after inversion below.
            scores.append(1.0 if "UNSOUND" in txt.upper() else 0.0)
        if i % (batch_size * 20) == 0:
            print(f"  {i}/{len(items)}", flush=True)
    return scores


VERIFIERS = {"prm": run_prm, "judge": run_judge}


# ---- analysis (CPU) --------------------------------------------------


def analyse(path):
    import numpy as np

    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = blob["verifier"]
    pos = blob["positive"]
    neg = blob["control"]

    def flagged(rec, thr):
        s = rec["score"]
        if s != s:
            return False
        # PRM: low score = predicted bad. Judge: 1.0 = flagged unsound.
        return s < thr if kind == "prm" else s >= 0.5

    print("=" * 80)
    print(f"Semantic verifier scope -- {kind}")
    print("=" * 80)
    print(f"population (arithmetic-blind inherited corruption)  {len(pos):,}")
    print(f"control (locally valid AND globally correct)        {len(neg):,}")
    print()
    print("Arithmetic verifier ceiling for comparison: scope 0.1999 at k=inf,")
    print("0.0000 at k=0.\n")

    thresholds = [0.3, 0.5, 0.7] if kind == "prm" else [0.5]
    print(f"{'threshold':<12}{'scope (TPR)':>14}{'false alarm':>14}{'scope - FA':>14}")
    print("-" * 80)
    for thr in thresholds:
        tpr = np.mean([flagged(r, thr) for r in pos]) if pos else float("nan")
        fpr = np.mean([flagged(r, thr) for r in neg]) if neg else float("nan")
        print(f"{thr:<12.2f}{tpr:>14.4f}{fpr:>14.4f}{tpr - fpr:>14.4f}")

    print()
    print("scope - false alarm is the number that matters. A verifier that")
    print("flags everything scores TPR 1.0 and is useless.")
    print()

    by_d = {}
    for r in pos:
        d = r.get("distance")
        by_d.setdefault(d, []).append(r)
    thr = 0.5
    print(f"{'distance':<12}{'n':>8}{'scope':>10}")
    print("-" * 80)
    for d in sorted(by_d, key=lambda x: (x is None, x)):
        g = by_d[d]
        if len(g) < 30:
            continue
        print(f"{str(d):<12}{len(g):>8}{np.mean([flagged(r, thr) for r in g]):>10.4f}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verifier", choices=list(VERIFIERS), default="prm")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--control-limit", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--load-4bit", action="store_true",
                    help="judge only; see the note in the module docstring")
    ap.add_argument("--analyse", metavar="RESULTS_JSON",
                    help="CPU-side analysis of a finished run")
    args = ap.parse_args()

    if args.analyse:
        analyse(args.analyse)
        return 0

    if not DATA.exists():
        print(f"missing {DATA}; run python scripts/download_data.py math-shepherd")
        return 1

    print("building population (arithmetic-blind inherited corruption)...")
    pos = build_population(limit=args.limit)
    print(f"  {len(pos):,} steps")
    print("building control (locally valid and globally correct)...")
    neg = build_control(limit=args.control_limit)
    print(f"  {len(neg):,} steps")

    fn = VERIFIERS[args.verifier]
    print(f"\nscoring with {args.verifier}...")
    pos_scores = fn(pos, args.batch_size, args.load_4bit)
    print("scoring control...")
    neg_scores = fn(neg, args.batch_size, args.load_4bit)

    for r, s in zip(pos, pos_scores, strict=True):
        r["score"] = s
    for r, s in zip(neg, neg_scores, strict=True):
        r["score"] = s

    OUT.mkdir(exist_ok=True)
    path = OUT / f"semantic_scope_{args.verifier}.json"
    path.write_text(
        json.dumps({"verifier": args.verifier, "positive": pos, "control": neg}),
        encoding="utf-8",
    )
    print(f"\nwrote {path}")
    analyse(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
