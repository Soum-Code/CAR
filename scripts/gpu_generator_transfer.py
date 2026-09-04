"""Re-measure the local/global gap on a second generator. [NEEDS GPU]

Every rate in this project comes from Math-Shepherd, i.e. from Mistral-7B-SFT:
local error 0.1011, global 0.3908, 69.8% of wrong steps locally valid. The
obvious objection is that those are properties of one weak 2023 model rather
than of multi-step reasoning, and a stronger generator would close the gap by
itself. Putting that objection to the test needs a materially stronger model
solving the same benchmark.

The generator is a parameter, not a constant. The thesis names Llama 3.1 8B;
the run that actually produced numbers used Qwen2.5-7B-Instruct, because Llama
3.1 is licence-gated on Kaggle and the consent did not come through. Both are
instruction-tuned ~7-8B models reporting GSM8K far above Mistral-7B-SFT's ~45%,
which is the property the argument depends on -- so the substitution costs the
exact model name and nothing else. Whichever is used is recorded in the run
metadata rather than assumed from this docstring.

WHAT IS MEASURED

Both per-step signals are rebuilt from scratch for the new generator:

  local   the step's own `<<expr=result>>` arithmetic, checked by the SAME
          `check_arithmetic` Math-Shepherd steps go through
  global  Math-Shepherd's hard estimation: sample K completions from the prefix
          ending at this step; `+` if any reaches the gold answer

Output is written in Math-Shepherd's own `label` format, so
`scripts/exp_measure_error_rate.py` analyses it unchanged. The comparison is
then between two datasets, not between two analysis implementations.

TWO GATES, BOTH BEFORE ANY NUMBER IS BELIEVED

  accuracy    few-shot solve rate must land in a plausible band for this model.
              A broken prompt shows up here as 20% and nowhere else.
  annotation  fraction of steps carrying `<<>>`. The local rate is only
              comparable if the model actually imitates the exemplar format;
              if it does not, the local rate is measured on a biased subset.

This is the same discipline the ch. 5 PRM run needed: on a borrowed model the
harness must reproduce known behaviour before a novel number from it counts.

NO QUANTISATION. Elsewhere the README allows 4-bit for a judge, because only
its verdict is used. Here the model's own error rate IS the measurement, and
quantisation changes it. There is deliberately no --load-4bit flag.

RUNNING THIS

    pip install -e ".[model]"
    python scripts/download_data.py gsm8k
    python scripts/gpu_generator_transfer.py --n 500 --rollouts 4

Writes runs/generated_<tag>.jsonl (Math-Shepherd format) and
runs/generated_<tag>_meta.json. Analyse on CPU with:

    python scripts/exp_measure_error_rate.py --data runs/generated_qwen25_7b.jsonl \
        --accuracy measured --name "Qwen2.5-7B-Instruct"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data.generated import (  # noqa: E402
    GeneratedSolution,
    annotation_rate,
    answers_match,
    build_fewshot,
    extract_answer,
    generation_prompt,
    local_validity,
    split_steps,
    to_shepherd_record,
    truncate_completion,
)

TRAIN = Path("data/raw/gsm8k/train.jsonl")
TEST = Path("data/raw/gsm8k/test.jsonl")
OUT = Path("runs")

MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Wide on purpose. Reported GSM8K figures for models in this class are greedy or
# maj@1 with 8 exemplars; we sample at temperature with 4, and both cost points.
# The gate is not trying to reproduce a leaderboard number -- it is trying to
# catch a prompt the model is not following, which reads as 0.2, not as 0.78.
ACCURACY_BAND = (0.55, 0.95)
MIN_ANNOTATION_RATE = 0.60


# ---- generation ------------------------------------------------------


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LEFT padding for batched decoder-only generation. Right padding puts pad
    # tokens between prompt and continuation, so the model generates from
    # padding -- the bug that made the first ch. 5 judge run untrustworthy.
    tok.padding_side = "left"

    # bf16 is emulated below sm_80, so gate on capability rather than on
    # is_bf16_supported(), which returns True on a P100.
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        dtype = torch.bfloat16
    print(f"  dtype={dtype}", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto"
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        ).eval()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  gpu{i} free {free / 1e9:.1f} / {total / 1e9:.1f} GB",
                  flush=True)
    return model, tok


def generate(model, tok, prompts, *, max_new_tokens=256, temperature=0.7,
             batch_size=24, max_length=1536, label=""):
    """Batched completion, returned in the caller's order.

    Prompts are sorted by length before batching. With left padding a batch
    costs the length of its longest member for every member, so mixing a
    2-step prefix with a 7-step one wastes most of the batch.
    """
    import torch

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[str] = [""] * len(prompts)
    t0 = time.time()

    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start : start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        cut = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            out[i] = truncate_completion(
                tok.decode(gen[j][cut:], skip_special_tokens=True)
            )
        if b % 10 == 0:
            done = start + len(idx)
            rate = done / max(1e-9, time.time() - t0)
            print(f"  {label} {done}/{len(prompts)}  {rate:.1f}/s", flush=True)
    return out


def sample_solutions(gen_fn, fewshot, rows):
    prompts = [generation_prompt(fewshot, r["question"]) for r in rows]
    texts = gen_fn(prompts, "solve")
    sols = []
    for r, text in zip(rows, texts, strict=True):
        steps = split_steps(text)
        sols.append(
            GeneratedSolution(
                example_id=r["example_id"],
                question=r["question"],
                gold=r["gold"],
                steps=steps,
                answer=extract_answer(text),
            )
        )
    return sols


# ---- global labels by rollout ----------------------------------------


def label_globally(gen_fn, fewshot, sols, k):
    """Math-Shepherd hard estimation, per step.

    The terminal step is labelled by the solution's own answer rather than by a
    rollout. Its prefix already contains all the reasoning, so the "rollout"
    would only re-read the answer off the page -- and Math-Shepherd's own
    `final_correct` is defined as the last step's label, which this keeps true
    by construction.
    """
    tasks, meta = [], []
    for si, sol in enumerate(sols):
        for ti in range(len(sol.steps) - 1):  # terminal handled separately
            prompt = generation_prompt(fewshot, sol.question, sol.steps[: ti + 1])
            for _ in range(k):
                tasks.append(prompt)
                meta.append((si, ti))

    print(f"  {len(tasks):,} rollouts over {len(sols):,} solutions", flush=True)
    texts = gen_fn(tasks, "rollout")

    reached: dict[tuple[int, int], bool] = {}
    for (si, ti), text in zip(meta, texts, strict=True):
        hit = answers_match(extract_answer(text), sols[si].gold)
        reached[(si, ti)] = reached.get((si, ti), False) or hit

    labels = []
    for si, sol in enumerate(sols):
        row = [reached.get((si, ti), False) for ti in range(len(sol.steps) - 1)]
        row.append(sol.final_correct)
        labels.append(row)
    return labels


# ---- stub generator, for proving the harness without a GPU -----------


class StubGenerator:
    """Replays GSM8K gold solutions, corrupting a known fraction of steps.

    Not a model and not pretending to be one. Its purpose is that every stage
    downstream of the model call -- prompt assembly, truncation, step
    splitting, answer extraction, rollout accounting, serialisation, the
    Math-Shepherd round trip, the analysis script -- can be exercised on CPU
    with a KNOWN answer before a Kaggle session is spent on it. The ch. 5 run
    burned three sessions on faults that were all visible without a GPU.

    Corruption propagates: once a prefix contains a corrupted value, every
    continuation from it reaches the wrong final answer. That is the behaviour
    the global labels are supposed to detect, so a dry run that did not have it
    would leave the rollout logic untested.
    """

    def __init__(self, rows, corrupt_p=0.15, seed=0):
        self.gold = {r["question"].strip(): r for r in rows}
        self.p = corrupt_p
        self.rng = random.Random(seed)

    @staticmethod
    def _split_prompt(prompt):
        """Recover (question, prefix) from a rendered prompt."""
        block = prompt.rsplit("Question:", 1)[-1]
        question, _, tail = block.partition("\nAnswer:")
        return question.strip(), tail.strip()

    def _corrupt(self, line):
        m = _CALC_LINE.search(line)
        if not m:
            return line, False
        try:
            wrong = f"{float(m.group(2)) + 1:g}"
        except ValueError:
            return line, False
        return line.replace(f"={m.group(2)}>>", f"={wrong}>>"), True

    def __call__(self, prompts, label=""):
        out = []
        for prompt in prompts:
            question, prefix = self._split_prompt(prompt)
            row = self.gold.get(question)
            if row is None:
                out.append("#### 0")
                continue
            lines = split_steps(row["solution"])
            done = len([x for x in prefix.splitlines() if x.strip()])
            # A prefix that no longer matches the gold text carries a corrupted
            # premise, so nothing downstream of it can reach the right answer.
            poisoned = prefix != "" and prefix != "\n".join(lines[:done])
            body = []
            for line in lines[done:]:
                if not poisoned and self.rng.random() < self.p:
                    line, hit = self._corrupt(line)
                    poisoned = poisoned or hit
                body.append(line)
            answer = row["gold"]
            if poisoned:
                answer = f"{float(answer) + 1:g}"
            out.append("\n".join(body + [f"#### {answer}"]))
        print(f"  {label} {len(out)} stub completions", flush=True)
        return out


_CALC_LINE = re.compile(r"<<([^>]*?)=([^>]*?)>>")


# ---- gates -----------------------------------------------------------


def check_gates(sols, strict=True):
    acc = sum(1 for s in sols if s.final_correct) / max(1, len(sols))
    ann = annotation_rate(sols)
    empty = sum(1 for s in sols if not s.steps) / max(1, len(sols))

    print(f"  solve rate       {acc:.4f}   band {ACCURACY_BAND}", flush=True)
    print(f"  annotation rate  {ann:.4f}   floor {MIN_ANNOTATION_RATE}", flush=True)
    print(f"  empty solutions  {empty:.4f}", flush=True)

    problems = []
    if not ACCURACY_BAND[0] <= acc <= ACCURACY_BAND[1]:
        problems.append(
            f"solve rate {acc:.4f} outside {ACCURACY_BAND}; the few-shot prompt "
            f"is probably not being followed"
        )
    if ann < MIN_ANNOTATION_RATE:
        problems.append(
            f"annotation rate {ann:.4f} below {MIN_ANNOTATION_RATE}; the local "
            f"error rate would be measured on a biased subset of steps"
        )
    if problems and strict:
        raise RuntimeError("; ".join(problems))
    for p in problems:
        print(f"  GATE FAILED: {p}", flush=True)
    return {"accuracy": acc, "annotation_rate": ann, "empty_rate": empty,
            "passed": not problems}


def round_trip(records):
    """Re-parse what we wrote and confirm it survives the Shepherd parser.

    The output is only useful because downstream code reads it as Math-Shepherd
    data. If a step text happens to break that parser, every rate computed from
    the file is silently wrong, so this checks rather than assumes.
    """
    from car.data.math_shepherd import parse_solution

    bad = 0
    for rec in records:
        sol = parse_solution(rec["label"])
        if sol is None or len(sol.steps) != rec["_n_steps"]:
            bad += 1
    print(f"  round-trip: {len(records) - bad}/{len(records)} parse back exactly",
          flush=True)
    if bad:
        raise RuntimeError(
            f"{bad} solutions do not survive the Math-Shepherd parser; "
            f"downstream rates would be computed on mangled steps"
        )


# ---- driver ----------------------------------------------------------


def load_rows(path, n, seed=0):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for i, r in enumerate(rows):
        gold = extract_answer(r["answer"])
        if gold is None:
            continue
        out.append({"example_id": f"gsm8k_test_{i}", "question": r["question"],
                    "gold": gold, "solution": r["answer"]})
    random.Random(seed).shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="qwen25_7b")
    ap.add_argument("--n", type=int, default=400, help="test problems")
    ap.add_argument("--rollouts", type=int, default=4, help="K per step")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gate", action="store_true",
                    help="record gate failures instead of aborting")
    ap.add_argument("--dry-run", action="store_true",
                    help="replay gold solutions instead of calling a model; "
                         "proves the harness end to end on CPU")
    ap.add_argument("--corrupt-p", type=float, default=0.15,
                    help="dry-run only: per-step corruption probability")
    args = ap.parse_args()

    for p in (TRAIN, TEST):
        if not p.exists():
            print(f"missing {p}; run python scripts/download_data.py gsm8k")
            return 1

    train = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    fewshot = build_fewshot(train, k=args.shots, seed=args.seed)
    rows = load_rows(TEST, args.n, seed=args.seed)
    print(f"{len(rows)} test problems, {args.shots}-shot prompt "
          f"({len(fewshot)} chars)\n", flush=True)

    if args.dry_run:
        print(f"DRY RUN: replaying gold solutions, corrupt_p={args.corrupt_p}\n",
              flush=True)
        gen_fn = StubGenerator(rows, corrupt_p=args.corrupt_p, seed=args.seed)
    else:
        print(f"loading {args.model}...", flush=True)
        model, tok = load_model(args.model)

        def gen_fn(prompts, label=""):
            return generate(model, tok, prompts, temperature=args.temperature,
                            batch_size=args.batch_size, label=label)

    print("\nsampling solutions...", flush=True)
    sols = sample_solutions(gen_fn, fewshot, rows)

    print("\ngates:", flush=True)
    gates = check_gates(sols, strict=not args.no_gate and not args.dry_run)

    print("\nlabelling globally by rollout...", flush=True)
    labels = label_globally(gen_fn, fewshot, sols, args.rollouts)

    records = []
    for sol, lab in zip(sols, labels, strict=True):
        if not sol.steps:
            continue
        rec = to_shepherd_record(sol.question, sol.steps, lab, sol.answer)
        rec["_n_steps"] = len(sol.steps)
        records.append(rec)

    print("\nverifying output:", flush=True)
    round_trip(records)

    OUT.mkdir(exist_ok=True)
    path = OUT / f"generated_{args.tag}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            rec.pop("_n_steps")
            fh.write(json.dumps(rec) + "\n")

    meta = {
        "model": "STUB (dry run)" if args.dry_run else args.model,
        "tag": args.tag, "n_problems": len(rows),
        "rollouts": args.rollouts, "temperature": args.temperature,
        "shots": args.shots, "seed": args.seed,
        "n_solutions": len(records),
        "n_steps": sum(len(s.steps) for s in sols),
        "gates": gates,
        "local_validity_definition": "<<expr=result>> only, as Math-Shepherd",
    }
    (OUT / f"generated_{args.tag}_meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {path} ({len(records):,} solutions)", flush=True)
    print(f"analyse with: python scripts/exp_measure_error_rate.py --data {path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
