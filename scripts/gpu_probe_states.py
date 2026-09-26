"""Train a probe on frozen internal states, and see if it ranks step error. [NEEDS GPU]

The one experiment Chapter 9 names as the test of its own negative result.
Everything in Chapter 7 is gated on AUROC 0.5742 -- neither token-level
uncertainty nor sampling-based semantic divergence ranks a globally-wrong step.
ReProbe (Ni et al., arXiv:2511.06209) reports that a sub-10M-parameter probe on
a frozen model's internal states matches PRMs up to 810x larger, which makes
this the cheapest way to find out whether the AUROC 0.56 result is about *these
signals* or about step-level uncertainty in general.

Chapter 9 commits to a prediction before the run, which is the point of making
it: ReProbe's margin is largest OUT of domain, and strong PRMs reach parity with
it on GSM8K. So a probe here should land near the ch. 5 PRM's 0.9033 rather than
above it -- good enough to calibrate on, and leaving 7.4's false-alarm problem
untouched.

WHAT IT DOES

One teacher-forced forward pass per solution, hidden states captured at each
step's final token for all 29 layers. Then a logistic probe per layer, trained
on dev-train and selected on dev-select -- never on test, and never on the
calibration split, which belongs to the conformal threshold.

Cost: the same forward pass gpu_score_uncertainty.py already does, so minutes.
The states are ~18 MB per layer in fp16 and stay in the session; only the
per-step scores and the metrics are written out.

RUNNING

    python scripts/gpu_probe_states.py \\
        --corpus runs/generated_qwen25_7b.jsonl \\
        --out runs/probe_qwen25_7b.json

Then feed the scores back through the real gate:

    python scripts/exp_gate_pipeline.py --probe runs/probe_qwen25_7b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data.generated import ANSWER_PREFIX, build_fewshot, generation_prompt  # noqa: E402
from car.data.math_shepherd import load_solutions  # noqa: E402
from car.data.splits import make_splits  # noqa: E402
from car.types import Example  # noqa: E402
from car.uncertainty.probe import (  # noqa: E402
    auroc,
    learning_curve,
    select_and_fit,
)

TRAIN = Path("data/raw/gsm8k/train.jsonl")
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Reported in ch. 7 on the same steps, for the comparison this run exists for.
BASELINE_AUROC = {"token-level": 0.5589, "semantic": 0.5740, "both": 0.5742}
PRM_SCOPE = 0.9033


def strip_answer_tail(text: str) -> str:
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
            "a fast tokenizer is required: step ends are located by character "
            "offset mapping"
        )
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        dtype = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", output_hidden_states=True
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto", output_hidden_states=True
        ).eval()
    print(f"  loaded {model_id}, dtype={dtype}", flush=True)
    return model, tok


def step_states(model, tok, prompt, steps, max_length=2048):
    """Hidden state at each step's FINAL token, every layer.

    The last token of a step is where the model has read the whole step and is
    about to commit to the next one, which is the position ReProbe reads and
    the only one at which "was that step sound" is a question the state could
    encode.
    """
    import torch

    parts, spans, cursor = [], [], 0
    for s in steps:
        if parts:
            cursor += 1
        parts.append(s)
        spans.append((cursor, cursor + len(s)))
        cursor += len(s)
    body = "\n".join(parts)

    enc = tok(prompt, add_special_tokens=True, return_tensors=None)
    n_prompt = len(enc["input_ids"])
    benc = tok(body, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"] + benc["input_ids"]
    offsets = benc["offset_mapping"]

    if len(ids) > max_length:
        drop = len(ids) - max_length
        ids, n_prompt = ids[drop:], n_prompt - drop
        if n_prompt < 1:
            return None

    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        hs = model(x, output_hidden_states=True).hidden_states  # tuple, n_layers+1

    out = []
    for lo, hi in spans:
        rows = [n_prompt + j for j, (a, b) in enumerate(offsets)
                if a < hi and b > lo and n_prompt + j < x.shape[1]]
        if not rows:
            out.append(None)
            continue
        last = rows[-1]
        out.append(np.stack([h[0, last].float().cpu().numpy() for h in hs]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path,
                    default=Path("runs/generated_qwen25_7b.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/probe_qwen25_7b.json"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-frac", type=float, default=0.3,
                    help="share of the DEV split held out to choose layer and C")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-states", type=Path, default=None,
                    help="write the hidden states, labels and per-step metadata "
                         "to an .npz. ~535 MB in fp16 for the 500-solution "
                         "corpus, and it makes every later probe question -- "
                         "more training data, a different target, a different "
                         "layer -- answerable without a GPU.")
    args = ap.parse_args()

    for p in (args.corpus, TRAIN):
        if not p.exists():
            print(f"missing {p}")
            return 1

    train_rows = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    fewshot = build_fewshot(train_rows, k=args.shots, seed=args.seed)
    sols = load_solutions(args.corpus)
    if args.limit:
        sols = sols[: args.limit]

    # Same hash splits as ch. 7, so the AUROCs are comparable step for step.
    examples = [Example(example_id=f"qwen_{i}", question=s.question, gold_answer="")
                for i, s in enumerate(sols)]
    splits = make_splits(examples, dev_frac=0.3, cal_frac=0.3, salt="car-v1")
    role = {}
    for name, group in (("dev", splits.dev), ("cal", splits.calibration),
                        ("test", splits.test)):
        for ex in group:
            role[ex.example_id] = name

    print(f"{len(sols)} solutions, {sum(len(s.steps) for s in sols)} steps", flush=True)
    model, tok = load_model(args.model)

    states, labels, which = [], [], []
    sol_id, position, n_in_sol = [], [], []
    for i, sol in enumerate(sols):
        texts = [s.text for s in sol.steps]
        texts[-1] = strip_answer_tail(texts[-1]) or texts[-1]
        got = step_states(model, tok, generation_prompt(fewshot, sol.question), texts)
        if got is None:
            continue
        kept = 0
        for j, (st, vec) in enumerate(zip(sol.steps, got, strict=True)):
            if vec is None:
                continue
            states.append(vec.astype(np.float32))
            labels.append(not st.global_ok)          # True = globally WRONG
            which.append(role[f"qwen_{i}"])
            sol_id.append(i)
            position.append(j)
            kept += 1
        n_in_sol.extend([kept] * kept)
        if i % 50 == 0:
            print(f"  {i}/{len(sols)}", flush=True)

    S = np.stack(states)                              # (n_steps, n_layers+1, hidden)
    y = np.asarray(labels, dtype=bool)
    which = np.asarray(which)
    sol_id = np.asarray(sol_id)
    position = np.asarray(position)
    n_in_sol = np.asarray(n_in_sol)

    # The FIRST globally-wrong step of each solution. Ch. 7.4 measured that this
    # is the only one a repair can rescue -- everything after it inherits
    # corruption a later fix does not undo -- so it is the target a step score
    # should arguably be trained against, and AUROC on `y` is not.
    # Computed over the KEPT steps so it matches exactly what the probe sees.
    first_bad = np.zeros_like(y)
    for s in np.unique(sol_id):
        rows = np.where(sol_id == s)[0]
        bad = rows[y[rows]]
        if len(bad):
            first_bad[bad[0]] = True

    print(f"\nstates {S.shape}, wrong-step rate {y.mean():.4f}", flush=True)
    print(f"first-bad steps {int(first_bad.sum())} "
          f"({first_bad.mean():.2%}) -- the scarcer target", flush=True)

    if args.save_states:
        # Last time only the scores were written out, which is why answering any
        # follow-up question needed another GPU session. fp16 halves the file and
        # costs nothing: the probe standardises before fitting anyway.
        args.save_states.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_states, states=S.astype(np.float16), global_wrong=y,
                 first_bad=first_bad, role=which, solution_id=sol_id,
                 position=position, n_steps_in_solution=n_in_sol)
        mb = args.save_states.stat().st_size / 1e6
        print(f"saved states -> {args.save_states} ({mb:.0f} MB); "
              f"every probe variant from here is CPU-only", flush=True)

    dev = np.where(which == "dev")[0]
    rng = np.random.default_rng(args.seed)
    dev = rng.permutation(dev)
    n_sel = max(1, int(len(dev) * args.select_frac))
    sel_idx, train_idx = dev[:n_sel], dev[n_sel:]
    test_idx = np.where(which == "test")[0]
    print(f"probe train {len(train_idx)} | select {len(sel_idx)} | "
          f"test {len(test_idx)}   (calibration untouched)", flush=True)

    by_layer = {L: S[:, L, :] for L in range(S.shape[1])}
    best = select_and_fit(by_layer, y, train_idx, sel_idx)
    test_auroc = auroc(best.model, best.scaler,
                       by_layer[best.layer][test_idx], y[test_idx])

    print("", flush=True)
    print("=" * 70, flush=True)
    print("PROBE RESULT", flush=True)
    print("=" * 70, flush=True)
    print(f"  selected layer {best.layer} of {S.shape[1] - 1}, C={best.C}", flush=True)
    print(f"  AUROC on the selection split   {best.auroc_select:.4f}", flush=True)
    print(f"  AUROC on TEST                  {test_auroc:.4f}", flush=True)
    print("", flush=True)
    for name, v in BASELINE_AUROC.items():
        print(f"  ch.7 baseline, {name:<14} {v:.4f}", flush=True)
    print(f"  ch.5 task PRM scope            {PRM_SCOPE:.4f}  "
          f"(ch. 9 predicted the probe lands near this, not above)", flush=True)

    print("\n  per-layer AUROC on the selection split:", flush=True)
    for L in sorted(best.per_layer):
        bar = "#" * int(max(0, best.per_layer[L] - 0.5) * 100)
        print(f"    layer {L:>2}  {best.per_layer[L]:.4f}  {bar}", flush=True)

    curve = learning_curve(by_layer[best.layer], y, train_idx, sel_idx, best.C,
                           seed=args.seed)
    print("\n  learning curve (is it starved, or saturated?):", flush=True)
    for n, a in curve:
        print(f"    n={n:>4}  AUROC {a:.4f}", flush=True)

    scores = best.score(by_layer[best.layer])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model,
        "layer": best.layer, "C": best.C, "n_layers": S.shape[1] - 1,
        "auroc_select": best.auroc_select, "auroc_test": float(test_auroc),
        "per_layer_select": {str(k): v for k, v in best.per_layer.items()},
        # NAMED for what it is: scored on the SELECTION split, the same data
        # the layer and C were chosen on. It is therefore biased and is NOT
        # evidence about training size -- a claim in ch. 7.6 rested on it
        # until round two caught that. exp_probe_variants.py produces the
        # held-out version.
        "learning_curve_on_selection_split": [[int(n), float(a)] for n, a in curve],
        "learning_curve": [[int(n), float(a)] for n, a in curve],
        "n_train": int(len(train_idx)), "n_select": int(len(sel_idx)),
        "n_test": int(len(test_idx)),
        "baselines": BASELINE_AUROC,
        "scores": [float(s) for s in scores],
        "split_of_step": which.tolist(),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
