"""Every headline AUROC in this thesis, with an interval around it. [CPU, ~3 min]

Chapters 5-7 argue from differences between AUROCs measured on one corpus of 500
solutions, and until now every one of them was a bare point estimate:

    0.5589   token-level composite          "does not rank global step error"
    0.5740   semantic divergence, numeric   "refuted"
    0.4904   semantic divergence, exact     "below chance"
    0.5742   both signals combined          "combining buys 0.0002"
    0.6968   probe on hidden states         "+0.12 over the best measured signal"

Four of those five sentences are claims about a *difference*, and two of them --
"buys 0.0002" and "does not rank" -- are claims that a difference is ABSENT.
Those are the ones that need an interval most, because a point estimate can be
small either because the effect is small or because the corpus is.

    python scripts/exp_significance.py

Every score is evaluated on the same steps, so every comparison is paired, and
the resampling unit is the solution. What that buys, and what it does not, is in
`car.eval.inference`.

RESOLUTION
----------
The last section measures something the point estimates hide. Semantic
divergence over K=5 samples is not a continuous score: its value depends only on
the block sizes of the sample partition, so it can take exactly seven values (the
partitions of 5), and most steps sit on one of two of them. Comparing its AUROC
to the probe's therefore confounds how much signal a score carries with how
finely it can express one. Quantising the probe onto divergence's own grid
separates the two and prices the resolution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car.data.splits import make_splits  # noqa: E402
from car.eval.inference import (  # noqa: E402
    auroc_ci,
    auroc_resolution_ceiling,
    bootstrap_p_value,
    design_effect,
    minimum_detectable_delta,
    paired_auroc_delta_ci,
)
from car.generation.replay import load_corpus  # noqa: E402
from car.uncertainty.composite import CompositeScorer  # noqa: E402

CORPUS = Path("runs/generated_qwen25_7b.jsonl")
SEM = Path("runs/uncertainty_qwen25_7b_sem.jsonl")
EXACT = Path("runs/uncertainty_qwen25_7b_exactmatch.jsonl")
ENTAIL = Path("runs/uncertainty_qwen25_7b_entail.jsonl")
PROBE = Path("runs/probe_qwen25_7b.json")
OUT = Path("runs/significance.json")

# Chapter 7's three weight sets, verbatim.
TOKEN = {"token_entropy": 1.0, "max_surprisal": 0.5, "mean_logprob": 1.0}
DIVERGENCE = {"semantic_divergence": 1.0}
BOTH = {**TOKEN, **DIVERGENCE}

# (label, baseline, contender). Each is a sentence in the thesis.
COMPARISONS = [
    ("numeric vs exact match (ch. 7: relation-dependent)",
     "semantic divergence (exact match)", "semantic divergence (numeric)"),
    ("both vs token-level (ch. 7: changed AUROC by 0.015)",
     "token-level composite", "both signals"),
    ("both vs divergence alone (ch. 7: buys 0.0002)",
     "semantic divergence (numeric)", "both signals"),
    ("probe vs best measured signal (ch. 7: +0.12)",
     "both signals", "probe on hidden states"),
    ("probe vs token-level", "token-level composite", "probe on hidden states"),
]


def build(corpus_path, sem_path, exact_path, entail_path, probe_path):
    """Every score as one array over the corpus's steps, in corpus order."""
    numeric = load_corpus(corpus_path, sem_path, prefix="qwen")
    exact = load_corpus(corpus_path, exact_path, prefix="qwen")
    by_id = {ex.example_id: ex for ex in numeric}

    splits = make_splits([ex.to_example() for ex in numeric], dev_frac=0.3,
                         cal_frac=0.3, salt="car-v1")
    test_ids = {ex.example_id for ex in splits.test}
    dev_features = [st.features for ex in splits.dev
                    for st in by_id[ex.example_id].steps]

    scorers = {name: CompositeScorer(w).fit(dev_features)
               for name, w in (("token-level composite", TOKEN),
                               ("both signals", BOTH))}

    cols: dict[str, list[float]] = {k: [] for k in
                                    ("token-level composite", "both signals",
                                     "semantic divergence (numeric)",
                                     "semantic divergence (exact match)")}
    correct, groups, in_test = [], [], []
    for gi, (ex, ex_x) in enumerate(zip(numeric, exact, strict=True)):
        assert ex.example_id == ex_x.example_id
        for st, st_x in zip(ex.steps, ex_x.steps, strict=True):
            cols["token-level composite"].append(
                scorers["token-level composite"].score(st.features))
            cols["both signals"].append(scorers["both signals"].score(st.features))
            cols["semantic divergence (numeric)"].append(
                st.features.semantic_divergence)
            cols["semantic divergence (exact match)"].append(
                st_x.features.semantic_divergence)
            correct.append(st.global_ok)
            groups.append(gi)
            in_test.append(ex.example_id in test_ids)

    if entail_path.exists():
        ent = load_corpus(corpus_path, entail_path, prefix="qwen")
        cols["semantic divergence (entailment)"] = [
            st.features.semantic_divergence for ex in ent for st in ex.steps]

    n = len(correct)
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        # The probe script walks the same corpus in the same order and writes one
        # score per step, so position is the join key. Asserted rather than
        # trusted: a silent off-by-one here would move every number below.
        if len(probe["scores"]) != n:
            raise SystemExit(f"probe has {len(probe['scores'])} scores, "
                             f"corpus has {n} steps")
        split_of_step = np.asarray(probe["split_of_step"])
        mismatch = int((np.asarray(in_test) != (split_of_step == "test")).sum())
        if mismatch:
            raise SystemExit(f"probe's test mask disagrees with this one on "
                             f"{mismatch} steps; the join is not positional")
        cols["probe on hidden states"] = probe["scores"]

    return ({k: np.asarray(v, dtype=float) for k, v in cols.items()},
            np.asarray(correct, dtype=bool), np.asarray(groups),
            np.asarray(in_test, dtype=bool))


#: Fitted on dev steps, so an all-steps AUROC for it is partly in-sample -- it
#: comes out at 0.8367 against 0.6968 on test, which is the winner's curse
#: chapter 7 already quotes, not a better measurement. Held out of that table
#: rather than printed with a caveat nobody reads.
IN_SAMPLE_OUTSIDE_TEST = {"probe on hidden states"}


def table(cols, correct, groups, mask, *, label, n_boot, seed, quoted,
          skip=()):
    print()
    print("=" * 100)
    print(f"{label}:  {int(mask.sum()):,} steps in {len(set(groups[mask])):,} "
          f"solutions, wrong-step rate {1 - correct[mask].mean():.4f}")
    print("=" * 100)
    print(f"{'score':<36}{'thesis':>9}{'AUROC':>9}{'95% CI (solution bootstrap)':>31}"
          f"{'deff':>8}{'vs chance':>10}")
    print("-" * 100)

    rows = {}
    for name, s in cols.items():
        if name in skip:
            continue
        ci = auroc_ci(s[mask], correct[mask], groups[mask], n_boot=n_boot, seed=seed)
        d = design_effect(s[mask], correct[mask], groups[mask],
                          n_boot=n_boot, seed=seed)
        verdict = "above" if ci.lo > 0.5 else ("below" if ci.hi < 0.5 else "—")
        print(f"{name:<36}{quoted.get(name, ''):>9}{ci.point:>9.4f}"
              f"{f'[{ci.lo:.4f}, {ci.hi:.4f}]':>31}{d['deff']:>8.2f}{verdict:>10}")
        rows[name] = {"auroc": ci.point, "lo": ci.lo, "hi": ci.hi, "se": ci.se,
                      "deff": d["deff"], "se_iid_steps": d["se_iid_steps"],
                      "excludes_chance": ci.excludes(0.5)}
    print()
    print("  deff: how much a step-level bootstrap would understate the variance.")
    print("  'vs chance': whether the whole interval clears 0.5.")
    return rows


def deltas(cols, correct, groups, mask, *, n_boot, seed):
    print()
    print("-" * 100)
    print("PAIRED DIFFERENCES — same steps, same resamples, so the shared "
          "corpus variability cancels")
    print("-" * 100)
    print(f"{'comparison':<52}{'delta':>9}{'95% CI':>27}{'p':>8}")
    print("-" * 100)

    out = []
    for label, base, cont in COMPARISONS:
        if base not in cols or cont not in cols:
            continue
        d = paired_auroc_delta_ci(cols[base][mask], cols[cont][mask],
                                  correct[mask], groups[mask],
                                  n_boot=n_boot, seed=seed)
        p = bootstrap_p_value(d)
        print(f"{label:<52}{d.point:>+9.4f}{f'[{d.lo:+.4f}, {d.hi:+.4f}]':>27}"
              f"{p:>8.3f}")
        out.append({"comparison": label, "baseline": base, "contender": cont,
                    "delta": d.point, "lo": d.lo, "hi": d.hi, "se": d.se,
                    "p": p, "excludes_zero": d.excludes(0.0)})
    return out


def resolution(cols, correct, mask):
    """What divergence's seven levels cost a score that has real signal."""
    grid = cols["semantic divergence (numeric)"][mask]
    levels = np.unique(grid)
    print()
    print("-" * 100)
    print("RESOLUTION — semantic divergence takes "
          f"{levels.size} distinct values on these steps")
    print("-" * 100)
    top = sorted(((int((grid == v).sum()), float(v)) for v in levels), reverse=True)
    share = ", ".join(f"{v + 0.0:.3f} ({n / grid.size:.0%})" for n, v in top[:4])
    print(f"  most common: {share}")
    print()
    print(f"{'score':<36}{'continuous':>12}{'on the grid':>14}{'cost':>9}")
    print("-" * 100)

    out = []
    for name in ("probe on hidden states", "token-level composite"):
        if name not in cols:
            continue
        r = auroc_resolution_ceiling(cols[name][mask], correct[mask], grid)
        print(f"{name:<36}{r['auroc_continuous']:>12.4f}"
              f"{r['auroc_on_grid']:>14.4f}{r['resolution_cost']:>9.4f}")
        out.append({"score": name} | r)
    print()
    print("  The score's ranking is held fixed and only its resolution is")
    print("  reduced to divergence's, so the cost is the price of the grid alone.")
    return {"n_levels": int(levels.size),
            "levels": [float(v) for v in levels],
            "level_share": {f"{v:.6f}": n / grid.size for n, v in top},
            "coarsened": out}


def power(diffs, test_rows):
    """What a fourth equivalence relation would have to do to be heard.

    The pending bidirectional-entailment run exists to answer one question: is
    chapter 7's negative result a property of sampling-based step uncertainty, or
    of `numeric_equivalence` standing in for the reference relation? The paired
    interval between the two relations that HAVE been measured prices that
    question before it is asked, because a third relation is measured against the
    same 925 steps with the same pairing and therefore the same SE.
    """
    row = next((d for d in diffs
                if d["baseline"] == "semantic divergence (exact match)"), None)
    if row is None:
        return {}
    mde = minimum_detectable_delta(row["se"])
    base = test_rows["semantic divergence (numeric)"]["auroc"]

    print()
    print("-" * 100)
    print("POWER — what a fourth relation would have to reach to change the answer")
    print("-" * 100)
    print(f"  paired SE between two relations on these steps   {row['se']:.4f}")
    print(f"  smallest detectable difference (80% power)       {mde:+.4f}")
    print(f"  so a new relation registers only above           "
          f"{base + mde:.4f} AUROC")
    print()
    print("  Chapter 7's score-quality sweep -- a separate experiment on")
    print("  synthetic scores -- puts the AUROC at which the verifier stops")
    print("  costing more than it recovers at ~0.65. That the bar above lands")
    print("  within 0.001 of it is a coincidence of two unrelated calculations,")
    print("  and a clarifying one: on this corpus a new relation can only be")
    print("  HEARD if it is already good enough to be USEFUL. A relation that")
    print("  does not clear it leaves the chapter's conclusion exactly where it")
    print("  is, whatever its point estimate.")
    return {"paired_se_between_relations": row["se"],
            "minimum_detectable_delta_80pct": mde,
            "auroc_a_new_relation_must_reach": base + mde,
            "verifier_crossing_auroc": 0.65}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--sem", type=Path, default=SEM)
    ap.add_argument("--exact", type=Path, default=EXACT)
    ap.add_argument("--entail", type=Path, default=ENTAIL)
    ap.add_argument("--probe", type=Path, default=PROBE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for p in (args.corpus, args.sem, args.exact):
        if not p.exists():
            print(f"missing {p}")
            return 1
    for p in (args.entail, args.probe):
        if not p.exists():
            print(f"note: {p} absent, its rows are skipped")

    cols, correct, groups, in_test = build(
        args.corpus, args.sem, args.exact, args.entail, args.probe)

    # The numbers chapter 7 prints, so a drift shows up in the table itself.
    quoted = {"token-level composite": "0.5589",
              "semantic divergence (numeric)": "0.5740",
              "both signals": "0.5742",
              "probe on hidden states": "0.6968"}

    kw = {"n_boot": args.n_boot, "seed": args.seed}
    test = table(cols, correct, groups, in_test, label="TEST SPLIT",
                 quoted=quoted, **kw)
    allsteps = table(cols, correct, groups, np.ones_like(in_test),
                     label="ALL STEPS (the probe is fitted on part of this, so "
                           "it is not here)", quoted={},
                     skip=IN_SAMPLE_OUTSIDE_TEST, **kw)
    diffs = deltas(cols, correct, groups, in_test, **kw)
    pwr = power(diffs, test)
    res = resolution(cols, correct, in_test)

    print()
    print("=" * 100)
    for row in diffs:
        held = "survives" if row["excludes_zero"] else "NOT established"
        print(f"  {held:<16} {row['comparison']}")
    print("=" * 100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n_boot": args.n_boot, "seed": args.seed, "unit": "solution",
         "n_steps": int(correct.size), "n_solutions": int(len(set(groups))),
         "test": test, "all_steps": allsteps, "paired": diffs,
         "power": pwr, "resolution": res}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
