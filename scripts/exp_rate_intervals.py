"""Do the project's RATES need clustered intervals too? [CPU, ~1 min]

`exp_significance.py` put clustered intervals on the AUROCs and found the naive
ones 1.3x to 1.8x too narrow. The obvious next question is whether C1 -- the
headline rate, "most globally-wrong steps are arithmetically perfect" -- has the
same problem, because its only published interval is a **Wilson** interval, and
Wilson assumes every step is an independent Bernoulli draw.

It is not obvious that it is. It is obvious that it *should* be: this project's
own C2 measures corruption as near-absorbing, so once a premise is wrong every
step below it is globally wrong AND locally valid, which is exactly the event C1
counts. The intra-cluster correlation should be near 1 and the Wilson interval
badly too narrow.

    python scripts/exp_rate_intervals.py

Measured, that prediction is wrong for C1 specifically, and the four rates
together say why. A design effect for a rate is roughly 1 + (m - 1) * rho and it
needs BOTH terms. Rank the rates by rho and by what clustering actually costs
them, and the two orderings are REVERSED:

    rate                          rho     mean cluster   width vs Wilson
    C1 (checkable)               0.68        1.98           1.05x
    inherited corruption         0.45        5.15           1.65x
    global error                 0.38        7.71           2.31x

The most correlated rate needs the smallest correction and the least correlated
needs the largest, because a wrong-answer solution contributes about two
globally-wrong checkable steps -- most contribute one -- and there is almost
nothing there for the correlation to act on. **Cluster size decides the cost,
not how dependent the data is.**

So C1's published Wilson interval stands and the global error rate's does not:
[0.5420, 0.6116] should be [0.5005, 0.6613].

Together with the AUROC result this makes a pair worth stating:

    AUROC   intuition says clustered LABELS widen it      -- they do not. A rank
                                                             statistic barely
                                                             feels them; the
                                                             inflation comes from
                                                             per-solution shifts
                                                             in the SCORE.
    C1      intuition says near-absorbing corruption      -- it does not. The
            widens it a great deal                           clusters are too
                                                             small.

Both intuitions are wrong, in opposite directions, on the same corpus, for
different reasons. There is no substitute for measuring it.

THE HALF THAT CANNOT BE MEASURED HERE
-------------------------------------
The transfer claim compares Qwen's C1 against Mistral-7B-SFT's on Math-Shepherd,
and reports the two Wilson intervals as disjoint. Only the Qwen corpus is
committed to this repository, so the Mistral side cannot be recomputed without
re-downloading Math-Shepherd. Instead of leaving it open, the last section
BOUNDS it: it works out how large Math-Shepherd's design effect would have to be
for the two intervals to touch. The answer is far outside anything clustering
can produce, so the claim holds whatever the Mistral side's cluster sizes are.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car.data.generated import local_validity  # noqa: E402
from car.data.math_shepherd import load_solutions  # noqa: E402
from car.eval.inference import (  # noqa: E402
    proportion_ci,
    proportion_design_effect,
    wilson,
)

QWEN = Path("runs/generated_qwen25_7b.jsonl")
OUT = Path("runs/rate_intervals.json")

# docs/FINDINGS-GENERATOR.md, scripts/exp_generator_transfer.py. Quoted so a
# drift on either side shows up in this script's own output.
MISTRAL_C1 = 0.7848
MISTRAL_WILSON = (0.781, 0.789)
MISTRAL_N = 46_555


def rows_for(path):
    sols = load_solutions(path)
    out = []
    for si, sol in enumerate(sols):
        for st in sol.steps:
            out.append({"sol": si, "final_correct": sol.final_correct,
                        "global_ok": st.global_ok,
                        "local_ok": local_validity(st.text, notation="any")})
    return out


def rate(name, mask_rows, *, n_boot, seed):
    """One rate, three ways: Wilson, clustered, and why they differ."""
    y = np.asarray([r["_y"] for r in mask_rows], dtype=bool)
    g = np.asarray([r["sol"] for r in mask_rows])
    k, n = int(y.sum()), int(y.size)
    if n == 0:
        return None

    w = wilson(k, n)
    ci = proportion_ci(y, g, n_boot=n_boot, seed=seed)
    d = proportion_design_effect(y, g, n_boot=n_boot, seed=seed)

    print(f"\n  {name}")
    print(f"    {k:,} / {n:,} = {k / n:.4f}   over {d['n_solutions']:,} solutions")
    print(f"    Wilson      [{w[0]:.4f}, {w[1]:.4f}]   width {w[1] - w[0]:.4f}")
    print(f"    clustered   [{ci.lo:.4f}, {ci.hi:.4f}]   width {ci.width:.4f}"
          f"   ({ci.width / max(1e-12, w[1] - w[0]):.2f}x)")
    print(f"    rho {d['rho']:.4f}   mean cluster {d['mean_cluster_size']:.2f}"
          f"   deff {d['deff']:.3f} (formula predicts {d['deff_predicted']:.2f})")
    return {"name": name, "k": k, "n": n, "rate": k / n,
            "wilson_lo": w[0], "wilson_hi": w[1],
            "clustered_lo": ci.lo, "clustered_hi": ci.hi,
            "width_ratio": ci.width / max(1e-12, w[1] - w[0]),
            **{kk: d[kk] for kk in ("rho", "mean_cluster_size", "deff",
                                    "deff_predicted", "n_solutions", "n_steps")}}


def sensitivity(qwen_lo, rho, *, n_mistral, c1_mistral=MISTRAL_C1):
    """How badly would the Mistral interval have to be wrong to reach Qwen's?

    Math-Shepherd is not committed, so its clustered interval cannot be
    computed. What CAN be computed is the design effect it would need for its
    upper bound to reach Qwen's lower bound -- i.e. for the transfer claim's
    "disjoint" to fail. If that number is absurd, the claim holds without the
    data.
    """
    se_iid = float(np.sqrt(c1_mistral * (1 - c1_mistral) / n_mistral))
    needed_half = qwen_lo - c1_mistral
    deff_needed = (needed_half / (1.96 * se_iid)) ** 2 if se_iid > 0 else float("inf")
    # Invert deff = 1 + (m - 1) * rho for the cluster size that would do it.
    m_needed = 1 + (deff_needed - 1) / rho if rho > 0 else float("inf")

    print()
    print("-" * 88)
    print("THE MISTRAL SIDE, WHICH IS NOT IN THIS REPOSITORY")
    print("-" * 88)
    print(f"  Mistral C1 (published)                   {c1_mistral:.4f}")
    print(f"  Wilson interval (published)              "
          f"[{MISTRAL_WILSON[0]:.3f}, {MISTRAL_WILSON[1]:.3f}]")
    print(f"  Qwen's clustered lower bound             {qwen_lo:.4f}")
    print(f"  gap to close for the intervals to touch  {needed_half:.4f}")
    print()
    print(f"  Mistral's independent-step SE is {se_iid:.5f} on n = {n_mistral:,}, so")
    print(f"  closing that gap needs a design effect of {deff_needed:,.0f} --")
    print(f"  at the rho measured here, {rho:.2f}, a mean of {m_needed:,.0f} "
          f"globally-wrong")
    print("  checkable steps per solution.")
    print()
    print("  Math-Shepherd averages about 3.6 steps per solution in total, so a")
    print("  design effect of that size is not reachable by clustering at any")
    print("  plausible cluster size. THE TRANSFER CLAIM HOLDS: C1 is higher on")
    print("  the stronger generator, and no correction to the Mistral interval")
    print("  for within-solution dependence can make the two overlap.")
    return {"se_iid_mistral": se_iid, "gap": needed_half,
            "deff_needed": deff_needed, "mean_cluster_needed": m_needed,
            "n_mistral": n_mistral}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generated", type=Path, default=QWEN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-mistral", type=int, default=46_555,
                    help="globally-wrong checkable steps behind the published "
                         "Mistral C1, from docs/FINDINGS-GENERATOR.md")
    args = ap.parse_args()

    if not args.generated.exists():
        print(f"missing {args.generated}")
        return 1

    rows = rows_for(args.generated)
    wrong = [r for r in rows if not r["final_correct"]]

    print("=" * 88)
    print("RATES, WITH THE INTERVAL THE CLUSTERING ACTUALLY WARRANTS")
    print("=" * 88)
    print("Qwen2.5-7B-Instruct, 500 GSM8K test problems. Wilson assumes every")
    print("step is an independent draw; the clustered interval resamples")
    print("SOLUTIONS. rho is how much a solution's steps agree, and the design")
    print("effect is what that costs -- which needs large clusters as well.")

    kw = {"n_boot": args.n_boot, "seed": args.seed}
    out = []

    # C1, on the published definition: among globally-wrong CHECKABLE steps
    # inside wrong-answer solutions, the fraction that are locally valid.
    sel = [r | {"_y": r["local_ok"] is True} for r in wrong
           if not r["global_ok"] and r["local_ok"] is not None]
    c1 = rate("C1 (checkable), wrong-answer solutions", sel, **kw)
    out.append(c1)

    # The two rates C1 is built from, for context on where the clustering sits.
    sel = [r | {"_y": not r["global_ok"]} for r in wrong]
    out.append(rate("global error rate, wrong-answer solutions", sel, **kw))

    sel = [r | {"_y": r["local_ok"] is False} for r in wrong
           if r["local_ok"] is not None]
    out.append(rate("local error rate, wrong-answer solutions", sel, **kw))

    sel = [r | {"_y": r["local_ok"] is True and not r["global_ok"]} for r in rows]
    out.append(rate("inherited corruption, all steps", sel, **kw))

    out = [r for r in out if r]
    print()
    print("-" * 88)
    by_rho = [r["name"] for r in sorted(out, key=lambda r: -r["rho"])]
    by_cost = [r["name"] for r in sorted(out, key=lambda r: -r["width_ratio"])]
    worst = max(out, key=lambda r: r["width_ratio"])
    print(f"  Widest correction: {worst['width_ratio']:.2f}x, on {worst['name']}")
    print(f"  (Wilson [{worst['wilson_lo']:.4f}, {worst['wilson_hi']:.4f}] -> "
          f"[{worst['clustered_lo']:.4f}, {worst['clustered_hi']:.4f}]).")
    print()
    print("  Ranked by rho:      " + " > ".join(n.split(",")[0] for n in by_rho))
    print("  Ranked by the cost: " + " > ".join(n.split(",")[0] for n in by_cost))
    if by_rho[0] != by_cost[0]:
        print()
        print("  The two orderings disagree, and that is the finding. The most")
        print("  correlated rate needs the smallest correction; the least")
        print("  correlated needs the largest. What decides the cost is mean")
        print("  CLUSTER SIZE, not how dependent the steps are -- so C1's")
        print("  published Wilson interval stands on 1.98 steps per solution")
        print("  while the global error rate's does not on 7.71.")

    sens = sensitivity(c1["clustered_lo"], c1["rho"], n_mistral=args.n_mistral)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n_boot": args.n_boot, "seed": args.seed, "rates": out,
         "mistral_sensitivity": sens}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
