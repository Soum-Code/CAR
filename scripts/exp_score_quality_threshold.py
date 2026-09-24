"""How good does the score have to be before the verifier is worth having? [CPU]

Chapter 7 leaves this open in two places and chapter 9 repeats it. The task PRM
detects 90.3% of globally-wrong steps and false-alarms on 9.87% of correct
ones. Behind the measured score (AUROC 0.5742) it is worth -4 points of
projected accuracy; behind the probe (0.6968) it is still negative; behind a
perfect score it is worth +17.6. The draft could only say the crossing is
"somewhere above 0.70 and unmeasured".

This measures it. `SyntheticAUROCScorer` makes score quality a dial under the
binormal model, and everything else in the pipeline is held fixed -- same
corpus, same splits, same split-conformal calibrator, same budget, same
verifier at its measured scope and false-alarm rate. The only thing that moves
between rows is how well the score ranks, so the crossing is attributable to
score quality and nothing else.

Three sweeps, because the interesting answer differs by question:

  A  alpha = 0.30, PRM at its measured 9.87% false alarm.
     Where does PROJECTED accuracy cross the no-gate baseline?
  B  alpha = 0.30, PRM with false alarms switched off.
     The ablation. If the crossing in A is driven by false alarms, B has no
     crossing to find -- the verifier is positive everywhere.
  C  alpha = 0.05, PRM at its measured false alarm.
     Does ANY score quality let the gate hold a binding alpha at this budget?
     Ch. 7 says no even for the oracle; this checks the whole range.

    python scripts/exp_score_quality_threshold.py
    python scripts/exp_score_quality_threshold.py --seeds 12   # tighter CIs

WHAT THIS IS NOT

The binormal draw is exchangeable across wrong steps: it is as likely to rank a
first-step error highly as a last-step one. A real score with a position bias
would land somewhere else, and ch. 6 measured that real step difficulty rises
with position. So the crossing here is where a *well-behaved* score of that
quality turns the verifier positive, and a real score of the same AUROC could
do worse. Reported as an estimate, not a threshold to design against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from car.baselines.policies import NeverVerify, QuantileGate  # noqa: E402
from car.conformal import SplitConformalCalibrator  # noqa: E402
from car.data.splits import assert_no_leakage, make_splits  # noqa: E402
from car.eval.metrics import step_detection_auroc  # noqa: E402
from car.generation.replay import attach_gold, load_corpus  # noqa: E402
from car.uncertainty.composite import CompositeScorer  # noqa: E402
from car.uncertainty.synthetic import SyntheticAUROCScorer  # noqa: E402
from car.verification.scoped import MEASURED_SCOPE, ScopedVerifier  # noqa: E402

# Imported rather than reimplemented: if this script ran the loop its own way,
# its rows would not be comparable to the ch. 7 tables, which is the entire
# point of the experiment.
from exp_gate_pipeline import CORPUS, FEATURES, WEIGHTS, run_condition  # noqa: E402

# Denser through 0.60-0.80, where the measured signals, the probe and the
# crossing all sit. Sparser above, where the shape is not in question.
GRID = (0.55, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.80,
        0.85, 0.90, 0.95, 0.99)
OUT = Path("runs/score_quality_threshold.json")

# Measured elsewhere in the project, printed alongside the sweep so the
# comparison is on the page rather than left to the reader.
REFERENCE = {
    "token + semantic (ch. 7)": (0.5742, 0.7912),
    "probe, layer 25 (sec. 7.6)": (0.6968, 0.7802),
    "oracle, binary (sec. 7.4)": (1.0000, 0.9780),
}


def sweep_point(auroc, *, seeds, by_id, splits, cal_y, labels, alpha, budget,
                scope, false_alarm, fixed_rate=None):
    """Run every seed at one score quality and return the per-seed rows.

    `fixed_rate` swaps the conformal calibrator for a raw quantile gate at that
    verification rate. Sweep A lets the calibrator choose the operating point,
    which is what a deployment would do -- but it means a better score also
    changes how many calls get made, so accuracy differences there are ranking
    and budget mixed together. Pinning the rate separates them.
    """
    rows = []
    for seed in range(seeds):
        scorer = SyntheticAUROCScorer(
            [by_id[k] for k in by_id], auroc, seed=seed)

        cal_s = np.asarray([
            scorer.score(st.features)
            for ex in splits.calibration for st in by_id[ex.example_id].steps
        ])
        test_s, test_y = [], []
        for ex in splits.test:
            for st in by_id[ex.example_id].steps:
                test_s.append(scorer.score(st.features))
                test_y.append(st.global_ok)

        calibrator = (
            QuantileGate(quantile=1.0 - fixed_rate).fit(cal_s)
            if fixed_rate is not None
            else SplitConformalCalibrator(alpha=alpha).fit(cal_s, cal_y)
        )
        verifier = ScopedVerifier(labels, scope=scope, false_alarm=false_alarm,
                                  label="task_prm", seed=seed)
        row = run_condition(
            f"auroc={auroc:.3f}", calibrator,
            by_id, splits.test, verifier, alpha=alpha, scorer=scorer,
            budget=budget, seed=seed)
        # The realised AUROC, not the target. They differ by sampling, and
        # conflating them would overstate how precisely the dial is set.
        row["auroc_measured"] = step_detection_auroc(
            np.asarray(test_s), np.asarray(test_y))
        row["auroc_target"] = auroc
        row["seed"] = seed
        rows.append(row)
    return rows


def position_correlation(scores_by_step, by_id, splits, *, wrong_only=False):
    """corr(normalised step position, score) on test steps.

    The diagnostic that explains sweep D. `SyntheticAUROCScorer` draws
    independently per step, so its correlation with position is zero by
    construction -- a real score's is not, and ch. 6 measured that real step
    difficulty RISES with position. A score that chases that difficulty ranks
    well on AUROC and badly on the projection, which only pays for the first
    bad step in a solution.
    """
    pos, val = [], []
    for ex in splits.test:
        steps = by_id[ex.example_id].steps
        for i, st in enumerate(steps):
            if wrong_only and st.global_ok:
                continue
            pos.append(i / max(1, len(steps) - 1))
            val.append(scores_by_step(st))
    if len(pos) < 3:
        return float("nan")
    return float(np.corrcoef(np.asarray(pos), np.asarray(val))[0, 1])


def summarise(rows):
    """Mean, sd and a 95% CI on the mean across seeds.

    The CI is what makes the crossing honest. Projected accuracy at adjacent
    grid points differs by less than the seed-to-seed spread, so a crossing
    read off the means alone would be quoting noise to three decimals.
    """
    keys = ("auroc_measured", "verification_rate", "calls_per_q",
            "selective_risk", "recall", "first_bad_recall",
            "projected_accuracy")
    out = {"auroc_target": rows[0]["auroc_target"], "n_seeds": len(rows)}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=float)
        out[k] = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        se = sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[k + "_sd"] = sd
        out[k + "_se"] = float(se)
        out[k + "_lo"] = float(np.mean(vals) - 1.96 * se)
        out[k + "_hi"] = float(np.mean(vals) + 1.96 * se)
    # Kept per-seed so a measured score can be placed as a percentile of the
    # synthetic distribution, which is a paired comparison: every draw runs on
    # the same test questions the measured score ran on.
    out["projected_accuracy_draws"] = [
        float(r["projected_accuracy"]) for r in rows]
    return out


def crossing(points, key, level):
    """Linearly interpolate where `key` first crosses `level` going up.

    Returns None when the sweep never crosses, which is itself a result -- it
    is the answer sweep C expects.
    """
    for lo, hi in zip(points, points[1:], strict=False):
        if lo[key] <= level < hi[key]:
            span = hi[key] - lo[key]
            if span <= 0:
                continue
            w = (level - lo[key]) / span
            return lo["auroc_target"] + w * (hi["auroc_target"] - lo["auroc_target"])
    return None


def crossing_interval(points, key, level):
    """Bracket the crossing by significance rather than by the mean.

    Returns (last AUROC whose whole CI is below `level`, first whose whole CI
    is above). Everything between is a score quality this experiment cannot
    tell apart from the baseline, and saying so is the result.
    """
    below = [p["auroc_target"] for p in points if p[key + "_hi"] < level]
    above = [p["auroc_target"] for p in points if p[key + "_lo"] > level]
    return (max(below) if below else None, min(above) if above else None)


def print_sweep(title, note, points, *, baseline=None, target=None):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)
    print(note)
    print()
    print(f"  {'AUROC':>7}{'measured':>10}{'verify%':>9}{'calls/q':>8}"
          f"{'sel.risk':>10}{'recall':>8}{'1st-bad':>9}{'PROJ acc':>10}"
          f"{'95% CI':>18}")
    print("  " + "-" * 98)
    for p in points:
        mark = ""
        if baseline is not None:
            if p["projected_accuracy_lo"] > baseline:
                mark = "  net +"
            elif p["projected_accuracy_hi"] < baseline:
                mark = "  net -"
            else:
                mark = "  ~same"
        if target is not None:
            mark = "  holds" if p["selective_risk_hi"] <= target else ""
        ci = (f"[{p['projected_accuracy_lo']:.4f},"
              f"{p['projected_accuracy_hi']:.4f}]")
        print(f"  {p['auroc_target']:>7.3f}{p['auroc_measured']:>10.4f}"
              f"{p['verification_rate']:>9.1%}{p['calls_per_q']:>8.2f}"
              f"{p['selective_risk']:>10.4f}{p['recall']:>8.4f}"
              f"{p['first_bad_recall']:>9.4f}"
              f"{p['projected_accuracy']:>10.4f}{ci:>18}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--gsm8k", type=Path, default=Path("data/raw/gsm8k/test.jsonl"))
    ap.add_argument("--alpha", type=float, default=0.30,
                    help="matches ch. 7 section 7.4, so the rows are comparable")
    ap.add_argument("--binding-alpha", type=float, default=0.05)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--match-rate", type=float, default=0.191,
                    help="verification rate for sweep D; the default is the "
                         "probe's measured rate, so the rows are comparable")
    ap.add_argument("--probe", type=Path,
                    default=Path("runs/probe_qwen25_7b.json"),
                    help="used only for the position diagnostic in sweep E")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    for p in (args.corpus, args.features):
        if not p.exists():
            print(f"missing {p}")
            return 1

    corpus = load_corpus(args.corpus, args.features, prefix="qwen")
    attach_gold(corpus, args.gsm8k)
    by_id = {ex.example_id: ex for ex in corpus}
    examples = [ex.to_example() for ex in corpus]
    splits = make_splits(examples, dev_frac=0.3, cal_frac=0.3, salt="car-v1")
    assert_no_leakage(splits)

    cal_y = np.asarray([
        st.global_ok for ex in splits.calibration
        for st in by_id[ex.example_id].steps
    ])
    labels = {
        (by_id[ex.example_id].question, i): st.global_ok
        for ex in splits.test
        for i, st in enumerate(by_id[ex.example_id].steps)
    }

    scope, fa = MEASURED_SCOPE["task_prm"]

    # Score accessors for the position diagnostic in sweep E. The composite
    # scaler is fitted on DEV only, exactly as the pipeline does it.
    comp_scorer = CompositeScorer(WEIGHTS).fit(
        [st.features for ex in splits.dev for st in by_id[ex.example_id].steps])
    def comp_pos(st):
        return comp_scorer.score(st.features)

    if args.probe.exists():
        probe_raw = json.loads(args.probe.read_text(encoding="utf-8"))["scores"]
        flat = [st for ex in corpus for st in ex.steps]
        probe_map = {id(st.features): s
                     for st, s in zip(flat, probe_raw, strict=True)}
        def probe_pos(st):
            return probe_map[id(st.features)]
    else:
        def probe_pos(st):
            return float("nan")

    n_first_bad = sum(
        1 for ex in splits.test
        if any(not st.global_ok for st in by_id[ex.example_id].steps))

    # The bar every row in sweep A has to clear. Taken from the pipeline itself
    # rather than quoted from the draft, so it cannot drift out of agreement.
    never = run_condition(
        "no gate", NeverVerify(), by_id, splits.test,
        ScopedVerifier(labels, scope=scope, false_alarm=fa, seed=0),
        alpha=args.alpha, scorer=SyntheticAUROCScorer(corpus, 0.5, seed=0),
        budget=args.budget, seed=0)
    baseline = never["projected_accuracy"]

    print("=" * 92)
    print("SCORE QUALITY SWEEP: where the task PRM stops being a liability")
    print("=" * 92)
    print(f"corpus    {args.corpus}   {len(corpus)} solutions")
    print(f"splits    {splits.summary()}")
    print(f"verifier  task PRM, scope={scope:.4f}, false alarm={fa:.4f}  "
          f"(measured, ch. 5)")
    print(f"budget    {args.budget} verification calls per question")
    print(f"seeds     {args.seeds} per grid point")
    print(f"baseline  projected accuracy with no gate at all: {baseline:.4f}")

    common = dict(seeds=args.seeds, by_id=by_id, splits=splits, cal_y=cal_y,
                  labels=labels, budget=args.budget, scope=scope)

    sweep_a = [summarise(sweep_point(a, alpha=args.alpha, false_alarm=fa, **common))
               for a in GRID]
    print_sweep(
        "A. alpha = 0.30, PRM at its measured 9.87% false-alarm rate",
        "PROJ acc is MODELLED, not measured. A row above the baseline is a\n"
        "score good enough to make this verifier worth calling.",
        sweep_a, baseline=baseline)
    x_a = crossing(sweep_a, "projected_accuracy", baseline)
    lo_a, hi_a = crossing_interval(sweep_a, "projected_accuracy", baseline)
    if x_a is None:
        print("\n  No crossing in the grid: the PRM is a net loss at every")
        print("  score quality tested, including 0.99.")
    else:
        print(f"\n  CROSSING at AUROC ~ {x_a:.3f} on the means.")
        print(f"  Significant at 95%: net-negative up to {lo_a}, net-positive "
              f"from {hi_a}.")
        print(f"  Between those the sweep cannot separate the gated system from")
        print(f"  the baseline, so the honest statement is an interval.")

    print()
    print("  Measured scores, for comparison (same corpus, budget, verifier):")
    print(f"    {'score':<28}{'AUROC':>8}{'PROJ acc':>11}{'sweep at that AUROC':>22}")
    for name, (a, acc) in REFERENCE.items():
        near = min(sweep_a, key=lambda p: abs(p["auroc_measured"] - a))
        fit = (f"{near['projected_accuracy']:.4f} @ {near['auroc_measured']:.3f}"
               if a < 0.99 else "-- off the grid --")
        print(f"    {name:<28}{a:>8.4f}{acc:>11.4f}{fit:>22}")

    sweep_b = [summarise(sweep_point(a, alpha=args.alpha, false_alarm=0.0, **common))
               for a in GRID]
    print_sweep(
        "B. ABLATION: same verifier, false alarms switched off",
        "Isolates the mechanism. If the crossing in A is about false alarms,\n"
        "there is nothing to cross here.",
        sweep_b, baseline=baseline)
    x_b = crossing(sweep_b, "projected_accuracy", baseline)
    print(f"\n  crossing: {'none -- positive throughout' if x_b is None else f'{x_b:.3f}'}")

    sweep_c = [summarise(sweep_point(a, alpha=args.binding_alpha, false_alarm=fa,
                                     **common))
               for a in GRID]
    print_sweep(
        f"C. alpha = {args.binding_alpha:.2f}, a target that actually binds",
        "Does any score quality let the gate hold it at this budget?",
        sweep_c, target=args.binding_alpha)
    held = [p for p in sweep_c if p["selective_risk_hi"] <= args.binding_alpha]
    if held:
        print(f"\n  Held from AUROC {held[0]['auroc_target']:.2f} upward.")
    else:
        best = min(sweep_c, key=lambda p: p["selective_risk"])
        print(f"\n  NEVER HELD. The best any score quality manages is "
              f"{best['selective_risk']:.4f} at AUROC "
              f"{best['auroc_target']:.2f} --")
        print(f"  {best['selective_risk'] / args.binding_alpha:.1f}x the target. "
              f"Score quality is not the binding constraint here; the")
        print(f"  budget is, exactly as the oracle row in section 7.3 says.")

    sweep_d = [summarise(sweep_point(a, alpha=args.alpha, false_alarm=fa,
                                     fixed_rate=args.match_rate, **common))
               for a in GRID]
    print_sweep(
        f"D. MATCHED BUDGET: verification pinned at {args.match_rate:.1%}, "
        f"the probe's rate",
        "Sweep A let the calibrator pick the operating point, so a better\n"
        "score changed both the ranking AND the number of calls. Here the rate\n"
        "is fixed, so only the ranking moves -- which is the comparison the\n"
        "probe row needs.",
        sweep_d, baseline=baseline)

    probe_auroc, probe_acc = REFERENCE["probe, layer 25 (sec. 7.6)"]
    near = min(sweep_d, key=lambda p: abs(p["auroc_measured"] - probe_auroc))
    draws = np.asarray(near["projected_accuracy_draws"])
    pct = float(np.mean(draws <= probe_acc))
    print()
    print("=" * 92)
    print("E. WHY AUROC IS NOT A SUFFICIENT DESCRIPTION OF A SCORE")
    print("=" * 92)
    print(f"  The probe reached projected accuracy {probe_acc:.4f} at AUROC "
          f"{probe_auroc:.4f}.")
    print(f"  A synthetic score of matched quality and budget reaches "
          f"{near['projected_accuracy']:.4f} "
          f"[{near['projected_accuracy_lo']:.4f},"
          f"{near['projected_accuracy_hi']:.4f}].")
    print(f"  The probe sits at the {pct:.1%} percentile of {len(draws)} "
          f"synthetic draws on the SAME")
    print(f"  {len(splits.test)} test questions, so the comparison is paired on "
          f"the question sample.")
    print()
    print(f"  overall recall     synthetic {near['recall']:.4f}   probe 0.2465"
          f"   -- the probe catches MORE")
    print(f"  first-bad recall   synthetic {near['first_bad_recall']:.4f}   "
          f"probe 0.3077   -- and rescues FEWER")
    print()
    print("  The mechanism, measured rather than inferred:")
    print(f"    corr(step position, probe score)      "
          f"{position_correlation(probe_pos, by_id, splits):+.4f}")
    print(f"    corr(step position, composite score)  "
          f"{position_correlation(comp_pos, by_id, splits):+.4f}")
    print(f"    same, among globally-wrong steps only: probe "
          f"{position_correlation(probe_pos, by_id, splits, wrong_only=True):+.4f}"
          f", composite "
          f"{position_correlation(comp_pos, by_id, splits, wrong_only=True):+.4f}")
    print()
    print("  The probe flags LATE steps; the composite flags EARLY ones. Ch. 6")
    print("  measured error rate rising with position, so chasing it is what")
    print("  earns the probe its AUROC -- and the projection only pays for the")
    print("  FIRST bad step, because everything after it inherits corruption a")
    print("  later repair does not undo. Two scores of equal AUROC are worth")
    print("  different amounts depending on WHICH errors they rank highly.")
    print()
    print(f"  Caveat: only {n_first_bad} of {len(splits.test)} test trajectories "
          f"contain a first bad step, so")
    print("  first-bad recall is a coarse statistic and single-condition")
    print("  differences in it are within noise. The sweep trend across 13 grid")
    print("  points is what carries the claim, not any one pair of rows.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "alpha": args.alpha,
        "binding_alpha": args.binding_alpha,
        "budget": args.budget,
        "seeds": args.seeds,
        "scope": scope,
        "false_alarm": fa,
        "baseline_projected_accuracy": baseline,
        "crossing_measured_fa": x_a,
        "crossing_ci_last_negative": lo_a,
        "crossing_ci_first_positive": hi_a,
        "crossing_no_fa": x_b,
        "match_rate": args.match_rate,
        "reference": {k: list(v) for k, v in REFERENCE.items()},
        "sweep_a": sweep_a,
        "sweep_b": sweep_b,
        "sweep_c": sweep_c,
        "sweep_d": sweep_d,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
