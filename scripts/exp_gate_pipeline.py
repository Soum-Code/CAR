"""The full CAR gate, end to end on real GSM8K output. [CPU]

Everything upstream of this has measured one piece at a time: the local/global
gap (ch. 4), verifier reach (ch. 5), allocation shape (ch. 6, in simulation).
This runs the actual control loop -- real generator uncertainty, real step
labels, real conformal calibration, real budget -- and asks the only question
the assembled system can answer:

    does calibrating a threshold on an uncertainty score actually bound the
    risk of the steps the gate lets through?

WHAT IS MEASURED AND WHAT IS MODELLED

The corpus is fixed, so the gate cannot change what the model writes. That
bounds the claims, and the boundary is where the honesty of this experiment
lives:

  MEASURED   selective risk on the accepted set -- of the steps the gate let
             through unverified, the fraction that are globally wrong. This is
             exactly the quantity conformal risk control promises to hold at
             alpha, so it is the end-to-end test that matters.
  MEASURED   verification rate, calls per question, budget-blocked steps,
             detection recall, score AUROC.
  MODELLED   final-answer accuracy. Requires assuming a caught error is
             repaired and downstream reasoning recovers. That is the
             propagation model's assumption, not an observation, and it is
             labelled PROJECTED wherever it appears.

VERIFIER REACH IS A PARAMETER, NOT A DETAIL

Ch. 5 found scope spanning 0.0000 to 0.9033 across verifier classes. A gate
paired with a low-scope verifier is not merely less useful: a missed detection
returns SUPPORTED, so the calibrator is fed a "no error here" label for a step
that has one, and converges on a threshold that is confident about a risk it
cannot see. Each condition is therefore run at each measured scope.

    python scripts/exp_gate_pipeline.py
    python scripts/exp_gate_pipeline.py --synthetic-signal 0.0   # null control

Requires runs/uncertainty_<tag>.jsonl from scripts/gpu_score_uncertainty.py.
Without it, use --synthetic-signal to exercise the loop on features whose
relationship to the labels is known by construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.agent.loop import CARAgent  # noqa: E402
from car.baselines.policies import (  # noqa: E402
    AlwaysVerify,
    NeverVerify,
    QuantileGate,
    RandomGate,
)
from car.conformal import AdaptiveCalibrator, SplitConformalCalibrator  # noqa: E402
from car.conformal.feasibility import min_verification_rate  # noqa: E402
from car.control.gate import ControlGate  # noqa: E402
from car.data.splits import assert_no_leakage, make_splits  # noqa: E402
from car.eval.metrics import (  # noqa: E402
    project_correct_after_repair,
    step_detection_auroc,
)
from car.generation.replay import (  # noqa: E402
    ReplayStepGenerator,
    answer_is_correct,
    attach_gold,
    load_corpus,
    score_final_answer,
)
from car.types import Decision, UncertaintyFeatures  # noqa: E402
from car.uncertainty.composite import CompositeScorer  # noqa: E402
from car.verification.scoped import MEASURED_SCOPE, ScopedVerifier  # noqa: E402

CORPUS = Path("runs/generated_qwen25_7b.jsonl")
FEATURES = Path("runs/uncertainty_qwen25_7b_sem.jsonl")

# semantic_divergence is now measured (scripts/gpu_semantic_divergence.py, K=5
# resampled continuations clustered by numeric equivalence), so it carries a
# weight. Zero it to reproduce the token-only condition; the two signals are
# nearly redundant (r = +0.44) and combining them moves test AUROC from 0.5740
# to 0.5742.
WEIGHTS = {
    "token_entropy": 1.0,
    "max_surprisal": 0.5,
    "mean_logprob": 1.0,
    "semantic_divergence": 1.0,
    "task_verifier_signal": 0.0,
}


# ---- synthetic features, for exercising the loop without a GPU -------------


def inject_synthetic(corpus, signal: float, seed: int = 0) -> None:
    """Overwrite features with draws whose relation to the labels is known.

    Mirrors MockBackend: a wrong step gets a higher mean, and `signal` is the
    separation in standard deviations. signal=0.0 is the null hypothesis --
    uncertainty is pure noise, and any apparent gain over a random gate at the
    same budget is an artifact of the harness rather than a finding.
    """
    rng = np.random.default_rng(seed)
    for ex in corpus:
        for st in ex.steps:
            shift = 0.0 if st.global_ok else signal
            st.features = UncertaintyFeatures(
                token_entropy=float(rng.normal(shift, 1.0)),
                max_surprisal=float(rng.normal(shift, 1.0)),
                mean_logprob=float(rng.normal(-shift, 1.0)),
            )


class ProbeScorer:
    """Scores from a trained probe on frozen internal states.

    Deployable, unlike the oracle: the probe reads the generator's own hidden
    states, not the label. Loaded from the JSON `gpu_probe_states.py` writes,
    keyed positionally -- record i of the score list is step i of the corpus in
    reading order -- so the length is asserted rather than assumed.
    """

    def __init__(self, corpus, path):
        import json

        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        flat = [st for ex in corpus for st in ex.steps]
        if len(blob["scores"]) != len(flat):
            raise ValueError(
                f"probe file has {len(blob['scores'])} scores but the corpus has "
                f"{len(flat)} steps; positional pairing would misalign every step"
            )
        self.meta = {k: blob[k] for k in ("layer", "C", "auroc_test", "auroc_select")}
        self._score = {id(st.features): s
                       for st, s in zip(flat, blob["scores"], strict=True)}

    def fit(self, dev_features):
        return self

    def score(self, feats) -> float:
        return float(self._score.get(id(feats), 0.0))


class OracleScorer:
    """A perfect uncertainty score. Not deployable; it reads the label.

    Chapter 3 promises this baseline and it is the only way to separate "the
    gate is bad" from "the task is hard at this budget". Every other component
    -- calibration, budget, verifier, the loop itself -- is unchanged, so the
    difference between this row and the real ones is attributable to the score
    and nothing else.

    Keyed on the identity of the feature object rather than its values, because
    `ReplayStepGenerator` hands back the same object each time and two distinct
    steps can carry numerically identical features.
    """

    def __init__(self, corpus):
        self._label = {
            id(st.features): st.global_ok for ex in corpus for st in ex.steps
        }

    def fit(self, dev_features):
        return self

    def score(self, feats) -> float:
        ok = self._label.get(id(feats))
        if ok is None:
            return 0.0
        return 0.0 if ok else 1.0


# ---- evaluation -----------------------------------------------------------


def evaluate(trajectories, alpha) -> dict:
    accepted_labels = []
    blocked = explored = 0
    caught = total_bad = 0
    first_caught = first_total = 0
    calls = 0

    for traj in trajectories:
        # `recall` counts any bad step caught, but the projection only rewards
        # catching the FIRST one -- everything after it inherits corruption
        # that repairing a later step does not undo. The two come apart when a
        # score's ranking correlates with step position, so both are reported.
        labelled = [r for r in traj.steps if r.label is not None]
        first_bad = next((r for r in labelled if r.label is False), None)
        if first_bad is not None:
            first_total += 1
            if first_bad.revised:
                first_caught += 1

        for r in traj.steps:
            if r.budget_blocked:
                blocked += 1
            if r.label is None:
                continue
            if not r.label:
                total_bad += 1
            if r.decision == Decision.VERIFY:
                calls += 1
                if r.revised and r.label is False:
                    caught += 1
                if r.forced_exploration:
                    explored += 1
            elif r.decision == Decision.CONTINUE:
                accepted_labels.append(r.label)

    n_steps = sum(len(t.steps) for t in trajectories)
    base = [t.correct for t in trajectories if t.correct is not None]
    proj = [p for p in (project_correct_after_repair(t) for t in trajectories)
            if p is not None]

    return {
        "n_steps": n_steps,
        "n_accepted": len(accepted_labels),
        # THE headline: risk among steps the gate let through unverified.
        "selective_risk": (
            float(np.mean([not x for x in accepted_labels])) if accepted_labels else float("nan")
        ),
        "target": alpha,
        "verification_rate": calls / max(1, n_steps),
        "calls_per_q": calls / max(1, len(trajectories)),
        "recall": caught / max(1, total_bad),
        "first_bad_recall": first_caught / max(1, first_total),
        "forced_explore": explored,
        "budget_blocked": blocked,
        "base_accuracy": float(np.mean(base)) if base else float("nan"),
        "projected_accuracy": float(np.mean(proj)) if proj else float("nan"),
    }


def run_condition(name, calibrator, corpus_by_id, examples, verifier, *, alpha,
                  scorer, budget, seed=0):
    gen = ReplayStepGenerator(list(corpus_by_id.values()))
    agent = CARAgent(
        generator=gen,
        scorer=scorer,
        calibrator=calibrator,
        gate=ControlGate(),
        verifier=verifier,
        budget_per_question=budget,
        max_steps=16,
        finalise=score_final_answer,
        score_answer=answer_is_correct,
    )
    trajs = agent.run_all(examples)
    row = evaluate(trajs, alpha)
    row["condition"] = name
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--probe", type=Path, default=None,
                    help="add a row scored by the trained internal-state probe "
                         "from scripts/gpu_probe_states.py")
    ap.add_argument("--synthetic-signal", type=float, default=None,
                    help="ignore --features and inject features with this "
                         "label separation; 0.0 is the null hypothesis")
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--gsm8k", type=Path, default=Path("data/raw/gsm8k/test.jsonl"),
                    help="source of gold answers, joined on question text")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"missing {args.corpus}")
        return 1
    feats = None if args.synthetic_signal is not None else args.features
    if feats is not None and not feats.exists():
        print(f"missing {feats}\n"
              f"run scripts/gpu_score_uncertainty.py, or pass --synthetic-signal")
        return 1

    corpus = load_corpus(args.corpus, feats, prefix="qwen")
    n_gold = attach_gold(corpus, args.gsm8k)
    if args.synthetic_signal is not None:
        inject_synthetic(corpus, args.synthetic_signal, seed=args.seed)
    by_id = {ex.example_id: ex for ex in corpus}

    examples = [ex.to_example() for ex in corpus]
    splits = make_splits(examples, dev_frac=0.3, cal_frac=0.3, salt="car-v1")
    assert_no_leakage(splits)

    print("=" * 92)
    print("CAR gate pipeline, end to end")
    print("=" * 92)
    src = (f"synthetic features, signal={args.synthetic_signal}"
           if args.synthetic_signal is not None else str(args.features))
    print(f"corpus   {args.corpus}   {len(corpus)} solutions, "
          f"{sum(len(e.steps) for e in corpus)} steps")
    print(f"features {src}")
    print(f"gold     {n_gold}/{len(corpus)} matched back to {args.gsm8k}")
    print(f"splits   {splits.summary()}")

    # Scaler fitted on DEV only. Fitting it on calibration or test would leak
    # into the threshold and void the conformal guarantee.
    dev_feats = [s.features for ex in splits.dev for s in by_id[ex.example_id].steps]
    scorer = CompositeScorer(WEIGHTS).fit(dev_feats)

    def scores_and_labels(split):
        s, y = [], []
        for ex in split:
            for st in by_id[ex.example_id].steps:
                s.append(scorer.score(st.features))
                y.append(st.global_ok)
        return np.asarray(s), np.asarray(y)

    cal_s, cal_y = scores_and_labels(splits.calibration)
    test_s, test_y = scores_and_labels(splits.test)

    # step_detection_auroc inverts internally: `labels` is True for a CORRECT
    # step. Passing ~test_y would double-invert and report 1 - AUROC.
    auroc = step_detection_auroc(test_s, test_y)
    base_risk = float(np.mean(~test_y))
    print(f"\nscore AUROC for detecting a globally-wrong step   {auroc:.4f}")
    print(f"base risk mu on test steps                        {base_risk:.4f}")
    floor = min_verification_rate(base_risk, args.alpha)
    print(f"Kotte floor at alpha={args.alpha:.2f}                          "
          f"{floor:.1%}   (budget here: "
          f"{args.budget / max(1e-9, np.mean([len(by_id[e.example_id].steps) for e in splits.test])):.1%})")
    if auroc < 0.55:
        print("\n  NOTE: AUROC near 0.5 means the score does not separate good")
        print("  steps from bad ones. Every score-based gate below then reduces")
        print("  to a random gate, and any apparent advantage is noise.")

    labels = {
        (by_id[ex.example_id].question, i): st.global_ok
        for ex in splits.test
        for i, st in enumerate(by_id[ex.example_id].steps)
    }

    # The oracle runs the SAME calibrator on a perfect score, so the gap between
    # it and "split conformal" is what a better signal could buy and nothing
    # else.
    oracle_scorer = OracleScorer(corpus)
    ocal_s = np.asarray([
        oracle_scorer.score(st.features)
        for ex in splits.calibration for st in by_id[ex.example_id].steps
    ])

    conditions = [
        ("cot (never verify)", lambda: NeverVerify(), scorer),
        ("always verify", lambda: AlwaysVerify(), scorer),
        ("random gate", lambda: RandomGate(rate=0.5, seed=args.seed), scorer),
        ("quantile gate", lambda: QuantileGate(quantile=1 - args.alpha).fit(cal_s),
         scorer),
        ("split conformal", lambda: SplitConformalCalibrator(alpha=args.alpha).fit(cal_s, cal_y),
         scorer),
        ("CAR (adaptive+ipw)", lambda: AdaptiveCalibrator(
            alpha=args.alpha, gamma=0.005, epsilon=0.2, update_mode="ipw",
            seed=args.seed).fit(cal_s, cal_y), scorer),
        ("CAR (adaptive, naive)", lambda: AdaptiveCalibrator(
            alpha=args.alpha, gamma=0.005, epsilon=0.0, update_mode="naive",
            seed=args.seed).fit(cal_s, cal_y), scorer),
        ("ORACLE score (ceiling)",
         lambda: SplitConformalCalibrator(alpha=args.alpha).fit(ocal_s, cal_y),
         oracle_scorer),
    ]

    if args.probe:
        probe_scorer = ProbeScorer(corpus, args.probe)
        pcal_s = np.asarray([
            probe_scorer.score(st.features)
            for ex in splits.calibration for st in by_id[ex.example_id].steps
        ])
        print(f"probe   layer {probe_scorer.meta['layer']}, "
              f"test AUROC {probe_scorer.meta['auroc_test']:.4f}")
        conditions.insert(-1, (
            "PROBE score",
            lambda: SplitConformalCalibrator(alpha=args.alpha).fit(pcal_s, cal_y),
            probe_scorer))

    # The last entry is an ABLATION, not a measured verifier: the task PRM with
    # its false-alarm rate set to zero. It exists to separate "the verifier
    # cannot see the error" from "the verifier sees it but breaks correct steps
    # on the way", which the measured rows cannot distinguish.
    arms = [(k, *MEASURED_SCOPE[k]) for k in
            ("arithmetic_local", "independent_judge", "task_prm")]
    arms.append(("task_prm ABLATION: no false alarms", MEASURED_SCOPE["task_prm"][0], 0.0))

    for kind, scope, fa in arms:
        print()
        print("=" * 92)
        print(f"VERIFIER: {kind}   scope={scope:.4f}  false alarm={fa:.4f}   "
              f"(measured, ch. 5)")
        print("=" * 92)
        print(f"  {'condition':<22}{'verify%':>9}{'calls/q':>9}"
              f"{'sel.risk':>10}{'target':>8}{'recall':>9}{'1st-bad':>9}"
              f"{'blocked':>9}{'PROJ acc':>10}")
        print("  " + "-" * 97)
        for name, make, sc in conditions:
            verifier = ScopedVerifier(labels, scope=scope, false_alarm=fa,
                                      label=kind, seed=args.seed)
            row = run_condition(name, make(), by_id, splits.test, verifier,
                                alpha=args.alpha, scorer=sc,
                                budget=args.budget, seed=args.seed)
            tgt = "-" if name in ("cot (never verify)", "always verify",
                                  "random gate") else f"{args.alpha:.2f}"
            print(f"  {name:<22}{row['verification_rate']:>9.1%}"
                  f"{row['calls_per_q']:>9.2f}{row['selective_risk']:>10.4f}"
                  f"{tgt:>8}{row['recall']:>9.4f}{row['first_bad_recall']:>9.4f}"
                  f"{row['budget_blocked']:>9,}"
                  f"{row['projected_accuracy']:>10.4f}")
            base = row["base_accuracy"]

        print(f"  {'-' * 88}")
        print(f"  {'corpus answer accuracy (no gate)':<40}{base:.4f}")
    # ---- does alpha bind at all? ------------------------------------------
    # Every row above "meets" alpha=0.30, and so does verifying nothing, because
    # the base risk is below the target. A guarantee satisfied by the empty
    # policy certifies nothing, so sweep alpha down to where it actually binds.
    print()
    print("=" * 92)
    print("ALPHA SWEEP: where does the target actually constrain anything?")
    print("=" * 92)
    print(f"base risk with no gate at all: {base_risk:.4f}. An alpha above that")
    print("is met by doing nothing, so only alpha below it tests the gate.")
    print()
    print(f"  {'alpha':<8}{'floor':>8}{'binds?':>9}{'split risk':>13}"
          f"{'split ver%':>12}{'CAR risk':>11}{'CAR ver%':>10}")
    print("  " + "-" * 82)
    for a in (0.05, 0.10, 0.15, 0.20, 0.30):
        v = ScopedVerifier(labels, scope=MEASURED_SCOPE["task_prm"][0],
                           false_alarm=0.0, label="prm", seed=args.seed)
        s_row = run_condition("split", SplitConformalCalibrator(alpha=a).fit(cal_s, cal_y),
                              by_id, splits.test, v, alpha=a, scorer=scorer,
                              budget=args.budget, seed=args.seed)
        v = ScopedVerifier(labels, scope=MEASURED_SCOPE["task_prm"][0],
                           false_alarm=0.0, label="prm", seed=args.seed)
        c_row = run_condition("CAR", AdaptiveCalibrator(
            alpha=a, gamma=0.005, epsilon=0.2, update_mode="ipw", seed=args.seed
        ).fit(cal_s, cal_y), by_id, splits.test, v, alpha=a, scorer=scorer,
            budget=args.budget, seed=args.seed)
        floor_a = min_verification_rate(base_risk, a)
        binds = "yes" if a < base_risk else "no"
        print(f"  {a:<8.2f}{floor_a:>8.1%}{binds:>9}{s_row['selective_risk']:>13.4f}"
              f"{s_row['verification_rate']:>12.1%}{c_row['selective_risk']:>11.4f}"
              f"{c_row['verification_rate']:>10.1%}")
    print()
    print("A row whose measured risk exceeds its alpha is a target the gate")
    print("could not hold at this budget with this score.")

    print()
    print("sel.risk is MEASURED: the fraction of steps the gate let through")
    print("unverified that are globally wrong. It is the quantity alpha is")
    print("supposed to bound. PROJ acc is MODELLED, not measured -- it assumes")
    print("a caught error is repaired and downstream reasoning recovers.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
