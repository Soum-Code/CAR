"""Command-line entry point.

    car verify-setup     check the install and run a smoke experiment
    car demo             reproduce the censored-feedback result
"""

from __future__ import annotations

import argparse
import sys


def _verify_setup() -> int:
    import numpy as np

    from car.agent import CARAgent
    from car.backends import MockBackend
    from car.conformal import AdaptiveCalibrator
    from car.control import ControlGate, InfluenceWeighting
    from car.data import assert_no_leakage, make_splits
    from car.eval import summarise
    from car.generation import MockStepGenerator
    from car.types import Example
    from car.uncertainty import CompositeScorer
    from car.verification import SimulatedVerifier

    print("CAR setup check")
    print("-" * 60)

    examples = [
        Example(example_id=f"q{i}", question=f"question {i}", gold_answer="yes")
        for i in range(120)
    ]
    splits = make_splits(examples)
    assert_no_leakage(splits)
    print(f"splits            {splits.summary()}")

    backend = MockBackend(seed=0, signal_strength=0.9)
    gen = MockStepGenerator(backend, n_steps=4, n_samples=5)

    def collect(exs):
        feats, labels = [], []
        for ex in exs:
            ctx = []
            for target in gen.plan(ex):
                g = gen.step(ex, target, ctx)
                feats.append(g.features)
                labels.append(g.true_label)
                ctx.append(g.step)
        return feats, np.array(labels)

    dev_f, dev_y = collect(splits.dev)
    cal_f, cal_y = collect(splits.calibration)

    scorer = CompositeScorer(
        weights={"token_entropy": 1.0, "max_surprisal": 0.5, "semantic_divergence": 1.0}
    ).fit(dev_f)
    cal_scores = scorer.score_many(cal_f)

    calibrator = AdaptiveCalibrator(
        alpha=0.1, gamma=0.005, epsilon=0.2, update_mode="ipw", seed=0
    )
    calibrator.fit(cal_scores, cal_y)
    print(f"initial threshold {calibrator.threshold:.4f}")

    agent = CARAgent(
        generator=gen,
        scorer=scorer,
        calibrator=calibrator,
        gate=ControlGate(influence_weighting=InfluenceWeighting(mode="sqrt")),
        # Simulated rather than deterministic: mock steps contain no arithmetic,
        # so CalculatorVerifier would return INSUFFICIENT every time and the
        # adaptive calibrator would never receive a label.
        verifier=SimulatedVerifier(backend, insufficient_rate=0.1, seed=0),
        budget_per_question=3,
    )
    trajs = agent.run_all(splits.test)

    test_scores = np.array([r.score for t in trajs for r in t.steps])
    test_labels = np.array(
        [r.label for t in trajs for r in t.steps if r.label is not None]
    )
    metrics = summarise(test_scores, test_labels, calibrator.threshold, trajs)

    print("-" * 60)
    for k, v in metrics.items():
        print(f"{k:<20}{v:.4f}" if isinstance(v, float) else f"{k:<20}{v}")
    print("-" * 60)
    print(f"exploration calls {calibrator.n_exploration_calls}")
    print(f"final threshold   {calibrator.threshold:.4f}")
    print("\nsetup OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="car", description="Conformalized Agentic Reasoning")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-setup", help="check the install and run a smoke experiment")
    sub.add_parser("demo", help="reproduce the censored-feedback result")

    args = parser.parse_args(argv)
    if args.command == "verify-setup":
        return _verify_setup()
    if args.command == "demo":
        from scripts.demo_censored_feedback import main as demo_main

        demo_main()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
