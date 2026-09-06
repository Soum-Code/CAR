"""Regression tests pinning the end-to-end pipeline's negative result.

Every refuted claim in this project has a test that keeps it refuted, so it
cannot quietly stop reproducing. The pipeline result is the strongest negative
the project has, and it rests on two numbers that a future change to the
scorer, the splits or the feature file could silently move.

See docs/FINDINGS-PIPELINE.md.
"""

from pathlib import Path

import numpy as np
import pytest

from car.agent.loop import CARAgent
from car.conformal import SplitConformalCalibrator
from car.data.splits import make_splits
from car.eval.metrics import step_detection_auroc
from car.generation.replay import (
    ReplayStepGenerator,
    answer_is_correct,
    load_corpus,
    score_final_answer,
)
from car.types import Decision
from car.uncertainty.composite import CompositeScorer
from car.verification.scoped import ScopedVerifier

CORPUS = Path("runs/generated_qwen25_7b.jsonl")
FEATURES = Path("runs/uncertainty_qwen25_7b.jsonl")

needs_run = pytest.mark.skipif(
    not (CORPUS.exists() and FEATURES.exists()),
    reason="generated corpus / uncertainty features not present",
)

WEIGHTS = {"token_entropy": 1.0, "max_surprisal": 0.5, "mean_logprob": 1.0}


@pytest.fixture(scope="module")
def prepared():
    corpus = load_corpus(CORPUS, FEATURES, prefix="qwen")
    by_id = {ex.example_id: ex for ex in corpus}
    examples = [ex.to_example() for ex in corpus]
    splits = make_splits(examples, dev_frac=0.3, cal_frac=0.3, salt="car-v1")
    scorer = CompositeScorer(WEIGHTS).fit(
        [s.features for ex in splits.dev for s in by_id[ex.example_id].steps]
    )

    def sl(split):
        s, y = [], []
        for ex in split:
            for st in by_id[ex.example_id].steps:
                s.append(scorer.score(st.features))
                y.append(st.global_ok)
        return np.asarray(s), np.asarray(y)

    return by_id, splits, scorer, sl


@needs_run
def test_generator_uncertainty_does_not_rank_global_step_error(prepared):
    """AUROC 0.5589. If this ever rises above 0.65 the finding has changed."""
    _, splits, _, sl = prepared
    test_s, test_y = sl(splits.test)
    auroc = step_detection_auroc(test_s, test_y)
    assert 0.50 < auroc < 0.65, f"AUROC moved to {auroc:.4f}"


@needs_run
def test_the_gate_misses_a_binding_alpha_target(prepared):
    """alpha=0.05, measured selective risk ~0.147 -- roughly 3x the target.

    This is the end-to-end refutation: split conformal holds its COVERAGE
    guarantee and the risk of what it accepts is untouched, because the score
    does not rank risk. A future version that passes this test has either
    fixed the score or broken the measurement.
    """
    by_id, splits, scorer, sl = prepared
    cal_s, cal_y = sl(splits.calibration)
    alpha = 0.05

    labels = {
        (by_id[ex.example_id].question, i): st.global_ok
        for ex in splits.test
        for i, st in enumerate(by_id[ex.example_id].steps)
    }
    agent = CARAgent(
        generator=ReplayStepGenerator(list(by_id.values())),
        scorer=scorer,
        calibrator=SplitConformalCalibrator(alpha=alpha).fit(cal_s, cal_y),
        verifier=ScopedVerifier(labels, scope=0.9033, false_alarm=0.0, seed=0),
        budget_per_question=2,
        max_steps=16,
        finalise=score_final_answer,
        score_answer=answer_is_correct,
    )
    trajs = agent.run_all(splits.test)
    accepted = [
        r.label
        for t in trajs
        for r in t.steps
        if r.decision == Decision.CONTINUE and r.label is not None
    ]
    risk = float(np.mean([not x for x in accepted]))
    assert risk > 2 * alpha, (
        f"selective risk {risk:.4f} is no longer far above the {alpha} target"
    )


@needs_run
def test_base_risk_is_below_most_of_the_alpha_sweep(prepared):
    """mu = 0.1578, so alpha >= 0.20 is met by verifying nothing.

    Worth pinning because it is the reason most of the sweep is vacuous, and a
    reader who misses it will read those rows as the gate succeeding.
    """
    _, splits, _, sl = prepared
    _, test_y = sl(splits.test)
    mu = float(np.mean(~test_y))
    assert 0.13 < mu < 0.19
    assert mu < 0.20
