"""End-to-end tests: uncertainty features, metrics, and the full agent loop
running on the mock backend.

Everything here runs on CPU in seconds. That is the point -- the entire control
and calibration stack is exercisable without a GPU, so the only thing the rented
GPU is needed for is generating real traces.
"""

import numpy as np
import pytest

from car.agent import CARAgent
from car.backends import MockBackend
from car.backends.cache import CachedBackend
from car.baselines import AlwaysVerify, NeverVerify
from car.conformal import AdaptiveCalibrator, SplitConformalCalibrator
from car.control import ControlGate, InfluenceWeighting
from car.eval import (
    empirical_coverage,
    expected_calibration_error,
    false_safe_rate,
    selective_risk,
    step_detection_auroc,
)
from car.generation import MockStepGenerator, parse_step
from car.generation.schema import StepParseError
from car.tracing import TraceWriter, flatten_steps, read_traces
from car.types import Decision, Example, ReasoningStep
from car.uncertainty import (
    CompositeScorer,
    cluster_by_equivalence,
    max_surprisal,
    normalised_semantic_divergence,
    token_entropy,
)
from car.verification import CalculatorVerifier, safe_eval


def _examples(n=40):
    return [
        Example(example_id=f"q{i}", question=f"question {i}", gold_answer="yes")
        for i in range(n)
    ]


def _build(calibrator, *, influence="none", n_examples=40, seed=0, budget=3, verifier="calculator"):
    backend = MockBackend(seed=seed, signal_strength=0.9, base_error_rate=0.3)
    gen = MockStepGenerator(backend, n_steps=4, n_samples=5)
    examples = _examples(n_examples)

    # Fit the scorer and calibrator on a dev/cal split, never on test.
    dev_feats, dev_labels = [], []
    for ex in examples[:20]:
        ctx = []
        for target in gen.plan(ex):
            g = gen.step(ex, target, ctx)
            dev_feats.append(g.features)
            dev_labels.append(g.true_label)
            ctx.append(g.step)

    scorer = CompositeScorer(
        weights={"token_entropy": 1.0, "max_surprisal": 0.5, "semantic_divergence": 1.0}
    ).fit(dev_feats)
    scores = scorer.score_many(dev_feats)
    calibrator.fit(scores, np.array(dev_labels))

    from car.verification import SimulatedVerifier

    verifiers = {
        "calculator": lambda: CalculatorVerifier(),
        "simulated": lambda: SimulatedVerifier(backend, seed=seed),
    }

    agent = CARAgent(
        generator=gen,
        scorer=scorer,
        calibrator=calibrator,
        gate=ControlGate(influence_weighting=InfluenceWeighting(mode=influence)),
        verifier=verifiers[verifier](),
        budget_per_question=budget,
    )
    return agent, examples[20:], scores, np.array(dev_labels)


# ---- uncertainty features -------------------------------------------


def test_features_are_finite_and_oriented():
    backend = MockBackend(seed=0)
    gen = backend.generate("some step", n=1)[0]
    assert np.isfinite(token_entropy(gen))
    assert np.isfinite(max_surprisal(gen))
    assert token_entropy(gen) >= 0


def test_semantic_divergence_is_zero_when_samples_agree():
    assert normalised_semantic_divergence([0, 0, 0, 0]) == 0.0


def test_semantic_divergence_is_one_when_all_samples_differ():
    """Maximum divergence: every sample lands in its own meaning cluster."""
    assert normalised_semantic_divergence([0, 1, 2, 3]) == pytest.approx(1.0)


def test_clustering_groups_identical_meanings():
    ids = cluster_by_equivalence(["Paris", "paris ", "London"])
    assert ids[0] == ids[1] != ids[2]


def test_composite_scorer_must_be_fitted_on_dev_only():
    with pytest.raises(ValueError):
        CompositeScorer().fit([])


def test_composite_scorer_roundtrips():
    backend = MockBackend(seed=1)
    feats = []
    from car.types import UncertaintyFeatures

    for i in range(20):
        g = backend.generate(f"s{i}", n=1)[0]
        feats.append(
            UncertaintyFeatures(
                token_entropy=token_entropy(g), max_surprisal=max_surprisal(g)
            )
        )
    s = CompositeScorer().fit(feats)
    s2 = CompositeScorer.from_dict(s.to_dict())
    assert s.score(feats[0]) == pytest.approx(s2.score(feats[0]))


# ---- schema ----------------------------------------------------------


def test_parse_step_reads_valid_json():
    step = parse_step('prefix {"step_id": 0, "claim": "x", "rationale": "y", '
                      '"dependency_ids": [], "tool_request": null} suffix')
    assert step.claim == "x"


def test_parse_step_raises_on_garbage():
    with pytest.raises(StepParseError):
        parse_step("no json at all here")


def test_parse_step_raises_on_malformed_json():
    with pytest.raises(StepParseError):
        parse_step('{"step_id": 0, "claim": ')


# ---- verification ----------------------------------------------------


def test_calculator_catches_bad_arithmetic():
    from car.types import Verdict

    v = CalculatorVerifier()
    bad = ReasoningStep(step_id=0, claim="47 * 3 = 131")
    assert v.verify(bad, "q").verdict == Verdict.CONTRADICTED


def test_calculator_confirms_good_arithmetic():
    from car.types import Verdict

    v = CalculatorVerifier()
    good = ReasoningStep(step_id=0, claim="47 * 3 = 141")
    assert v.verify(good, "q").verdict == Verdict.SUPPORTED


def test_safe_eval_refuses_code_execution():
    """Retrieved and generated text is untrusted; eval must not be reachable."""
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval("__import__('os').system('echo pwned')")


# ---- metrics ---------------------------------------------------------


def test_auroc_is_one_for_a_perfect_score():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([True, True, False, False])
    assert step_detection_auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_is_half_for_an_uninformative_score():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    labels = rng.random(4000) > 0.5
    assert step_detection_auroc(scores, labels) == pytest.approx(0.5, abs=0.05)


def test_selective_risk_and_coverage_move_in_opposite_directions():
    """Loosening the gate accepts more correct steps AND more wrong ones. The
    trade-off the whole project is about."""
    rng = np.random.default_rng(1)
    n = 2000
    correct = rng.random(n) > 0.3
    scores = np.where(correct, rng.normal(0, 1, n), rng.normal(3, 1, n))

    tight, loose = 0.0, 3.0
    assert empirical_coverage(scores, correct, loose) > empirical_coverage(
        scores, correct, tight
    )
    assert selective_risk(scores, correct, loose) > selective_risk(
        scores, correct, tight
    )


def test_false_safe_rate_is_zero_when_nothing_is_accepted():
    scores = np.array([1.0, 2.0, 3.0])
    labels = np.array([True, False, False])
    assert false_safe_rate(scores, labels, threshold=-10.0) == 0.0


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    conf = np.array([0.0] * 500 + [1.0] * 500)
    correct = np.array([False] * 500 + [True] * 500)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-9)


# ---- agent loop ------------------------------------------------------


def test_agent_runs_end_to_end():
    agent, test_examples, _, _ = _build(SplitConformalCalibrator(alpha=0.1))
    trajs = agent.run_all(test_examples)

    assert len(trajs) == len(test_examples)
    assert all(t.n_steps > 0 for t in trajs)
    assert all(t.verification_calls <= 3 for t in trajs), "budget must be respected"


def test_never_verify_spends_nothing():
    agent, test_examples, _, _ = _build(NeverVerify())
    trajs = agent.run_all(test_examples)
    assert sum(t.verification_calls for t in trajs) == 0


def test_always_verify_saturates_the_budget():
    agent, test_examples, _, _ = _build(AlwaysVerify(), budget=4)
    trajs = agent.run_all(test_examples)
    # 4 steps per question, budget 4, so every step should trigger a call.
    assert all(t.verification_calls == 4 for t in trajs)


def test_verification_happens_before_commitment():
    """Structural check on the central architectural claim: a step that was
    verified must carry its verdict in the same record, not a later one."""
    agent, test_examples, _, _ = _build(AlwaysVerify(), budget=4)
    for t in agent.run_all(test_examples):
        for r in t.steps:
            if r.decision == Decision.VERIFY:
                assert r.verdict is not None


def test_adaptive_agent_logs_exploration_provenance():
    cal = AdaptiveCalibrator(alpha=0.1, epsilon=0.3, update_mode="ipw", seed=0)
    agent, test_examples, _, _ = _build(cal, budget=4)
    trajs = agent.run_all(test_examples)

    explored = [r for t in trajs for r in t.steps if r.forced_exploration]
    assert explored, "epsilon=0.3 should produce forced verifications"
    assert all(r.decision == Decision.VERIFY for r in explored)


def test_adaptive_loop_actually_closes():
    """The calibrator must receive labels and move.

    Regression test for a real failure: paired with a verifier that cannot
    judge the steps it is given, every verdict comes back INSUFFICIENT, no
    label is ever produced, and the threshold silently never updates. The run
    still looks healthy, which is what makes it dangerous.
    """
    cal = AdaptiveCalibrator(alpha=0.1, epsilon=0.4, update_mode="ipw", seed=0)
    agent, test_examples, _, _ = _build(cal, budget=4, verifier="simulated")
    start = cal.threshold
    agent.run_all(test_examples)

    assert cal.n_updates > 0, "calibrator received no usable labels"
    assert cal.threshold != start, "threshold never moved"


def test_insufficient_verdict_yields_no_label():
    """'I could not check' must not be recorded as 'it was fine'.

    A verifier that always abstains should leave the calibrator untouched.
    """
    from car.verification import SimulatedVerifier

    cal = AdaptiveCalibrator(alpha=0.1, epsilon=1.0, update_mode="ipw", seed=0)
    agent, test_examples, _, _ = _build(cal, budget=4)
    agent.verifier = SimulatedVerifier(
        MockBackend(seed=0), insufficient_rate=1.0, seed=0
    )
    start = cal.threshold
    agent.run_all(test_examples)

    assert cal.n_updates == 0
    assert cal.threshold == start


def test_simulated_verifier_error_rate_is_honoured():
    """Verifier fallibility must be controllable -- a perfect checker is an
    unrealistic assumption and the ablations need to vary it."""
    from car.types import Verdict
    from car.verification import SimulatedVerifier

    backend = MockBackend(seed=0)
    perfect = SimulatedVerifier(backend, error_rate=0.0, seed=0)
    broken = SimulatedVerifier(backend, error_rate=1.0, seed=0)

    step = ReasoningStep(step_id=0, claim="[q1] target_0")
    a = perfect.verify(step, "q").verdict
    b = broken.verify(step, "q").verdict
    assert a != b, "error_rate=1.0 must invert every verdict"
    assert {a, b} == {Verdict.SUPPORTED, Verdict.CONTRADICTED}


def test_influence_gate_reallocates_calls_toward_early_steps():
    """With influence weighting on, verification should shift earlier in the
    chain, where a mistake costs the most downstream."""

    def mean_verified_position(mode):
        agent, ex, _, _ = _build(SplitConformalCalibrator(alpha=0.3), influence=mode)
        trajs = agent.run_all(ex)
        pos = [
            r.step.step_id
            for t in trajs
            for r in t.steps
            if r.decision == Decision.VERIFY
        ]
        return float(np.mean(pos)) if pos else float("nan")

    assert mean_verified_position("sqrt") < mean_verified_position("none")


# ---- tracing ---------------------------------------------------------


def test_traces_roundtrip_and_flatten(tmp_path):
    agent, test_examples, _, _ = _build(SplitConformalCalibrator(alpha=0.1))
    path = tmp_path / "trace.jsonl"

    with TraceWriter(path, manifest={"alpha": 0.1, "backend": "mock"}) as w:
        agent.trace = w
        agent.run_all(test_examples)

    loaded = read_traces(path)
    assert len(loaded) == len(test_examples)

    rows = flatten_steps(loaded)
    assert rows and "forced_exploration" in rows[0]
    assert all("score" in r and "decision" in r for r in rows)


def test_cache_avoids_recomputation(tmp_path):
    """The GPU/CPU split depends on this: generate once, analyse many times."""
    cached = CachedBackend(MockBackend(seed=0), cache_dir=tmp_path / "gen")
    a = cached.generate("prompt", n=2)
    b = cached.generate("prompt", n=2)

    assert cached.hits == 1 and cached.misses == 1
    assert np.allclose(a[0].token_entropies, b[0].token_entropies)


def test_cache_key_separates_sampling_params(tmp_path):
    """A stale cache must never silently serve results for other settings."""
    cached = CachedBackend(MockBackend(seed=0), cache_dir=tmp_path / "gen")
    cached.generate("prompt", n=1, temperature=0.7)
    cached.generate("prompt", n=1, temperature=1.0)
    assert cached.misses == 2
