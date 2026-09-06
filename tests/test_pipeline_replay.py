"""Tests for replaying a generated corpus through the real agent loop.

The value of replay is that `CARAgent` runs unmodified, so what these tests
protect is the joins: corpus to features, corpus to gold answers, and the
modelled projection that must not invent movement the gate did not cause.
"""

import json

import pytest

from car.agent.loop import CARAgent
from car.baselines.policies import AlwaysVerify, NeverVerify
from car.data.generated import to_shepherd_record
from car.eval.metrics import project_correct_after_repair, projected_accuracy
from car.generation.replay import (
    ReplayStepGenerator,
    answer_is_correct,
    attach_gold,
    final_answer_of,
    load_corpus,
    score_final_answer,
)
from car.types import Decision, Verdict
from car.uncertainty.composite import CompositeScorer
from car.verification.scoped import MEASURED_SCOPE, ScopedVerifier


def write_corpus(tmp_path, records):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def two_solutions():
    return [
        to_shepherd_record(
            "Q one?", ["a = <<1+1=2>>2", "b = <<2+2=4>>4"], [True, True], "4"
        ),
        to_shepherd_record(
            "Q two?", ["c = <<3*3=9>>9", "d = <<9-1=7>>7"], [True, False], "7"
        ),
    ]


# ---- loading ----------------------------------------------------------


def test_load_corpus_reads_steps_and_labels(tmp_path):
    corpus = load_corpus(write_corpus(tmp_path, two_solutions()))
    assert len(corpus) == 2
    assert [s.global_ok for s in corpus[1].steps] == [True, False]
    assert corpus[0].example_id == "gen_0"


def test_load_corpus_rejects_a_feature_file_of_the_wrong_length(tmp_path):
    """Positional pairing must fail loudly, not misalign every step."""
    cp = write_corpus(tmp_path, two_solutions())
    fp = tmp_path / "feat.jsonl"
    fp.write_text(json.dumps({"n_steps": 2, "steps": [{}, {}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="misalign"):
        load_corpus(cp, fp)


def test_load_corpus_rejects_a_step_count_mismatch(tmp_path):
    cp = write_corpus(tmp_path, two_solutions())
    fp = tmp_path / "feat.jsonl"
    fp.write_text(
        json.dumps({"n_steps": 2, "steps": [{}, {}]}) + "\n"
        + json.dumps({"n_steps": 1, "steps": [{}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="steps in the corpus"):
        load_corpus(cp, fp)


def test_features_are_attached_in_order(tmp_path):
    cp = write_corpus(tmp_path, two_solutions())
    fp = tmp_path / "feat.jsonl"
    rows = [
        {"n_steps": 2, "steps": [{"token_entropy": 1.0}, {"token_entropy": 2.0}]},
        {"n_steps": 2, "steps": [{"token_entropy": 3.0}, {"token_entropy": 4.0}]},
    ]
    fp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    corpus = load_corpus(cp, fp)
    assert [s.features.token_entropy for s in corpus[0].steps] == [1.0, 2.0]
    assert [s.features.token_entropy for s in corpus[1].steps] == [3.0, 4.0]


def test_attach_gold_matches_on_normalised_question(tmp_path):
    cp = write_corpus(tmp_path, two_solutions())
    corpus = load_corpus(cp)
    gsm = tmp_path / "test.jsonl"
    gsm.write_text(
        json.dumps({"question": "Q  one?\n", "answer": "blah\n#### 4"}) + "\n"
        + json.dumps({"question": "Q two?", "answer": "blah\n#### 8"}),
        encoding="utf-8",
    )
    assert attach_gold(corpus, gsm) == 2
    assert corpus[0].gold_answer == "4"
    assert corpus[1].gold_answer == "8"


def test_final_answer_is_read_off_the_last_step(tmp_path):
    corpus = load_corpus(write_corpus(tmp_path, two_solutions()))
    assert final_answer_of(corpus[0]) == "4"
    assert final_answer_of(corpus[1]) == "7"


def test_answer_correctness_is_numeric(tmp_path):
    assert answer_is_correct("4", "4.0") is True
    assert answer_is_correct("4", "8") is False
    assert answer_is_correct(None, "4") is None


# ---- the loop ---------------------------------------------------------


def build_agent(corpus, calibrator, verifier, budget=4):
    scorer = CompositeScorer({"token_entropy": 1.0})
    scorer.fit([s.features for ex in corpus for s in ex.steps])
    return CARAgent(
        generator=ReplayStepGenerator(corpus),
        scorer=scorer,
        calibrator=calibrator,
        verifier=verifier,
        budget_per_question=budget,
        finalise=score_final_answer,
        score_answer=answer_is_correct,
    )


def test_replay_serves_labels_to_the_loop(tmp_path):
    corpus = load_corpus(write_corpus(tmp_path, two_solutions()))
    attach_gold_stub(corpus)
    agent = build_agent(corpus, NeverVerify(), None)
    trajs = agent.run_all([ex.to_example() for ex in corpus])
    assert [r.label for r in trajs[1].steps] == [True, False]
    assert all(r.decision == Decision.CONTINUE for t in trajs for r in t.steps)


def attach_gold_stub(corpus):
    for ex in corpus:
        ex.gold_answer = final_answer_of(ex) or ""


def test_replay_answer_comes_from_the_corpus_not_the_last_claim(tmp_path):
    corpus = load_corpus(write_corpus(tmp_path, two_solutions()))
    attach_gold_stub(corpus)
    agent = build_agent(corpus, NeverVerify(), None)
    trajs = agent.run_all([ex.to_example() for ex in corpus])
    assert trajs[0].final_answer == "4"
    assert trajs[0].correct is True


# ---- scoped verifier --------------------------------------------------


def labels_for(corpus):
    return {
        (ex.question, i): st.global_ok
        for ex in corpus
        for i, st in enumerate(ex.steps)
    }


def test_full_scope_verifier_detects_every_bad_step(tmp_path):
    corpus = load_corpus(write_corpus(tmp_path, two_solutions()))
    attach_gold_stub(corpus)
    v = ScopedVerifier(labels_for(corpus), scope=1.0)
    agent = build_agent(corpus, AlwaysVerify(), v)
    trajs = agent.run_all([ex.to_example() for ex in corpus])
    revised = [r for t in trajs for r in t.steps if r.revised]
    assert len(revised) == 1
    assert revised[0].label is False


def test_zero_scope_verifier_says_supported_not_insufficient(tmp_path):
    """The modelling choice that matters.

    A verifier that misses an error does not announce the miss -- it says the
    step looks fine. INSUFFICIENT would give the calibrator no label; SUPPORTED
    gives it a WRONG label, which is the mechanism by which a low-scope
    verifier makes calibration confident about a risk it cannot see.
    """
    corpus = load_corpus(write_corpus(tmp_path, two_solutions()))
    attach_gold_stub(corpus)
    v = ScopedVerifier(labels_for(corpus), scope=0.0)
    agent = build_agent(corpus, AlwaysVerify(), v)
    trajs = agent.run_all([ex.to_example() for ex in corpus])
    verdicts = [r.verdict for t in trajs for r in t.steps]
    assert Verdict.INSUFFICIENT not in verdicts
    assert all(v is Verdict.SUPPORTED for v in verdicts)
    assert v.misses == 1


def test_false_alarm_rate_is_honoured(tmp_path):
    corpus = load_corpus(write_corpus(tmp_path, two_solutions()))
    v = ScopedVerifier(labels_for(corpus), scope=0.0, false_alarm=1.0)
    agent = build_agent(corpus, AlwaysVerify(), v)
    agent.run_all([ex.to_example() for ex in corpus])
    assert v.false_alarms == 3  # the three globally-correct steps


def test_measured_scope_table_matches_chapter_5():
    assert MEASURED_SCOPE["task_prm"] == (0.9033, 0.0987)
    assert MEASURED_SCOPE["independent_judge"] == (0.2283, 0.0200)
    assert MEASURED_SCOPE["arithmetic_local"][0] == 0.0
    with pytest.raises(ValueError):
        ScopedVerifier.from_measured({}, "nonexistent")


# ---- the modelled projection -----------------------------------------


def test_projection_reduces_to_observed_accuracy_without_repairs(tmp_path):
    """Scope 0, no false alarms: the projection must not invent movement."""
    corpus = load_corpus(write_corpus(tmp_path, two_solutions()))
    attach_gold_stub(corpus)
    v = ScopedVerifier(labels_for(corpus), scope=0.0)
    agent = build_agent(corpus, AlwaysVerify(), v)
    trajs = agent.run_all([ex.to_example() for ex in corpus])
    observed = sum(1 for t in trajs if t.correct) / len(trajs)
    assert projected_accuracy(trajs) == pytest.approx(observed)


def test_projection_rescues_a_wrong_answer_only_via_the_first_bad_step(tmp_path):
    from car.types import ReasoningStep, StepRecord, Trajectory, UncertaintyFeatures

    def rec(label, revised):
        return StepRecord(
            step=ReasoningStep(step_id=0, claim="x"),
            features=UncertaintyFeatures(), score=0.0, threshold=0.0,
            decision=Decision.VERIFY, label=label, revised=revised,
        )

    # wrong answer, first bad step caught -> rescued
    t = Trajectory(example_id="a", question="q", correct=False,
                   steps=[rec(False, True), rec(False, False)])
    assert project_correct_after_repair(t) is True

    # wrong answer, only a LATER bad step caught -> not rescued: the first one
    # already corrupted everything downstream
    t = Trajectory(example_id="b", question="q", correct=False,
                   steps=[rec(False, False), rec(False, True)])
    assert project_correct_after_repair(t) is False


def test_projection_loses_a_right_answer_to_a_false_alarm():
    from car.types import ReasoningStep, StepRecord, Trajectory, UncertaintyFeatures

    good = StepRecord(
        step=ReasoningStep(step_id=0, claim="x"), features=UncertaintyFeatures(),
        score=0.0, threshold=0.0, decision=Decision.VERIFY, label=True, revised=True,
    )
    t = Trajectory(example_id="c", question="q", correct=True, steps=[good])
    assert project_correct_after_repair(t) is False


def test_projection_keeps_a_right_answer_that_contains_a_bad_step():
    """Math-Shepherd labels are optimistic; a '-' step can still end correct."""
    from car.types import ReasoningStep, StepRecord, Trajectory, UncertaintyFeatures

    bad = StepRecord(
        step=ReasoningStep(step_id=0, claim="x"), features=UncertaintyFeatures(),
        score=0.0, threshold=0.0, decision=Decision.CONTINUE, label=False,
    )
    t = Trajectory(example_id="d", question="q", correct=True, steps=[bad])
    assert project_correct_after_repair(t) is True
