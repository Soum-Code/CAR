"""Tests for the generated-corpus harness used by the Llama re-measurement.

The whole value of `car.data.generated` is that its output is read back by
`car.data.math_shepherd`, so the two generators' rates come out of one analysis
implementation. Most of what is worth testing is therefore round-tripping and
the definitions staying identical, not the string formatting.
"""

import json
from pathlib import Path

import pytest

from car.data.generated import (
    GeneratedSolution,
    annotation_rate,
    answers_match,
    build_fewshot,
    extract_answer,
    generation_prompt,
    local_validity,
    split_steps,
    to_shepherd_label,
    to_shepherd_record,
    truncate_completion,
)
from car.data.math_shepherd import parse_solution

TRAIN = Path("data/raw/gsm8k/train.jsonl")
needs_gsm8k = pytest.mark.skipif(not TRAIN.exists(), reason="GSM8K not downloaded")

SOLUTION = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
    "#### 72"
)


# ---- parsing ---------------------------------------------------------


def test_split_steps_drops_the_answer_line():
    assert split_steps(SOLUTION) == [
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.",
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.",
    ]


def test_extract_answer_prefers_the_hash_marker():
    assert extract_answer(SOLUTION) == "72"


def test_extract_answer_falls_back_to_the_last_number():
    assert extract_answer("so he has 15 left") == "15"


def test_extract_answer_normalises_commas_and_currency():
    assert extract_answer("#### $1,250") == "1250"


def test_answers_match_is_numeric_not_textual():
    assert answers_match("72", "72.0")
    assert answers_match("1,250", "1250")
    assert not answers_match("72", "27")
    assert not answers_match(None, "72")


def test_truncate_completion_cuts_the_invented_next_problem():
    """Few-shot completion does not stop; it starts the next question."""
    raw = SOLUTION + "\n\nQuestion: Bob has 3 apples\nAnswer: he has 3"
    assert truncate_completion(raw) == SOLUTION


def test_truncate_completion_stops_after_the_answer_marker():
    assert truncate_completion("a = <<1+1=2>>2\n#### 2\nand then more") == (
        "a = <<1+1=2>>2\n#### 2"
    )


# ---- local validity --------------------------------------------------


def test_local_validity_uses_the_calculator_markers():
    assert local_validity("Natalia sold 48/2 = <<48/2=24>>24 clips.") is True
    assert local_validity("Natalia sold 48/2 = <<48/2=25>>25 clips.") is False


def test_local_validity_is_none_without_annotation_by_default():
    """The default definition is Math-Shepherd's, so the rates stay comparable."""
    assert local_validity("He then had 20 apples left.") is None
    assert local_validity("So 100 - 60 = 40 dollars.") is None


def test_local_validity_prose_fallback_is_opt_in():
    assert local_validity("So 100 - 60 = 40 dollars.", allow_prose=True) is True
    assert local_validity("So 100 - 60 = 30 dollars.", allow_prose=True) is False


def test_local_validity_prose_fallback_needs_an_operator():
    """`x = 5` is a definition, not a claim a calculator can refute."""
    assert local_validity("Let the total = 5.", allow_prose=True) is None


def test_annotation_rate_counts_steps_not_solutions():
    sols = [
        GeneratedSolution("a", "q", "72", ["x = <<1+1=2>>2", "then done"], "2"),
    ]
    assert annotation_rate(sols) == 0.5


# ---- round trip into Math-Shepherd format ----------------------------


def test_shepherd_round_trip_preserves_steps_and_labels():
    steps = ["Natalia sold 48/2 = <<48/2=24>>24 clips.", "48+24 = <<48+24=72>>72."]
    label = to_shepherd_label("Q?", steps, [True, False], "72")
    sol = parse_solution(label)
    assert sol is not None
    assert sol.question == "Q?"
    assert [s.global_ok for s in sol.steps] == [True, False]
    assert sol.steps[0].local_ok is True
    assert sol.steps[1].local_ok is True


def test_shepherd_round_trip_carries_a_local_error_through():
    steps = ["13 x 8 / 13 = <<13*8/13=6>>6 slices."]
    sol = parse_solution(to_shepherd_label("Q?", steps, [False], "6"))
    assert sol.steps[0].local_ok is False
    assert sol.steps[0].global_ok is False


def test_shepherd_round_trip_survives_a_step_ending_in_an_operator():
    """A trailing +/- would be eaten as the label and shift every step."""
    sol = parse_solution(to_shepherd_label("Q?", ["the total is 5 +"], [True], "5"))
    assert sol is not None
    assert len(sol.steps) == 1
    assert sol.steps[0].global_ok is True


def test_shepherd_round_trip_survives_newlines_inside_a_step():
    sol = parse_solution(to_shepherd_label("Q?", ["line one\nline two"], [True], "5"))
    assert len(sol.steps) == 1


def test_final_step_carries_the_answer_the_way_shepherd_does():
    label = to_shepherd_label("Q?", ["a", "b"], [True, True], "72")
    assert "The answer is: 72" in label
    assert label.count("The answer is:") == 1


def test_label_count_mismatch_is_an_error():
    with pytest.raises(ValueError):
        to_shepherd_label("Q?", ["a", "b"], [True], "1")


def test_record_is_tagged_gsm8k_so_the_loader_keeps_it():
    rec = to_shepherd_record("Q?", ["a"], [True], "1")
    assert rec["task"] == "GSM8K"
    assert json.loads(json.dumps(rec))["label"].startswith("Q?")


def test_final_correct_follows_the_last_step_label():
    """Math-Shepherd defines it that way; the generated corpus must agree."""
    sol = parse_solution(to_shepherd_label("Q?", ["a", "b"], [True, False], "9"))
    assert sol.final_correct is False
    sol = parse_solution(to_shepherd_label("Q?", ["a", "b"], [False, True], "9"))
    assert sol.final_correct is True


# ---- prompting -------------------------------------------------------


def test_generation_prompt_continues_from_a_prefix():
    p = generation_prompt("SHOTS", "How many?", ["Step text one."])
    assert p.endswith("Step text one.\n")
    assert "Question: How many?" in p


def test_generation_prompt_from_scratch_ends_open():
    p = generation_prompt("SHOTS", "How many?")
    assert p.endswith("Answer: ")


@needs_gsm8k
def test_fewshot_exemplars_all_carry_calculator_markers():
    rows = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    shots = build_fewshot(rows, k=4, seed=0)
    assert shots.count("Question:") == 4
    assert shots.count("####") == 4
    assert "<<" in shots


@needs_gsm8k
def test_fewshot_is_deterministic_given_a_seed():
    rows = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert build_fewshot(rows, 4, seed=3) == build_fewshot(rows, 4, seed=3)
    assert build_fewshot(rows, 4, seed=3) != build_fewshot(rows, 4, seed=4)
