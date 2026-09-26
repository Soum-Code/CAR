"""Pins probe round two — and the two corrections it went through.

The first version of this experiment concluded "doubling the data does not move
it, C9c refuted". Both halves were artifacts: the two arms had selected
different layers, so the comparison was not about data quantity, and the
learning curve walked an unshuffled pool whose composition drifted with its
size. The corrected answer has the opposite sign and is not significant.

So these tests guard the method as much as the numbers. A comparison that
claims to isolate training size must hold the configuration fixed; a learning
curve must sample its pool; a gain must name its comparator.
"""

import json
from pathlib import Path

import pytest

RUN = Path("runs/probe_variants.json")

needs_run = pytest.mark.skipif(
    not RUN.exists(), reason="run scripts/exp_probe_variants.py first")


def blob():
    return json.loads(RUN.read_text(encoding="utf-8"))


def by_variant():
    return {r["variant"][0]: r for r in blob()["variants"]}


# ---- gap 1: what more data is actually worth ------------------------------


@needs_run
def test_data_quantity_is_measured_at_fixed_configuration():
    """The correction. A and B select different layers (25 vs 28), so their
    difference is not a measurement of training size."""
    b = blob()
    d = b["fixed_config_doubling"]
    assert set(d) == {"A-config", "B-config"}
    assert d["A-config"]["layer"] != d["B-config"]["layer"], (
        "if both arms now pick the same layer the confound is gone, but the "
        "prose explaining it needs updating")
    for r in d.values():
        assert r["n_big"] > 1.9 * r["n_small"]


@needs_run
def test_more_data_helps_slightly_and_not_significantly():
    """C9c is UNSUPPORTED, not refuted: the sign is positive everywhere and
    every interval spans zero. Either half flipping changes the chapter."""
    for tag, r in blob()["fixed_config_doubling"].items():
        d = r["delta"]
        assert d["mean"] > 0, f"{tag}: sign went negative"
        assert d["mean"] < 0.05, f"{tag}: gain got large enough to support C9c"
        assert d["ci"][0] < 0 < d["ci"][1], f"{tag}: interval no longer spans zero"


@needs_run
def test_the_learning_curve_pool_is_sampled_not_concatenated():
    """The second artifact: an unshuffled pool made the curve's composition
    drift with its size, so 'size' was confounded with 'which split'."""
    b = blob()
    curve = b["learning_curve_pooled"]
    assert len(curve) >= 4
    # Rising end to end, which the confounded version reported as falling.
    assert curve[-1][1] > curve[0][1]


# ---- gap 2: the training target -------------------------------------------


@needs_run
def test_the_first_bad_gain_is_reported_against_both_comparators():
    """The pooled global probe has the lowest first-bad recall here, so quoting
    the gain against it alone picks the flattering baseline."""
    g = blob()["first_bad_gain_by_comparator"]
    assert {"C_vs_A", "C_vs_B"} <= set(g)
    assert g["C_vs_B"]["ci"][0] > 0, "the favourable comparison should exclude zero"
    assert g["C_vs_A"]["ci"][0] < 0, (
        "against the published probe the interval spans zero; if that changes "
        "the chapter can drop its comparator caveat")
    assert g["C_vs_A"]["mean"] > 0, "direction should still be positive"


@needs_run
def test_the_right_target_eliminates_the_auroc_advantage():
    """Not 'halves'. 0.5735 is below the 0.5742 token+semantic baseline whose
    failure is chapter 7's central negative result."""
    v = by_variant()
    baseline = blob()["reference"]["token + semantic"]
    assert v["C"]["auroc_test"] < baseline
    assert v["C"]["first_bad_recall"] == max(
        r["first_bad_recall"] for r in blob()["variants"])


@needs_run
def test_variant_c_shows_no_winners_curse():
    """It is the best of 145 configurations chosen on 19 positives, so the gap
    is worth recording even though it came out clean."""
    w = blob()["variant_c_winners_curse"]
    assert w["n_configurations"] == 145
    assert w["n_select_positives"] < 25
    assert w["test"] >= w["selection"] - 0.05


@needs_run
def test_residualising_position_moves_the_positional_component():
    v = by_variant()
    assert abs(v["D"]["score_position_corr"]) < abs(v["B"]["score_position_corr"])


# ---- the limit, and reproducibility ---------------------------------------


@needs_run
def test_first_bad_positive_counts_are_recorded():
    """49 pooled, not 68: the other 19 sit in the selection split. The wrong
    number was in the thesis once."""
    b = blob()
    assert b["n_first_bad"] == 108
    assert b["n_first_bad_test"] == 40
    assert b["n_first_bad_train_pool"] == 49
    assert b["n_first_bad_train_dev"] == 27
    assert (b["n_first_bad_train_pool"] + b["n_first_bad_select"]
            + b["n_first_bad_test"] == b["n_first_bad"])


@needs_run
def test_the_gate_safe_result_is_null_and_says_so():
    gs = blob()["gate_safe"]
    assert gs["firstbad"]["n_pos_train"] == 27
    d = gs["first_bad_recall_diff"]
    assert d["mean"] > 0 and d["ci"][0] < 0 < d["ci"][1]
    assert gs["firstbad"]["first_bad_recall"] > gs["global"]["first_bad_recall"]


@needs_run
def test_every_quoted_block_is_produced_by_the_script():
    """An earlier version hand-wrote the gate-safe and bootstrap blocks into
    this artifact from a throwaway shell snippet, so the thesis quoted numbers
    no committed code produced. Each key here must come from exp_probe_variants.py."""
    src = Path("scripts/exp_probe_variants.py").read_text(encoding="utf-8")
    for key in ("fixed_config_doubling", "learning_curve_pooled",
                "first_bad_gain_by_comparator", "variant_c_winners_curse",
                "gate_safe", "n_first_bad_train_pool"):
        assert key in blob(), f"{key} missing from the artifact"
        assert f'"{key}"' in src, f"{key} is in the artifact but no code writes it"


@needs_run
def test_gate_score_files_exist_for_reproducing_the_gate_table():
    """section 3's table is reproduced with
    `exp_gate_pipeline.py --probe runs/gate_probe_<tag>.json`, so the inputs
    have to be committed rather than described."""
    n_steps = blob()["n_steps"]
    for tag in ("global", "firstbad"):
        p = Path(f"runs/gate_probe_{tag}.json")
        assert p.exists(), f"{p} missing -- rerun with --emit-gate-scores"
        d = json.loads(p.read_text(encoding="utf-8"))
        assert len(d["scores"]) == n_steps
        assert d["target"] == tag
