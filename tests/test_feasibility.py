"""Tests for the Kotte feasibility floor, and that the config respects it."""

from pathlib import Path

import pytest
import yaml

from car.conformal import (
    MU_GLOBAL_GSM8K,
    MU_LOCAL_GSM8K,
    alpha_report,
    check_alpha,
    feasible_alpha,
    min_verification_rate,
)

CONFIG = Path("configs/default.yaml")


# ---- the bound -------------------------------------------------------


def test_no_floor_when_mu_below_alpha():
    assert min_verification_rate(mu=0.2, alpha=0.3) == 0.0


def test_floor_matches_the_closed_form():
    # (0.39 - 0.10) / (1 - 0.10)
    assert min_verification_rate(mu=0.39, alpha=0.10) == pytest.approx(0.29 / 0.90)


def test_floor_grows_as_alpha_tightens():
    floors = [min_verification_rate(MU_GLOBAL_GSM8K, a) for a in (0.40, 0.30, 0.20, 0.10)]
    assert floors == sorted(floors), "tighter alpha must demand more verification"


def test_measured_gsm8k_floors():
    """The numbers the config's alpha choice rests on."""
    assert min_verification_rate(MU_GLOBAL_GSM8K, 0.10) == pytest.approx(0.323, abs=0.002)
    assert min_verification_rate(MU_GLOBAL_GSM8K, 0.30) == pytest.approx(0.130, abs=0.002)
    assert min_verification_rate(MU_GLOBAL_GSM8K, 0.40) == 0.0


def test_local_and_global_risk_give_different_answers():
    """The reason `risk:` must be stated explicitly in the config.

    The same alpha=0.10 costs essentially nothing against local risk and a
    third of the budget against global risk -- a ~260x difference in floor.

    Note mu_local = 0.1011 sits just ABOVE 0.10, so the local floor is not
    literally zero: alpha=0.10 is marginally infeasible even against local
    risk, by 0.12%. Close enough to the boundary that measurement noise
    decides it, which is itself worth knowing before quoting alpha=0.10 as
    safe.
    """
    local_floor = min_verification_rate(MU_LOCAL_GSM8K, 0.10)
    global_floor = min_verification_rate(MU_GLOBAL_GSM8K, 0.10)

    assert local_floor < 0.01, "local floor should be negligible"
    assert global_floor > 0.30, "global floor should dominate the budget"
    assert global_floor > 100 * local_floor


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        min_verification_rate(mu=1.5, alpha=0.1)
    with pytest.raises(ValueError):
        min_verification_rate(mu=0.3, alpha=0.0)


def test_feasible_alpha_removes_the_floor():
    a = feasible_alpha(MU_GLOBAL_GSM8K)
    assert min_verification_rate(MU_GLOBAL_GSM8K, a) == 0.0


# ---- the runtime guard -----------------------------------------------


def test_check_alpha_raises_when_budget_is_below_the_floor():
    """A config that cannot reach its own target must fail at setup, not
    produce a flat results table."""
    with pytest.raises(ValueError, match="infeasible configuration"):
        check_alpha(alpha=0.10, mu=MU_GLOBAL_GSM8K, budget_fraction=0.05)


def test_check_alpha_passes_with_adequate_budget():
    r = check_alpha(alpha=0.30, mu=MU_GLOBAL_GSM8K, budget_fraction=0.56)
    assert not r.feasible  # a floor exists...
    assert r.min_verification_rate == pytest.approx(0.130, abs=0.002)  # ...but is affordable


def test_report_renders():
    assert "base risk mu" in alpha_report(MU_GLOBAL_GSM8K)


# ---- the shipped config ----------------------------------------------


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_config_alpha_is_no_longer_the_infeasible_default():
    """Regression: alpha was 0.1, which charges a 32% verification entry fee
    against measured global risk."""
    assert _config()["conformal"]["alpha"] > 0.1


def test_config_states_which_risk_it_targets():
    c = _config()["conformal"]
    assert c["risk"] in ("global", "local")
    assert "base_risk_mu" in c


def test_shipped_config_budget_clears_its_own_floor():
    """The config must be internally consistent: its budget has to exceed the
    floor implied by its own alpha and mu."""
    c = _config()
    conf, ctrl = c["conformal"], c["control"]
    check_alpha(
        alpha=conf["alpha"],
        mu=conf["base_risk_mu"],
        budget_fraction=ctrl["budget_fraction"],
    )


def test_config_sweeps_alpha_rather_than_reporting_one_point():
    sweep = _config()["conformal"]["alpha_sweep"]
    assert len(sweep) >= 3
    assert _config()["conformal"]["alpha"] in sweep


def test_config_does_not_default_to_influence_weighting():
    """Refuted four times over; it must not be the shipped default."""
    ctrl = _config()["control"]
    assert ctrl["allocation"] != "influence"
    assert ctrl["influence"] == "none"
