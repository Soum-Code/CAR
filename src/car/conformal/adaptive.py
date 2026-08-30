"""Adaptive calibration under censored feedback -- the core research component.

The problem
-----------
Gibbs & Candes (NeurIPS 2021) give an online update that tracks long-run
coverage under distribution shift:

    q_{t+1} = q_t + gamma * (alpha - err_t)

It assumes err_t is observed at every step. In CAR it is not, and the way it is
missing is unusually hostile:

    score > threshold  ->  VERIFY   ->  label observed
    score <= threshold ->  CONTINUE ->  label NEVER observed

The risk we want to control is the error rate among ACCEPTED steps. Those are
exactly the steps we get no feedback on. The estimand and the observable set
are disjoint by construction, and the censoring mechanism is the very threshold
we are trying to update -- a loop that blinds itself as it loosens.

This is the "selective labels" problem (Lakkaraju, Kleinberg, Leskovec, Ludwig,
Mullainathan, KDD 2017), where a judge's bail decision determines whether the
outcome is ever seen. Their fix exploits variation in leniency across judges.
We have one policy, so we manufacture that variation ourselves.

The approach
------------
With probability epsilon, verify regardless of what the gate says. Those forced
verifications are the only labels drawn from the accept region under a known
propensity, so they are the only ones that can support an unbiased risk
estimate. The cost is epsilon of the verification budget spent on exploration.

Three update modes are implemented so the effect is measurable rather than
asserted:

  naive             Update on every observed label, ignoring provenance. This
                    is what you get by importing ACI unchanged. It estimates
                    risk on the REJECT region and applies it to the ACCEPT
                    region -- biased toward runaway conservatism. Expected to
                    fail; included because demonstrating that failure is the
                    motivating figure of the paper.

  exploration_only  Update only on forced-exploration labels from the accept
                    region. Unbiased, but the effective learning rate is
                    gamma * epsilon, so it adapts slowly.

  ipw               Same labels, inverse-propensity weighted by 1/epsilon.
                    Recovers the full-feedback update magnitude in expectation
                    at the cost of higher variance.

Status
------
The convergence rate this is designed to test -- selective risk approaching
alpha at O(1/sqrt(epsilon * T)) with budget overhead epsilon * T -- is a
HYPOTHESIS. It is not proven here and must not be reported as a guarantee. This
module is the instrument for testing it empirically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

UpdateMode = Literal["naive", "exploration_only", "ipw"]


@dataclass
class ThresholdTrace:
    """History of the threshold and the signals that moved it.

    Kept because the adaptive dynamics are the object of study; a single final
    threshold tells you nothing about whether it was stable or oscillating.
    """

    thresholds: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    explored: list[bool] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "threshold": np.asarray(self.thresholds),
            "error": np.asarray(self.errors),
            "weight": np.asarray(self.weights),
            "explored": np.asarray(self.explored, dtype=bool),
        }


class AdaptiveCalibrator:
    """Online threshold with forced exploration to break feedback censoring.

    Parameters
    ----------
    alpha:
        Target risk level for accepted steps.
    gamma:
        Step size of the online update.
    epsilon:
        Forced-exploration rate. Setting this to 0.0 reproduces the censored
        setting exactly, which is the ablation showing why exploration is
        needed.
    update_mode:
        See module docstring.
    clip:
        (low, high) bounds on the threshold. Without clipping a biased update
        can drive the threshold off to a degenerate always-verify or
        never-verify policy and the run tells you nothing.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        gamma: float = 0.05,
        epsilon: float = 0.1,
        update_mode: UpdateMode = "ipw",
        clip: tuple[float, float] | None = None,
        seed: int = 0,
    ) -> None:
        if update_mode not in ("naive", "exploration_only", "ipw"):
            raise ValueError(f"unknown update_mode: {update_mode!r}")
        if update_mode == "ipw" and epsilon <= 0:
            raise ValueError("ipw update requires epsilon > 0 (division by propensity)")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_mode = update_mode
        self.clip = clip
        self._rng = np.random.default_rng(seed)

        self._threshold: float | None = None
        self.trace = ThresholdTrace()
        self.n_updates = 0
        self.n_exploration_calls = 0

    # ---- setup -------------------------------------------------------

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("calibrator used before fit()")
        return self._threshold

    def fit(self, scores: np.ndarray, labels: np.ndarray | None = None) -> "AdaptiveCalibrator":
        """Initialise from a calibration set, then adapt online from there.

        Starting from a split-conformal threshold rather than an arbitrary
        value means the online phase begins somewhere defensible; the adaptive
        layer is a correction, not a from-scratch search.
        """
        from car.conformal.split import conformal_quantile

        s = np.asarray(scores, dtype=float)
        if labels is not None:
            lab = np.asarray(labels, dtype=bool)
            s_fit = s[lab]
            if s_fit.size == 0:
                raise ValueError("no correct steps in calibration set")
        else:
            s_fit = s

        self._threshold = conformal_quantile(s_fit, self.alpha)
        if self.clip is None:
            # Default bounds from the observed score range, widened a little so
            # the threshold can move but cannot run away.
            lo, hi = float(s.min()), float(s.max())
            span = hi - lo + 1e-9
            self.clip = (lo - 0.5 * span, hi + 0.5 * span)
        self.trace.thresholds.append(self._threshold)
        return self

    # ---- online loop -------------------------------------------------

    def should_explore(self) -> bool:
        """Draw the forced-verification coin for this step.

        Must be called BEFORE the gate decision and its result logged, so that
        the propensity of observing each label is known exactly at analysis
        time. Deciding to explore after seeing the gate fire would destroy the
        very property that makes these labels unbiased.
        """
        if self.epsilon <= 0.0:
            return False
        explore = bool(self._rng.random() < self.epsilon)
        if explore:
            self.n_exploration_calls += 1
        return explore

    def update(
        self,
        observed_error: int | float,
        *,
        was_exploration: bool = False,
        was_accepted: bool = False,
    ) -> None:
        """Apply the online threshold update for one observed label.

        Parameters
        ----------
        observed_error:
            1 if the step turned out to be wrong, 0 otherwise.
        was_exploration:
            Whether this label came from a forced verification.
        was_accepted:
            Whether the gate would have ACCEPTED this step (score <= threshold).
            This is what identifies a label as belonging to the accept region --
            the region whose risk we are trying to control.
        """
        if self._threshold is None:
            raise RuntimeError("calibrator used before fit()")

        weight = self._update_weight(was_exploration=was_exploration, was_accepted=was_accepted)
        if weight == 0.0:
            return

        err = float(observed_error)
        # err > alpha  ->  negative step  ->  threshold falls  ->  gate tightens
        # ->  more steps get verified. Direction check: this is the whole
        # safety property, so it is asserted by test_adaptive_direction.
        self._threshold += self.gamma * weight * (self.alpha - err)
        if self.clip is not None:
            self._threshold = float(np.clip(self._threshold, *self.clip))

        self.n_updates += 1
        self.trace.thresholds.append(self._threshold)
        self.trace.errors.append(err)
        self.trace.weights.append(weight)
        self.trace.explored.append(was_exploration)

    def _update_weight(self, *, was_exploration: bool, was_accepted: bool) -> float:
        if self.update_mode == "naive":
            # Uses whatever label happens to arrive. Mostly reject-region
            # labels, which is precisely the bias being demonstrated.
            return 1.0

        if self.update_mode == "exploration_only":
            return 1.0 if (was_exploration and was_accepted) else 0.0

        # ipw: only accept-region labels carry information about accept-region
        # risk, and they arrive with propensity epsilon.
        if was_accepted and was_exploration:
            return 1.0 / self.epsilon
        return 0.0

    # ---- diagnostics -------------------------------------------------

    def exploration_overhead(self, total_steps: int) -> float:
        """Fraction of steps spent on forced verification."""
        return self.n_exploration_calls / total_steps if total_steps else 0.0

    def summary(self) -> dict:
        return {
            "update_mode": self.update_mode,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "final_threshold": self._threshold,
            "n_updates": self.n_updates,
            "n_exploration_calls": self.n_exploration_calls,
        }
