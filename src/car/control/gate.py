"""The control gate: CONTINUE, VERIFY, or ABSTAIN.

Decision order matters and is not arbitrary:

1. Draw the forced-exploration coin FIRST, before looking at the score. If
   exploration were conditioned on the gate outcome, the resulting labels would
   no longer have a known propensity and the whole censored-feedback correction
   would collapse.
2. Compute the gate value: uncertainty scaled by downstream influence.
3. Compare against the (possibly budget-adjusted) threshold.
4. Fall back to CONTINUE when the budget is exhausted -- and record that, since
   a forced CONTINUE is not the same event as a confident CONTINUE.
"""

from __future__ import annotations

from dataclasses import dataclass

from car.control.budget import Budget, BudgetPolicy
from car.control.influence import InfluenceWeighting
from car.types import Decision


@dataclass
class GateOutcome:
    decision: Decision
    gate_value: float
    threshold: float
    influence: float
    forced_exploration: bool
    budget_blocked: bool = False


class ControlGate:
    """Maps (uncertainty score, influence, budget) to an action.

    Parameters
    ----------
    influence_weighting:
        Set mode="none" to recover a pure-uncertainty gate. That is the
        ablation isolating what structure-awareness buys.
    abstain_threshold:
        Gate value above which the step is abandoned rather than verified --
        the model is so uncertain that verification is unlikely to rescue it.
        None disables abstention.
    """

    def __init__(
        self,
        *,
        influence_weighting: InfluenceWeighting | None = None,
        budget_policy: BudgetPolicy | None = None,
        abstain_threshold: float | None = None,
    ) -> None:
        self.influence_weighting = influence_weighting or InfluenceWeighting(mode="none")
        self.budget_policy = budget_policy or BudgetPolicy(mode="none")
        self.abstain_threshold = abstain_threshold

    def decide(
        self,
        score: float,
        threshold: float,
        *,
        n_descendants: int = 0,
        budget: Budget | None = None,
        force_explore: bool = False,
    ) -> GateOutcome:
        influence = self.influence_weighting(n_descendants)
        gate_value = score * influence

        effective_threshold = threshold
        if budget is not None:
            effective_threshold = threshold * self.budget_policy.scale(budget)

        # Forced exploration: verify regardless of the gate. Recorded as such
        # so downstream analysis knows this label has propensity epsilon.
        if force_explore:
            if budget is None or budget.can_afford():
                return GateOutcome(
                    decision=Decision.VERIFY,
                    gate_value=gate_value,
                    threshold=effective_threshold,
                    influence=influence,
                    forced_exploration=True,
                )

        if self.abstain_threshold is not None and gate_value > self.abstain_threshold:
            return GateOutcome(
                decision=Decision.ABSTAIN,
                gate_value=gate_value,
                threshold=effective_threshold,
                influence=influence,
                forced_exploration=False,
            )

        if gate_value > effective_threshold:
            if budget is not None and not budget.can_afford():
                # Wanted to verify, could not pay. This is a distinct failure
                # mode from a confident CONTINUE and must not be conflated
                # with one in the results.
                return GateOutcome(
                    decision=Decision.CONTINUE,
                    gate_value=gate_value,
                    threshold=effective_threshold,
                    influence=influence,
                    forced_exploration=False,
                    budget_blocked=True,
                )
            return GateOutcome(
                decision=Decision.VERIFY,
                gate_value=gate_value,
                threshold=effective_threshold,
                influence=influence,
                forced_exploration=False,
            )

        return GateOutcome(
            decision=Decision.CONTINUE,
            gate_value=gate_value,
            threshold=effective_threshold,
            influence=influence,
            forced_exploration=False,
        )

    @staticmethod
    def was_accepted(outcome: GateOutcome) -> bool:
        """Whether the gate WOULD have accepted this step on its own merits.

        Distinct from `decision == CONTINUE`: a step verified by forced
        exploration may still have been one the gate would have accepted, and
        that is exactly the case whose label the adaptive calibrator needs.
        """
        return outcome.gate_value <= outcome.threshold
