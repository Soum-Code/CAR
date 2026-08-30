"""Verification budget accounting.

Verification is a resource. The gate must be able to answer "can I afford this
call?" and, ideally, "should I save it for a step that matters more?".
"""

from __future__ import annotations


class Budget:
    """Per-question verification budget.

    `spent_on_exploration` is tracked separately because forced-exploration
    calls are a cost of the METHOD, not of the policy being evaluated. Any
    accuracy-vs-cost plot that hides exploration calls is overstating CAR's
    efficiency, so the accounting keeps them visible.
    """

    def __init__(self, total: int) -> None:
        if total < 0:
            raise ValueError("budget must be non-negative")
        self.total = total
        self.spent = 0
        self.spent_on_exploration = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def fraction_used(self) -> float:
        return self.spent / self.total if self.total else 1.0

    def can_afford(self, cost: int = 1) -> bool:
        return self.remaining >= cost

    def spend(self, cost: int = 1, *, exploration: bool = False) -> bool:
        """Deduct if affordable. Returns whether the spend happened."""
        if not self.can_afford(cost):
            return False
        self.spent += cost
        if exploration:
            self.spent_on_exploration += cost
        return True

    def reset(self) -> None:
        self.spent = 0
        self.spent_on_exploration = 0

    def summary(self) -> dict[str, int | float]:
        return {
            "total": self.total,
            "spent": self.spent,
            "spent_on_exploration": self.spent_on_exploration,
            "spent_on_gate": self.spent - self.spent_on_exploration,
            "remaining": self.remaining,
            "fraction_used": self.fraction_used,
        }


class BudgetPolicy:
    """Optional threshold tightening as budget depletes.

    With calls running out, the gate should become choosier -- spend the last
    call on a step that really needs it. `scale` multiplies the threshold; a
    value above 1 raises the bar for triggering verification.

    Set mode="none" for the clean deterministic gate. The doc's CMDP framing is
    the formal model for this; the implementation here is the deterministic
    approximation it explicitly permits.
    """

    def __init__(self, mode: str = "none", strength: float = 0.5) -> None:
        if mode not in ("none", "linear"):
            raise ValueError(f"unknown budget policy: {mode!r}")
        self.mode = mode
        self.strength = strength

    def scale(self, budget: Budget) -> float:
        if self.mode == "none":
            return 1.0
        # As the budget drains, raise the effective threshold so only the most
        # uncertain / most influential steps still trigger a call.
        return 1.0 + self.strength * budget.fraction_used
