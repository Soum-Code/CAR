"""Is this risk target attainable at all?

Kotte (arXiv:2606.29054), Proposition 3: when the base risk `mu` exceeds the
target `alpha`, ANY distribution-free method must abstain on at least

    (mu - alpha) / (1 - alpha)

of examples. No choice of score, calibrator or gate escapes it. That makes it a
question to answer BEFORE running anything, not a result to discover after.

For CAR "abstain" covers verify-or-abstain -- anything other than accepting a
step unchecked -- so the bound is a hard floor on the verification rate, and
therefore on cost.

Measured on GSM8K (Math-Shepherd, Mistral-7B-SFT, 93k steps; see
docs/FINDINGS-PROPAGATION.md):

    mu_global ~ 0.39    steps that do not lead to a correct answer
    mu_local  ~ 0.10    steps whose arithmetic is actually wrong

Those give very different answers, which is the whole point of the local/global
distinction: `alpha = 0.10` is comfortable against local risk and forces a 32%
verification floor against global risk.
"""

from __future__ import annotations

from dataclasses import dataclass

# Measured base risks on GSM8K. See docs/FINDINGS-PROPAGATION.md, Addendum 4.
# Post-stratified to the model's reported accuracy; re-measure for a different
# generator before relying on these.
MU_GLOBAL_GSM8K = 0.3908
MU_LOCAL_GSM8K = 0.1011


@dataclass
class Feasibility:
    alpha: float
    mu: float
    feasible: bool
    min_verification_rate: float

    def __str__(self) -> str:
        if self.feasible:
            return (
                f"alpha={self.alpha:.2f} is attainable at mu={self.mu:.4f} "
                f"(no floor: mu <= alpha)"
            )
        return (
            f"alpha={self.alpha:.2f} at mu={self.mu:.4f} forces verifying at least "
            f"{self.min_verification_rate:.1%} of steps before any method is admissible"
        )


def min_verification_rate(mu: float, alpha: float) -> float:
    """Lower bound on the fraction of steps that must be verified or abstained on."""
    if not 0.0 <= mu <= 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if mu <= alpha:
        return 0.0
    return (mu - alpha) / (1.0 - alpha)


def check_alpha(alpha: float, mu: float, *, budget_fraction: float | None = None) -> Feasibility:
    """Assess a target, and raise if the budget provably cannot reach it.

    A configuration whose verification budget sits below the floor cannot hit
    its own risk target no matter how good the uncertainty score is. Failing
    loudly at setup beats discovering it from a flat results table.
    """
    floor = min_verification_rate(mu, alpha)
    result = Feasibility(
        alpha=alpha, mu=mu, feasible=mu <= alpha, min_verification_rate=floor
    )
    if budget_fraction is not None and budget_fraction < floor:
        raise ValueError(
            f"infeasible configuration: alpha={alpha:.2f} at mu={mu:.4f} requires "
            f"verifying >= {floor:.1%} of steps, but the budget allows "
            f"{budget_fraction:.1%}. Raise alpha, raise the budget, or improve "
            f"the base model."
        )
    return result


def feasible_alpha(mu: float, *, margin: float = 0.0) -> float:
    """Smallest alpha needing no verification floor at this base risk.

    `margin` adds headroom above mu. Returns the value at which the Kotte floor
    vanishes -- not a recommendation to use it, since a large alpha is a weak
    guarantee.
    """
    return min(0.999, mu + margin)


def alpha_report(mu: float, alphas=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50)) -> str:
    """Human-readable feasibility table, for run manifests and logs."""
    lines = [f"base risk mu = {mu:.4f}", f"{'alpha':>8}{'floor':>12}{'status':>14}"]
    for a in alphas:
        f = min_verification_rate(mu, a)
        status = "no floor" if f == 0.0 else "floored"
        lines.append(f"{a:>8.2f}{f:>11.1%}{status:>14}")
    return "\n".join(lines)
