"""Where should a verification budget go?

Follow-up to exp_propagation.py, which found front-loaded ("verify early")
allocation LOSING to uniform. That contradicts hypothesis H3 in the project
spec, which asserts unconditionally that early verification beats end-only
verification.

This script establishes the condition under which H3 is actually true.

The asymmetry driving everything:

    entering corruption   e * (1 - v_t)                      no scope term
    escaping corruption   v_t * scope * decay^(age-1) * (1-e) scope AND decay

With constant scope a late check is a catch-all -- verifying step 7 repairs
anything from steps 0-6, so late calls dominate. But Singh & Pawar
(arXiv:2608.14588) measured detectability DECAYING as errors propagate: escape
probabilities of 24.6% / 48.3% / 89.3% across successive boundaries. Constant
scope is the unrealistic case.

Run: python scripts/exp_allocation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.propagation import (  # noqa: E402
    PropagationChain,
    final_error_probability,
    optimal_schedule,
    schedule_policy,
)

LENGTH = 8
LOCAL_ERR = 0.15
BUDGET = 3.0
N = 20000

# Fitted from the escape probabilities reported in arXiv:2608.14588.
# Detection = 1 - escape: 0.754, 0.517, 0.107 at ages 1, 2, 3.
# Ratios 0.517/0.754 = 0.686 and 0.107/0.517 = 0.207 -> geometric mean ~0.377.
SNOWBALL_DECAY = 0.377


def shapes(length, budget):
    front = np.array([length - i - 1 for i in range(length)], dtype=float)
    back = np.array([i + 1 for i in range(length)], dtype=float)
    return {
        "front-loaded": list(np.clip(front / front.sum() * budget, 0, 1)),
        "uniform": [budget / length] * length,
        "back-loaded": list(np.clip(back / back.sum() * budget, 0, 1)),
    }


def simulate(schedule, scope, decay, outcome, n=N, seed=1):
    chain = PropagationChain(
        length=LENGTH,
        local_error_rate=LOCAL_ERR,
        verifier_scope=scope,
        scope_decay=decay,
    )
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        r = chain.run(schedule_policy(schedule), rng, budget=LENGTH)
        errs.append(not r.final_correct(outcome))
    return float(np.mean(errs))


def validate():
    print("=" * 84)
    print("Closed form vs simulation (with decay, terminal outcome)")
    print("=" * 84)
    print(f"{'scope':<8}{'decay':<8}{'v':<7}{'closed':>11}{'simulated':>12}{'diff':>9}")
    print("-" * 84)
    worst = 0.0
    for scope in (0.0, 0.5, 1.0):
        for decay in (1.0, SNOWBALL_DECAY):
            for v in (0.25, 0.5):
                cf = final_error_probability(
                    LENGTH, v, scope, LOCAL_ERR, scope_decay=decay
                )
                sim = simulate([v] * LENGTH, scope, decay, "terminal", n=8000)
                worst = max(worst, abs(cf - sim))
                print(f"{scope:<8.1f}{decay:<8.3f}{v:<7.2f}"
                      f"{cf:>11.4f}{sim:>12.4f}{abs(cf-sim):>9.4f}")
    print(f"\nmax discrepancy: {worst:.4f}\n")


def allocation_table(decay, outcome, label):
    print("=" * 84)
    print(f"Allocation shape -- {label}")
    print("=" * 84)
    print(f"final-answer error, budget = {BUDGET:.0f} of {LENGTH} steps, "
          f"decay = {decay}, outcome = {outcome}\n")
    print(f"{'scope':<9}{'front':>11}{'uniform':>11}{'back':>11}"
          f"{'optimal':>11}{'winner':>13}{'opt gain':>10}")
    print("-" * 84)
    for scope in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = shapes(LENGTH, BUDGET)
        vals = {
            k: final_error_probability(
                LENGTH, v, scope, LOCAL_ERR, scope_decay=decay, outcome=outcome
            )
            for k, v in s.items()
        }
        opt = optimal_schedule(
            LENGTH, BUDGET, scope, LOCAL_ERR, scope_decay=decay, outcome=outcome
        )
        o = final_error_probability(
            LENGTH, opt, scope, LOCAL_ERR, scope_decay=decay, outcome=outcome
        )
        winner = min(vals, key=vals.get)
        gain = min(vals.values()) - o
        print(f"{scope:<9.2f}{vals['front-loaded']:>11.4f}{vals['uniform']:>11.4f}"
              f"{vals['back-loaded']:>11.4f}{o:>11.4f}{winner:>13}{gain:>10.4f}")
    print()


def optimal_shapes(decay, outcome, label):
    print("=" * 84)
    print(f"Optimal per-step schedule -- {label}")
    print("=" * 84)
    print(f"budget {BUDGET:.0f} of {LENGTH}, decay = {decay}, outcome = {outcome}\n")
    print(f"{'scope':<8}" + "".join(f"{f's{i}':>8}" for i in range(LENGTH)) + "   shape")
    print("-" * 84)
    for scope in (0.0, 0.25, 0.5, 1.0):
        sched = optimal_schedule(
            LENGTH, BUDGET, scope, LOCAL_ERR, scope_decay=decay, outcome=outcome
        )
        half = LENGTH // 2
        tilt = sum(sched[:half]) - sum(sched[half:])
        shape = "front" if tilt > 0.15 else ("back" if tilt < -0.15 else "flat")
        print(f"{scope:<8.2f}" + "".join(f"{x:>8.2f}" for x in sched) + f"   {shape}")
    print()


def main():
    validate()
    allocation_table(1.0, "terminal", "A: constant scope (no decay)")
    allocation_table(SNOWBALL_DECAY, "terminal",
                     f"B: decaying scope (decay={SNOWBALL_DECAY}, from arXiv:2608.14588)")
    allocation_table(1.0, "conjunctive", "C: conjunctive outcome (repair cannot undo)")
    optimal_shapes(1.0, "terminal", "A: constant scope")
    optimal_shapes(SNOWBALL_DECAY, "terminal", "B: decaying scope")

    print("=" * 84)
    print("Verdict on spec hypothesis H3 ('verify early beats verify late')")
    print("=" * 84)
    for decay, outcome, name in [
        (1.0, "terminal", "constant scope, terminal"),
        (SNOWBALL_DECAY, "terminal", "decaying scope, terminal"),
        (1.0, "conjunctive", "conjunctive outcome"),
    ]:
        s = shapes(LENGTH, BUDGET)
        f = final_error_probability(LENGTH, s["front-loaded"], 1.0, LOCAL_ERR,
                                    scope_decay=decay, outcome=outcome)
        b = final_error_probability(LENGTH, s["back-loaded"], 1.0, LOCAL_ERR,
                                    scope_decay=decay, outcome=outcome)
        verdict = "H3 HOLDS" if f < b else ("tie" if abs(f - b) < 1e-6 else "H3 FALSE")
        print(f"  {name:<32} front={f:.4f}  back={b:.4f}   {verdict}")
    print()


if __name__ == "__main__":
    main()
