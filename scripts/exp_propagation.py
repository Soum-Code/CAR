"""Stress-test of the surviving contribution (POSITIONING.md section 4.1).

Three questions, each of which can kill the thesis if it comes back the wrong
way. Run: python scripts/exp_propagation.py

Q1. Does controlling LOCAL selective risk bound final-answer error?
    If yes, CAR reduces to CSA with a different loss and there is no thesis.

Q2. Does verifier scope change the picture qualitatively?
    If a purely local verifier can still bound final error given enough budget,
    the local/global distinction is quantitative decoration, not structure.

Q3. Does influence-weighted allocation beat uniform AT MATCHED COST?
    If not, the second contribution is dead too and only the framing survives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.propagation import (  # noqa: E402
    PropagationChain,
    always,
    influence_weighted,
    never,
    survival_probability,
    uniform_random,
)

N_TRIALS = 20000
LENGTH = 8
LOCAL_ERR = 0.15


def evaluate(chain: PropagationChain, policy, n=N_TRIALS, seed=0, budget=None):
    rng = np.random.default_rng(seed)
    final, local_r, global_r, calls = [], [], [], []
    for _ in range(n):
        r = chain.run(policy, rng, budget=budget)
        final.append(r.final_correct)
        local_r.append(r.local_selective_risk)
        global_r.append(r.global_selective_risk)
        calls.append(r.verification_calls)
    return {
        "final_accuracy": float(np.mean(final)),
        "final_error": 1.0 - float(np.mean(final)),
        "local_risk": float(np.mean(local_r)),
        "global_risk": float(np.mean(global_r)),
        "calls": float(np.mean(calls)),
    }


def q1_local_vs_global():
    print("=" * 78)
    print("Q1. Does controlling LOCAL selective risk bound final-answer error?")
    print("=" * 78)
    print("Purely local verifier (scope=0.0), chain length 8, local error 15%.\n")

    chain = PropagationChain(
        length=LENGTH, local_error_rate=LOCAL_ERR, verifier_scope=0.0
    )
    print(f"{'policy':<22}{'local risk':>12}{'global risk':>13}"
          f"{'final err':>12}{'calls':>9}")
    print("-" * 78)
    for name, pol in [
        ("never verify", never),
        ("uniform 25%", uniform_random(0.25)),
        ("uniform 50%", uniform_random(0.50)),
        ("uniform 75%", uniform_random(0.75)),
        ("always verify", always),
    ]:
        m = evaluate(chain, pol)
        print(f"{name:<22}{m['local_risk']:>12.4f}{m['global_risk']:>13.4f}"
              f"{m['final_error']:>12.4f}{m['calls']:>9.2f}")

    print("\nReading: local risk -> 0 as budget grows, but final error does not.")
    print("A local verifier repairs every step it inspects and still cannot see")
    print("that a premise was already corrupt.\n")


def q2_verifier_scope():
    print("=" * 78)
    print("Q2. Does verifier SCOPE change the picture qualitatively?")
    print("=" * 78)
    print("Final-answer error vs verification budget, by verifier scope.\n")

    rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"{'scope':<10}" + "".join(f"{f'v={r:.0%}':>12}" for r in rates))
    print("-" * 78)
    for scope in (0.0, 0.25, 0.5, 1.0):
        chain = PropagationChain(
            length=LENGTH, local_error_rate=LOCAL_ERR, verifier_scope=scope
        )
        row = ""
        for r in rates:
            pol = never if r == 0.0 else (always if r == 1.0 else uniform_random(r))
            row += f"{evaluate(chain, pol, n=8000)['final_error']:>12.4f}"
        print(f"{scope:<10.2f}{row}")

    print("\nClosed form (survival_probability), scope=0 vs scope=1:")
    for scope in (0.0, 1.0):
        vals = [survival_probability(LENGTH, r, scope, LOCAL_ERR) for r in rates]
        print(f"  scope={scope:.1f}: " + "  ".join(f"{v:.4f}" for v in vals))
    print()


def q3_allocation():
    print("=" * 78)
    print("Q3. Does influence-weighted allocation beat uniform AT MATCHED COST?")
    print("=" * 78)
    print("Global verifier (scope=1.0) -- the regime where budget actually helps.\n")

    chain = PropagationChain(
        length=LENGTH, local_error_rate=LOCAL_ERR, verifier_scope=1.0
    )
    print(f"{'budget':<10}{'uniform':>14}{'influence-lin':>16}"
          f"{'influence-sqrt':>16}{'delta':>10}")
    print("-" * 78)
    for base in (0.125, 0.25, 0.375, 0.5):
        u = evaluate(chain, uniform_random(base), n=20000, seed=1)
        il = evaluate(chain, influence_weighted(base, "linear"), n=20000, seed=1)
        isq = evaluate(chain, influence_weighted(base, "sqrt"), n=20000, seed=1)
        best = min(il["final_error"], isq["final_error"])
        print(f"{base:<10.3f}{u['final_error']:>14.4f}{il['final_error']:>16.4f}"
              f"{isq['final_error']:>16.4f}{u['final_error'] - best:>10.4f}")
        print(f"{'  (calls)':<10}{u['calls']:>14.2f}{il['calls']:>16.2f}"
              f"{isq['calls']:>16.2f}")

    print("\nCalls must match across columns, otherwise this is just 'verify more'.\n")


def main():
    q1_local_vs_global()
    q2_verifier_scope()
    q3_allocation()


if __name__ == "__main__":
    main()
