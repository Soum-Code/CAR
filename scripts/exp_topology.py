"""Does influence-weighted allocation survive on non-chain topologies?

FINDINGS-PROPAGATION.md rejected influence weighting, but that test could not
actually see it: on a linear chain, descendant count is a strictly decreasing
function of position, so "influence-weighted" IS "front-loaded". The chain also
routes every error through the terminal node, which hands a structural win to
back-loading that has nothing to do with reasoning.

This script re-runs the comparison where influence and position come apart.

Run: python scripts/exp_topology.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.topology import (  # noqa: E402
    SCHEDULES,
    DAGPropagation,
    bushy,
    chain,
    converging_tree,
    diamond,
    parallel_chains,
)

LOCAL_ERR = 0.15
N_TRIALS = 30000
SNOWBALL_DECAY = 0.377

TOPOLOGIES = [
    chain(10),
    parallel_chains(3, 3),
    converging_tree(3),
    diamond(3, 3),
    bushy(13, seed=0),
    bushy(13, seed=3),
]


def evaluate(dag, schedule, scope, decay, n=N_TRIALS, seed=1, repair=1.0):
    sim = DAGPropagation(
        dag,
        local_error_rate=LOCAL_ERR,
        verifier_scope=scope,
        scope_decay=decay,
        repair_success=repair,
    )
    rng = np.random.default_rng(seed)
    err, calls = [], []
    for _ in range(n):
        r = sim.run(schedule, rng, budget=dag.n)
        err.append(not r.terminal_correct)
        calls.append(r.verification_calls)
    return float(np.mean(err)), float(np.mean(calls))


def structure_table():
    print("=" * 88)
    print("Topology structure: is influence distinguishable from position?")
    print("=" * 88)
    print(f"{'topology':<22}{'nodes':>7}{'corr(pos,infl)':>16}"
          f"{'influence profile':>40}")
    print("-" * 88)
    for dag in TOPOLOGIES:
        prof = dag.influence_profile().astype(int)
        shown = " ".join(str(x) for x in prof[:14])
        if dag.n > 14:
            shown += " ..."
        r = dag.position_influence_correlation()
        print(f"{dag.name:<22}{dag.n:>7}{r:>16.3f}   {shown}")
    print("\ncorr near -1 => influence IS position (chain); the policies are")
    print("indistinguishable there. Values near 0 are where the test bites.\n")


def allocation_table(scope, decay, label):
    print("=" * 88)
    print(f"Allocation comparison -- {label}")
    print("=" * 88)
    print(f"final-answer error at matched budget (37.5% of nodes), "
          f"scope={scope}, decay={decay}\n")
    names = ["uniform", "front", "back", "influence", "depth", "cut"]
    print(f"{'topology':<22}" + "".join(f"{n:>11}" for n in names) + f"{'winner':>12}")
    print("-" * 88)

    wins: dict[str, int] = dict.fromkeys(names, 0)
    for dag in TOPOLOGIES:
        budget = 0.375 * dag.n
        vals, calls = {}, {}
        for nm in names:
            sched = SCHEDULES[nm](dag, budget)
            vals[nm], calls[nm] = evaluate(dag, sched, scope, decay)
        w = min(vals, key=vals.get)
        wins[w] += 1
        spread = max(calls.values()) - min(calls.values())
        flag = "" if spread < 0.15 else f"  [calls vary {spread:.2f}]"
        print(f"{dag.name:<22}" + "".join(f"{vals[n]:>11.4f}" for n in names)
              + f"{w:>12}{flag}")

    print(f"\nwins: {dict((k, v) for k, v in wins.items() if v)}\n")
    return wins


def head_to_head():
    print("=" * 88)
    print("Influence vs front-loading, isolated")
    print("=" * 88)
    print("If these are equal on a chain and differ elsewhere, the earlier")
    print("chain-only rejection of influence weighting was uninformative.\n")
    print(f"{'topology':<22}{'influence':>12}{'front':>12}{'uniform':>12}"
          f"{'infl-front':>13}{'infl-unif':>12}")
    print("-" * 88)
    for dag in TOPOLOGIES:
        budget = 0.375 * dag.n
        i, _ = evaluate(dag, SCHEDULES["influence"](dag, budget), 1.0, 1.0)
        f, _ = evaluate(dag, SCHEDULES["front"](dag, budget), 1.0, 1.0)
        u, _ = evaluate(dag, SCHEDULES["uniform"](dag, budget), 1.0, 1.0)
        print(f"{dag.name:<22}{i:>12.4f}{f:>12.4f}{u:>12.4f}"
              f"{i - f:>13.4f}{i - u:>12.4f}")
    print("\n(negative = influence weighting is better)\n")


def budget_sweep():
    print("=" * 88)
    print("Budget sweep on the topology where influence looks strongest")
    print("=" * 88)
    dag = parallel_chains(3, 3)
    print(f"{dag.name}, scope=1.0, decay=1.0\n")
    print(f"{'budget':<10}{'uniform':>12}{'front':>12}{'back':>12}"
          f"{'influence':>12}{'cut':>12}{'winner':>12}")
    print("-" * 88)
    for frac in (0.15, 0.25, 0.375, 0.5, 0.65):
        b = frac * dag.n
        vals = {}
        for nm in ("uniform", "front", "back", "influence", "cut"):
            vals[nm], _ = evaluate(dag, SCHEDULES[nm](dag, b), 1.0, 1.0, n=20000)
        w = min(vals, key=vals.get)
        print(f"{frac:<10.3f}" + "".join(
            f"{vals[n]:>12.4f}" for n in ("uniform", "front", "back", "influence", "cut")
        ) + f"{w:>12}")
    print()


def main():
    structure_table()
    head_to_head()
    allocation_table(1.0, 1.0, "global verifier, no decay")
    allocation_table(1.0, SNOWBALL_DECAY, f"global verifier, decay={SNOWBALL_DECAY}")
    allocation_table(0.3, 1.0, "weak scope (0.3)")
    budget_sweep()


if __name__ == "__main__":
    main()
