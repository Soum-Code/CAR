"""Does 'verify early' survive once REGENERATION cost is counted?

exp_allocation.py established that H3 ("verify early beats verify late") is
false on final-answer accuracy under every model variant tested. Front-loading
never won.

But those experiments measured only accuracy and verification calls. They
ignored the cost channel that actually distinguishes early from late: work
thrown away.

Catching a corrupted premise at step 1 costs one revision. Catching the same
premise at step 7 means steps 2-7 were all generated on top of it and must be
redone. Late verification is cheap in CALLS and expensive in TOKENS.

This is the honest test of whether influence-weighted allocation has a
defensible objective. Run: python scripts/exp_cost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.propagation import PropagationChain, schedule_policy  # noqa: E402

LENGTH = 8
LOCAL_ERR = 0.15
BUDGET = 3.0
N = 40000
SNOWBALL_DECAY = 0.377


def shapes(length, budget):
    front = np.array([length - i - 1 for i in range(length)], dtype=float)
    back = np.array([i + 1 for i in range(length)], dtype=float)
    return {
        "front-loaded": list(np.clip(front / front.sum() * budget, 0, 1)),
        "uniform": [budget / length] * length,
        "back-loaded": list(np.clip(back / back.sum() * budget, 0, 1)),
    }


def evaluate(schedule, scope, decay, n=N, seed=1):
    chain = PropagationChain(
        length=LENGTH,
        local_error_rate=LOCAL_ERR,
        verifier_scope=scope,
        scope_decay=decay,
    )
    rng = np.random.default_rng(seed)
    err, regen, calls = [], [], []
    for _ in range(n):
        r = chain.run(schedule_policy(schedule), rng, budget=LENGTH)
        err.append(not r.final_correct("terminal"))
        regen.append(r.regenerated_steps)
        calls.append(r.verification_calls)
    return {
        "error": float(np.mean(err)),
        "regen": float(np.mean(regen)),
        "calls": float(np.mean(calls)),
    }


def main():
    for decay, label in [
        (1.0, "constant scope"),
        (SNOWBALL_DECAY, f"decaying scope ({SNOWBALL_DECAY}, arXiv:2608.14588)"),
    ]:
        print("=" * 86)
        print(f"Accuracy vs regeneration cost -- {label}")
        print("=" * 86)
        print(f"budget {BUDGET:.0f} of {LENGTH} steps, global verifier (scope=1.0)\n")
        print(f"{'shape':<16}{'final err':>11}{'v-calls':>10}{'regen steps':>13}"
              f"{'total cost':>12}{'err x cost':>12}")
        print("-" * 86)

        rows = {}
        for name, sched in shapes(LENGTH, BUDGET).items():
            m = evaluate(sched, 1.0, decay)
            total = LENGTH + m["calls"] + m["regen"]
            rows[name] = (m, total)
            print(f"{name:<16}{m['error']:>11.4f}{m['calls']:>10.2f}"
                  f"{m['regen']:>13.2f}{total:>12.2f}{m['error'] * total:>12.4f}")

        cheap = min(rows, key=lambda k: rows[k][1])
        accurate = min(rows, key=lambda k: rows[k][0]["error"])
        print(f"\n  lowest total cost : {cheap}")
        print(f"  lowest error      : {accurate}")
        if cheap != accurate:
            print("  -> the two objectives DISAGREE; the choice is a real trade-off")
        else:
            print("  -> same winner on both; no trade-off to exploit")
        print()

    print("=" * 86)
    print("Regeneration cost as a function of WHERE the budget goes")
    print("=" * 86)
    print("global verifier, constant scope, varying budget\n")
    print(f"{'budget':<9}" + "".join(f"{s:>26}" for s in
                                     ("front-loaded", "back-loaded")))
    print(f"{'':<9}" + "".join(f"{'err':>9}{'regen':>9}{'cost':>8}" for _ in range(2)))
    print("-" * 86)
    for b in (1.0, 2.0, 3.0, 4.0, 6.0):
        s = shapes(LENGTH, b)
        out = ""
        for name in ("front-loaded", "back-loaded"):
            m = evaluate(s[name], 1.0, 1.0, n=20000)
            out += f"{m['error']:>9.4f}{m['regen']:>9.2f}{LENGTH + m['calls'] + m['regen']:>8.2f}"
        print(f"{b:<9.1f}{out}")
    print()


if __name__ == "__main__":
    main()
