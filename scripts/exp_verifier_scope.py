"""Measure verifier scope: can a verifier see inherited corruption, and at what cost?

`scope` has been a free parameter in every propagation result. It is the
variable C3 claims controls the whole design, and no published work reports it.

Reframing that makes it measurable without a GPU: scope is not a property of a
verifier type, it is a property of how much context the verifier re-examines.
A calculator checking only step t cannot know its operands came from a wrong
step t-2. The same calculator re-checking t-k..t can. So reach is a window.

Measured here on real Math-Shepherd GSM8K solutions:

  Q1  What is the scope of a naive step-local verifier?     (expect ~0)
  Q2  How does scope grow with lookback window k?
  Q3  What does reach cost, in steps re-examined per call?
  Q4  Is there a ceiling below 1, and what causes it?

Run: python scripts/exp_verifier_scope.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data.math_shepherd import load_solutions  # noqa: E402
from car.verification.lookback import (  # noqa: E402
    LookbackVerifier,
    corruption_distance,
    inherited_steps,
)

DATA = Path("data/raw/mathshepherd/strided.jsonl")
WINDOWS = [0, 1, 2, 3, 5, None]


def _label(k):
    return "inf" if k is None else str(k)


def population(sols):
    """The steps whose detectability defines scope."""
    out = []
    for sol in sols:
        for s in inherited_steps(sol):
            d = corruption_distance(sol, s.index)
            out.append((sol, s, d))
    return out


def q1_q2_scope_by_window(sols, pop):
    print("=" * 84)
    print("Q1/Q2. Scope as a function of lookback window")
    print("=" * 84)
    print("Population: steps that are locally VALID but globally WRONG --")
    print(f"arithmetically perfect, wrong only because a premise was. n = {len(pop):,}\n")

    print(f"{'window k':<12}{'scope':>10}{'detected':>12}{'cost/call':>12}"
          f"{'unverifiable in window':>26}")
    print("-" * 84)
    results = {}
    for k in WINDOWS:
        v = LookbackVerifier(k=k)
        det = 0
        unver = 0
        for sol, step, _ in pop:
            r = v.verify_step(sol, step.index)
            det += r.detected
            unver += r.n_unverifiable
        scope = det / max(1, len(pop))
        results[k] = scope
        print(f"{_label(k):<12}{scope:>10.4f}{det:>12,}{v.cost_per_call():>12.2f}"
              f"{unver / max(1, len(pop)):>26.2f}")

    print()
    print(f"naive step-local verifier (k=0) scope = {results[0]:.4f}")
    print(f"unbounded-lookback verifier     scope = {results[None]:.4f}")
    print()
    return results


def q3_decay_by_distance(sols, pop):
    print("=" * 84)
    print("Q3. Detection vs propagation distance -- the decay curve")
    print("=" * 84)
    print("d = hops from the nearest upstream arithmetic error.")
    print("Comparable to Singh & Pawar's escape probabilities (24.6/48.3/89.3%).\n")

    by_d = {}
    for sol, step, d in pop:
        if d is None:
            continue
        by_d.setdefault(d, []).append((sol, step))

    ks = [0, 1, 2, 3, None]
    print(f"{'d':<6}{'n':>9}" + "".join(f"{'k=' + _label(k):>10}" for k in ks))
    print("-" * 84)
    for d in sorted(by_d):
        group = by_d[d]
        if len(group) < 50:
            continue
        row = ""
        for k in ks:
            v = LookbackVerifier(k=k)
            det = sum(v.verify_step(sol, s.index).detected for sol, s in group)
            row += f"{det / len(group):>10.3f}"
        print(f"{d:<6}{len(group):>9,}{row}")

    print()
    print("Reading: a verifier with window k detects corruption originating")
    print("within k hops. Beyond that it is blind, which is exactly the decay")
    print("the propagation model parameterises as scope * decay^(d-1).")
    print()
    return by_d


def q4_ceiling(sols, pop):
    print("=" * 84)
    print("Q4. Why scope does not reach 1, even with unlimited lookback")
    print("=" * 84)

    v = LookbackVerifier(k=None)
    missed = [(sol, s) for sol, s, _ in pop if not v.verify_step(sol, s.index).detected]
    print(f"inherited-corruption steps                 {len(pop):,}")
    print(f"  missed even at unlimited lookback        {len(missed):,} "
          f"({len(missed) / max(1, len(pop)):.1%})")
    print()

    # Why were they missed? No arithmetic error exists upstream to find.
    no_upstream_error = 0
    unverifiable_upstream = 0
    for sol, s in missed:
        prefix = [x for x in sol.steps if x.index < s.index]
        if not any(x.local_ok is False for x in prefix):
            no_upstream_error += 1
        if any(x.local_ok is None for x in prefix):
            unverifiable_upstream += 1

    print("of the missed steps:")
    print(f"  no upstream ARITHMETIC error at all      {no_upstream_error:,} "
          f"({no_upstream_error / max(1, len(missed)):.1%})")
    print(f"  have an unverifiable step upstream       {unverifiable_upstream:,} "
          f"({unverifiable_upstream / max(1, len(missed)):.1%})")
    print()
    print("The first number is the real ceiling. These steps are globally wrong")
    print("with no arithmetic mistake anywhere upstream -- the error is in the")
    print("SETUP (wrong quantity chosen, wrong operation, misread problem), not")
    print("in the calculation. No arithmetic verifier, at any window size, can")
    print("see them. Closing that gap needs a semantically different verifier,")
    print("not a wider window.")
    print()

    all_steps = [s for sol in sols for s in sol.steps]
    uncheckable = sum(1 for s in all_steps if s.local_ok is None)
    print(f"context: {uncheckable / len(all_steps):.1%} of all steps carry no "
          f"calculator annotation")
    print()


def q5_cost_benefit(sols, pop):
    print("=" * 84)
    print("Q5. Is reach worth its cost?")
    print("=" * 84)
    print("Reach is not free: window k re-examines up to k+1 steps per call.\n")
    print(f"{'window k':<12}{'scope':>10}{'cost/call':>12}{'scope per step':>16}")
    print("-" * 84)
    for k in WINDOWS:
        v = LookbackVerifier(k=k)
        det = sum(v.verify_step(sol, s.index).detected for sol, s, _ in pop)
        scope = det / max(1, len(pop))
        cost = v.cost_per_call()
        print(f"{_label(k):<12}{scope:>10.4f}{cost:>12.2f}{scope / cost:>16.4f}")
    print()


def main():
    if not DATA.exists():
        print(f"missing {DATA}; run python scripts/download_data.py")
        return
    print("loading Math-Shepherd GSM8K solutions...")
    sols = load_solutions(DATA)
    pop = population(sols)
    print(f"{len(sols):,} solutions, {len(pop):,} inherited-corruption steps\n")

    dist = Counter(d for _, _, d in pop if d is not None)
    print(f"distance distribution: {dict(sorted(dist.items())[:8])}")
    none_d = sum(1 for _, _, d in pop if d is None)
    print(f"no upstream arithmetic error: {none_d:,} "
          f"({none_d / max(1, len(pop)):.1%})\n")

    q1_q2_scope_by_window(sols, pop)
    q3_decay_by_distance(sols, pop)
    q4_ceiling(sols, pop)
    q5_cost_benefit(sols, pop)


if __name__ == "__main__":
    main()
