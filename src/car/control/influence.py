"""Downstream influence from the reasoning dependency graph.

The observation this module exists for: uncertainty alone is the wrong trigger.
A step that is uncertain but that nothing else depends on is cheap to get wrong.
A step that is only mildly uncertain but carries five downstream conclusions is
expensive to get wrong. Verification should follow expected downstream loss,
not raw uncertainty.

    verification_value ~ P(step is wrong) * downstream_influence

`dependency_ids` in the step schema is what makes this computable, and computing
it is also what makes "verify early" a PREDICTION of the framework rather than
a separate empirical observation -- early steps have more descendants by
construction, so they score higher.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from car.types import ReasoningStep


def build_adjacency(steps: Sequence[ReasoningStep]) -> dict[int, list[int]]:
    """Map each step_id to the ids of steps that directly depend on it."""
    children: dict[int, list[int]] = {s.step_id: [] for s in steps}
    for s in steps:
        for dep in s.dependency_ids:
            if dep in children:
                children[dep].append(s.step_id)
    return children


def descendant_counts(steps: Sequence[ReasoningStep]) -> dict[int, int]:
    """Number of steps transitively reachable from each step.

    Computed by reverse topological sweep. Steps are emitted in increasing
    step_id and may only depend on earlier ids (enforced by the schema), so
    iterating in reverse order guarantees children are resolved first.
    """
    children = build_adjacency(steps)
    order = sorted((s.step_id for s in steps), reverse=True)
    reachable: dict[int, set[int]] = {}
    for sid in order:
        acc: set[int] = set()
        for c in children.get(sid, []):
            acc.add(c)
            acc |= reachable.get(c, set())
        reachable[sid] = acc
    return {sid: len(v) for sid, v in reachable.items()}


class InfluenceWeighting:
    """Turns a descendant count into a multiplier on the uncertainty score.

    Modes
    -----
    none    Ignore structure entirely. This is the pure-uncertainty gate, i.e.
            the ablation that isolates what influence-awareness contributes.
    linear  1 + n_descendants. Aggressive; long chains dominate.
    sqrt    1 + sqrt(n). Damped, usually the sensible default.
    log     1 + log(1 + n). Most conservative.
    """

    def __init__(self, mode: str = "sqrt", cap: float | None = 4.0) -> None:
        if mode not in ("none", "linear", "sqrt", "log"):
            raise ValueError(f"unknown influence mode: {mode!r}")
        self.mode = mode
        self.cap = cap

    def __call__(self, n_descendants: int) -> float:
        n = max(0, int(n_descendants))
        if self.mode == "none":
            return 1.0
        if self.mode == "linear":
            w = 1.0 + n
        elif self.mode == "sqrt":
            w = 1.0 + np.sqrt(n)
        else:
            w = 1.0 + np.log1p(n)
        return float(min(w, self.cap) if self.cap is not None else w)


def prospective_influence(step_id: int, n_planned_steps: int, mode: str = "sqrt") -> float:
    """Influence estimate available BEFORE later steps exist.

    The gate has to decide at step t, when the descendants of step t have not
    been generated yet. Using the true descendant count online would be
    reading the future. This approximates it from position in the plan: an
    early step in an n-step plan can influence at most n - t - 1 later steps.

    `descendant_counts` remains the exact quantity for offline analysis.
    """
    remaining = max(0, n_planned_steps - step_id - 1)
    return InfluenceWeighting(mode=mode)(remaining)
