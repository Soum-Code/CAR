"""Reasoning DAGs and topology-aware error propagation.

Why this module exists: on a linear chain, `descendant_count(t)` decreases
monotonically with `t`, so "influence-weighted allocation" and "front-loaded
allocation" are *the same policy*. The chain experiments in
FINDINGS-PROPAGATION.md therefore could not distinguish them, and rejecting
influence weighting on that evidence was premature.

A chain also has a structural quirk that hands the result to back-loading:
every error eventually flows through the terminal step, so a single late check
with global scope is a catch-all. That is a property of chains, not of
reasoning.

On a wide DAG both effects disappear:

  * influence decouples from position -- a mid-graph hub can outrank an early
    leaf
  * corruption on one branch is invisible to a verifier on a parallel branch,
    so no single late check dominates

Propagation semantics here follow EDGES, not step order:

    premise_corrupt(v) = any parent u with global_correct(u) == False
    global_correct(v)  = local_valid(v) and not premise_corrupt(v)

Repairing a node cuts corruption on paths *through that node* only. Parallel
paths from the same bad ancestor keep carrying it. That is the property a chain
cannot express and the reason this test is worth running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class DAG:
    """A reasoning graph in topological order.

    Nodes are `0..n-1` and every parent index is strictly less than its child,
    so a single forward pass respects dependencies.
    """

    n: int
    parents: tuple[tuple[int, ...], ...]
    terminal: int
    name: str = ""

    def __post_init__(self) -> None:
        for v, ps in enumerate(self.parents):
            for p in ps:
                if p >= v:
                    raise ValueError(f"node {v} has non-topological parent {p}")

    @property
    def children(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {v: [] for v in range(self.n)}
        for v, ps in enumerate(self.parents):
            for p in ps:
                out[p].append(v)
        return out

    @lru_cache(maxsize=None)
    def descendants(self, v: int) -> frozenset[int]:
        """Transitive descendants of `v` -- the blast radius if `v` is wrong."""
        ch = self.children
        seen: set[int] = set()
        stack = list(ch[v])
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(ch[u])
        return frozenset(seen)

    @lru_cache(maxsize=None)
    def ancestors(self, v: int) -> frozenset[int]:
        seen: set[int] = set()
        stack = list(self.parents[v])
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(self.parents[u])
        return frozenset(seen)

    def influence(self, v: int) -> int:
        """Descendant count. The quantity influence weighting bets on."""
        return len(self.descendants(v))

    def reaches_terminal(self, v: int) -> bool:
        return v == self.terminal or self.terminal in self.descendants(v)

    def hop_distance(self, u: int, v: int) -> int | None:
        """Shortest directed hop count u -> v, or None if unreachable.

        Used for scope decay: how far a corrupted premise has travelled before
        a verifier looks at it.
        """
        if u == v:
            return 0
        ch = self.children
        frontier, dist, seen = [u], 0, {u}
        while frontier:
            dist += 1
            nxt = []
            for x in frontier:
                for y in ch[x]:
                    if y == v:
                        return dist
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
            frontier = nxt
        return None

    def influence_profile(self) -> np.ndarray:
        return np.array([self.influence(v) for v in range(self.n)], dtype=float)

    def position_influence_correlation(self) -> float:
        """Pearson r between node index and influence.

        Near -1 means influence is just position in disguise (a chain), and
        influence weighting cannot be distinguished from front-loading. Values
        near 0 are where the comparison becomes meaningful.
        """
        inf = self.influence_profile()
        pos = np.arange(self.n, dtype=float)
        if inf.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(pos, inf)[0, 1])


# ---- topology constructors -------------------------------------------


def chain(n: int = 8) -> DAG:
    """0 -> 1 -> ... -> n-1. Influence is perfectly anti-correlated with position."""
    parents = tuple(() if v == 0 else (v - 1,) for v in range(n))
    return DAG(n=n, parents=parents, terminal=n - 1, name=f"chain({n})")


def parallel_chains(n_branches: int = 3, branch_len: int = 3) -> DAG:
    """Independent branches that merge into a single conclusion.

    The realistic shape for multi-hop factual QA: gather several facts along
    separate lines of reasoning, then combine. Branch heads sit at different
    positions but carry identical influence, which is exactly the confound the
    chain could not break.
    """
    parents: list[tuple[int, ...]] = []
    heads = []
    for b in range(n_branches):
        for i in range(branch_len):
            if i == 0:
                parents.append(())
                heads.append(len(parents) - 1)
            else:
                parents.append((len(parents) - 1,))
    tails = [b * branch_len + branch_len - 1 for b in range(n_branches)]
    parents.append(tuple(tails))  # merge node
    n = len(parents)
    return DAG(
        n=n,
        parents=tuple(parents),
        terminal=n - 1,
        name=f"parallel({n_branches}x{branch_len})",
    )


def converging_tree(depth: int = 3) -> DAG:
    """Leaves pairwise combine upward into one conclusion.

    Bottom-up aggregation: many independent premises, progressively merged.
    Influence is graded by level rather than by position, and many nodes at the
    same level share influence while sitting at very different indices.
    """
    levels: list[list[int]] = []
    parents: list[tuple[int, ...]] = []
    width = 2**depth
    levels.append([])
    for _ in range(width):
        parents.append(())
        levels[0].append(len(parents) - 1)

    cur = levels[0]
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            parents.append((cur[i], cur[i + 1]))
            nxt.append(len(parents) - 1)
        cur = nxt
    n = len(parents)
    return DAG(n=n, parents=tuple(parents), terminal=n - 1,
               name=f"converging(d={depth})")


def diamond(n_layers: int = 3, width: int = 2) -> DAG:
    """Repeated split-and-merge. Every layer is a bottleneck."""
    parents: list[tuple[int, ...]] = [()]
    prev_join = 0
    for _ in range(n_layers):
        branch = []
        for _ in range(width):
            parents.append((prev_join,))
            branch.append(len(parents) - 1)
        parents.append(tuple(branch))
        prev_join = len(parents) - 1
    n = len(parents)
    return DAG(n=n, parents=tuple(parents), terminal=n - 1,
               name=f"diamond({n_layers}x{width})")


def bushy(n: int = 13, seed: int = 0, max_parents: int = 2) -> DAG:
    """Random DAG with a single terminal sink.

    Guards against reading too much into three hand-built shapes.
    """
    rng = np.random.default_rng(seed)
    parents: list[tuple[int, ...]] = [()]
    for v in range(1, n - 1):
        k = int(rng.integers(1, min(max_parents, v) + 1))
        ps = rng.choice(v, size=k, replace=False)
        parents.append(tuple(sorted(int(p) for p in ps)))
    # Terminal absorbs every node that has no children yet.
    have_children = {p for ps in parents for p in ps}
    dangling = [v for v in range(n - 1) if v not in have_children]
    parents.append(tuple(dangling) if dangling else (n - 2,))
    return DAG(n=n, parents=tuple(parents), terminal=n - 1, name=f"bushy({n},s={seed})")


# ---- DAG-aware propagation -------------------------------------------


@dataclass
class DAGResult:
    correct: dict[int, bool] = field(default_factory=dict)
    verified: set[int] = field(default_factory=set)
    verification_calls: int = 0
    terminal_correct: bool = False
    # Hops from the ORIGIN of a node's corruption, i.e. the first node that
    # went wrong on the path reaching it. Not the distance to the nearest
    # incorrect ancestor: once corruption spreads, every intermediate node is
    # itself incorrect and that distance is always 1, which silently disables
    # any decay.
    origin_dist: dict[int, int] = field(default_factory=dict)

    @property
    def n_wrong(self) -> int:
        return sum(1 for v in self.correct.values() if not v)


class DAGPropagation:
    """Propagation over an arbitrary reasoning DAG.

    Same failure decomposition as `car.propagation`, but corruption now flows
    along edges rather than along step order, and a repair only cuts the paths
    that pass through the repaired node.
    """

    def __init__(
        self,
        dag: DAG,
        *,
        local_error_rate: float = 0.15,
        verifier_scope: float = 1.0,
        scope_decay: float = 1.0,
        repair_success: float = 1.0,
        verifiable: set[int] | None = None,
    ) -> None:
        self.dag = dag
        self.local_error_rate = local_error_rate
        self.verifier_scope = verifier_scope
        self.scope_decay = scope_decay
        self.repair_success = repair_success
        # The terminal node is NOT verifiable by default. A verifier that can
        # definitively check the final answer is an oracle on the task -- if
        # you had one you would not need the reasoning chain. Allowing it makes
        # "always verify the last node" a trivially winning policy and drives
        # measured error to exactly zero, which is an artifact, not a result.
        self.verifiable = (
            verifiable
            if verifiable is not None
            else set(range(dag.n)) - {dag.terminal}
        )

    def run(self, schedule, rng: np.random.Generator, budget: int | None = None) -> DAGResult:
        """Execute one pass. `schedule[v]` is P(verify node v)."""
        dag = self.dag
        res = DAGResult()
        remaining = budget if budget is not None else dag.n

        for v in range(dag.n):
            local_valid = rng.random() >= self.local_error_rate
            bad_parents = [p for p in dag.parents[v] if not res.correct.get(p, True)]
            premise_corrupt = bool(bad_parents)

            # Distance from where this corruption originally started.
            inherited_dist = (
                min(res.origin_dist.get(p, 0) + 1 for p in bad_parents)
                if bad_parents
                else 0
            )

            can_verify = v in self.verifiable
            if can_verify and rng.random() < schedule[v] and remaining > 0:
                res.verified.add(v)
                remaining -= 1
                res.verification_calls += 1

                sees_local = not local_valid
                sees_inherited = False
                if premise_corrupt:
                    eff = self.verifier_scope * (
                        self.scope_decay ** max(0, inherited_dist - 1)
                    )
                    sees_inherited = rng.random() < eff

                if sees_local or sees_inherited:
                    if rng.random() < self.repair_success:
                        local_valid = True
                        if sees_inherited:
                            # Cut corruption on paths THROUGH v only. Parallel
                            # routes from the same bad ancestor are untouched --
                            # the property a chain cannot represent.
                            premise_corrupt = False
                            inherited_dist = 0

            res.correct[v] = local_valid and not premise_corrupt
            if not res.correct[v]:
                # A locally invalid step starts a fresh corruption here;
                # otherwise it carries forward whatever it inherited.
                res.origin_dist[v] = 0 if not local_valid else inherited_dist

        res.terminal_correct = res.correct[dag.terminal]
        return res


# ---- allocation schedules --------------------------------------------


def _verifiable_mask(dag: DAG, verifiable: set[int] | None) -> np.ndarray:
    keep = verifiable if verifiable is not None else set(range(dag.n)) - {dag.terminal}
    return np.array([v in keep for v in range(dag.n)], dtype=bool)


def _normalise(
    w: np.ndarray, budget: float, mask: np.ndarray | None = None
) -> list[float]:
    """Scale weights so expected calls equal `budget`, capped at 1 per node.

    Redistributes any mass lost to the cap, so every schedule really does spend
    the same budget -- otherwise a comparison is just "verify more". Weight on
    unverifiable nodes is zeroed BEFORE normalising, so no budget is silently
    thrown away on a node that can never be checked.
    """
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    if mask is not None:
        w = np.where(mask, w, 0.0)
    if w.sum() <= 0:
        w = mask.astype(float) if mask is not None else np.ones_like(w)
    p = w / w.sum() * budget
    for _ in range(100):
        over = p > 1.0
        if not over.any():
            break
        excess = (p[over] - 1.0).sum()
        p[over] = 1.0
        room = (~over) & (p > 0)
        if not room.any() or p[room].sum() <= 0:
            break
        p[room] += excess * p[room] / p[room].sum()
    return list(np.clip(p, 0.0, 1.0))


def uniform_schedule(dag: DAG, budget: float, verifiable=None) -> list[float]:
    m = _verifiable_mask(dag, verifiable)
    return _normalise(np.ones(dag.n), budget, m)


def front_schedule(dag: DAG, budget: float, verifiable=None) -> list[float]:
    m = _verifiable_mask(dag, verifiable)
    return _normalise(np.array([dag.n - v for v in range(dag.n)], dtype=float), budget, m)


def back_schedule(dag: DAG, budget: float, verifiable=None) -> list[float]:
    m = _verifiable_mask(dag, verifiable)
    return _normalise(np.array([v + 1 for v in range(dag.n)], dtype=float), budget, m)


def influence_schedule(
    dag: DAG, budget: float, verifiable=None, power: float = 1.0
) -> list[float]:
    """P(verify v) proportional to v's descendant count.

    The policy under test. On a chain this collapses to `front_schedule`; on a
    wide DAG it is a genuinely different allocation.
    """
    m = _verifiable_mask(dag, verifiable)
    w = np.array([dag.influence(v) + 1.0 for v in range(dag.n)])
    return _normalise(w**power, budget, m)


def depth_schedule(dag: DAG, budget: float, verifiable=None) -> list[float]:
    """Proportional to ancestor count -- the mirror image of influence.

    Included so that "structure helps" cannot be claimed merely because SOME
    structural signal beats uniform.
    """
    m = _verifiable_mask(dag, verifiable)
    w = np.array([len(dag.ancestors(v)) + 1.0 for v in range(dag.n)])
    return _normalise(w, budget, m)


def cut_schedule(dag: DAG, budget: float, verifiable=None) -> list[float]:
    """Concentrate on bottlenecks: nodes carrying many paths to the terminal.

    Weight = |ancestors(v)| * |descendants(v)|, which peaks in the middle of the
    graph where much of the flow is funnelled, rather than at either end.
    """
    m = _verifiable_mask(dag, verifiable)
    w = np.array(
        [
            (len(dag.ancestors(v)) + 1.0) * (dag.influence(v) + 1.0)
            if dag.reaches_terminal(v)
            else 0.0
            for v in range(dag.n)
        ]
    )
    return _normalise(w, budget, m)


SCHEDULES = {
    "uniform": uniform_schedule,
    "front": front_schedule,
    "back": back_schedule,
    "influence": influence_schedule,
    "depth": depth_schedule,
    "cut": cut_schedule,
}
