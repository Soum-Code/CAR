"""Semantic divergence at the level of a single reasoning step.

Farquhar et al. (Nature 2024) compute semantic entropy over whole answers by
clustering samples into meaning-equivalence classes with bidirectional
entailment, then taking entropy over the cluster distribution. CAR applies the
same principle to one intermediate step.

Worth knowing before you rely on this: recent work (arXiv 2602.02427) reports
that sampling-agreement methods are weaker at pinpointing *intermediate* step
uncertainty than they are at whole-answer uncertainty. That is an open question
for this project, not a settled result -- which is exactly why the equivalence
function is pluggable and why the composite score does not assume this feature
carries the load.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Given two texts, return True if they mean the same thing.
EquivalenceFn = Callable[[str, str], bool]


def exact_match_equivalence(a: str, b: str) -> bool:
    """Cheapest possible baseline: normalised string equality.

    Deliberately weak. It exists so the pipeline runs without an NLI model,
    and as an ablation floor showing what lexical-only clustering buys you.
    """
    return a.strip().lower() == b.strip().lower()


def cluster_by_equivalence(
    texts: Sequence[str], equivalent: EquivalenceFn = exact_match_equivalence
) -> list[int]:
    """Greedy transitive clustering into meaning classes.

    Assigns each text to the first existing cluster whose representative it is
    equivalent to. O(n * k) comparisons for n samples and k clusters, which is
    fine for the n=5-10 regime semantic entropy actually uses.
    """
    reps: list[str] = []
    assignments: list[int] = []
    for t in texts:
        for ci, rep in enumerate(reps):
            if equivalent(t, rep):
                assignments.append(ci)
                break
        else:
            reps.append(t)
            assignments.append(len(reps) - 1)
    return assignments


def semantic_entropy(
    cluster_ids: Sequence[int], logprobs: Sequence[float] | None = None
) -> float:
    """Shannon entropy over the cluster distribution.

    If per-sample sequence logprobs are supplied, clusters are weighted by
    their summed probability mass (the Rao-Blackwellised variant); otherwise
    clusters are weighted by raw sample counts (the discrete variant).
    """
    ids = np.asarray(list(cluster_ids))
    if ids.size == 0:
        return 0.0

    n_clusters = int(ids.max()) + 1
    if logprobs is None:
        counts = np.bincount(ids, minlength=n_clusters).astype(float)
        probs = counts / counts.sum()
    else:
        lp = np.asarray(list(logprobs), dtype=float)
        # Stabilise before exponentiating; step logprobs can be very negative.
        w = np.exp(lp - lp.max())
        mass = np.bincount(ids, weights=w, minlength=n_clusters)
        total = mass.sum()
        if total <= 0:
            return 0.0
        probs = mass / total

    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def normalised_semantic_divergence(
    cluster_ids: Sequence[int], logprobs: Sequence[float] | None = None
) -> float:
    """Semantic entropy scaled to [0, 1] by the maximum entropy for n samples.

    Normalising matters because the composite score fuses this with token
    entropy, and an unnormalised feature would let sample count silently
    dominate the weighting.
    """
    ids = list(cluster_ids)
    n = len(ids)
    if n <= 1:
        return 0.0
    h = semantic_entropy(ids, logprobs)
    return float(h / np.log(n))
