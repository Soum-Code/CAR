"""Token-level uncertainty features.

All of these are cheap -- they fall out of a single forward pass. The expensive
signal (semantic divergence) lives in `semantic.py`.
"""

from __future__ import annotations

import numpy as np

from car.backends.base import Generation


def token_entropy(gen: Generation) -> float:
    """Mean entropy of the next-token distribution over content tokens.

    Averaged rather than summed so it does not scale with step length --
    a long step is not automatically an uncertain one.
    """
    _, ent = gen.content_slice()
    return float(ent.mean()) if ent.size else 0.0


def max_surprisal(gen: Generation) -> float:
    """Largest -log p over content tokens: the single most surprising token.

    Catches locally anomalous tokens that averaging would wash out -- a step
    that is confident everywhere except on the one number that matters.
    """
    lp, _ = gen.content_slice()
    return float(-lp.min()) if lp.size else 0.0


def mean_logprob(gen: Generation) -> float:
    """Average log-probability of selected content tokens.

    Returned as-is (negative). Callers that want "higher = more uncertain"
    should negate it; `composite.py` handles orientation centrally.
    """
    lp, _ = gen.content_slice()
    return float(lp.mean()) if lp.size else 0.0


def perplexity(gen: Generation) -> float:
    lp, _ = gen.content_slice()
    return float(np.exp(-lp.mean())) if lp.size else 1.0
