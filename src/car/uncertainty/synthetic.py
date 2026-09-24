"""A score whose AUROC is set by construction, for sweeping score quality.

Chapters 7 and 9 both end on the same open number. The measured signals reach
AUROC 0.5742, the probe 0.6968, and at both of those the task PRM is a net
*loss* -- its 9.87% false-alarm rate breaks more correct steps than its 90.3%
scope rescues. An oracle score turns the same verifier into a 17.6-point gain.
So somewhere between 0.70 and 1.0 the verifier stops being a liability, and the
draft could only say "somewhere".

This module closes that by making score quality a dial. Under the binormal
model a globally-wrong step draws its score from N(d, 1) and a correct step
from N(0, 1), independently, so

    AUROC = P(S_wrong > S_correct) = P(N(d, 2) > 0) = Phi(d / sqrt(2))

and the separation needed for a target AUROC inverts in closed form:

    d = sqrt(2) * Phi^-1(AUROC)

Nothing else in the pipeline changes, which is the point -- the same corpus,
calibrator, budget and verifier, with only the score's discriminative power
moved. That makes the crossing point attributable to score quality and to
nothing else.

TWO LIMITS, BOTH LOAD-BEARING

*This is a shape, not a signal.* A real score at AUROC 0.75 need not be
binormal, and the errors it ranks highly need not be the errors that matter.
The binormal draw is exchangeable across wrong steps, so it is equally likely
to catch a first-step error as a last-step one. A real score with a position
bias would land elsewhere. The crossing here is therefore an estimate for a
*well-behaved* score of that quality, and a real one could do worse.

*The AUROC is a population parameter, not the sample value.* Each draw lands
near the target, not on it. The measured test AUROC is reported alongside every
row so the two can be compared rather than conflated.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

__all__ = ["separation_for_auroc", "SyntheticAUROCScorer"]


def separation_for_auroc(auroc: float) -> float:
    """Binormal mean separation `d` giving the requested population AUROC.

    Raises rather than clipping at 0.5 and 1.0: an AUROC below 0.5 is a score
    pointing the wrong way, which is a bug at the call site and not something
    to silently absorb, and 1.0 needs infinite separation (use the oracle).
    """
    if not 0.5 <= auroc < 1.0:
        raise ValueError(
            f"auroc must be in [0.5, 1.0), got {auroc}. 1.0 requires infinite "
            f"separation -- use OracleScorer for the perfect-score row."
        )
    return float(np.sqrt(2.0) * ndtri(auroc))


class SyntheticAUROCScorer:
    """Assigns every step a score drawn to hit a target AUROC.

    Keyed on the identity of the feature object rather than its values, for the
    same reason `OracleScorer` is: `ReplayStepGenerator` hands back the same
    object on each visit, and two distinct steps can carry numerically
    identical features.

    Not deployable. Like the oracle it reads the label -- it just reads it
    noisily, with the noise calibrated so the resulting ranking is exactly as
    good as a score of the stated quality.
    """

    def __init__(self, corpus, auroc: float, seed: int = 0) -> None:
        self.auroc = float(auroc)
        self.separation = separation_for_auroc(auroc)
        rng = np.random.default_rng(seed)
        self._score: dict[int, float] = {}
        for ex in corpus:
            for st in ex.steps:
                shift = 0.0 if st.global_ok else self.separation
                self._score[id(st.features)] = float(rng.normal(shift, 1.0))

    def fit(self, dev_features):  # noqa: ARG002 - interface parity
        """No-op. The draw is already calibrated; there is nothing to learn."""
        return self

    def score(self, feats) -> float:
        return self._score.get(id(feats), 0.0)
