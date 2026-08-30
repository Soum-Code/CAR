"""Deterministic simulator backend for CPU development.

This is not a toy. It is the instrument that lets us validate the conformal
layer against a *known* ground truth before pointing it at a real LLM.

The trick: we draw a latent "difficulty" for each step, make step correctness
a Bernoulli draw whose probability is a known monotone function of that
difficulty, and then emit token scores that are noisy observations of it. So
the true relationship between uncertainty and correctness is something we
control exactly. If the conformal calibrator cannot recover the right
threshold here, the bug is in our code, not in the language model.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from car.backends.base import Generation


def _seed_from(*parts: object) -> int:
    """Stable seed from arbitrary inputs, so runs are reproducible across
    machines and Python's hash randomisation cannot leak in."""
    blob = json.dumps([str(p) for p in parts], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2**31)


class MockBackend:
    """Simulates a reasoning LLM with controllable, known error behaviour.

    Parameters
    ----------
    signal_strength:
        How informative uncertainty is about correctness. 1.0 means the score
        separates correct from incorrect steps well; 0.0 means uncertainty is
        pure noise (useful as a null hypothesis -- CAR should show no gain
        over a random gate in that regime, and if it does, we have a leak).
    base_error_rate:
        Marginal probability that a step is wrong.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        signal_strength: float = 0.8,
        base_error_rate: float = 0.25,
        n_content_tokens: int = 24,
        vocab_entropy_scale: float = 2.5,
    ) -> None:
        self.seed = seed
        self.signal_strength = float(np.clip(signal_strength, 0.0, 1.0))
        self.base_error_rate = float(np.clip(base_error_rate, 0.0, 1.0))
        self.n_content_tokens = n_content_tokens
        self.vocab_entropy_scale = vocab_entropy_scale

    @property
    def name(self) -> str:
        return f"mock(sig={self.signal_strength},err={self.base_error_rate})"

    def latent_correctness(self, prompt: str, sample_idx: int = 0) -> tuple[bool, float]:
        """Ground-truth step correctness and the latent difficulty behind it.

        Exposed so the harness can build oracle baselines and labelled
        calibration sets without re-deriving the simulator's internals.
        """
        rng = np.random.default_rng(_seed_from(self.seed, prompt, sample_idx))
        difficulty = rng.beta(2.0, 2.0)
        # Error probability rises with difficulty, centred on base_error_rate.
        p_err = np.clip(
            self.base_error_rate + self.signal_strength * (difficulty - 0.5),
            0.01,
            0.99,
        )
        is_correct = rng.random() > p_err
        return bool(is_correct), float(difficulty)

    def generate(
        self, prompt: str, *, n: int = 1, temperature: float = 1.0
    ) -> list[Generation]:
        out: list[Generation] = []
        for i in range(n):
            _, difficulty = self.latent_correctness(prompt, sample_idx=0)
            rng = np.random.default_rng(_seed_from(self.seed, prompt, "gen", i))

            # Harder steps get higher entropy and more negative logprobs, but
            # noisily -- a confident-but-wrong step is exactly the failure mode
            # we care about, so the mapping must not be deterministic.
            noise = rng.normal(0.0, 1.0 - 0.6 * self.signal_strength)
            level = np.clip(difficulty + 0.25 * noise, 0.0, 1.0)

            k = self.n_content_tokens
            entropies = np.abs(
                rng.normal(level * self.vocab_entropy_scale, 0.4, size=k)
            )
            logprobs = -np.abs(rng.normal(level * 1.5, 0.5, size=k))

            # Temperature widens the sampled distribution, as it would for a
            # real model, so semantic-divergence experiments respond to it.
            if temperature != 1.0:
                entropies = entropies * float(temperature)

            out.append(
                Generation(
                    text=f"[mock step | difficulty={difficulty:.3f} | sample={i}]",
                    token_logprobs=logprobs,
                    token_entropies=entropies,
                    content_token_mask=np.ones(k, dtype=bool),
                )
            )
        return out

    def semantic_cluster_id(self, prompt: str, sample_idx: int) -> int:
        """Which meaning-cluster a given sample falls into.

        Real semantic entropy needs an NLI model to decide equivalence. The
        simulator short-circuits that: harder steps scatter across more
        clusters, which is the behaviour semantic entropy is meant to detect.
        """
        _, difficulty = self.latent_correctness(prompt, sample_idx=0)
        rng = np.random.default_rng(_seed_from(self.seed, prompt, "cluster", sample_idx))
        n_clusters = 1 + int(round(difficulty * 3))
        return int(rng.integers(0, n_clusters))
