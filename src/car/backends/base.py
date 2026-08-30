"""Language-model backend interface.

Everything downstream of this file works on `Generation` objects and never
touches a tokenizer, so the conformal / gate / metric code is testable on CPU
against the mock backend and runs unchanged against a real model on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Generation:
    """One sampled completion plus the token scores we need for uncertainty.

    `token_logprobs` is the log-probability of each *selected* token.
    `token_entropies` is the entropy of the full next-token distribution at
    each position -- this needs the whole logit vector, which is why it must
    come from the backend rather than being recomputed later.
    """

    text: str
    token_logprobs: np.ndarray = field(default_factory=lambda: np.array([]))
    token_entropies: np.ndarray = field(default_factory=lambda: np.array([]))
    # Character span of the claim+rationale inside `text`. Uncertainty is
    # computed only over these tokens, never over JSON syntax.
    content_token_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.token_logprobs) != len(self.token_entropies):
            raise ValueError(
                f"logprob/entropy length mismatch: "
                f"{len(self.token_logprobs)} vs {len(self.token_entropies)}"
            )

    def content_slice(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (logprobs, entropies) restricted to content tokens."""
        if self.content_token_mask is None:
            return self.token_logprobs, self.token_entropies
        m = self.content_token_mask.astype(bool)
        return self.token_logprobs[m], self.token_entropies[m]


@runtime_checkable
class LMBackend(Protocol):
    """Minimal surface a language model must expose for CAR."""

    def generate(self, prompt: str, *, n: int = 1, temperature: float = 1.0) -> list[Generation]:
        """Sample `n` completions with per-token scores attached."""
        ...

    @property
    def name(self) -> str: ...
