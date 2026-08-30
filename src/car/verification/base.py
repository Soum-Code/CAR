"""Verifier interface.

The design rule, and it is the one that decides whether any of this works:
the verifier must draw on an information source or a deterministic procedure
that is meaningfully DIFFERENT from the generator.

Huang et al. (ICLR 2024) showed that a model asked to check its own reasoning
without external feedback does not reliably improve and sometimes gets worse --
the same parameters produced the error and are being asked to catch it. So a
second LLM with the same weights is not a verifier. It is a same-model critic,
and it belongs in the ablation table as a negative control, not in the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from car.types import ReasoningStep, Verdict


@dataclass
class VerificationResult:
    verdict: Verdict
    # Identifiers of the evidence used, so any decision can be reproduced and
    # audited later without re-running retrieval.
    evidence_ids: list[str] = field(default_factory=list)
    detail: str = ""
    revised_claim: str | None = None
    cost: int = 1


@runtime_checkable
class Verifier(Protocol):
    def verify(self, step: ReasoningStep, question: str) -> VerificationResult: ...

    @property
    def name(self) -> str: ...


class OracleVerifier:
    """Perfect verifier backed by known ground-truth labels.

    Not deployable, and not meant to be. It defines the upper bound on what any
    gating policy could achieve, which is what separates "our gate is good" from
    "this task is easy". Every accuracy-vs-cost plot should carry this line.
    """

    def __init__(self, labels: dict[tuple[str, int], bool]) -> None:
        self.labels = labels
        self.calls = 0

    @property
    def name(self) -> str:
        return "oracle"

    def verify(self, step: ReasoningStep, question: str) -> VerificationResult:
        self.calls += 1
        key = (question, step.step_id)
        correct = self.labels.get(key)
        if correct is None:
            return VerificationResult(Verdict.INSUFFICIENT, detail="no oracle label")
        return VerificationResult(
            Verdict.SUPPORTED if correct else Verdict.CONTRADICTED,
            detail="oracle",
        )
