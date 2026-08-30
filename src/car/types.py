"""Core data types for CAR.

Every object that crosses a module boundary is defined here. The types are
deliberately strict: a reasoning step that does not validate is a bug we want
to see immediately, not a silently malformed row in a results table.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Decision(str, Enum):
    """What the control gate decided to do with a step."""

    CONTINUE = "CONTINUE"
    VERIFY = "VERIFY"
    ABSTAIN = "ABSTAIN"


class Verdict(str, Enum):
    """What an external verifier concluded about a step."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class ReasoningStep(BaseModel):
    """One atomic claim or computational transformation.

    The generator is prompted to emit exactly this shape. `dependency_ids`
    is what makes the reasoning chain a DAG rather than a flat list, which is
    what the influence-aware gate needs.
    """

    step_id: int = Field(ge=0)
    claim: str
    rationale: str = ""
    dependency_ids: list[int] = Field(default_factory=list)
    tool_request: str | None = None

    @model_validator(mode="after")
    def _no_self_dependency(self) -> ReasoningStep:
        if self.step_id in self.dependency_ids:
            raise ValueError(f"step {self.step_id} depends on itself")
        if any(d > self.step_id for d in self.dependency_ids):
            raise ValueError(
                f"step {self.step_id} depends on a later step: {self.dependency_ids}"
            )
        return self


class UncertaintyFeatures(BaseModel):
    """Raw uncertainty signals for one step, before they are combined.

    Kept separate from the combined score so ablations ("token-only",
    "semantic-only") are a matter of zeroing weights, not changing code paths.
    """

    token_entropy: float = 0.0
    max_surprisal: float = 0.0
    mean_logprob: float = 0.0
    semantic_divergence: float = 0.0
    task_verifier_signal: float = 0.0

    def as_vector(self, keys: list[str]) -> list[float]:
        return [getattr(self, k) for k in keys]


class StepRecord(BaseModel):
    """Full audit trail for a single step.

    This is the unit that gets written to the JSONL trace and later loaded for
    analysis. Everything needed to recompute a decision must be in here --
    if a field is missing, that experiment is not reproducible.
    """

    step: ReasoningStep
    features: UncertaintyFeatures
    score: float
    threshold: float
    influence: float = 1.0
    gate_value: float = 0.0

    decision: Decision
    # True when this step was verified because of forced exploration rather
    # than because the gate fired. This flag is the whole reason the censored
    # feedback problem is tractable -- these are the only unbiased labels.
    forced_exploration: bool = False

    verdict: Verdict | None = None
    revised: bool = False

    # Ground-truth step correctness. Only populated for offline analysis or
    # when a deterministic checker is available; the online policy must never
    # read this field.
    label: bool | None = None

    budget_remaining: int = 0
    latency_s: float = 0.0


class Trajectory(BaseModel):
    """One question, start to finish."""

    example_id: str
    question: str
    steps: list[StepRecord] = Field(default_factory=list)
    final_answer: str | None = None
    gold_answer: str | None = None
    correct: bool | None = None
    verification_calls: int = 0
    aborted: bool = False

    @property
    def n_steps(self) -> int:
        return len(self.steps)


class Example(BaseModel):
    """A dataset item, normalised across StrategyQA / GSM8K / etc."""

    example_id: str
    question: str
    gold_answer: str
    # StrategyQA ships an annotated decomposition and evidence paragraphs;
    # GSM8K ships a worked solution. Both land here, shape depends on dataset.
    decomposition: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
