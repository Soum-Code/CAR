"""Step generation with uncertainty features attached."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from car.backends.base import LMBackend
from car.generation.schema import (
    STEP_PROMPT_TEMPLATE,
    StepParseError,
    format_context,
    parse_step,
)
from car.types import Example, ReasoningStep, UncertaintyFeatures
from car.uncertainty.features import max_surprisal, mean_logprob, token_entropy
from car.uncertainty.semantic import normalised_semantic_divergence


@dataclass
class GeneratedStep:
    step: ReasoningStep
    features: UncertaintyFeatures
    # Ground-truth correctness, available only in simulation or when a
    # deterministic checker applies. The control policy must never read it;
    # it exists so offline analysis and oracle baselines are possible.
    true_label: bool | None = None
    parse_failed: bool = False


class StepGenerator(Protocol):
    def plan(self, example: Example) -> list[str]: ...

    def step(
        self, example: Example, target: str, context: list[ReasoningStep]
    ) -> GeneratedStep: ...


class LLMStepGenerator:
    """Real generator: prompts the model, parses JSON, computes features.

    `n_samples` controls semantic sampling. This is the dominant compute cost
    in the whole project -- n_samples extra forward passes per step, per
    question. Budget it explicitly before scaling up the dataset.
    """

    def __init__(
        self,
        backend: LMBackend,
        *,
        n_samples: int = 5,
        temperature: float = 0.7,
        max_steps: int = 6,
    ) -> None:
        self.backend = backend
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_steps = max_steps

    def plan(self, example: Example) -> list[str]:
        # StrategyQA ships an annotated decomposition; use it when present
        # rather than asking the model to re-derive a plan we already have.
        if example.decomposition:
            return example.decomposition[: self.max_steps]
        return [f"step {i}" for i in range(self.max_steps)]

    def step(
        self, example: Example, target: str, context: list[ReasoningStep]
    ) -> GeneratedStep:
        prompt = STEP_PROMPT_TEMPLATE.format(
            question=example.question,
            context=format_context(context),
            next_id=len(context),
        )

        gens = self.backend.generate(prompt, n=1, temperature=0.0)
        primary = gens[0]

        try:
            step = parse_step(primary.text, expected_id=len(context))
            parse_failed = False
        except StepParseError:
            step = ReasoningStep(
                step_id=len(context),
                claim=primary.text.strip()[:200],
                rationale="",
                dependency_ids=list(range(len(context))),
            )
            parse_failed = True

        feats = UncertaintyFeatures(
            token_entropy=token_entropy(primary),
            max_surprisal=max_surprisal(primary),
            mean_logprob=mean_logprob(primary),
        )

        if self.n_samples > 1:
            samples = self.backend.generate(
                prompt, n=self.n_samples, temperature=self.temperature
            )
            texts = [g.text for g in samples]
            from car.uncertainty.semantic import cluster_by_equivalence

            clusters = cluster_by_equivalence(texts)
            seq_lp = [float(g.token_logprobs.sum()) for g in samples]
            feats.semantic_divergence = normalised_semantic_divergence(clusters, seq_lp)

        return GeneratedStep(step=step, features=feats, parse_failed=parse_failed)


class MockStepGenerator:
    """Simulation generator built on MockBackend.

    Produces steps whose true correctness is known, which is what lets the
    conformal layer be validated against ground truth before any GPU is
    involved.
    """

    def __init__(self, backend, *, n_steps: int = 4, n_samples: int = 5) -> None:
        self.backend = backend
        self.n_steps = n_steps
        self.n_samples = n_samples

    def plan(self, example: Example) -> list[str]:
        return [f"target_{i}" for i in range(self.n_steps)]

    def step(
        self, example: Example, target: str, context: list[ReasoningStep]
    ) -> GeneratedStep:
        key = f"{example.example_id}::{target}"
        gen = self.backend.generate(key, n=1)[0]
        correct, _ = self.backend.latent_correctness(key)

        feats = UncertaintyFeatures(
            token_entropy=token_entropy(gen),
            max_surprisal=max_surprisal(gen),
            mean_logprob=mean_logprob(gen),
        )

        if self.n_samples > 1:
            clusters = [
                self.backend.semantic_cluster_id(key, i) for i in range(self.n_samples)
            ]
            feats.semantic_divergence = normalised_semantic_divergence(clusters)

        step = ReasoningStep(
            step_id=len(context),
            claim=f"[{example.example_id}] {target}",
            rationale="simulated",
            # Chain topology: each step depends on the previous one. This is
            # what makes the trajectory a dependent sequence rather than iid
            # draws, which is the condition the adaptive layer must survive.
            dependency_ids=[len(context) - 1] if context else [],
        )
        return GeneratedStep(step=step, features=feats, true_label=correct)
