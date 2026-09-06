"""Replay a pre-generated corpus through the live agent loop.

The README's GPU/CPU split says generation runs once on a GPU and everything
else reads the result on CPU. This is the piece that makes that true for the
gate itself: `ReplayStepGenerator` satisfies the same `StepGenerator` protocol
as `LLMStepGenerator`, so `CARAgent` runs unmodified and the calibration,
gating, budget and baseline code paths are the real ones rather than a parallel
offline reimplementation.

What replay can and cannot establish
------------------------------------
The step text is fixed. The gate therefore cannot change what the model writes,
so a verified-and-corrected step does not cause the following steps to be
regenerated from the corrected premise.

That bounds the claims honestly:

  MEASURABLE   selective risk on the accepted set -- of the steps the gate let
               through unverified, what fraction are globally wrong. This is
               exactly the quantity conformal risk control promises to bound at
               alpha, so it is the end-to-end test that matters.
  MEASURABLE   verification rate, budget spend, score AUROC, coverage.
  PROJECTED    final-answer accuracy after repair. Requires assuming a caught
               error is repaired and downstream reasoning recovers, which is
               the propagation model's assumption, not an observation. Anything
               derived this way must be labelled as modelling.

Trying to report measured final-answer accuracy off a fixed corpus would be
claiming an intervention that never happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from car.data.math_shepherd import ShepherdSolution, load_solutions
from car.generation.step_generator import GeneratedStep
from car.types import Example, ReasoningStep, UncertaintyFeatures

FEATURE_KEYS = ("token_entropy", "max_surprisal", "mean_logprob", "semantic_divergence")


@dataclass
class ReplayStep:
    text: str
    global_ok: bool
    local_ok: bool | None
    features: UncertaintyFeatures


@dataclass
class ReplayExample:
    example_id: str
    question: str
    gold_answer: str
    final_answer: str | None
    steps: list[ReplayStep] = field(default_factory=list)

    def to_example(self) -> Example:
        return Example(
            example_id=self.example_id,
            question=self.question,
            gold_answer=self.gold_answer or "",
            decomposition=[s.text for s in self.steps],
            metadata={"final_answer": self.final_answer or final_answer_of(self)},
        )


def _features_from(row: dict | None) -> UncertaintyFeatures:
    if not row:
        return UncertaintyFeatures()
    return UncertaintyFeatures(**{k: float(row.get(k, 0.0)) for k in FEATURE_KEYS})


def load_corpus(
    corpus_path: str | Path,
    features_path: str | Path | None = None,
    *,
    notation: str = "any",
    prefix: str = "gen",
) -> list[ReplayExample]:
    """Load a Math-Shepherd-format corpus, optionally with per-step features.

    Identity is positional: record *i* of the corpus pairs with record *i* of
    the feature file. That is fragile enough to be worth checking rather than
    trusting, so the step counts are asserted per record -- a silent
    misalignment would attach one step's uncertainty to another step's label
    and quietly destroy every correlation being measured.
    """
    from car.data.generated import local_validity

    sols: list[ShepherdSolution] = load_solutions(corpus_path)

    feats: list[dict] = []
    if features_path is not None:
        with Path(features_path).open(encoding="utf-8") as fh:
            feats = [json.loads(line) for line in fh if line.strip()]
        if len(feats) != len(sols):
            raise ValueError(
                f"feature file has {len(feats)} records but the corpus has "
                f"{len(sols)}; positional pairing would misalign every step"
            )

    out = []
    for i, sol in enumerate(sols):
        frow = feats[i] if feats else None
        if frow is not None and len(frow.get("steps", [])) != len(sol.steps):
            raise ValueError(
                f"record {i}: {len(sol.steps)} steps in the corpus but "
                f"{len(frow.get('steps', []))} in the feature file"
            )
        steps = [
            ReplayStep(
                text=s.text,
                global_ok=s.global_ok,
                local_ok=local_validity(s.text, notation=notation),
                features=_features_from(frow["steps"][j] if frow else None),
            )
            for j, s in enumerate(sol.steps)
        ]
        out.append(
            ReplayExample(
                example_id=f"{prefix}_{i}",
                question=sol.question,
                gold_answer=(frow or {}).get("gold", ""),
                final_answer=(frow or {}).get("answer"),
                steps=steps,
            )
        )
    return out


def attach_gold(corpus: list[ReplayExample], gsm8k_path: str | Path) -> int:
    """Join gold answers back onto a generated corpus, by question text.

    The Math-Shepherd `label` format has no field for the gold answer or the
    example id, so a generated corpus loses both. Positional joining against
    the sampling order would work only until someone changes a seed or filters
    a record; matching on the question text is order-independent and fails
    visibly (as an unmatched count) rather than silently pairing the wrong
    answer with the wrong question.

    Returns the number of examples matched.
    """
    from car.data.generated import extract_answer

    rows = [
        json.loads(x)
        for x in Path(gsm8k_path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    gold = {" ".join(r["question"].split()): extract_answer(r["answer"]) for r in rows}
    n = 0
    for ex in corpus:
        g = gold.get(" ".join(ex.question.split()))
        if g is not None:
            ex.gold_answer = g
            n += 1
    return n


def final_answer_of(example: ReplayExample) -> str | None:
    """The answer the corpus's own last step states.

    Replay cannot regenerate an answer, so the trajectory's answer is whatever
    the recorded solution concluded. `to_shepherd_label` appends
    "The answer is: N" to the final step, which is where it is read from.
    """
    from car.data.generated import ANSWER_PREFIX, extract_answer

    if not example.steps:
        return None
    tail = example.steps[-1].text
    i = tail.find(ANSWER_PREFIX)
    if i == -1:
        return None
    return extract_answer(tail[i + len(ANSWER_PREFIX):])


class ReplayStepGenerator:
    """Serves a fixed corpus through the `StepGenerator` protocol.

    `true_label` is the corpus's global (+/-) label, so oracle baselines and
    offline analysis see real correctness. The online policy must not read it,
    which is the same contract the live generator has.
    """

    def __init__(self, corpus: list[ReplayExample], *, chain_topology: bool = True):
        self._by_id = {ex.example_id: ex for ex in corpus}
        self.chain_topology = chain_topology

    def plan(self, example: Example) -> list[str]:
        ex = self._by_id[example.example_id]
        return [f"step_{i}" for i in range(len(ex.steps))]

    def step(
        self, example: Example, target: str, context: list[ReasoningStep]
    ) -> GeneratedStep:
        ex = self._by_id[example.example_id]
        i = len(context)
        rs = ex.steps[i]
        # Chain topology. The generated corpora carry no annotated dependency
        # structure, and derived operand matching does not survive a model that
        # writes LaTeX. A chain is GSM8K's dominant shape and matches the
        # `remaining planned steps` influence proxy the loop already uses --
        # and influence weighting is off by default anyway, having lost to
        # uniform in five separate tests.
        deps = [i - 1] if (self.chain_topology and i > 0) else []
        step = ReasoningStep(
            step_id=i,
            claim=rs.text,
            rationale="",
            dependency_ids=deps,
        )
        return GeneratedStep(step=step, features=rs.features, true_label=rs.global_ok)


def score_final_answer(example: Example, accepted: list[ReasoningStep]) -> str | None:
    """The corpus's own answer, since replay cannot regenerate it."""
    if not accepted:
        return None
    return example.metadata.get("final_answer")


def answer_is_correct(answer: str | None, gold: str | None) -> bool | None:
    from car.data.generated import answers_match

    if answer is None or not gold:
        return None
    return answers_match(answer, gold)
