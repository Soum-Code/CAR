"""Structured step schema and parsing.

Free-form paragraphs are not a reproducible unit of reasoning -- you cannot
label them consistently and you cannot tell what depends on what. The generator
is therefore constrained to emit JSON:

    {"step_id": 0, "claim": "...", "rationale": "...",
     "dependency_ids": [], "tool_request": null}

Segmentation rules:
  * one step = one atomic claim or one computational transformation
  * dependency_ids names the earlier steps used as premises
  * uncertainty is measured over claim and rationale tokens only, never JSON
    syntax -- punctuation is highly predictable and would dilute the signal
"""

from __future__ import annotations

import json
import re

from car.types import ReasoningStep

STEP_PROMPT_TEMPLATE = """You are solving a question one reasoning step at a time.

Question: {question}

Steps established so far:
{context}

Emit the NEXT single reasoning step as JSON with exactly these keys:
  step_id        integer, must be {next_id}
  claim          one atomic factual or computational assertion
  rationale      brief justification, one sentence
  dependency_ids list of earlier step_ids used as premises
  tool_request   a tool name if this step needs one, else null

Emit only the JSON object.
JSON:"""


_JSON_BLOCK = re.compile(r"\{.*?\}", re.S)


class StepParseError(ValueError):
    """Raised when generator output cannot be read as a valid step."""


def parse_step(text: str, expected_id: int | None = None) -> ReasoningStep:
    """Extract and validate a ReasoningStep from raw model output.

    Parse failures are raised rather than silently repaired. A malformed step is
    a real event that should be counted in the results -- quietly patching it
    hides how often the generator breaks its own schema.
    """
    match = _JSON_BLOCK.search(text)
    if not match:
        raise StepParseError(f"no JSON object found in output: {text[:120]!r}")

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise StepParseError(f"invalid JSON: {exc}") from exc

    if expected_id is not None and obj.get("step_id") != expected_id:
        obj["step_id"] = expected_id  # generators drift on counters; ids are ours

    try:
        return ReasoningStep.model_validate(obj)
    except Exception as exc:
        raise StepParseError(f"step failed validation: {exc}") from exc


def content_span(text: str, step: ReasoningStep) -> tuple[int, int] | None:
    """Character span of claim+rationale within the raw generation.

    Used to build the content token mask so uncertainty ignores JSON scaffolding.
    Returns None when the claim cannot be located, in which case the caller
    should fall back to scoring all tokens and record that it did.
    """
    start = text.find(step.claim)
    if start < 0:
        return None
    end = start + len(step.claim)
    if step.rationale:
        r = text.find(step.rationale)
        if r >= 0:
            end = max(end, r + len(step.rationale))
    return start, end


def format_context(steps: list[ReasoningStep]) -> str:
    if not steps:
        return "(none yet)"
    return "\n".join(f"  [{s.step_id}] {s.claim}" for s in steps)
