"""Deterministic arithmetic verifier for GSM8K-style steps.

This is the strongest kind of verifier available: it does not share parameters,
priors, or failure modes with the generator. If the model claims 47 * 3 = 131,
this catches it every time.
"""

from __future__ import annotations

import ast
import operator
import re

from car.types import ReasoningStep, Verdict
from car.verification.base import VerificationResult

# Whitelisted operators. Anything outside this set is rejected rather than
# evaluated -- `eval` on model output is an obvious injection route.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_EQUATION = re.compile(r"([-+*/().\d\s%]+?)\s*=\s*(-?[\d.,]+)")


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression without exposing the interpreter."""
    node = ast.parse(expr.strip(), mode="eval").body
    return _eval_node(node)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int | float):
            raise ValueError(f"non-numeric constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"expression node not allowed: {type(node).__name__}")


class CalculatorVerifier:
    """Extracts `<expression> = <value>` from a step and checks the arithmetic."""

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance
        self.calls = 0

    @property
    def name(self) -> str:
        return "calculator"

    def verify(self, step: ReasoningStep, question: str) -> VerificationResult:
        self.calls += 1
        text = f"{step.claim} {step.rationale}"
        matches = _EQUATION.findall(text)
        if not matches:
            return VerificationResult(
                Verdict.INSUFFICIENT, detail="no arithmetic equation found"
            )

        for lhs, rhs in matches:
            try:
                got = safe_eval(lhs)
                want = float(rhs.replace(",", ""))
            except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
                return VerificationResult(
                    Verdict.INSUFFICIENT, detail=f"unparseable: {exc}"
                )
            if abs(got - want) > self.tolerance:
                return VerificationResult(
                    Verdict.CONTRADICTED,
                    detail=f"{lhs.strip()} = {got:g}, step claims {want:g}",
                    revised_claim=f"{lhs.strip()} = {got:g}",
                )
        return VerificationResult(Verdict.SUPPORTED, detail="arithmetic checks out")
