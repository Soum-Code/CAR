"""Verification layer: the external signals that make selective gating worthwhile."""

from car.verification.base import OracleVerifier, VerificationResult, Verifier
from car.verification.calculator import CalculatorVerifier, safe_eval
from car.verification.retrieval import RetrievalVerifier, SameModelCritic
from car.verification.scoped import MEASURED_SCOPE, ScopedVerifier
from car.verification.simulated import SimulatedVerifier

__all__ = [
    "MEASURED_SCOPE",
    "CalculatorVerifier",
    "OracleVerifier",
    "RetrievalVerifier",
    "SameModelCritic",
    "ScopedVerifier",
    "SimulatedVerifier",
    "VerificationResult",
    "Verifier",
    "safe_eval",
]
