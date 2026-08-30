"""Verification layer: the external signals that make selective gating worthwhile."""

from car.verification.base import OracleVerifier, VerificationResult, Verifier
from car.verification.calculator import CalculatorVerifier, safe_eval
from car.verification.retrieval import RetrievalVerifier, SameModelCritic
from car.verification.simulated import SimulatedVerifier

__all__ = [
    "CalculatorVerifier",
    "OracleVerifier",
    "RetrievalVerifier",
    "SameModelCritic",
    "SimulatedVerifier",
    "VerificationResult",
    "Verifier",
    "safe_eval",
]
