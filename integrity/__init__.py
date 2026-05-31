"""
integrity/__init__.py — DarkPassenger Integrity Layer

Public exports for the circuit breaker and communication rules engine.
"""

from integrity.circuit_breaker import CircuitBreaker, TransformedOutput
from integrity.communication_rules import (
    CommunicationRulesEngine,
    PreFlightResult,
    RuleViolation,
    RuleCategory,
    check_rules,
)

__all__ = [
    "CircuitBreaker",
    "TransformedOutput",
    "CommunicationRulesEngine",
    "PreFlightResult",
    "RuleViolation",
    "RuleCategory",
    "check_rules",
]