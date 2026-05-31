"""
dp_types/__init__.py — DarkPassenger Type Contracts

Core data structures for the integrity layer.
"""

from dp_types.integrity_types import (
    CriticalityLevel,
    OVERRIDE_LEVELS,
    ATTENUATED_LEVELS,
    ProtectedField,
    GhostMindOutput,
    ValidationStatus,
    ValidationViolation,
    ValidationResult,
)

__all__ = [
    "CriticalityLevel",
    "OVERRIDE_LEVELS",
    "ATTENUATED_LEVELS",
    "ProtectedField",
    "GhostMindOutput",
    "ValidationStatus",
    "ValidationViolation",
    "ValidationResult",
]
