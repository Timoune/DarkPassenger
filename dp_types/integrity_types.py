"""
types/integrity_types.py — DarkPassenger Integrity Layer Data Contracts

These types are the shared language between GhostMind and DarkPassenger.
They define what a GhostMind output looks like, what the circuit breaker
checks, and what it certifies before any personality transformation is allowed.

GhostMind → GhostMindOutput → CircuitBreaker → (pass/fail) → DarkPassenger
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# CriticalityLevel — how urgent is this output?
# ---------------------------------------------------------------------------

class CriticalityLevel(str, Enum):
    """
    Severity classification assigned by GhostMind before output.

    NORMAL       — standard conversation; full personality transformation allowed
    TECHNICAL    — technical operation; personality attenuated to 50%
    HIGH_RISK    — high-risk decision warning; minimal personality, full clarity
    EMERGENCY    — emergency situation; personality bypassed entirely
    SECURITY     — security alert; raw GhostMind output only
    SYSTEM_FAIL  — system failure; raw GhostMind output only
    """
    NORMAL      = "normal"
    TECHNICAL   = "technical"
    HIGH_RISK   = "high_risk"
    EMERGENCY   = "emergency"
    SECURITY    = "security"
    SYSTEM_FAIL = "system_fail"


# Levels that trigger the Critical Response Override — bypass ALL personality
OVERRIDE_LEVELS = frozenset({
    CriticalityLevel.EMERGENCY,
    CriticalityLevel.SECURITY,
    CriticalityLevel.SYSTEM_FAIL,
})

# Levels that require maximum expression attenuation (50% or less)
ATTENUATED_LEVELS = frozenset({
    CriticalityLevel.TECHNICAL,
    CriticalityLevel.HIGH_RISK,
})


# ---------------------------------------------------------------------------
# ProtectedField — a named value that must survive transformation unchanged
# ---------------------------------------------------------------------------

@dataclass
class ProtectedField:
    """
    A single named value that DarkPassenger must never alter.

    GhostMind stamps these before handing off to DarkPassenger.
    The CircuitBreaker extracts and checksums them; the Validator
    confirms they survive transformation intact.

    Attributes
    ----------
    key         : human-readable field name for audit logs
    value       : the actual protected value (any serialisable type)
    field_type  : category tag — used to route validation logic
    checksum    : SHA-256 hex digest of str(value), set by GhostMindOutput
    """
    key:        str
    value:      Any
    field_type: str       # "number" | "warning" | "risk" | "tool_result" | "decision"
    checksum:   str = ""  # populated by GhostMindOutput.finalize()


# ---------------------------------------------------------------------------
# GhostMindOutput — what GhostMind hands to DarkPassenger
# ---------------------------------------------------------------------------

@dataclass
class GhostMindOutput:
    """
    The canonical handoff object from GhostMind to DarkPassenger.

    GhostMind creates this, marks criticality, stamps protected fields,
    calls finalize(), then passes it to the CircuitBreaker.

    Attributes
    ----------
    content          : the raw text GhostMind wants to communicate
    criticality      : how this output should be treated by DarkPassenger
    protected_fields : values that must survive transformation unchanged
    warnings         : explicit warning strings (always preserved verbatim)
    tool_results     : tool outputs (always preserved verbatim)
    risk_assessment  : risk level string (always preserved verbatim)
    security_flags   : security-related signals; any truthy value → override
    uncertainty_score: GhostMind's confidence [0.0–1.0]; feeds expression attenuation
    source_module    : which GhostMind module produced this (for audit)
    conversation_id  : links to DecisionLedger entry
    _finalized       : guard flag; set by finalize(), blocks post-hoc mutation
    """
    content:           str
    criticality:       CriticalityLevel       = CriticalityLevel.NORMAL
    protected_fields:  List[ProtectedField]   = field(default_factory=list)
    warnings:          List[str]              = field(default_factory=list)
    tool_results:      List[str]              = field(default_factory=list)
    risk_assessment:   Optional[str]          = None
    security_flags:    List[str]              = field(default_factory=list)
    uncertainty_score: float                  = 0.0   # 0 = certain, 1 = fully uncertain
    source_module:     str                    = "unknown"
    conversation_id:   str                    = ""
    _finalized:        bool                   = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Convenience constructors for common criticality levels
    # ------------------------------------------------------------------

    @classmethod
    def normal(cls, content: str, **kwargs) -> "GhostMindOutput":
        return cls(content=content, criticality=CriticalityLevel.NORMAL, **kwargs)

    @classmethod
    def technical(cls, content: str, **kwargs) -> "GhostMindOutput":
        return cls(content=content, criticality=CriticalityLevel.TECHNICAL, **kwargs)

    @classmethod
    def emergency(cls, content: str, **kwargs) -> "GhostMindOutput":
        return cls(content=content, criticality=CriticalityLevel.EMERGENCY, **kwargs)

    @classmethod
    def security_alert(cls, content: str, **kwargs) -> "GhostMindOutput":
        return cls(content=content, criticality=CriticalityLevel.SECURITY, **kwargs)

    @classmethod
    def system_failure(cls, content: str, **kwargs) -> "GhostMindOutput":
        return cls(content=content, criticality=CriticalityLevel.SYSTEM_FAIL, **kwargs)

    # ------------------------------------------------------------------
    # Field helpers
    # ------------------------------------------------------------------

    def add_number(self, key: str, value: float | int) -> "GhostMindOutput":
        """Register a numerical value that must not be altered."""
        self.protected_fields.append(
            ProtectedField(key=key, value=value, field_type="number")
        )
        return self

    def add_warning(self, text: str) -> "GhostMindOutput":
        """Add a warning string that must survive verbatim."""
        self.warnings.append(text)
        return self

    def add_tool_result(self, result: str) -> "GhostMindOutput":
        """Add a tool output that must survive verbatim."""
        self.tool_results.append(result)
        return self

    def set_risk(self, assessment: str) -> "GhostMindOutput":
        """Set the risk assessment string."""
        self.risk_assessment = assessment
        return self

    def add_security_flag(self, flag: str) -> "GhostMindOutput":
        """Add a security flag; triggers override regardless of criticality."""
        self.security_flags.append(flag)
        return self

    def finalize(self) -> "GhostMindOutput":
        """
        Lock the output and compute checksums for all protected fields.

        Must be called before passing to CircuitBreaker.
        Subsequent mutation attempts raise RuntimeError.
        """
        import hashlib
        for pf in self.protected_fields:
            pf.checksum = hashlib.sha256(str(pf.value).encode()).hexdigest()
        self._finalized = True
        return self

    def __setattr__(self, name: str, value: Any):
        # Allow _finalized itself and initial construction to bypass the guard
        if name != "_finalized" and getattr(self, "_finalized", False):
            raise RuntimeError(
                f"GhostMindOutput is finalized — cannot mutate field '{name}'."
            )
        super().__setattr__(name, value)

    @property
    def requires_override(self) -> bool:
        """True if the Critical Response Override must fire."""
        return (
            self.criticality in OVERRIDE_LEVELS
            or bool(self.security_flags)
        )

    @property
    def expression_confidence_cap(self) -> float:
        """
        Maximum allowed Expression Confidence given this output's criticality.

        Maps to the DarkPassenger spec defaults:
            NORMAL      → 0.90
            TECHNICAL   → 0.50
            HIGH_RISK   → 0.50
            EMERGENCY   → 0.05
            SECURITY    → 0.0  (override; personality bypassed)
            SYSTEM_FAIL → 0.0  (override; personality bypassed)
        """
        caps = {
            CriticalityLevel.NORMAL:      0.90,
            CriticalityLevel.TECHNICAL:   0.50,
            CriticalityLevel.HIGH_RISK:   0.50,
            CriticalityLevel.EMERGENCY:   0.05,
            CriticalityLevel.SECURITY:    0.00,
            CriticalityLevel.SYSTEM_FAIL: 0.00,
        }
        return caps.get(self.criticality, 0.90)


# ---------------------------------------------------------------------------
# ValidationResult — what the CircuitBreaker reports
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    PASS          = "pass"           # safe to transform
    FAIL_MEANING  = "fail_meaning"   # protected fields altered or missing
    FAIL_STYLE    = "fail_style"     # style validation heuristic failed
    FAIL_CRITICAL = "fail_critical"  # integrity safeguard triggered
    BYPASS        = "bypass"         # Critical Response Override fired


@dataclass
class ValidationViolation:
    """A single violation found during validation."""
    stage:       str    # "meaning" | "style" | "integrity"
    field:       str    # which field/check failed
    expected:    Any
    found:       Any
    description: str


@dataclass
class ValidationResult:
    """
    Full report from the CircuitBreaker / ValidationPipeline.

    Attributes
    ----------
    status          : overall verdict
    violations      : list of individual violations found
    override_active : True when Critical Response Override fired
    safe_output     : the certified-safe text to deliver (may be raw GhostMind
                      content if override fired, or transformed content if pass)
    attempts        : how many regeneration attempts were made
    """
    status:          ValidationStatus
    violations:      List[ValidationViolation] = field(default_factory=list)
    override_active: bool                      = False
    safe_output:     str                       = ""
    attempts:        int                       = 1

    @property
    def passed(self) -> bool:
        return self.status in (ValidationStatus.PASS, ValidationStatus.BYPASS)

    @property
    def failed(self) -> bool:
        return not self.passed
