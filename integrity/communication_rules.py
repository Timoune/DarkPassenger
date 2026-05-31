"""
integrity/communication_rules.py — DarkPassenger Communication Rules Layer

From the DarkPassenger spec §12: Communication Rules Layer

This module defines the IMMUTABLE CORE — a set of rules that can never
be overridden by configuration, persona profiles, or adaptive tuning.

Unlike the CircuitBreaker (which validates after the fact), the
CommunicationRules provides a pre-flight check that DarkPassenger must
consult before beginning any transformation. It also defines the
Integrity Rules that govern what DarkPassenger must NEVER do.

Two layers:
    1. ImmutableCore        — what DarkPassenger may NEVER alter
    2. IntegrityRules       — what DarkPassenger must ALWAYS do

Both layers are enforced together by the CommunicationRulesEngine.
The CircuitBreaker validates the output; this module validates the intent
before transformation begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from dp_types.integrity_types import (
    GhostMindOutput,
    CriticalityLevel,
    OVERRIDE_LEVELS,
)


# ---------------------------------------------------------------------------
# Rule violation types
# ---------------------------------------------------------------------------

class RuleCategory(str, Enum):
    IMMUTABLE_CORE  = "immutable_core"
    INTEGRITY_RULE  = "integrity_rule"
    PRE_FLIGHT      = "pre_flight"


@dataclass
class RuleViolation:
    """A pre-flight rule violation; blocks transformation from starting."""
    category:    RuleCategory
    rule_id:     str
    description: str
    severity:    str = "hard_block"   # "hard_block" | "warning"


# ---------------------------------------------------------------------------
# Pre-flight check result
# ---------------------------------------------------------------------------

@dataclass
class PreFlightResult:
    """
    Result of the CommunicationRulesEngine pre-flight check.

    Attributes
    ----------
    allowed          : True if transformation may proceed
    violations       : any rule violations found
    override_active  : True if Critical Response Override applies
    max_expression   : maximum Expression Confidence allowed (0.0–1.0)
    """
    allowed:         bool
    violations:      List[RuleViolation]
    override_active: bool
    max_expression:  float

    @property
    def blocked(self) -> bool:
        return not self.allowed


# ---------------------------------------------------------------------------
# CommunicationRulesEngine
# ---------------------------------------------------------------------------

class CommunicationRulesEngine:
    """
    Enforces the Communication Rules Layer before any transformation starts.

    This is NOT a post-hoc validator (that's the CircuitBreaker).
    This is a PRE-FLIGHT check: DarkPassenger asks "may I transform this?"
    before spending cycles on personality application.

    The engine answers three questions:
        1. Does the Critical Response Override apply? (bypass everything)
        2. Are there hard-block rules that forbid transformation?
        3. What is the maximum Expression Confidence allowed?
    """

    def __init__(self, logger=None):
        self.logger = logger

    # ── Main entry point ──────────────────────────────────────────────────────

    def pre_flight(self, ghost_output: GhostMindOutput) -> PreFlightResult:
        """
        Run all pre-flight checks on a GhostMindOutput.

        Returns a PreFlightResult. If result.blocked is True, DarkPassenger
        must not attempt transformation and must deliver raw content instead.
        """
        try:
            return self._run_checks(ghost_output)
        except Exception as exc:
            if self.logger:
                self.logger.error(
                    "communication_rules_preflight_error", error=str(exc)
                )
            # Fail safe: block transformation on any unexpected error
            return PreFlightResult(
                allowed=False,
                violations=[RuleViolation(
                    category=RuleCategory.PRE_FLIGHT,
                    rule_id="PREFLIGHT_EXCEPTION",
                    description=f"Pre-flight check raised: {exc}. Blocking as safe default.",
                )],
                override_active=True,
                max_expression=0.0,
            )

    def _run_checks(self, ghost: GhostMindOutput) -> PreFlightResult:
        violations: List[RuleViolation] = []

        # ── Critical Response Override check ──────────────────────────────
        if ghost.requires_override:
            self._log(
                "override_active",
                criticality=ghost.criticality.value,
                security_flags=ghost.security_flags,
            )
            return PreFlightResult(
                allowed=False,
                violations=[],
                override_active=True,
                max_expression=0.0,
            )

        # ── Immutable Core checks ─────────────────────────────────────────
        violations.extend(self._check_immutable_core(ghost))

        # ── Integrity Rule checks ─────────────────────────────────────────
        violations.extend(self._check_integrity_rules(ghost))

        hard_blocks = [v for v in violations if v.severity == "hard_block"]

        if hard_blocks:
            self._log(
                "preflight_blocked",
                violation_count=len(hard_blocks),
                rule_ids=[v.rule_id for v in hard_blocks],
            )
            return PreFlightResult(
                allowed=False,
                violations=violations,
                override_active=False,
                max_expression=0.0,
            )

        max_expression = self._compute_max_expression(ghost)

        self._log(
            "preflight_passed",
            criticality=ghost.criticality.value,
            max_expression=max_expression,
            warnings_count=len(violations),
        )

        return PreFlightResult(
            allowed=True,
            violations=violations,
            override_active=False,
            max_expression=max_expression,
        )

    # ── Immutable Core ────────────────────────────────────────────────────────

    def _check_immutable_core(
        self, ghost: GhostMindOutput
    ) -> List[RuleViolation]:
        """
        Check the immutable core rules from spec §12.

        From the spec — DarkPassenger may NEVER alter:
            - Numerical Values
            - Security Restrictions
            - Risk Assessments
            - Conclusions
            - Tool Results
            - Warnings
            - Uncertainty Indicators
        """
        violations = []

        # IC-01: If security flags are set, override is required — no transform
        if ghost.security_flags:
            violations.append(RuleViolation(
                category=RuleCategory.IMMUTABLE_CORE,
                rule_id="IC-01",
                description=(
                    "Security flags are set. Transformation is not allowed. "
                    "The Critical Response Override must fire."
                ),
                severity="hard_block",
            ))

        # IC-02: HIGH_RISK content with unacknowledged risk assessment
        if (
            ghost.criticality == CriticalityLevel.HIGH_RISK
            and not ghost.risk_assessment
            and not ghost.warnings
        ):
            violations.append(RuleViolation(
                category=RuleCategory.IMMUTABLE_CORE,
                rule_id="IC-02",
                description=(
                    "HIGH_RISK criticality without risk_assessment or warnings. "
                    "GhostMind must set risk context before DarkPassenger transforms."
                ),
                severity="hard_block",
            ))

        # IC-03: Unfinalized output (missing checksums)
        if ghost.protected_fields and not ghost._finalized:
            violations.append(RuleViolation(
                category=RuleCategory.IMMUTABLE_CORE,
                rule_id="IC-03",
                description=(
                    "GhostMindOutput has protected fields but was not finalized. "
                    "Call finalize() before passing to DarkPassenger."
                ),
                severity="hard_block",
            ))

        return violations

    # ── Integrity Rules ───────────────────────────────────────────────────────

    def _check_integrity_rules(
        self, ghost: GhostMindOutput
    ) -> List[RuleViolation]:
        """
        Check the Integrity Rules from spec §12.

        From the spec — DarkPassenger must:
            - Never suppress warnings
            - Never fabricate information
            - Never contradict GhostMind
            - Never hide failure states
            - Never reinterpret tool results
            - Never remove critical information

        These checks are pre-flight (intent-level). The CircuitBreaker
        validates the actual output after transformation.
        """
        violations = []

        # IR-01: Content appears to be a failure/error state
        content_lower = ghost.content.lower()
        failure_indicators = [
            "error:", "failed:", "failure:", "exception:", "[error]",
            "system failure", "critical failure", "fatal error",
        ]
        has_failure = any(fi in content_lower for fi in failure_indicators)

        if has_failure and ghost.criticality == CriticalityLevel.NORMAL:
            # Not a hard block, but a warning — criticality should be elevated
            violations.append(RuleViolation(
                category=RuleCategory.INTEGRITY_RULE,
                rule_id="IR-01",
                description=(
                    "Content appears to contain a failure state but criticality "
                    "is NORMAL. Consider elevating to TECHNICAL or HIGH_RISK so "
                    "expression confidence is appropriately attenuated."
                ),
                severity="warning",
            ))

        # IR-02: Tool results present but criticality is NORMAL with no protection
        if ghost.tool_results and not ghost.protected_fields:
            violations.append(RuleViolation(
                category=RuleCategory.INTEGRITY_RULE,
                rule_id="IR-02",
                description=(
                    "Tool results are present but no protected_fields are set. "
                    "Key values from tool outputs should be registered as "
                    "protected_fields so the CircuitBreaker can verify them."
                ),
                severity="warning",
            ))

        return violations

    # ── Expression Confidence cap ─────────────────────────────────────────────

    def _compute_max_expression(self, ghost: GhostMindOutput) -> float:
        """
        Compute the maximum Expression Confidence allowed.

        From spec §4 — entropy-linked confidence:
            Higher certainty → stronger personality expression
            Lower certainty → reduced personality expression

        This combines the criticality-based cap with the uncertainty score:
            final_cap = criticality_cap × (1.0 - uncertainty_score)
        """
        criticality_cap = ghost.expression_confidence_cap
        uncertainty_attenuation = 1.0 - ghost.uncertainty_score
        return round(criticality_cap * uncertainty_attenuation, 4)

    # ── Audit ─────────────────────────────────────────────────────────────────

    def _log(self, event: str, **kwargs):
        if self.logger:
            self.logger.info(f"comm_rules_{event}", **kwargs)


# ---------------------------------------------------------------------------
# Convenience function for quick pre-flight checks
# ---------------------------------------------------------------------------

_default_engine = CommunicationRulesEngine()


def check_rules(ghost_output: GhostMindOutput) -> PreFlightResult:
    """
    Module-level convenience wrapper for pre-flight checks.

    Uses a default CommunicationRulesEngine with no logger.
    For production use, instantiate CommunicationRulesEngine with your logger.
    """
    return _default_engine.pre_flight(ghost_output)
