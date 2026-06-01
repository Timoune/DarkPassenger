"""
core/expression_confidence.py — DarkPassenger Expression Confidence System

Computes the global Expression Confidence scalar that attenuates the
PersonaVector according to:

    1. The criticality of the GhostMind output  (spec §4, default levels)
    2. GhostMind's uncertainty score            (spec §4, entropy-linked)

Entropy-Linked Confidence (spec §4)
─────────────────────────────────────
    "Expression Confidence is inversely related to GhostMind uncertainty."

    Higher certainty  → Stronger personality expression
    Lower certainty   → Reduced personality expression
    Very high uncertainty → Personality fades; raw GhostMind output becomes
                            more visible

The formula used here:

    effective_confidence = criticality_cap × uncertainty_multiplier

where:

    criticality_cap       — per CriticalityLevel cap (0.90 / 0.50 / 0.05 / 0.00)
    uncertainty_multiplier — inverse sigmoid of uncertainty_score

The inverse sigmoid ensures:
    - uncertainty=0.0  → multiplier=1.00  (full personality, GhostMind is certain)
    - uncertainty=0.50 → multiplier≈0.50  (half personality)
    - uncertainty=0.90 → multiplier≈0.18  (personality nearly gone)
    - uncertainty=1.00 → multiplier=0.00  (complete fade)

This means personality never masks uncertainty. More uncertainty = less
performance, more transparency.

Spec §4 default levels (used as criticality_cap floor):
    Normal Conversation    → 0.90
    Technical Operations   → 0.50
    High-Risk Actions      → 0.50
    Emergency Situations   → 0.05
    Security / SysFail     → 0.00  (always override — no personality at all)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from dp_types.integrity_types import CriticalityLevel, GhostMindOutput


# ── Spec-defined default confidence caps per criticality level ────────────────

_CRITICALITY_CAPS: dict[CriticalityLevel, float] = {
    CriticalityLevel.NORMAL:      0.90,
    CriticalityLevel.TECHNICAL:   0.50,
    CriticalityLevel.HIGH_RISK:   0.50,
    CriticalityLevel.EMERGENCY:   0.05,
    CriticalityLevel.SECURITY:    0.00,
    CriticalityLevel.SYSTEM_FAIL: 0.00,
}

# Uncertainty above this threshold causes an accelerated fade
_UNCERTAINTY_KNEE: float = 0.50

# Steepness of the inverse sigmoid curve
# Higher = sharper transition around the knee
_SIGMOID_STEEPNESS: float = 6.0


# ── ConfidenceCalculation ─────────────────────────────────────────────────────

@dataclass
class ConfidenceCalculation:
    """
    Full diagnostics from one ExpressionConfidenceSystem.compute() call.

    Attributes
    ──────────
    final_confidence:
        The effective scalar applied to the PersonaVector. Range [0.0, 1.0].

    criticality_cap:
        The hard ceiling imposed by CriticalityLevel.

    uncertainty_multiplier:
        The inverse-sigmoid multiplier derived from GhostMind's uncertainty
        score. Always in [0.0, 1.0].

    uncertainty_score:
        The raw uncertainty_score from GhostMindOutput (echoed for audit).

    criticality:
        The CriticalityLevel that determined criticality_cap.

    capped_by_uncertainty:
        True if uncertainty_multiplier < 1.0 (personality was attenuated by
        uncertainty, not just criticality).

    reason:
        Human-readable explanation.
    """
    final_confidence:       float
    criticality_cap:        float
    uncertainty_multiplier: float
    uncertainty_score:      float
    criticality:            CriticalityLevel
    capped_by_uncertainty:  bool
    reason:                 str


# ── ExpressionConfidenceSystem ────────────────────────────────────────────────

class ExpressionConfidenceSystem:
    """
    Computes the global Expression Confidence scalar for a GhostMind output.

    Usage:
        ecs = ExpressionConfidenceSystem()
        result = ecs.compute(ghost_output)
        attenuated_vector = persona_vector.scale(result.final_confidence)

    Or as a simple scalar:
        confidence = ecs.scalar(ghost_output)
        attenuated_vector = persona_vector.scale(confidence)
    """

    def __init__(
        self,
        sigmoid_steepness: float = _SIGMOID_STEEPNESS,
        logger=None,
    ):
        """
        Args:
            sigmoid_steepness:
                Controls the sharpness of the uncertainty fade curve.
                Higher values = sharper fade at the knee point.
                Must be > 0.

            logger:
                Optional logger. Receives confidence events at DEBUG level.
        """
        if sigmoid_steepness <= 0:
            raise ValueError(
                f"sigmoid_steepness must be > 0, got {sigmoid_steepness!r}"
            )
        self._steepness = sigmoid_steepness
        self._logger    = logger

    # ── Primary interface ─────────────────────────────────────────────────────

    def compute(self, ghost_output: GhostMindOutput) -> ConfidenceCalculation:
        """
        Compute full Expression Confidence for a GhostMindOutput.

        Returns a ConfidenceCalculation with the effective scalar and
        full diagnostics.
        """
        criticality   = ghost_output.criticality
        uncertainty   = max(0.0, min(1.0, ghost_output.uncertainty_score))
        crit_cap      = _CRITICALITY_CAPS.get(criticality, 0.90)

        # If the criticality cap is 0.0, no further calculation needed.
        if crit_cap == 0.0:
            return ConfidenceCalculation(
                final_confidence=0.0,
                criticality_cap=0.0,
                uncertainty_multiplier=1.0,   # n/a — criticality dominates
                uncertainty_score=uncertainty,
                criticality=criticality,
                capped_by_uncertainty=False,
                reason=f"criticality_override:{criticality.value}_cap=0.0",
            )

        # Compute entropy-linked uncertainty multiplier
        unc_multiplier = self._uncertainty_multiplier(uncertainty)

        # Effective confidence = criticality cap × uncertainty attenuation
        effective = crit_cap * unc_multiplier
        # Clamp to [0, criticality_cap] — never exceeds the cap
        effective = max(0.0, min(crit_cap, effective))

        capped_by_unc = unc_multiplier < 0.95  # meaningful attenuation, not floating-point noise

        reason = (
            f"criticality={criticality.value}"
            f":cap={crit_cap:.2f}"
            f":uncertainty={uncertainty:.3f}"
            f":unc_multiplier={unc_multiplier:.3f}"
            f":final={effective:.3f}"
        )

        self._log(f"expression_confidence: {reason}")

        return ConfidenceCalculation(
            final_confidence=round(effective, 6),
            criticality_cap=crit_cap,
            uncertainty_multiplier=unc_multiplier,
            uncertainty_score=uncertainty,
            criticality=criticality,
            capped_by_uncertainty=capped_by_unc,
            reason=reason,
        )

    def scalar(self, ghost_output: GhostMindOutput) -> float:
        """
        Convenience wrapper — returns just the final_confidence float.
        """
        return self.compute(ghost_output).final_confidence

    # ── Uncertainty multiplier formula ────────────────────────────────────────

    def _uncertainty_multiplier(self, uncertainty: float) -> float:
        """
        Map uncertainty_score ∈ [0.0, 1.0] to a multiplier ∈ [0.0, 1.0].

        Uses an inverse logistic (sigmoid) centered at _UNCERTAINTY_KNEE:

            multiplier = 1 - sigmoid(steepness × (uncertainty - knee))

        Behaviour:
            uncertainty = 0.0  → multiplier ≈ 1.00  (full personality)
            uncertainty = 0.50 → multiplier ≈ 0.50  (half personality)
            uncertainty = 1.0  → multiplier ≈ 0.00  (no personality)

        The exact mid-point is at uncertainty = _UNCERTAINTY_KNEE (0.50),
        which gives multiplier = 0.50 by construction.
        """
        x          = self._steepness * (uncertainty - _UNCERTAINTY_KNEE)
        sigmoid    = 1.0 / (1.0 + math.exp(-x))
        multiplier = 1.0 - sigmoid
        return max(0.0, min(1.0, multiplier))

    # ── Override / cap query helpers ──────────────────────────────────────────

    @staticmethod
    def criticality_cap(criticality: CriticalityLevel) -> float:
        """
        Return the expression confidence cap for a given CriticalityLevel.

        Useful for pre-flight checks without a full GhostMindOutput.
        """
        return _CRITICALITY_CAPS.get(criticality, 0.90)

    @staticmethod
    def requires_override(ghost_output: GhostMindOutput) -> bool:
        """
        Return True if this output should bypass all personality expression.

        Mirrors GhostMindOutput.requires_override for convenience.
        """
        return ghost_output.requires_override

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self._logger is not None:
            try:
                self._logger.debug(message)
            except Exception:
                pass
