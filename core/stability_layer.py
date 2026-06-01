"""
core/stability_layer.py — DarkPassenger Communication Stability Layer

Prevents personality jitter by detecting abrupt PersonaVector shifts and
applying weighted smoothing when the change has no corresponding contextual
justification.

Architecture  (spec §5)
──────────────────────
The Stability Layer sits between PersonaVector generation (Stage 3) and
Expression Confidence Attenuation (Stage 5) in the TransformationPipeline.

Inputs:
    current_vector   — freshly computed PersonaVector for this response
    previous_vector  — PersonaVector from the previous response (or None)
    current_state    — RuntimeState for this response
    previous_state   — RuntimeState snapshot from the previous response (or None)

Output:
    A smoothed PersonaVector — either the raw vector (when no jitter detected
    or a legitimate context shift is happening) or a blend of previous and
    current (when unexplained drift is detected).

Key rules from the spec
───────────────────────
1.  Detect abrupt trait changes:
        Euclidean distance between vectors > drift_threshold → examine further.

2.  Detect context change:
        If overlay, intent, topic, or relationship changed significantly,
        allow the shift through — it is intentional.

3.  Smoothing formula:
        smoothed = prev.blend(current, (1 - smoothing_factor))
        smoothing_factor=0.5 means the result is halfway between prev and current.

4.  Context Override Rule (spec §5):
        The Stability Layer must NEVER block legitimate transitions
        (e.g. Normal → Emergency, Teaching → Focused).
        It only smooths unexplained noise.

5.  Warm-up exemption:
        The first N responses in a session (response_index ≤ WARMUP_RESPONSES)
        skip smoothing — there is no settled identity yet to protect.

6.  Emergency / Override pass-through:
        If either state has an EMERGENCY overlay or override is active,
        the current vector is returned without modification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from core.persona_vector import OverlayType, PersonaVector
from core.runtime_state import RuntimeState


# ── Constants ─────────────────────────────────────────────────────────────────

# Responses at the start of a session before the fingerprint is settled
WARMUP_RESPONSES: int = 2

# If the overlay shifted to or from EMERGENCY, always allow the full transition
_HARD_PASS_OVERLAYS = frozenset({OverlayType.EMERGENCY})


# ── StabilityCheckResult ──────────────────────────────────────────────────────

@dataclass
class StabilityCheckResult:
    """
    The full diagnostic from one Stability Layer pass.

    Attributes
    ──────────
    output_vector:
        The vector that should be used downstream. This is either the raw
        current vector (no smoothing needed) or the smoothed blend.

    smoothing_applied:
        True if the smoothing formula was actually applied.

    distance:
        Euclidean distance between current and previous vectors.
        0.0 if there was no previous vector.

    context_shift_detected:
        True if a legitimate context change justifies the vector shift.
        When True, smoothing is NOT applied even if distance > threshold.

    override_pass_through:
        True if smoothing was skipped because of Emergency / warm-up rules.

    reason:
        Human-readable explanation of what happened.
    """
    output_vector:         PersonaVector
    smoothing_applied:     bool
    distance:              float
    context_shift_detected: bool
    override_pass_through: bool
    reason:                str


# ── CommunicationStabilityLayer ───────────────────────────────────────────────

class CommunicationStabilityLayer:
    """
    Smooths the active PersonaVector to prevent personality jitter.

    Construction:
        layer = CommunicationStabilityLayer(
            smoothing_factor=0.50,   # from StabilityParameters
            drift_threshold=0.30,    # from StabilityParameters
        )

    Usage (called as stability_hook in TransformationPipeline):
        smoothed_vector = layer(current_vector, previous_vector, current_state)

    Or with full diagnostics:
        result = layer.check(current_vector, previous_vector, current_state,
                             previous_state)
        smoothed_vector = result.output_vector
    """

    def __init__(
        self,
        smoothing_factor: float = 0.50,
        drift_threshold:  float = 0.30,
        logger=None,
    ):
        """
        Args:
            smoothing_factor:
                Controls how much the previous vector is preserved when jitter
                is detected. 0.0 = no smoothing (always use current);
                1.0 = completely locked to previous.
                Recommended range: 0.3–0.7.

            drift_threshold:
                Euclidean distance above which the Stability Layer examines
                whether a context shift is responsible. Below this threshold
                the vector passes unchanged regardless of other factors.

            logger:
                Optional logger. Receives stability events at DEBUG level.
        """
        if not 0.0 <= smoothing_factor <= 1.0:
            raise ValueError(
                f"smoothing_factor {smoothing_factor!r} must be in [0.0, 1.0]"
            )
        if drift_threshold <= 0.0:
            raise ValueError(
                f"drift_threshold {drift_threshold!r} must be > 0"
            )

        self.smoothing_factor = smoothing_factor
        self.drift_threshold  = drift_threshold
        self._logger          = logger

        # Session-scoped previous state kept by the layer itself.
        # The pipeline also passes the previous vector directly, but the layer
        # keeps its own copy to support use as a standalone callable.
        self._prev_vector: Optional[PersonaVector] = None
        self._prev_state:  Optional[RuntimeState]  = None

    # ── Public callable interface (matches StabilityHookFn signature) ─────────

    def __call__(
        self,
        current_vector:  PersonaVector,
        previous_vector: PersonaVector,
        current_state:   RuntimeState,
    ) -> PersonaVector:
        """
        Minimal hook-compatible interface.

        Returns the smoothed PersonaVector. For full diagnostics use check().
        """
        result = self.check(
            current_vector=current_vector,
            previous_vector=previous_vector,
            current_state=current_state,
            previous_state=self._prev_state,
        )
        # Update internal state for next call
        self._prev_vector = result.output_vector
        self._prev_state  = current_state.copy()
        return result.output_vector

    # ── Full diagnostic interface ─────────────────────────────────────────────

    def check(
        self,
        current_vector:  PersonaVector,
        previous_vector: Optional[PersonaVector],
        current_state:   RuntimeState,
        previous_state:  Optional[RuntimeState] = None,
    ) -> StabilityCheckResult:
        """
        Full stability check with diagnostics.

        This is the primary implementation; __call__ delegates here.

        Args:
            current_vector:  The newly computed PersonaVector.
            previous_vector: The PersonaVector from the last response, or None
                             on the first response of a session.
            current_state:   The RuntimeState for this response.
            previous_state:  The RuntimeState snapshot from the last response,
                             or None. Used to detect overlay/intent changes.

        Returns:
            StabilityCheckResult with the output vector and full diagnostics.
        """
        # ── 1. First response / no previous vector → always pass through ──────
        if previous_vector is None:
            return StabilityCheckResult(
                output_vector=current_vector,
                smoothing_applied=False,
                distance=0.0,
                context_shift_detected=False,
                override_pass_through=True,
                reason="first_response_no_previous_vector",
            )

        # ── 2. Warm-up exemption ──────────────────────────────────────────────
        if current_state.response_index <= WARMUP_RESPONSES:
            return StabilityCheckResult(
                output_vector=current_vector,
                smoothing_applied=False,
                distance=current_vector.distance(previous_vector),
                context_shift_detected=False,
                override_pass_through=True,
                reason=f"warmup_period_response_index={current_state.response_index}",
            )

        # ── 3. Emergency / hard-pass overlay detection ────────────────────────
        if self._is_emergency_transition(current_state, previous_state):
            return StabilityCheckResult(
                output_vector=current_vector,
                smoothing_applied=False,
                distance=current_vector.distance(previous_vector),
                context_shift_detected=True,
                override_pass_through=True,
                reason="emergency_overlay_transition_pass_through",
            )

        # ── 4. Compute distance ───────────────────────────────────────────────
        distance = current_vector.distance(previous_vector)

        # ── 5. Below threshold → no action ───────────────────────────────────
        if distance <= self.drift_threshold:
            return StabilityCheckResult(
                output_vector=current_vector,
                smoothing_applied=False,
                distance=distance,
                context_shift_detected=False,
                override_pass_through=False,
                reason=f"distance={distance:.4f}_below_threshold={self.drift_threshold}",
            )

        # ── 6. Above threshold → check for legitimate context shift ───────────
        context_shift = self._detect_context_shift(current_state, previous_state)

        if context_shift:
            # Legitimate transition — allow it through without smoothing
            self._log(
                f"stability: context_shift detected (distance={distance:.4f}), "
                f"allowing transition reason={context_shift}"
            )
            return StabilityCheckResult(
                output_vector=current_vector,
                smoothing_applied=False,
                distance=distance,
                context_shift_detected=True,
                override_pass_through=False,
                reason=f"context_shift:{context_shift}",
            )

        # ── 7. Unexplained drift → apply smoothing ────────────────────────────
        smoothed = self._smooth(previous_vector, current_vector)
        smoothed_distance = smoothed.distance(previous_vector)

        self._log(
            f"stability: smoothing applied "
            f"(raw_distance={distance:.4f}, "
            f"smoothed_distance={smoothed_distance:.4f}, "
            f"factor={self.smoothing_factor})"
        )

        return StabilityCheckResult(
            output_vector=smoothed,
            smoothing_applied=True,
            distance=distance,
            context_shift_detected=False,
            override_pass_through=False,
            reason=(
                f"smoothing_applied:raw_distance={distance:.4f}"
                f":smoothed_distance={smoothed_distance:.4f}"
            ),
        )

    # ── Session management ────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """
        Clear session state. Call when a new conversation begins.
        """
        self._prev_vector = None
        self._prev_state  = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _smooth(
        self,
        previous: PersonaVector,
        current:  PersonaVector,
    ) -> PersonaVector:
        """
        Apply smoothing formula from the spec example (§5):

            smoothed = prev.blend(current, 1 - smoothing_factor)

        At smoothing_factor=0.50: result is 50% prev + 50% current.
        At smoothing_factor=0.70: result is 70% prev + 30% current (more stable).
        At smoothing_factor=0.30: result is 30% prev + 70% current (more responsive).

        This matches the spec example:
            Previous:  humor=0.80, warmth=0.60, directness=0.70
            Computed:  humor=0.20, warmth=0.90, directness=0.40
            Smoothed:  humor=0.65, warmth=0.68, directness=0.63
            (at smoothing_factor ≈ 0.70 → prev*0.70 + current*0.30)
        """
        blend_weight = 1.0 - self.smoothing_factor
        return previous.blend(current, weight=blend_weight)

    def _is_emergency_transition(
        self,
        current_state:  RuntimeState,
        previous_state: Optional[RuntimeState],
    ) -> bool:
        """
        Return True if either state involves an emergency overlay.

        The spec's Context Override Rule states that Normal → Emergency and
        Emergency → Normal transitions must never be blocked or smoothed.
        """
        current_overlay  = current_state.current_overlay
        previous_overlay = previous_state.current_overlay if previous_state else None

        # Direct emergency overlay
        if current_overlay in _HARD_PASS_OVERLAYS:
            return True

        # Exiting emergency — also allow the full recovery
        if previous_overlay in _HARD_PASS_OVERLAYS and current_overlay not in _HARD_PASS_OVERLAYS:
            return True

        # Check blended overlays for emergency components
        if OverlayType.EMERGENCY in current_state.current_overlay_blends:
            return True

        return False

    def _detect_context_shift(
        self,
        current_state:  RuntimeState,
        previous_state: Optional[RuntimeState],
    ) -> Optional[str]:
        """
        Determine whether a meaningful context change explains the vector shift.

        Returns a description string if a shift is detected, or None if the
        vector change appears to be unexplained jitter.

        Checks (in priority order):
            1. Overlay changed (single or blend)
            2. Communication intent changed
            3. Relationship changed
            4. Topic changed significantly (different and non-None)
            5. Complexity changed
        """
        if previous_state is None:
            return "no_previous_state"

        # 1. Overlay changed
        if current_state.current_overlay != previous_state.current_overlay:
            return (
                f"overlay_change:"
                f"{previous_state.current_overlay}→{current_state.current_overlay}"
            )

        # Blended overlays changed
        if current_state.current_overlay_blends != previous_state.current_overlay_blends:
            return "blended_overlay_change"

        # 2. Intent changed
        if current_state.current_intent != previous_state.current_intent:
            return (
                f"intent_change:"
                f"{previous_state.current_intent.value}"
                f"→{current_state.current_intent.value}"
            )

        # 3. Relationship changed
        if current_state.active_relationship != previous_state.active_relationship:
            return (
                f"relationship_change:"
                f"{previous_state.active_relationship.value}"
                f"→{current_state.active_relationship.value}"
            )

        # 4. Topic changed (both non-None, and different)
        if (
            current_state.current_topic is not None
            and previous_state.current_topic is not None
            and current_state.current_topic != previous_state.current_topic
        ):
            return "topic_change"

        # 5. Complexity changed
        if current_state.current_complexity != previous_state.current_complexity:
            return f"complexity_change:{previous_state.current_complexity}→{current_state.current_complexity}"

        return None

    def _log(self, message: str) -> None:
        if self._logger is not None:
            try:
                self._logger.debug(message)
            except Exception:
                pass
