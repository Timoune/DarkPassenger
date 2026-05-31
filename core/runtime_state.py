"""
core/runtime_state.py — DarkPassenger Runtime State Manager

Manages the temporary, session-scoped communication context that
DarkPassenger uses while producing a single response.

This is NOT memory. It is NOT reasoning. It is NOT emotion.
It is NOT persistent. It does not survive between sessions.

It answers the question: "What is the current context of THIS response?"

Contents:
    - current_topic            : what subject is being discussed
    - current_intent           : what the communication is trying to do
    - current_overlay          : active expression modifier
    - current_overlay_blends   : multi-overlay blend weights
    - current_complexity       : assessed complexity of this response
    - current_expression_confidence : the active confidence cap (from pre-flight)
    - active_relationship      : who we're talking to
    - response_index           : how many responses have been generated this session

The RuntimeStateManager owns one RuntimeState instance and provides
thread-safe update methods, snapshot support (for stability comparison),
and session-reset semantics.

Spec reference: DarkPassenger-Plan.txt §9
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from core.persona_vector import (
    OverlayType,
    RelationshipContext,
    CommunicationIntent,
)


# ── RuntimeState ──────────────────────────────────────────────────────────────

@dataclass
class RuntimeState:
    """
    The live, temporary context for the current response generation cycle.

    All fields are reset by RuntimeStateManager.reset_session().
    None of these fields are persisted anywhere — they are working memory
    for the current response only.

    Fields
    ──────
    current_topic:
        Short string describing the subject being discussed.
        Used by the Stability Layer (Part 5) to detect context shifts.

    current_intent:
        The CommunicationIntent for this response.
        Inferred from GhostMind's output or set explicitly.

    current_overlay:
        The active single OverlayType, or None if no overlay is active.

    current_overlay_blends:
        Multi-overlay blend weights, e.g. {TEACHING: 0.7, FOCUSED: 0.3}.
        When non-empty, this takes precedence over current_overlay.

    current_complexity:
        Assessed complexity of the current content: "low", "medium", "high".
        Influences Speech Fingerprint pacing (Part 7-8).

    current_expression_confidence:
        The active Expression Confidence cap for this response.
        Set from CommunicationRulesEngine.pre_flight() output.
        Range: 0.0 (no personality) → 1.0 (full personality).

    active_relationship:
        The RelationshipContext for the current user.
        Set at session start; may be updated if user identity changes.

    response_index:
        Count of responses generated this session (0-based).
        Used by the Stability Layer to determine warm-up vs. settled state.

    session_start_ts:
        Unix timestamp of when reset_session() was last called.
        Not used for reasoning — available for audit/logging.
    """
    current_topic:               Optional[str]             = None
    current_intent:              CommunicationIntent        = CommunicationIntent.INFORM
    current_overlay:             Optional[OverlayType]      = None
    current_overlay_blends:      Dict[OverlayType, float]  = field(default_factory=dict)
    current_complexity:          str                        = "medium"
    current_expression_confidence: float                   = 1.0
    active_relationship:         RelationshipContext        = RelationshipContext.UNKNOWN
    response_index:              int                        = 0
    session_start_ts:            float                      = field(
        default_factory=time.time
    )

    def has_blends(self) -> bool:
        """Return True if multi-overlay blends are configured."""
        return bool(self.current_overlay_blends)

    def effective_overlay(self) -> Optional[OverlayType]:
        """
        The overlay that should be passed to PersonaVectorEngine.build().

        Returns None if blends are active (caller should use build_blended()),
        or current_overlay if a single overlay is set.
        """
        if self.has_blends():
            return None
        return self.current_overlay

    def copy(self) -> "RuntimeState":
        """Return a deep copy (snapshot) of the current state."""
        return copy.deepcopy(self)


# ── RuntimeStateManager ───────────────────────────────────────────────────────

class RuntimeStateManager:
    """
    Owns and manages the single RuntimeState for a DarkPassenger session.

    Provides:
        - Type-safe update methods
        - Snapshot support for Stability Layer comparisons
        - Session reset
        - Response lifecycle hooks (begin_response / end_response)
    """

    def __init__(
        self,
        relationship: RelationshipContext = RelationshipContext.UNKNOWN,
    ):
        self._state = RuntimeState(active_relationship=relationship)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> RuntimeState:
        """Read-only access to the current state. Use update methods to modify."""
        return self._state

    # ── Update methods ────────────────────────────────────────────────────────

    def set_topic(self, topic: Optional[str]) -> None:
        """Update the current discussion topic."""
        self._state.current_topic = topic

    def set_intent(self, intent: CommunicationIntent) -> None:
        """Update the communication intent."""
        self._state.current_intent = intent

    def set_overlay(self, overlay: Optional[OverlayType], weight: float = 1.0) -> None:
        """
        Set a single expression overlay, clearing any blends.

        Args:
            overlay: The overlay to activate, or None to clear.
            weight:  Not stored directly here — used by the pipeline when
                     calling PersonaVectorEngine. Provided as context.
        """
        self._state.current_overlay = overlay
        self._state.current_overlay_blends = {}

    def set_blended_overlays(self, blends: Dict[OverlayType, float]) -> None:
        """
        Set multi-overlay blends, clearing any single overlay.

        Args:
            blends: {OverlayType: weight} — weights need not sum to 1.0.
        """
        self._state.current_overlay_blends = dict(blends)
        self._state.current_overlay = None

    def set_complexity(self, complexity: str) -> None:
        """
        Set the assessed complexity. Valid values: "low", "medium", "high".

        Invalid values are silently coerced to "medium".
        """
        if complexity not in ("low", "medium", "high"):
            complexity = "medium"
        self._state.current_complexity = complexity

    def set_expression_confidence(self, confidence: float) -> None:
        """
        Set the active expression confidence cap (0.0–1.0).

        This is set from CommunicationRulesEngine.pre_flight().max_expression
        and should not normally be set manually.
        """
        self._state.current_expression_confidence = max(0.0, min(1.0, confidence))

    def set_relationship(self, relationship: RelationshipContext) -> None:
        """Update the active relationship context."""
        self._state.active_relationship = relationship

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def begin_response(self) -> None:
        """
        Called at the start of each response generation cycle.

        Increments the response counter. Does NOT clear per-response fields
        (topic, overlay, etc.) — those persist across responses in a session
        and are updated explicitly when context shifts.
        """
        self._state.response_index += 1

    def end_response(self) -> None:
        """
        Called at the end of each response generation cycle.

        Currently a no-op — reserved for future cleanup hooks.
        Per-response fields (overlay, expression_confidence) are intentionally
        kept until the next begin_response() or explicit update.
        """
        pass

    def reset_session(
        self,
        relationship: Optional[RelationshipContext] = None,
    ) -> None:
        """
        Reset to a clean session state.

        Called when a new conversation begins. Clears all temporary context.
        The relationship context is preserved if a new one is not provided.
        """
        rel = relationship if relationship is not None else self._state.active_relationship
        self._state = RuntimeState(active_relationship=rel)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> RuntimeState:
        """
        Return a deep copy of the current state.

        Used by the Communication Stability Layer (Part 5) to compare
        the previous state against the incoming state before building
        the PersonaVector, detecting abrupt context shifts.
        """
        return self._state.copy()

    # ── Convenience ───────────────────────────────────────────────────────────

    def effective_overlay(self) -> Optional[OverlayType]:
        """Delegate to RuntimeState.effective_overlay()."""
        return self._state.effective_overlay()

    def is_blended(self) -> bool:
        """Return True if multi-overlay blends are active."""
        return self._state.has_blends()

    def summary(self) -> dict:
        """
        Return a dict summary of the current state for logging/audit.
        """
        return {
            "topic":                 self._state.current_topic,
            "intent":                self._state.current_intent.value,
            "overlay":               (
                self._state.current_overlay.value
                if self._state.current_overlay else None
            ),
            "blends":                {
                k.value: v
                for k, v in self._state.current_overlay_blends.items()
            },
            "complexity":            self._state.current_complexity,
            "expression_confidence": self._state.current_expression_confidence,
            "relationship":          self._state.active_relationship.value,
            "response_index":        self._state.response_index,
        }

    def __repr__(self) -> str:
        return f"RuntimeStateManager(response_index={self._state.response_index}, relationship={self._state.active_relationship.value!r})"
