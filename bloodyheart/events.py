"""
bloodyheart/events.py — BloodyHeart Event Contracts for DarkPassenger

Every message that crosses the CoreBus is a BusEvent. This module defines:

  BusEvent           — the envelope that wraps every inter-module message
  EventPriority      — P0 (Security) through P4 (Maintenance)
  DPEventType        — the versioned event types DarkPassenger produces/consumes
  DPEventSchemas     — the schema registry for DarkPassenger events

Architecture (BloodyHeart-Plan.txt §2.1, §2.3)
────────────────────────────────────────────────
All inter-module communication in Mini Von flows through the CoreBus.
Modules never communicate directly. Every event carries:

  tx_id      — unique transaction ID (UUID)
  timestamp  — wall-clock UTC timestamp
  source     — originating module name
  destination— target module name ("*" = broadcast)
  priority   — P0–P4 queue assignment
  event_type — versioned type string (e.g. "dp.output.ready.v1")
  payload    — dict of event-specific data
  version    — schema version for compatibility validation

DarkPassenger's role in the CoreBus:
  RECEIVES: ghostmind.output.ready.v1  (GhostMind → DarkPassenger gate)
  PRODUCES: dp.output.ready.v1         (DarkPassenger → Voicy / EchoLink)
            dp.config.updated.v1       (DarkPassenger → all subscribers)
            dp.health.report.v1        (DarkPassenger → BloodyHeart monitor)

Spec reference: BloodyHeart-Plan.txt §2.1, §2.2, §2.3
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, Optional


# ── EventPriority ─────────────────────────────────────────────────────────────

class EventPriority(IntEnum):
    """
    Five-level priority queue assignment (BloodyHeart §2.1).

    Lower integer = higher priority. The CoreBus scheduler always drains
    lower-numbered queues before higher-numbered ones.
    """
    SECURITY    = 0   # Lockdowns, permission updates, security violations
    HUMAN       = 1   # User messages, direct commands, human intervention
    AUTONOMOUS  = 2   # Task execution, scheduled operations, tool invocations
    COGNITIVE   = 3   # GhostMind planning, reflection cycles, simulations
    MAINTENANCE = 4   # Memory consolidation, metrics, cleanup


# ── DPEventType — versioned event type strings ────────────────────────────────

class DPEventType:
    """
    Registry of versioned event type strings for DarkPassenger.

    Naming convention:  <subsystem>.<noun>.<verb>.<vN>
    All DarkPassenger-originated events are prefixed "dp.".
    Events DarkPassenger consumes from other modules use their own prefix.
    """

    # ── Events DarkPassenger CONSUMES ──────────────────────────────────────────

    # GhostMind has produced output and dispatches it to DarkPassenger.
    # Payload: GhostMindOutputPayload
    GHOSTMIND_OUTPUT_READY   = "ghostmind.output.ready.v1"

    # BloodyHeart instructs DarkPassenger to update its persona config at runtime.
    # Payload: PersonaConfigUpdatePayload
    DP_CONFIG_UPDATE         = "dp.config.update.v1"

    # BloodyHeart requests a health/status report.
    # Payload: {} (empty)
    DP_HEALTH_REQUEST        = "dp.health.request.v1"

    # BloodyHeart broadcasts an emergency lockdown — DarkPassenger must activate
    # EMERGENCY overlay immediately and flush any in-flight transformations.
    # Payload: EmergencyPayload
    SYSTEM_EMERGENCY         = "system.emergency.v1"

    # BloodyHeart broadcasts a safe-mode level change.
    # Payload: SafeModePayload
    SYSTEM_SAFE_MODE         = "system.safe_mode.v1"

    # ── Events DarkPassenger PRODUCES ─────────────────────────────────────────

    # DarkPassenger has produced a certified, transformed response.
    # Destination: "voicy" or "echolink" depending on delivery channel.
    # Payload: DPOutputReadyPayload
    DP_OUTPUT_READY          = "dp.output.ready.v1"

    # DarkPassenger's persona configuration was successfully updated.
    # Payload: PersonaConfigUpdatedPayload
    DP_CONFIG_UPDATED        = "dp.config.updated.v1"

    # DarkPassenger health report (response to DP_HEALTH_REQUEST).
    # Payload: DPHealthPayload
    DP_HEALTH_REPORT         = "dp.health.report.v1"

    # DarkPassenger validation failed — raw GhostMind content was forwarded.
    # Payload: DPValidationFailurePayload
    DP_VALIDATION_FAILURE    = "dp.validation.failure.v1"

    # DarkPassenger rejected an incoming event (schema mismatch, trust violation).
    # Payload: DPEventRejectedPayload
    DP_EVENT_REJECTED        = "dp.event.rejected.v1"


# ── BusEvent — the CoreBus envelope ──────────────────────────────────────────

@dataclass
class BusEvent:
    """
    The universal CoreBus message envelope (BloodyHeart §2.2).

    Every inter-module message in Mini Von is a BusEvent. The payload
    is event-type-specific. BloodyHeart validates the payload against
    the schema registry before dispatching.

    Fields
    ──────
    event_type:
        Versioned type string from DPEventType (or another module's registry).

    source:
        Name of the originating module (e.g. "ghostmind", "darkpassenger").

    destination:
        Name of the target module, or "*" for broadcast.

    priority:
        EventPriority queue assignment. The CoreBus scheduler uses this.

    payload:
        Event-specific data dict. Schema is validated against DPEventSchemas.

    tx_id:
        UUID identifying this transaction. Immutable after creation.
        Used for EventJournal replay and audit correlation.

    timestamp_utc:
        ISO-8601 UTC timestamp set at creation time.

    version:
        Schema version string (e.g. "1"). Used for compatibility checking.

    correlation_id:
        Optional UUID linking this event to a parent transaction chain.
        E.g. the same transaction that started with ghostmind.output.ready.v1.
    """
    event_type:     str
    source:         str
    destination:    str
    priority:       EventPriority
    payload:        Dict[str, Any]
    tx_id:          str            = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc:  str            = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version:        str            = "1"
    correlation_id: Optional[str]  = None

    @classmethod
    def create(
        cls,
        event_type: str,
        source: str,
        destination: str,
        priority: EventPriority,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> "BusEvent":
        """
        Factory: create a BusEvent with auto-generated tx_id and timestamp.

        This is the preferred construction path (over the dataclass directly)
        because it ensures tx_id and timestamp are always fresh.

        Args:
            event_type:     Versioned type string.
            source:         Originating module name.
            destination:    Target module name, or "*".
            priority:       EventPriority queue assignment.
            payload:        Event-specific data dict.
            correlation_id: Optional parent tx_id for chain correlation.
        """
        return cls(
            event_type=event_type,
            source=source,
            destination=destination,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id,
        )

    def reply(
        self,
        event_type: str,
        source: str,
        payload: Dict[str, Any],
        priority: Optional[EventPriority] = None,
    ) -> "BusEvent":
        """
        Create a reply event correlated to this event.

        The new event's destination is this event's source,
        and its correlation_id is this event's tx_id.

        Args:
            event_type: Type of the reply event.
            source:     Module sending the reply (usually the current module).
            payload:    Reply-specific data.
            priority:   Defaults to the same priority as the originating event.
        """
        return BusEvent.create(
            event_type=event_type,
            source=source,
            destination=self.source,
            priority=priority if priority is not None else self.priority,
            payload=payload,
            correlation_id=self.tx_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable dict for EventJournal storage."""
        return {
            "tx_id":          self.tx_id,
            "timestamp_utc":  self.timestamp_utc,
            "event_type":     self.event_type,
            "source":         self.source,
            "destination":    self.destination,
            "priority":       int(self.priority),
            "version":        self.version,
            "correlation_id": self.correlation_id,
            "payload":        self.payload,
        }


# ── Typed payload helpers ─────────────────────────────────────────────────────
# These are not dataclasses — they are dict-shape documentation and factory
# functions. The CoreBus works with plain dicts for maximum flexibility.
# Schema validation is done by DPEventSchemas.

def make_ghostmind_output_payload(
    *,
    content: str,
    criticality: str,
    uncertainty: float,
    session_id: str,
    relationship: str,
    intent: str,
    overlay: Optional[str],
    protected_fields: list,
) -> Dict[str, Any]:
    """
    Payload for ghostmind.output.ready.v1.

    Args:
        content:          The raw GhostMind response string.
        criticality:      CriticalityLevel.value (e.g. "normal", "emergency").
        uncertainty:      GhostMind's self-assessed uncertainty 0.0–1.0.
        session_id:       Session UUID for audit correlation.
        relationship:     RelationshipContext.value for this response.
        intent:           CommunicationIntent.value for this response.
        overlay:          OverlayType.value, or None.
        protected_fields: List of {key, value, field_type} dicts.
    """
    return {
        "content":          content,
        "criticality":      criticality,
        "uncertainty":      uncertainty,
        "session_id":       session_id,
        "relationship":     relationship,
        "intent":           intent,
        "overlay":          overlay,
        "protected_fields": protected_fields,
    }


def make_dp_output_ready_payload(
    *,
    content: str,
    session_id: str,
    response_id: str,
    override_active: bool,
    expression_confidence: float,
    elapsed_ms: float,
    fast_path_active: bool,
    destination_channel: str,
) -> Dict[str, Any]:
    """
    Payload for dp.output.ready.v1.

    destination_channel: "voicy" | "echolink" | "ui"
    """
    return {
        "content":               content,
        "session_id":            session_id,
        "response_id":           response_id,
        "override_active":       override_active,
        "expression_confidence": expression_confidence,
        "elapsed_ms":            elapsed_ms,
        "fast_path_active":      fast_path_active,
        "destination_channel":   destination_channel,
    }


def make_persona_config_update_payload(
    *,
    profile_id: Optional[str] = None,
    trait_overrides: Optional[Dict[str, float]] = None,
    overlay: Optional[str] = None,
    relationship: Optional[str] = None,
    requested_by: str = "ui",
) -> Dict[str, Any]:
    """
    Payload for dp.config.update.v1.

    All fields are optional — only supplied fields are applied.
    requested_by: "ui" | "admin" | "api"
    """
    return {
        "profile_id":      profile_id,
        "trait_overrides": trait_overrides or {},
        "overlay":         overlay,
        "relationship":    relationship,
        "requested_by":    requested_by,
    }


# ── DPEventSchemas — schema validation registry ───────────────────────────────

class DPEventSchemas:
    """
    Schema registry for DarkPassenger-related BusEvents (BloodyHeart §2.3).

    BloodyHeart validates every incoming event against this registry before
    dispatching it to DarkPassenger. Events that fail validation are rejected
    and a dp.event.rejected.v1 is emitted.

    Schemas are expressed as {field_name: type} dicts.
    Required fields must be present; type checking is shallow (isinstance).
    Optional fields (prefixed "_opt_") are validated if present.
    """

    _SCHEMAS: Dict[str, Dict[str, type]] = {
        DPEventType.GHOSTMIND_OUTPUT_READY: {
            "content":          str,
            "criticality":      str,
            "uncertainty":      float,
            "session_id":       str,
            "relationship":     str,
            "intent":           str,
            # overlay is optional (None allowed)
        },
        DPEventType.DP_CONFIG_UPDATE: {
            # All fields optional — but dict must be present
        },
        DPEventType.DP_HEALTH_REQUEST: {
            # Empty payload allowed
        },
        DPEventType.SYSTEM_EMERGENCY: {
            "reason": str,
        },
        DPEventType.SYSTEM_SAFE_MODE: {
            "level": int,
        },
    }

    @classmethod
    def validate(cls, event: BusEvent) -> tuple[bool, str]:
        """
        Validate a BusEvent payload against the registered schema.

        Returns:
            (True, "")           if valid
            (False, reason_str)  if invalid
        """
        schema = cls._SCHEMAS.get(event.event_type)
        if schema is None:
            # Unknown event type — pass through (extensible ecosystem)
            return True, ""

        for field_name, expected_type in schema.items():
            if field_name not in event.payload:
                return False, f"Missing required field: '{field_name}'"
            val = event.payload[field_name]
            if val is not None and not isinstance(val, expected_type):
                return False, (
                    f"Field '{field_name}' expected {expected_type.__name__}, "
                    f"got {type(val).__name__}"
                )
        return True, ""
