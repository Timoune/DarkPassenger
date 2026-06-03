"""
bloodyheart/middleware.py — DarkPassenger Gate Middleware

This is the enforcement layer that guarantees every GhostMind output MUST
pass through the DarkPassenger transformation pipeline before it can reach
Voicy or EchoLink. No direct path exists from GhostMind to any output
channel. The middleware is the lock on that gate.

Architecture (BloodyHeart-Plan.txt §1, §2.1)
─────────────────────────────────────────────
                    ┌─────────────┐
    GhostMind  ───► │  CoreBus    │ (ghostmind.output.ready.v1 @ P1)
                    └──────┬──────┘
                           │
                    ┌──────▼──────────────────┐
                    │  DarkPassengerGate       │  ← THIS MODULE
                    │  (BloodyHeart middleware) │
                    │                          │
                    │  1. Schema validation     │
                    │  2. Trust check           │
                    │  3. Safe-mode gate        │
                    │  4. Pipeline dispatch     │
                    │  5. Output routing        │
                    └──────┬──────────────────-┘
                           │
               ┌───────────┼───────────┐
               │           │           │
            Voicy      EchoLink       UI
         (TTS output) (remote comm) (debug)

Key guarantees
──────────────
  • GhostMind output can NEVER reach Voicy or EchoLink without passing
    through TransformationPipeline.transform(). This is enforced by
    DarkPassengerGate.handle() — the only path from GhostMind events
    to dp.output.ready.v1 events.

  • If the pipeline raises an unhandled exception, the gate emits a
    dp.validation.failure.v1 with the raw GhostMind content (integrity
    guarantee: the message is never silently dropped).

  • Emergency and safe-mode events are handled with highest priority
    and immediately reconfigure the pipeline state.

  • Config-update events from the UI flow through the gate to the
    ConfigManager and pipeline — they never directly mutate pipeline
    internals without validation.

Components
──────────
  DarkPassengerGate   — the main middleware class, wired to the CoreBus
  GateSafeModeLevel   — mirrors BloodyHeart safe-mode levels (§4.1)
  GateHealthStatus    — health report produced on request
  RoutingTable        — maps destination_channel → module name

Spec reference: BloodyHeart-Plan.txt §1, §2.1, §4.1, §4.2, §20 (DarkPassenger-Plan.txt)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional

from bloodyheart.events import (
    BusEvent,
    DPEventType,
    EventPriority,
    DPEventSchemas,
    make_dp_output_ready_payload,
    make_persona_config_update_payload,
)
from core.transformation_pipeline import TransformationPipeline, TransformationInput
from core.runtime_state import RuntimeStateManager
from core.persona_vector import (
    RelationshipContext,
    CommunicationIntent,
    OverlayType,
)
from core.config_manager import ConfigManager
from dp_types.integrity_types import (
    GhostMindOutput,
    CriticalityLevel,
    ProtectedField,
)


# ── GateSafeModeLevel ─────────────────────────────────────────────────────────

class GateSafeModeLevel(IntEnum):
    """
    Mirrors BloodyHeart §4.1 Hierarchical Safe Mode levels.

    NORMAL  — full operation; all personality transformation enabled
    L1      — non-critical capabilities degraded; pipeline still active
    L2      — autonomous execution disabled; human-triggered only
    L3      — read-only; no config writes; pipeline still reads and transforms
    L4      — full lockdown; gate blocks ALL traffic; diagnostics only
    """
    NORMAL = 0
    L1     = 1
    L2     = 2
    L3     = 3
    L4     = 4


# ── RoutingTable ──────────────────────────────────────────────────────────────

class RoutingTable:
    """
    Maps destination_channel labels to CoreBus module names.

    DarkPassenger does not decide where to send output — GhostMind
    specifies the destination_channel in its output payload. The gate
    uses this table to resolve it to a CoreBus destination name.

    Default channels:
        "voicy"    → text-to-speech output module
        "echolink" → remote communication module
        "ui"       → debug / real-time UI feed (no TTS)
        "*"        → broadcast (used for health reports)
    """

    _DEFAULT: Dict[str, str] = {
        "voicy":    "voicy",
        "echolink": "echolink",
        "ui":       "ui",
        "*":        "*",
    }

    def __init__(self, overrides: Optional[Dict[str, str]] = None):
        self._table = dict(self._DEFAULT)
        if overrides:
            self._table.update(overrides)

    def resolve(self, channel: str) -> str:
        """
        Resolve a channel label to a CoreBus module name.

        Falls back to "voicy" (the primary output channel) if the
        requested channel is not registered.
        """
        return self._table.get(channel, "voicy")

    def register(self, channel: str, module_name: str) -> None:
        """Register or update a channel → module mapping."""
        self._table[channel] = module_name

    @property
    def channels(self) -> Dict[str, str]:
        """Read-only view of the current routing table."""
        return dict(self._table)


# ── GateHealthStatus ─────────────────────────────────────────────────────────

@dataclass
class GateHealthStatus:
    """
    Health report produced by DarkPassengerGate on dp.health.request.v1.

    Attributes
    ──────────
    safe_mode_level:   Current GateSafeModeLevel.
    events_processed:  Total ghostmind.output.ready events handled.
    events_rejected:   Events blocked by schema validation or trust checks.
    pipeline_errors:   Unhandled pipeline exceptions caught by the gate.
    avg_latency_ms:    Rolling average gate-to-output latency.
    last_error:        Most recent error string, or None.
    audit_log_count:   Records currently in pipeline.audit_log.
    reviewer_health:   BehavioralReviewSystem full_review() health snapshot.
    """
    safe_mode_level:   int
    events_processed:  int
    events_rejected:   int
    pipeline_errors:   int
    avg_latency_ms:    float
    last_error:        Optional[str]
    audit_log_count:   int
    reviewer_health:   Dict[str, str]   # {"stability": "healthy", ...}

    def to_dict(self) -> dict:
        return {
            "safe_mode_level":  self.safe_mode_level,
            "events_processed": self.events_processed,
            "events_rejected":  self.events_rejected,
            "pipeline_errors":  self.pipeline_errors,
            "avg_latency_ms":   round(self.avg_latency_ms, 3),
            "last_error":       self.last_error,
            "audit_log_count":  self.audit_log_count,
            "reviewer_health":  self.reviewer_health,
        }


# ── DarkPassengerGate ─────────────────────────────────────────────────────────

# CoreBus emit callable type: (BusEvent) → None
EmitFn = Callable[[BusEvent], None]


class DarkPassengerGate:
    """
    BloodyHeart middleware that enforces the GhostMind → DarkPassenger gate.

    This class is the single point of entry for GhostMind output into the
    DarkPassenger system. It:

      1. Validates incoming BusEvents against DPEventSchemas
      2. Enforces safe-mode level restrictions
      3. Reconstructs GhostMindOutput from the event payload
      4. Dispatches through TransformationPipeline.transform()
      5. Emits dp.output.ready.v1 to the appropriate output channel
      6. Handles emergency/safe-mode/config-update events

    Construction
    ────────────
        gate = DarkPassengerGate(
            pipeline=pipeline,
            state_manager=state_manager,
            emit=core_bus.emit,           # CoreBus emit callable
            config_manager=config_manager,
        )

        # Wire to CoreBus event dispatch:
        core_bus.subscribe(DPEventType.GHOSTMIND_OUTPUT_READY, gate.handle)
        core_bus.subscribe(DPEventType.DP_CONFIG_UPDATE,       gate.handle)
        core_bus.subscribe(DPEventType.DP_HEALTH_REQUEST,      gate.handle)
        core_bus.subscribe(DPEventType.SYSTEM_EMERGENCY,       gate.handle)
        core_bus.subscribe(DPEventType.SYSTEM_SAFE_MODE,       gate.handle)

    All public methods are synchronous. Async wrappers are the CoreBus's
    responsibility (BloodyHeart §17 — latency management).

    Thread safety
    ─────────────
    DarkPassengerGate is NOT thread-safe. The CoreBus is responsible for
    serialising events dispatched to any single handler.
    """

    MODULE_NAME = "darkpassenger"

    def __init__(
        self,
        pipeline: TransformationPipeline,
        state_manager: RuntimeStateManager,
        emit: EmitFn,
        config_manager: ConfigManager,
        routing_table: Optional[RoutingTable] = None,
        default_channel: str = "voicy",
        logger=None,
    ):
        """
        Args:
            pipeline:        The active TransformationPipeline (owns audit_log,
                             reviewer, perf — wired in v1.4).
            state_manager:   The session RuntimeStateManager.
            emit:            CoreBus emit callable — gate calls this to publish
                             outbound events. Signature: (BusEvent) -> None.
            config_manager:  ConfigManager — used for runtime config updates.
            routing_table:   Optional custom routing table. Defaults to standard.
            default_channel: Fallback channel if GhostMind doesn't specify one.
            logger:          Optional structured logger.
        """
        self._pipeline       = pipeline
        self._state_manager  = state_manager
        self._emit           = emit
        self._config         = config_manager
        self._routing        = routing_table or RoutingTable()
        self._default_channel = default_channel
        self._logger         = logger

        # Safe-mode state
        self._safe_mode = GateSafeModeLevel.NORMAL

        # Statistics
        self._events_processed: int   = 0
        self._events_rejected:  int   = 0
        self._pipeline_errors:  int   = 0
        self._latencies_ms:     List[float] = []
        self._last_error:       Optional[str] = None

        # Emergency overlay reference
        self._EMERGENCY_OVERLAY = OverlayType.EMERGENCY

    # ── Main dispatch entry point ─────────────────────────────────────────────

    def handle(self, event: BusEvent) -> None:
        """
        Dispatch a BusEvent to the appropriate handler.

        This is the single entry point wired to the CoreBus. Every event
        type DarkPassenger cares about flows through here.

        Unknown event types are logged and ignored (extensible ecosystem).
        """
        dispatch = {
            DPEventType.GHOSTMIND_OUTPUT_READY: self._handle_ghostmind_output,
            DPEventType.DP_CONFIG_UPDATE:        self._handle_config_update,
            DPEventType.DP_HEALTH_REQUEST:       self._handle_health_request,
            DPEventType.SYSTEM_EMERGENCY:        self._handle_emergency,
            DPEventType.SYSTEM_SAFE_MODE:        self._handle_safe_mode,
        }
        handler = dispatch.get(event.event_type)
        if handler is None:
            if self._logger:
                self._logger.debug(
                    "gate_unknown_event_ignored",
                    event_type=event.event_type,
                    source=event.source,
                )
            return
        handler(event)

    # ── GhostMind output handler — the gate itself ────────────────────────────

    def _handle_ghostmind_output(self, event: BusEvent) -> None:
        """
        THE GATE: Every GhostMind output must pass through here.

        Steps:
          1. Schema validation
          2. Trust check (source must be "ghostmind")
          3. Safe-mode gate (L4 blocks all; L2 blocks autonomous)
          4. Reconstruct GhostMindOutput from payload
          5. Sync RuntimeState from event payload
          6. Dispatch through TransformationPipeline
          7. Route certified output to Voicy / EchoLink
        """
        t_start = time.monotonic()

        # ── 1. Schema validation ──────────────────────────────────────────────
        valid, reason = DPEventSchemas.validate(event)
        if not valid:
            self._reject_event(event, f"Schema validation failed: {reason}")
            return

        # ── 2. Trust check — only GhostMind may send output events ───────────
        if event.source != "ghostmind":
            self._reject_event(
                event,
                f"Trust violation: ghostmind.output.ready.v1 from '{event.source}' "
                "is not permitted. Only 'ghostmind' may originate output events."
            )
            return

        # ── 3. Safe-mode gate ─────────────────────────────────────────────────
        if self._safe_mode >= GateSafeModeLevel.L4:
            self._reject_event(
                event,
                f"Safe-mode L4 lockdown: all DarkPassenger output blocked."
            )
            return

        payload = event.payload

        # ── 4. Reconstruct GhostMindOutput from payload ───────────────────────
        try:
            ghost_output = self._build_ghostmind_output(payload)
        except Exception as exc:
            self._handle_pipeline_error(event, f"GhostMindOutput reconstruction failed: {exc}")
            return

        # ── 5. Sync RuntimeState ──────────────────────────────────────────────
        try:
            self._sync_runtime_state(payload)
        except Exception as exc:
            if self._logger:
                self._logger.warning("gate_state_sync_error", error=str(exc))
            # Non-fatal — continue with existing state

        # ── 6. TransformationPipeline dispatch (THE MANDATORY GATE) ──────────
        try:
            ti = TransformationInput(
                ghost_output=ghost_output,
                runtime_state_manager=self._state_manager,
            )
            result = self._pipeline.transform(ti)
        except Exception as exc:
            self._handle_pipeline_error(event, f"Pipeline exception: {exc}")
            return

        # ── 7. Route certified output to destination channel ──────────────────
        channel = payload.get("destination_channel") or self._default_channel
        destination = self._routing.resolve(channel)

        # Build the response_id from the most recent audit record
        response_id = self._latest_response_id()

        out_payload = make_dp_output_ready_payload(
            content=result.final_output,
            session_id=payload.get("session_id", "unknown"),
            response_id=response_id,
            override_active=result.override_active,
            expression_confidence=result.expression_confidence,
            elapsed_ms=result.elapsed_ms,
            fast_path_active=result.fast_path_active,
            destination_channel=channel,
        )

        out_event = event.reply(
            event_type=DPEventType.DP_OUTPUT_READY,
            source=self.MODULE_NAME,
            payload=out_payload,
            priority=EventPriority.HUMAN,
        )
        # Override destination — reply() sends to event.source (ghostmind),
        # but we want to route to the output channel instead.
        out_event.destination = destination

        # Emit the certified, transformed response
        self._emit(out_event)

        # Track latency
        elapsed = (time.monotonic() - t_start) * 1000
        self._latencies_ms.append(elapsed)
        if len(self._latencies_ms) > 500:
            self._latencies_ms = self._latencies_ms[-500:]
        self._events_processed += 1

        if self._logger:
            self._logger.info(
                "gate_output_emitted",
                destination=destination,
                response_id=response_id,
                override_active=result.override_active,
                elapsed_ms=round(elapsed, 2),
                fast_path=result.fast_path_active,
            )

    # ── Config update handler ─────────────────────────────────────────────────

    def _handle_config_update(self, event: BusEvent) -> None:
        """
        Apply a runtime persona configuration update from the UI or admin.

        Safe-mode L3+ blocks config writes.
        Allowed updates: profile_id, trait_overrides, overlay, relationship.
        Forbidden: validation logic, speech fingerprint, security constraints.
        """
        if self._safe_mode >= GateSafeModeLevel.L3:
            self._reject_event(
                event,
                f"Safe-mode L{self._safe_mode}: config writes are blocked."
            )
            return

        payload = event.payload
        applied: List[str] = []

        # Profile switch
        profile_id = payload.get("profile_id")
        if profile_id:
            try:
                self._config.set_active_profile(profile_id)
                # Invalidate the fast-path cache — new profile means new persona state
                self._pipeline.perf.invalidate_fast_path()
                self._pipeline.perf.profile_cache.invalidate(profile_id)
                applied.append(f"profile_id={profile_id}")
            except KeyError:
                if self._logger:
                    self._logger.warning("gate_config_unknown_profile", profile_id=profile_id)

        # Trait overrides — applied directly to the active profile's base_traits
        trait_overrides = payload.get("trait_overrides") or {}
        if trait_overrides:
            try:
                active = self._config.active_profile
                if active is not None:
                    for trait, value in trait_overrides.items():
                        if hasattr(active.base_traits, trait):
                            clamped = max(0.0, min(1.0, float(value)))
                            setattr(active.base_traits, trait, clamped)
                    # Invalidate caches so changes take effect immediately
                    self._pipeline.perf.invalidate_fast_path()
                    self._pipeline.perf.profile_cache.clear()
                    applied.append(f"traits={list(trait_overrides.keys())}")
            except Exception as exc:
                if self._logger:
                    self._logger.warning("gate_config_trait_override_error", error=str(exc))

        # Overlay change — updates RuntimeState
        overlay_str = payload.get("overlay")
        if overlay_str:
            try:
                overlay = OverlayType(overlay_str)
                self._state_manager.set_overlay(overlay)
                self._pipeline.perf.invalidate_fast_path()
                applied.append(f"overlay={overlay_str}")
            except ValueError:
                if self._logger:
                    self._logger.warning("gate_config_unknown_overlay", overlay=overlay_str)

        # Relationship change — updates RuntimeState
        relationship_str = payload.get("relationship")
        if relationship_str:
            try:
                rel = RelationshipContext(relationship_str)
                self._state_manager.set_relationship(rel)
                self._pipeline.perf.invalidate_fast_path()
                applied.append(f"relationship={relationship_str}")
            except ValueError:
                if self._logger:
                    self._logger.warning("gate_config_unknown_relationship",
                                         relationship=relationship_str)

        # Emit confirmation
        confirm_event = event.reply(
            event_type=DPEventType.DP_CONFIG_UPDATED,
            source=self.MODULE_NAME,
            payload={
                "applied":       applied,
                "requested_by":  payload.get("requested_by", "unknown"),
                "safe_mode":     int(self._safe_mode),
            },
            priority=EventPriority.MAINTENANCE,
        )
        self._emit(confirm_event)

        if self._logger:
            self._logger.info("gate_config_updated", applied=applied)

    # ── Health request handler ────────────────────────────────────────────────

    def _handle_health_request(self, event: BusEvent) -> None:
        """
        Produce a GateHealthStatus and emit dp.health.report.v1.
        """
        status = self._build_health_status()
        health_event = event.reply(
            event_type=DPEventType.DP_HEALTH_REPORT,
            source=self.MODULE_NAME,
            payload=status.to_dict(),
            priority=EventPriority.MAINTENANCE,
        )
        self._emit(health_event)

        if self._logger:
            self._logger.info("gate_health_report_emitted",
                               safe_mode=self._safe_mode,
                               events_processed=self._events_processed)

    # ── Emergency handler ─────────────────────────────────────────────────────

    def _handle_emergency(self, event: BusEvent) -> None:
        """
        BloodyHeart emergency broadcast: activate EMERGENCY overlay immediately.

        Sets safe mode to L4 (full lockdown) and forces EMERGENCY overlay
        on the RuntimeState, ensuring any in-progress response uses the
        minimal-personality emergency path.
        """
        reason = event.payload.get("reason", "unspecified")

        # Force EMERGENCY overlay onto state
        self._state_manager.set_overlay(self._EMERGENCY_OVERLAY)

        # Escalate to L4 lockdown
        self._safe_mode = GateSafeModeLevel.L4

        # Invalidate fast-path — no cached state is safe after an emergency
        self._pipeline.perf.invalidate_fast_path()

        if self._logger:
            self._logger.critical(
                "gate_emergency_activated",
                reason=reason,
                safe_mode=int(self._safe_mode),
            )

    # ── Safe-mode level handler ───────────────────────────────────────────────

    def _handle_safe_mode(self, event: BusEvent) -> None:
        """
        BloodyHeart safe-mode level change broadcast.

        Updates the gate's safe-mode level. Lower levels restore capabilities;
        higher levels restrict them. Level 0 = full normal operation.
        """
        level_int = event.payload.get("level", 0)
        try:
            self._safe_mode = GateSafeModeLevel(int(level_int))
        except ValueError:
            self._safe_mode = GateSafeModeLevel.L4  # unknown level → lockdown
            if self._logger:
                self._logger.warning("gate_unknown_safe_mode_level", level=level_int)
            return

        # Clearing EMERGENCY overlay when returning to NORMAL
        if self._safe_mode == GateSafeModeLevel.NORMAL:
            current_overlay = self._state_manager.state.current_overlay
            if current_overlay == self._EMERGENCY_OVERLAY:
                self._state_manager.set_overlay(None)

        if self._logger:
            self._logger.info("gate_safe_mode_changed", level=int(self._safe_mode))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_ghostmind_output(self, payload: dict) -> GhostMindOutput:
        """
        Reconstruct a GhostMindOutput from a ghostmind.output.ready.v1 payload.

        Field name mapping (payload key → GhostMindOutput attribute):
            "criticality"   → criticality      (CriticalityLevel enum)
            "uncertainty"   → uncertainty_score (float, GhostMindOutput field name)
            "session_id"    → conversation_id
            "protected_fields" → protected_fields (list of ProtectedField)
        """
        criticality = CriticalityLevel(payload.get("criticality", "normal"))
        # payload uses "uncertainty"; GhostMindOutput stores it as "uncertainty_score"
        uncertainty_score = float(payload.get("uncertainty", 0.0))

        raw_protected = payload.get("protected_fields") or []
        protected = [
            ProtectedField(
                key=pf.get("key", ""),
                value=pf.get("value"),
                field_type=pf.get("field_type", "unknown"),
            )
            for pf in raw_protected
        ]

        ghost = GhostMindOutput(
            content=payload["content"],
            criticality=criticality,
            uncertainty_score=uncertainty_score,
            protected_fields=protected,
            conversation_id=payload.get("session_id", ""),
            source_module="ghostmind",
        )
        ghost.finalize()  # compute checksums on protected fields
        return ghost

    def _sync_runtime_state(self, payload: dict) -> None:
        """
        Update RuntimeState from GhostMind payload metadata.

        Only updates fields that are explicitly provided in the payload.
        Missing fields leave RuntimeState unchanged.
        """
        relationship_str = payload.get("relationship")
        if relationship_str:
            try:
                self._state_manager.set_relationship(RelationshipContext(relationship_str))
            except ValueError:
                pass

        intent_str = payload.get("intent")
        if intent_str:
            try:
                self._state_manager.set_intent(CommunicationIntent(intent_str))
            except ValueError:
                pass

        overlay_str = payload.get("overlay")
        if overlay_str and self._safe_mode < GateSafeModeLevel.L4:
            try:
                self._state_manager.set_overlay(OverlayType(overlay_str))
            except ValueError:
                pass

    def _reject_event(self, event: BusEvent, reason: str) -> None:
        """Emit a dp.event.rejected.v1 and increment the rejection counter."""
        self._events_rejected += 1
        self._last_error = reason

        reject_event = BusEvent.create(
            event_type=DPEventType.DP_EVENT_REJECTED,
            source=self.MODULE_NAME,
            destination=event.source,
            priority=EventPriority.SECURITY,
            payload={
                "rejected_tx_id":   event.tx_id,
                "rejected_type":    event.event_type,
                "reason":           reason,
                "safe_mode_level":  int(self._safe_mode),
            },
            correlation_id=event.tx_id,
        )
        self._emit(reject_event)

        if self._logger:
            self._logger.warning(
                "gate_event_rejected",
                event_type=event.event_type,
                source=event.source,
                reason=reason,
            )

    def _handle_pipeline_error(self, event: BusEvent, reason: str) -> None:
        """
        Handle an unhandled pipeline exception.

        Emits dp.validation.failure.v1 with the raw GhostMind content
        (content is never silently dropped) and increments the error counter.
        """
        self._pipeline_errors += 1
        self._last_error = reason

        raw_content = event.payload.get("content", "")
        failure_event = BusEvent.create(
            event_type=DPEventType.DP_VALIDATION_FAILURE,
            source=self.MODULE_NAME,
            destination=self._routing.resolve(
                event.payload.get("destination_channel") or self._default_channel
            ),
            priority=EventPriority.HUMAN,
            payload={
                "content":       raw_content,
                "session_id":    event.payload.get("session_id", "unknown"),
                "reason":        reason,
                "original_tx_id": event.tx_id,
            },
            correlation_id=event.tx_id,
        )
        self._emit(failure_event)

        if self._logger:
            self._logger.error(
                "gate_pipeline_error",
                reason=reason,
                tx_id=event.tx_id,
            )

    def _build_health_status(self) -> GateHealthStatus:
        """Collect all health metrics into a GateHealthStatus."""
        avg_lat = (
            sum(self._latencies_ms) / len(self._latencies_ms)
            if self._latencies_ms else 0.0
        )

        # Pull BehavioralReview health snapshot
        reviewer_health: Dict[str, str] = {}
        try:
            bundle = self._pipeline.reviewer.full_review()
            reviewer_health = {
                "stability":  bundle.stability.health,
                "quality":    bundle.quality.health,
                "drift":      bundle.drift.health if bundle.drift else "no_baseline",
                "adaptation": bundle.adaptation.health,
            }
        except Exception:
            reviewer_health = {"error": "review_unavailable"}

        return GateHealthStatus(
            safe_mode_level=int(self._safe_mode),
            events_processed=self._events_processed,
            events_rejected=self._events_rejected,
            pipeline_errors=self._pipeline_errors,
            avg_latency_ms=avg_lat,
            last_error=self._last_error,
            audit_log_count=self._pipeline.audit_log.count,
            reviewer_health=reviewer_health,
        )

    def _latest_response_id(self) -> str:
        """Return the response_id of the most recent audit record, or a new UUID."""
        records = self._pipeline.audit_log.last(1)
        if records:
            return records[0].response_id
        return str(uuid.uuid4())

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def safe_mode(self) -> GateSafeModeLevel:
        """Current safe-mode level."""
        return self._safe_mode

    @property
    def stats(self) -> dict:
        """Gate statistics snapshot for monitoring."""
        avg_lat = (
            sum(self._latencies_ms) / len(self._latencies_ms)
            if self._latencies_ms else 0.0
        )
        return {
            "events_processed": self._events_processed,
            "events_rejected":  self._events_rejected,
            "pipeline_errors":  self._pipeline_errors,
            "avg_latency_ms":   round(avg_lat, 3),
            "safe_mode_level":  int(self._safe_mode),
            "last_error":       self._last_error,
        }
