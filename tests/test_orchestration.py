"""
tests/test_orchestration.py — v1.5 Orchestration Integration Tests

Covers:
  bloodyheart.events     — BusEvent, EventPriority, DPEventType, DPEventSchemas
  bloodyheart.middleware  — DarkPassengerGate (gate enforcement, safe-mode,
                           config updates, health reports, emergency handling)
  ui.config_interface    — PersonaConfigInterface, controls, PresetManager,
                           platform adapters
"""

from __future__ import annotations

import json
import uuid
from typing import List
from unittest.mock import MagicMock, patch, call

# ── Module imports ────────────────────────────────────────────────────────────
from bloodyheart.events import (
    BusEvent, EventPriority, DPEventType, DPEventSchemas,
    make_ghostmind_output_payload, make_dp_output_ready_payload,
    make_persona_config_update_payload,
)
from bloodyheart.middleware import (
    DarkPassengerGate, GateSafeModeLevel, GateHealthStatus, RoutingTable,
)
from ui.config_interface import (
    PersonaConfigInterface, UIControlState,
    SliderControl, ToggleControl, DropdownControl, BlendControl,
    PersonaPreset, PresetManager,
    DesktopConfigAdapter, WebConfigAdapter, MobileConfigAdapter, DiscordConfigAdapter,
)
from core.persona_vector import OverlayType, RelationshipContext, CommunicationIntent
from core.runtime_state import RuntimeStateManager

# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def ok(msg): global PASS; PASS += 1; print(f"  ✓  {msg}")
def fail(msg, e=""): global FAIL; FAIL += 1; print(f"  ✗  {msg}: {e}")
def check(name, fn):
    try: fn(); ok(name)
    except Exception as e: fail(name, e)
def assert_(v, m=""): 
    if not v: raise AssertionError(m or f"Expected truthy, got {v!r}")
def raises(exc, fn):
    try: fn(); raise AssertionError(f"{exc.__name__} not raised")
    except exc: pass

def make_emit() -> tuple:
    """Return (emit_fn, emitted_events_list)."""
    events: List[BusEvent] = []
    def emit(e: BusEvent): events.append(e)
    return emit, events

def make_ghostmind_event(
    content="Hello world",
    criticality="normal",
    uncertainty=0.1,
    session_id="sess-1",
    relationship="owner",
    intent="inform",
    overlay=None,
    destination_channel="voicy",
    source="ghostmind",
) -> BusEvent:
    payload = make_ghostmind_output_payload(
        content=content, criticality=criticality, uncertainty=uncertainty,
        session_id=session_id, relationship=relationship, intent=intent,
        overlay=overlay, protected_fields=[],
    )
    payload["destination_channel"] = destination_channel
    return BusEvent.create(
        event_type=DPEventType.GHOSTMIND_OUTPUT_READY,
        source=source,
        destination="darkpassenger",
        priority=EventPriority.HUMAN,
        payload=payload,
    )

def make_gate(emit=None, pipeline=None, state_manager=None, config=None):
    """Build a DarkPassengerGate with lightweight mocks."""
    emit = emit or (lambda e: None)
    if pipeline is None:
        pipeline = MagicMock()
        # TransformationResult mock
        result = MagicMock()
        result.final_output = "transformed output"
        result.override_active = False
        result.expression_confidence = 0.9
        result.elapsed_ms = 50.0
        result.fast_path_active = False
        result.pipeline_warnings = []
        result.stages_executed = ["input_reception", "final_response"]
        result.relationship = "owner"
        result.intent = "inform"
        pipeline.transform.return_value = result
        pipeline.audit_log = MagicMock()
        pipeline.audit_log.count = 0
        pipeline.audit_log.last.return_value = []
        pipeline.perf = MagicMock()
        pipeline.reviewer = MagicMock()
        bundle = MagicMock()
        bundle.stability.health = "healthy"
        bundle.quality.health = "healthy"
        bundle.drift = None
        bundle.adaptation.health = "no_action"
        pipeline.reviewer.full_review.return_value = bundle
    if state_manager is None:
        state_manager = RuntimeStateManager()
    if config is None:
        config = MagicMock()
        config.active_profile = None
    return DarkPassengerGate(
        pipeline=pipeline,
        state_manager=state_manager,
        emit=emit,
        config_manager=config,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BusEvent tests
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ BusEvent ═══")

def t_bus_event_create():
    e = BusEvent.create(
        event_type=DPEventType.GHOSTMIND_OUTPUT_READY,
        source="ghostmind", destination="darkpassenger",
        priority=EventPriority.HUMAN, payload={"content": "hi"},
    )
    assert_(e.event_type == DPEventType.GHOSTMIND_OUTPUT_READY)
    assert_(e.source == "ghostmind")
    assert_(e.destination == "darkpassenger")
    assert_(e.priority == EventPriority.HUMAN)
    assert_(isinstance(e.tx_id, str) and len(e.tx_id) > 0)
    assert_(isinstance(e.timestamp_utc, str))
check("BusEvent.create sets all fields", t_bus_event_create)

def t_bus_event_unique_tx_ids():
    e1 = BusEvent.create("t","a","b",EventPriority.HUMAN,{})
    e2 = BusEvent.create("t","a","b",EventPriority.HUMAN,{})
    assert_(e1.tx_id != e2.tx_id)
check("BusEvent.create generates unique tx_ids", t_bus_event_unique_tx_ids)

def t_bus_event_reply():
    orig = BusEvent.create("orig","ghostmind","dp",EventPriority.HUMAN,{})
    reply = orig.reply("reply_type","darkpassenger",{"result":"ok"})
    assert_(reply.destination == "ghostmind")
    assert_(reply.correlation_id == orig.tx_id)
    assert_(reply.source == "darkpassenger")
check("BusEvent.reply sets destination and correlation_id", t_bus_event_reply)

def t_bus_event_to_dict():
    e = BusEvent.create("t","a","b",EventPriority.MAINTENANCE,{"x":1})
    d = e.to_dict()
    assert_(isinstance(d, dict))
    assert_(json.dumps(d))  # JSON-serialisable
    assert_(d["event_type"] == "t")
    assert_(d["priority"] == 4)
check("BusEvent.to_dict serialises cleanly", t_bus_event_to_dict)

def t_event_priority_ordering():
    assert_(EventPriority.SECURITY < EventPriority.HUMAN)
    assert_(EventPriority.HUMAN < EventPriority.AUTONOMOUS)
    assert_(EventPriority.AUTONOMOUS < EventPriority.COGNITIVE)
    assert_(EventPriority.COGNITIVE < EventPriority.MAINTENANCE)
check("EventPriority ordering is correct (P0<P1<P2<P3<P4)", t_event_priority_ordering)


# ══════════════════════════════════════════════════════════════════════════════
# DPEventSchemas tests
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ DPEventSchemas ═══")

def t_schema_valid_ghostmind_event():
    e = make_ghostmind_event()
    valid, reason = DPEventSchemas.validate(e)
    assert_(valid, f"Expected valid, got: {reason}")
check("ghostmind.output.ready.v1 with valid payload passes", t_schema_valid_ghostmind_event)

def t_schema_missing_field():
    e = BusEvent.create(
        DPEventType.GHOSTMIND_OUTPUT_READY, "ghostmind", "dp",
        EventPriority.HUMAN, {"content": "hi"},  # missing criticality etc.
    )
    valid, reason = DPEventSchemas.validate(e)
    assert_(not valid)
    assert_("criticality" in reason or "Missing" in reason)
check("ghostmind.output.ready.v1 with missing field fails", t_schema_missing_field)

def t_schema_unknown_event_type_passes():
    e = BusEvent.create("unknown.module.v99","x","y",EventPriority.MAINTENANCE,{})
    valid, reason = DPEventSchemas.validate(e)
    assert_(valid, "Unknown types should pass through")
check("unknown event type passes schema validation (extensible)", t_schema_unknown_event_type_passes)


# ══════════════════════════════════════════════════════════════════════════════
# RoutingTable tests
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ RoutingTable ═══")

def t_routing_defaults():
    rt = RoutingTable()
    assert_(rt.resolve("voicy")    == "voicy")
    assert_(rt.resolve("echolink") == "echolink")
    assert_(rt.resolve("ui")       == "ui")
check("RoutingTable default mappings correct", t_routing_defaults)

def t_routing_fallback():
    rt = RoutingTable()
    assert_(rt.resolve("unknown_channel") == "voicy")
check("RoutingTable falls back to voicy for unknown channel", t_routing_fallback)

def t_routing_register():
    rt = RoutingTable()
    rt.register("telegram", "telegram_bot")
    assert_(rt.resolve("telegram") == "telegram_bot")
check("RoutingTable.register adds custom mapping", t_routing_register)

def t_routing_overrides():
    rt = RoutingTable(overrides={"voicy": "custom_tts"})
    assert_(rt.resolve("voicy") == "custom_tts")
check("RoutingTable accepts constructor overrides", t_routing_overrides)


# ══════════════════════════════════════════════════════════════════════════════
# DarkPassengerGate — gate enforcement
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ DarkPassengerGate — gate enforcement ═══")

def t_gate_emits_dp_output_ready():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(make_ghostmind_event())
    output_events = [e for e in events if e.event_type == DPEventType.DP_OUTPUT_READY]
    assert_(len(output_events) == 1, f"Expected 1 dp.output.ready, got {len(output_events)}")
check("gate emits dp.output.ready.v1 on valid ghostmind output", t_gate_emits_dp_output_ready)

def t_gate_routes_to_voicy():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(make_ghostmind_event(destination_channel="voicy"))
    out = [e for e in events if e.event_type == DPEventType.DP_OUTPUT_READY]
    assert_(len(out) == 1)
    assert_(out[0].destination == "voicy")
check("gate routes output to voicy channel", t_gate_routes_to_voicy)

def t_gate_routes_to_echolink():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(make_ghostmind_event(destination_channel="echolink"))
    out = [e for e in events if e.event_type == DPEventType.DP_OUTPUT_READY]
    assert_(len(out) == 1 and out[0].destination == "echolink")
check("gate routes output to echolink channel", t_gate_routes_to_echolink)

def t_gate_calls_pipeline_transform():
    emit, _ = make_emit()
    pipeline = MagicMock()
    result = MagicMock()
    result.final_output = "ok"
    result.override_active = False
    result.expression_confidence = 0.9
    result.elapsed_ms = 30.0
    result.fast_path_active = False
    result.pipeline_warnings = []
    result.stages_executed = []
    result.relationship = "owner"
    result.intent = "inform"
    pipeline.transform.return_value = result
    pipeline.audit_log = MagicMock()
    pipeline.audit_log.count = 0
    pipeline.audit_log.last.return_value = []
    pipeline.perf = MagicMock()
    pipeline.reviewer = MagicMock()
    pipeline.reviewer.full_review.return_value = MagicMock(
        stability=MagicMock(health="healthy"),
        quality=MagicMock(health="healthy"),
        drift=None,
        adaptation=MagicMock(health="no_action"),
    )
    gate = make_gate(emit=emit, pipeline=pipeline)
    gate.handle(make_ghostmind_event())
    assert_(pipeline.transform.called, "pipeline.transform must be called")
check("gate always calls pipeline.transform (gate guarantee)", t_gate_calls_pipeline_transform)

def t_gate_rejects_non_ghostmind_source():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(make_ghostmind_event(source="rogue_module"))
    rejected = [e for e in events if e.event_type == DPEventType.DP_EVENT_REJECTED]
    assert_(len(rejected) == 1)
    assert_("Trust violation" in rejected[0].payload["reason"])
check("gate rejects output.ready.v1 from non-ghostmind source", t_gate_rejects_non_ghostmind_source)

def t_gate_rejects_invalid_schema():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    bad_event = BusEvent.create(
        DPEventType.GHOSTMIND_OUTPUT_READY, "ghostmind", "dp",
        EventPriority.HUMAN, {"content": "hi"},  # missing required fields
    )
    gate.handle(bad_event)
    rejected = [e for e in events if e.event_type == DPEventType.DP_EVENT_REJECTED]
    assert_(len(rejected) == 1)
check("gate rejects event with schema validation failure", t_gate_rejects_invalid_schema)

def t_gate_rejection_increments_counter():
    emit, _ = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(make_ghostmind_event(source="bad_actor"))
    gate.handle(make_ghostmind_event(source="bad_actor"))
    assert_(gate.stats["events_rejected"] == 2)
check("gate rejection counter increments correctly", t_gate_rejection_increments_counter)

def t_gate_processes_counter_increments():
    emit, _ = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(make_ghostmind_event())
    gate.handle(make_ghostmind_event())
    assert_(gate.stats["events_processed"] == 2)
check("gate processed counter increments correctly", t_gate_processes_counter_increments)

def t_gate_output_payload_fields():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(make_ghostmind_event(session_id="sess-42"))
    out = [e for e in events if e.event_type == DPEventType.DP_OUTPUT_READY]
    assert_(len(out) == 1)
    p = out[0].payload
    for field in ["content","session_id","response_id","override_active",
                  "expression_confidence","elapsed_ms","fast_path_active",
                  "destination_channel"]:
        assert_(field in p, f"Missing field: {field}")
    assert_(p["session_id"] == "sess-42")
check("dp.output.ready.v1 payload has all required fields", t_gate_output_payload_fields)

def t_gate_correlates_output_to_input():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    input_event = make_ghostmind_event()
    gate.handle(input_event)
    out = [e for e in events if e.event_type == DPEventType.DP_OUTPUT_READY]
    assert_(len(out) == 1)
    assert_(out[0].correlation_id == input_event.tx_id)
check("gate output event correlates to input tx_id", t_gate_correlates_output_to_input)


# ══════════════════════════════════════════════════════════════════════════════
# DarkPassengerGate — safe-mode
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ DarkPassengerGate — safe-mode ═══")

def t_gate_l4_blocks_all_output():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    # Trigger L4 via safe_mode event
    gate.handle(BusEvent.create(DPEventType.SYSTEM_SAFE_MODE,"bloodyheart","dp",
                                EventPriority.SECURITY, {"level": 4}))
    gate.handle(make_ghostmind_event())
    out = [e for e in events if e.event_type == DPEventType.DP_OUTPUT_READY]
    rejected = [e for e in events if e.event_type == DPEventType.DP_EVENT_REJECTED]
    assert_(len(out) == 0, "L4 should block all output")
    assert_(len(rejected) >= 1)
check("gate L4 safe-mode blocks all GhostMind output", t_gate_l4_blocks_all_output)

def t_gate_l3_blocks_config_writes():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(BusEvent.create(DPEventType.SYSTEM_SAFE_MODE,"bloodyheart","dp",
                                EventPriority.SECURITY, {"level": 3}))
    cfg_event = BusEvent.create(DPEventType.DP_CONFIG_UPDATE,"ui","dp",
                                EventPriority.HUMAN,
                                make_persona_config_update_payload(profile_id="default"))
    gate.handle(cfg_event)
    rejected = [e for e in events if e.event_type == DPEventType.DP_EVENT_REJECTED]
    assert_(len(rejected) >= 1)
    assert_("config writes" in rejected[0].payload["reason"].lower() or
            "L3" in rejected[0].payload["reason"])
check("gate L3 safe-mode blocks config writes", t_gate_l3_blocks_config_writes)

def t_gate_safe_mode_level_property():
    gate = make_gate()
    assert_(gate.safe_mode == GateSafeModeLevel.NORMAL)
    gate.handle(BusEvent.create(DPEventType.SYSTEM_SAFE_MODE,"bh","dp",
                                EventPriority.SECURITY, {"level": 2}))
    assert_(gate.safe_mode == GateSafeModeLevel.L2)
check("gate safe_mode property reflects current level", t_gate_safe_mode_level_property)

def t_gate_normal_restores_from_l4():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(BusEvent.create(DPEventType.SYSTEM_SAFE_MODE,"bh","dp",
                                EventPriority.SECURITY, {"level": 4}))
    assert_(gate.safe_mode == GateSafeModeLevel.L4)
    gate.handle(BusEvent.create(DPEventType.SYSTEM_SAFE_MODE,"bh","dp",
                                EventPriority.SECURITY, {"level": 0}))
    assert_(gate.safe_mode == GateSafeModeLevel.NORMAL)
    # Now output should work again
    gate.handle(make_ghostmind_event())
    out = [e for e in events if e.event_type == DPEventType.DP_OUTPUT_READY]
    assert_(len(out) == 1)
check("gate restores to NORMAL after safe-mode cleared", t_gate_normal_restores_from_l4)


# ══════════════════════════════════════════════════════════════════════════════
# DarkPassengerGate — emergency
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ DarkPassengerGate — emergency ═══")

def t_gate_emergency_sets_l4():
    gate = make_gate()
    gate.handle(BusEvent.create(DPEventType.SYSTEM_EMERGENCY,"bloodyheart","dp",
                                EventPriority.SECURITY, {"reason": "security breach"}))
    assert_(gate.safe_mode == GateSafeModeLevel.L4)
check("emergency event sets safe-mode to L4", t_gate_emergency_sets_l4)

def t_gate_emergency_sets_emergency_overlay():
    state_mgr = RuntimeStateManager()
    gate = make_gate(state_manager=state_mgr)
    gate.handle(BusEvent.create(DPEventType.SYSTEM_EMERGENCY,"bloodyheart","dp",
                                EventPriority.SECURITY, {"reason": "test"}))
    assert_(state_mgr.state.current_overlay == OverlayType.EMERGENCY)
check("emergency event activates EMERGENCY overlay", t_gate_emergency_sets_emergency_overlay)

def t_gate_emergency_blocks_subsequent_output():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(BusEvent.create(DPEventType.SYSTEM_EMERGENCY,"bloodyheart","dp",
                                EventPriority.SECURITY, {"reason": "test"}))
    gate.handle(make_ghostmind_event())
    out = [e for e in events if e.event_type == DPEventType.DP_OUTPUT_READY]
    assert_(len(out) == 0, "Emergency L4 should block output")
check("emergency blocks subsequent GhostMind output", t_gate_emergency_blocks_subsequent_output)


# ══════════════════════════════════════════════════════════════════════════════
# DarkPassengerGate — config updates
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ DarkPassengerGate — config updates ═══")

def t_gate_config_update_emits_confirmed():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    cfg_event = BusEvent.create(
        DPEventType.DP_CONFIG_UPDATE, "ui", "dp", EventPriority.HUMAN,
        make_persona_config_update_payload(overlay="focused"),
    )
    gate.handle(cfg_event)
    confirmed = [e for e in events if e.event_type == DPEventType.DP_CONFIG_UPDATED]
    assert_(len(confirmed) == 1)
check("config update emits dp.config.updated.v1 confirmation", t_gate_config_update_emits_confirmed)

def t_gate_config_overlay_updates_state():
    emit, _ = make_emit()
    state_mgr = RuntimeStateManager()
    gate = make_gate(emit=emit, state_manager=state_mgr)
    cfg_event = BusEvent.create(
        DPEventType.DP_CONFIG_UPDATE, "ui", "dp", EventPriority.HUMAN,
        make_persona_config_update_payload(overlay="teaching"),
    )
    gate.handle(cfg_event)
    assert_(state_mgr.state.current_overlay == OverlayType.TEACHING)
check("config update: overlay change updates RuntimeState", t_gate_config_overlay_updates_state)

def t_gate_config_relationship_updates_state():
    emit, _ = make_emit()
    state_mgr = RuntimeStateManager()
    gate = make_gate(emit=emit, state_manager=state_mgr)
    cfg_event = BusEvent.create(
        DPEventType.DP_CONFIG_UPDATE, "ui", "dp", EventPriority.HUMAN,
        make_persona_config_update_payload(relationship="friend"),
    )
    gate.handle(cfg_event)
    assert_(state_mgr.state.active_relationship == RelationshipContext.FRIEND)
check("config update: relationship change updates RuntimeState", t_gate_config_relationship_updates_state)

def t_gate_config_trait_override_applies():
    emit, _ = make_emit()
    pipeline = MagicMock()
    result = MagicMock()
    result.final_output = "ok"; result.override_active = False
    result.expression_confidence = 0.9; result.elapsed_ms = 30.0
    result.fast_path_active = False; result.pipeline_warnings = []
    result.stages_executed = []; result.relationship = "owner"; result.intent = "inform"
    pipeline.transform.return_value = result
    pipeline.audit_log = MagicMock(); pipeline.audit_log.count = 0
    pipeline.audit_log.last.return_value = []
    pipeline.perf = MagicMock()
    pipeline.reviewer = MagicMock()
    pipeline.reviewer.full_review.return_value = MagicMock(
        stability=MagicMock(health="healthy"), quality=MagicMock(health="healthy"),
        drift=None, adaptation=MagicMock(health="no_action"),
    )
    # Active profile mock with base_traits
    profile = MagicMock()
    profile.base_traits = MagicMock()
    profile.base_traits.humor = 0.70
    from core.config_manager import ConfigManager
    config = MagicMock(spec=ConfigManager)
    config.active_profile = profile

    gate = make_gate(emit=emit, pipeline=pipeline, config=config)
    cfg_event = BusEvent.create(
        DPEventType.DP_CONFIG_UPDATE, "ui", "dp", EventPriority.HUMAN,
        make_persona_config_update_payload(trait_overrides={"humor": 0.90}),
    )
    gate.handle(cfg_event)
    # The caches should have been invalidated
    assert_(pipeline.perf.invalidate_fast_path.called)
check("config update: trait overrides invalidate fast-path cache", t_gate_config_trait_override_applies)


# ══════════════════════════════════════════════════════════════════════════════
# DarkPassengerGate — health report
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ DarkPassengerGate — health report ═══")

def t_gate_health_report_emitted():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    health_req = BusEvent.create(
        DPEventType.DP_HEALTH_REQUEST, "bloodyheart", "dp",
        EventPriority.MAINTENANCE, {},
    )
    gate.handle(health_req)
    reports = [e for e in events if e.event_type == DPEventType.DP_HEALTH_REPORT]
    assert_(len(reports) == 1)
check("health request emits dp.health.report.v1", t_gate_health_report_emitted)

def t_gate_health_report_payload_fields():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    gate.handle(BusEvent.create(DPEventType.DP_HEALTH_REQUEST,"bh","dp",
                                EventPriority.MAINTENANCE,{}))
    rpt = [e for e in events if e.event_type == DPEventType.DP_HEALTH_REPORT][0]
    for f in ["safe_mode_level","events_processed","events_rejected","pipeline_errors",
              "avg_latency_ms","audit_log_count","reviewer_health"]:
        assert_(f in rpt.payload, f"Missing field: {f}")
check("health report payload has all required fields", t_gate_health_report_payload_fields)

def t_gate_health_report_correlates_to_request():
    emit, events = make_emit()
    gate = make_gate(emit=emit)
    req = BusEvent.create(DPEventType.DP_HEALTH_REQUEST,"bh","dp",EventPriority.MAINTENANCE,{})
    gate.handle(req)
    rpt = [e for e in events if e.event_type == DPEventType.DP_HEALTH_REPORT][0]
    assert_(rpt.correlation_id == req.tx_id)
check("health report correlation_id matches request tx_id", t_gate_health_report_correlates_to_request)


# ══════════════════════════════════════════════════════════════════════════════
# SliderControl, ToggleControl, DropdownControl, BlendControl
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ UI Controls ═══")

def t_slider_clamps():
    s = SliderControl("humor", 0.7)
    s2 = s.set(1.5)  # above max
    assert_(s2.value == 1.0)
    s3 = s.set(-0.5)  # below min
    assert_(s3.value == 0.0)
check("SliderControl.set clamps to [min, max]", t_slider_clamps)

def t_slider_immutable_update():
    s = SliderControl("humor", 0.5)
    s2 = s.set(0.8)
    assert_(s.value == 0.5, "Original should be unchanged")
    assert_(s2.value == 0.8)
check("SliderControl.set returns new instance (immutable)", t_slider_immutable_update)

def t_slider_to_dict():
    s = SliderControl("humor", 0.7, label="Humor Level")
    d = s.to_dict()
    assert_(d["type"] == "slider")
    assert_(d["name"] == "humor")
    assert_(d["value"] == 0.7)
    assert_(d["label"] == "Humor Level")
check("SliderControl.to_dict has correct structure", t_slider_to_dict)

def t_slider_auto_label():
    s = SliderControl("analytical_depth", 0.5)
    assert_(s.label == "Analytical Depth")
check("SliderControl auto-generates label from name", t_slider_auto_label)

def t_toggle_set():
    t = ToggleControl("feature", True)
    t2 = t.set(False)
    assert_(t.value is True and t2.value is False)
check("ToggleControl.set returns new instance", t_toggle_set)

def t_dropdown_invalid_option():
    d = DropdownControl("overlay", "none", options=["none","focused","teaching"])
    raises(ValueError, lambda: d.set("invalid_option"))
check("DropdownControl.set raises ValueError for invalid option", t_dropdown_invalid_option)

def t_dropdown_valid_option():
    d = DropdownControl("overlay", "none", options=["none","focused","teaching"])
    d2 = d.set("teaching")
    assert_(d2.value == "teaching")
check("DropdownControl.set accepts valid option", t_dropdown_valid_option)

def t_dropdown_to_dict():
    d = DropdownControl("overlay","none",options=["none","focused"])
    dd = d.to_dict()
    assert_(dd["type"] == "dropdown")
    assert_("options" in dd)
check("DropdownControl.to_dict has correct structure", t_dropdown_to_dict)

def t_blend_set_weight():
    b = BlendControl()
    b2 = b.set_weight("focused", 0.7)
    assert_(b2.weights["focused"] == 0.7)
    assert_(b.weights["focused"] == 0.0, "Original unchanged")
check("BlendControl.set_weight is immutable", t_blend_set_weight)

def t_blend_clamps_weight():
    b = BlendControl()
    b2 = b.set_weight("focused", 1.5)
    assert_(b2.weights["focused"] == 1.0)
check("BlendControl.set_weight clamps to [0,1]", t_blend_clamps_weight)

def t_blend_active_blends():
    b = BlendControl()
    b = b.set_weight("focused", 0.7)
    b = b.set_weight("teaching", 0.3)
    b = b.set_blend_active(True)
    active = b.active_blends()
    assert_("focused" in active and "teaching" in active)
    assert_(all(v > 0 for v in active.values()))
check("BlendControl.active_blends returns non-zero weights when active", t_blend_active_blends)

def t_blend_inactive_returns_empty():
    b = BlendControl()
    b = b.set_weight("focused", 0.7)
    # blend_active defaults to False
    assert_(b.active_blends() == {})
check("BlendControl.active_blends returns empty when inactive", t_blend_inactive_returns_empty)


# ══════════════════════════════════════════════════════════════════════════════
# PersonaConfigInterface
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ PersonaConfigInterface ═══")

def make_interface(emit=None):
    emit = emit or (lambda e: None)
    return PersonaConfigInterface(
        emit=emit,
        available_profiles=["default","professional","darkpassenger"],
        initial_trait_values={"humor": 0.70, "directness": 0.90},
    )

def t_interface_initial_state():
    iface = make_interface()
    state = iface.control_state
    assert_(isinstance(state, UIControlState))
    assert_("humor" in state.trait_sliders)
    assert_(state.trait_sliders["humor"].value == 0.70)
    assert_(state.trait_sliders["directness"].value == 0.90)
check("PersonaConfigInterface builds correct initial state", t_interface_initial_state)

def t_interface_set_trait_buffers():
    iface = make_interface()
    iface.set_trait("humor", 0.85)
    assert_(iface.control_state.trait_sliders["humor"].value == 0.85)
    assert_(iface.has_pending_changes)
check("set_trait buffers change and updates control state", t_interface_set_trait_buffers)

def t_interface_apply_emits_event():
    emit, events = make_emit()
    iface = make_interface(emit=emit)
    iface.set_trait("humor", 0.85)
    result = iface.apply()
    assert_(result is True)
    cfg_events = [e for e in events if e.event_type == DPEventType.DP_CONFIG_UPDATE]
    assert_(len(cfg_events) >= 1)
check("apply() emits dp.config.update.v1", t_interface_apply_emits_event)

def t_interface_apply_empty_returns_false():
    emit, events = make_emit()
    iface = make_interface(emit=emit)
    result = iface.apply()
    assert_(result is False)
    assert_(len(events) == 0)
check("apply() returns False when nothing pending", t_interface_apply_empty_returns_false)

def t_interface_set_trait_immediate():
    emit, events = make_emit()
    iface = make_interface(emit=emit)
    iface.set_trait("warmth", 0.6, immediate=True)
    cfg = [e for e in events if e.event_type == DPEventType.DP_CONFIG_UPDATE]
    assert_(len(cfg) >= 1)
    assert_(not iface.has_pending_changes)
check("set_trait(immediate=True) emits event immediately", t_interface_set_trait_immediate)

def t_interface_set_overlay():
    emit, events = make_emit()
    iface = make_interface(emit=emit)
    iface.set_overlay("teaching")
    assert_(iface.control_state.overlay_dropdown.value == "teaching")
    iface.apply()
    cfg = [e for e in events if e.event_type == DPEventType.DP_CONFIG_UPDATE]
    assert_(len(cfg) >= 1)
    assert_(cfg[0].payload["overlay"] == "teaching")
check("set_overlay buffers and emits overlay update", t_interface_set_overlay)

def t_interface_set_overlay_none():
    emit, events = make_emit()
    iface = make_interface(emit=emit)
    iface.set_overlay("none", immediate=True)
    cfg = [e for e in events if e.event_type == DPEventType.DP_CONFIG_UPDATE]
    assert_(len(cfg) >= 1)
    # "none" should be sent as None to the gate
    assert_(cfg[0].payload["overlay"] is None)
check("set_overlay('none') sends None overlay to gate", t_interface_set_overlay_none)

def t_interface_set_relationship():
    emit, events = make_emit()
    iface = make_interface(emit=emit)
    iface.set_relationship("friend", immediate=True)
    cfg = [e for e in events if e.event_type == DPEventType.DP_CONFIG_UPDATE]
    assert_(len(cfg) >= 1)
    assert_(cfg[0].payload["relationship"] == "friend")
check("set_relationship emits relationship update", t_interface_set_relationship)

def t_interface_unknown_trait_ignored():
    iface = make_interface()
    # Should not raise
    iface.set_trait("nonexistent_trait", 0.5)
    assert_(not iface.has_pending_changes)
check("set_trait with unknown trait is ignored gracefully", t_interface_unknown_trait_ignored)

def t_interface_reset_to_defaults():
    emit, events = make_emit()
    iface = make_interface(emit=emit)
    iface.set_trait("humor", 0.95)
    iface.apply()
    events.clear()
    iface.reset_to_defaults(immediate=True)
    cfg = [e for e in events if e.event_type == DPEventType.DP_CONFIG_UPDATE]
    assert_(len(cfg) >= 1)
    overrides = cfg[0].payload.get("trait_overrides") or {}
    assert_(all(v == 0.5 for v in overrides.values()), f"Expected all 0.5, got {overrides}")
check("reset_to_defaults resets sliders to 0.5 and emits", t_interface_reset_to_defaults)


# ══════════════════════════════════════════════════════════════════════════════
# PresetManager
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ PresetManager ═══")

def t_preset_builtins_present():
    pm = PresetManager()
    for name in ["DarkPassenger","Professional","Mentor","Minimalist"]:
        assert_(pm.get(name) is not None, f"Missing builtin: {name}")
check("PresetManager has all 4 built-in presets", t_preset_builtins_present)

def t_preset_save_and_get():
    pm = PresetManager()
    p = PersonaPreset(name="MyPreset", profile_id="default",
                      trait_values={"humor": 0.8}, overlay="relaxed")
    pm.save(p)
    fetched = pm.get("MyPreset")
    assert_(fetched is not None)
    assert_(fetched.trait_values["humor"] == 0.8)
check("PresetManager.save + get roundtrip", t_preset_save_and_get)

def t_preset_cannot_overwrite_builtin():
    pm = PresetManager()
    raises(ValueError, lambda: pm.save(PersonaPreset("DarkPassenger","x",{})))
check("PresetManager.save raises ValueError for builtin name", t_preset_cannot_overwrite_builtin)

def t_preset_delete_user():
    pm = PresetManager()
    pm.save(PersonaPreset("Temp","default",{}))
    pm.delete("Temp")
    assert_(pm.get("Temp") is None)
check("PresetManager.delete removes user preset", t_preset_delete_user)

def t_preset_cannot_delete_builtin():
    pm = PresetManager()
    raises(ValueError, lambda: pm.delete("Professional"))
check("PresetManager.delete raises ValueError for builtin", t_preset_cannot_delete_builtin)

def t_preset_export_import_json():
    pm = PresetManager()
    pm.save(PersonaPreset("ExportTest","default",{"humor":0.9},"teaching","desc"))
    exported = pm.export_json()
    pm2 = PresetManager()
    count = pm2.import_json(exported)
    assert_(count == 1)
    p = pm2.get("ExportTest")
    assert_(p is not None and p.trait_values["humor"] == 0.9)
check("PresetManager export/import JSON roundtrip", t_preset_export_import_json)

def t_preset_load_into_interface():
    emit, events = make_emit()
    iface = make_interface(emit=emit)
    loaded = iface.load_preset("DarkPassenger", immediate=True)
    assert_(loaded is True)
    cfg = [e for e in events if e.event_type == DPEventType.DP_CONFIG_UPDATE]
    assert_(len(cfg) >= 1)
    # Trait overrides should include DarkPassenger's traits
    overrides = cfg[0].payload.get("trait_overrides") or {}
    assert_("humor" in overrides or "directness" in overrides)
check("load_preset('DarkPassenger') applies traits and emits update", t_preset_load_into_interface)

def t_preset_load_nonexistent_returns_false():
    iface = make_interface()
    result = iface.load_preset("GhostPreset_That_Does_Not_Exist")
    assert_(result is False)
check("load_preset returns False for unknown preset", t_preset_load_nonexistent_returns_false)

def t_preset_save_current_state():
    iface = make_interface()
    iface.set_trait("humor", 0.77)
    preset = iface.save_preset("Snapshot", description="test")
    assert_(isinstance(preset, PersonaPreset))
    assert_(preset.trait_values["humor"] == 0.77)
    assert_(preset.name == "Snapshot")
check("save_preset captures current control state", t_preset_save_current_state)


# ══════════════════════════════════════════════════════════════════════════════
# Platform Adapters
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ Platform Adapters ═══")

def t_desktop_adapter():
    iface = make_interface()
    d = DesktopConfigAdapter.serialise(iface.control_state, ["DarkPassenger","Professional"])
    assert_(d["platform"] == "desktop")
    assert_(isinstance(d["controls"], list))
    assert_(len(d["controls"]) > 0)
    assert_(d["preset_names"] == ["DarkPassenger","Professional"])
check("DesktopConfigAdapter.serialise produces correct structure", t_desktop_adapter)

def t_web_adapter_json():
    iface = make_interface()
    json_str = WebConfigAdapter.serialise(iface.control_state, ["Minimalist"])
    d = json.loads(json_str)
    assert_(d["platform"] == "web")
    assert_("groups" in d)
    assert_("personality" in d["groups"])
    assert_("context" in d["groups"])
    assert_("advanced" in d["groups"])
    assert_("presets" in d["groups"])
check("WebConfigAdapter.serialise produces valid JSON with groups", t_web_adapter_json)

def t_web_adapter_parse_update():
    update = WebConfigAdapter.parse_update('{"control":"humor","value":0.85}')
    assert_(update["control"] == "humor")
    assert_(update["value"] == 0.85)
check("WebConfigAdapter.parse_update deserialises correctly", t_web_adapter_parse_update)

def t_mobile_adapter():
    iface = make_interface()
    d = MobileConfigAdapter.serialise(iface.control_state)
    assert_(d["platform"] == "mobile")
    assert_("primary" in d and "secondary" in d and "context" in d)
    primary_names = [c["name"] for c in d["primary"]]
    assert_("humor" in primary_names)
    assert_("directness" in primary_names)
check("MobileConfigAdapter.serialise has primary/secondary split", t_mobile_adapter)

def t_discord_adapter():
    iface = make_interface()
    d = DiscordConfigAdapter.serialise(iface.control_state)
    assert_("humor" in d)
    assert_("overlay" in d)
    assert_("profile" in d)
check("DiscordConfigAdapter.serialise produces flat dict", t_discord_adapter)

def t_discord_slash_command_manifest():
    manifest = DiscordConfigAdapter.slash_command_help()
    assert_(manifest["name"] == "persona")
    assert_(isinstance(manifest["options"], list))
    option_names = [o["name"] for o in manifest["options"]]
    assert_("humor" in option_names)
    assert_("overlay" in option_names)
    assert_("preset" in option_names)
check("DiscordConfigAdapter.slash_command_help returns valid manifest", t_discord_slash_command_manifest)


# ══════════════════════════════════════════════════════════════════════════════
# Payload factories
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ Payload factories ═══")

def t_ghostmind_payload_factory():
    p = make_ghostmind_output_payload(
        content="hi", criticality="normal", uncertainty=0.1,
        session_id="s1", relationship="owner", intent="inform",
        overlay=None, protected_fields=[],
    )
    for k in ["content","criticality","uncertainty","session_id","relationship","intent"]:
        assert_(k in p, f"Missing: {k}")
check("make_ghostmind_output_payload has all fields", t_ghostmind_payload_factory)

def t_dp_output_payload_factory():
    p = make_dp_output_ready_payload(
        content="out", session_id="s1", response_id="r1",
        override_active=False, expression_confidence=0.9,
        elapsed_ms=50.0, fast_path_active=False, destination_channel="voicy",
    )
    for k in ["content","session_id","response_id","override_active",
              "expression_confidence","elapsed_ms","destination_channel"]:
        assert_(k in p, f"Missing: {k}")
check("make_dp_output_ready_payload has all fields", t_dp_output_payload_factory)

def t_config_update_payload_factory():
    p = make_persona_config_update_payload(
        profile_id="default", trait_overrides={"humor": 0.8},
        overlay="focused", relationship="owner", requested_by="ui",
    )
    for k in ["profile_id","trait_overrides","overlay","relationship","requested_by"]:
        assert_(k in p, f"Missing: {k}")
    assert_(p["trait_overrides"]["humor"] == 0.8)
check("make_persona_config_update_payload has all fields", t_config_update_payload_factory)


# ══════════════════════════════════════════════════════════════════════════════
# Package-level import check
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ Package imports ═══")

def t_bloodyheart_package():
    import bloodyheart
    for sym in ["BusEvent","EventPriority","DPEventType","DPEventSchemas",
                "DarkPassengerGate","GateSafeModeLevel","GateHealthStatus","RoutingTable"]:
        assert_(hasattr(bloodyheart, sym), f"bloodyheart missing: {sym}")
check("bloodyheart package exports all public symbols", t_bloodyheart_package)

def t_ui_package():
    import ui
    for sym in ["PersonaConfigInterface","UIControlState","SliderControl",
                "ToggleControl","DropdownControl","BlendControl",
                "PersonaPreset","PresetManager","DesktopConfigAdapter",
                "WebConfigAdapter","MobileConfigAdapter","DiscordConfigAdapter"]:
        assert_(hasattr(ui, sym), f"ui missing: {sym}")
check("ui package exports all public symbols", t_ui_package)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*48}")
print(f"v1.5 orchestration suite: {PASS} passed, {FAIL} failed")
if FAIL:
    import sys; sys.exit(1)
