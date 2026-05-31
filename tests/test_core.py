"""
tests/test_core.py — DarkPassenger Core Infrastructure Tests

Covers:
    - PersonaVector math (blend, scale, distance, clamp, add_delta)
    - ExpressionBudget (from_vector, validate, effective_weight)
    - PersonaVectorEngine (build, build_blended, overlay/context/intent effects)
    - PersonaProfile (from_dict, to_dict, roundtrip)
    - ConfigManager (load, validate, active profile, update, save/reload)
    - RuntimeState (defaults, update, snapshot, session lifecycle)
    - RuntimeStateManager (all update methods, reset, begin/end response)
    - TransformationPipeline (full pipeline, override, hard block, stub warnings,
                              circuit breaker integration, extension hooks)
"""

import json
import math
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.persona_vector import (
    PersonaVector,
    ExpressionBudget,
    OverlayType,
    RelationshipContext,
    CommunicationIntent,
    PersonaVectorEngine,
)
from core.config_manager import (
    ConfigManager,
    PersonaProfile,
    StabilityParameters,
    CommunicationHabits,
    ValidationError,
    _validate_profile_dict,
    CURRENT_SCHEMA_VERSION,
)
from core.runtime_state import RuntimeState, RuntimeStateManager
from core.transformation_pipeline import (
    TransformationPipeline,
    TransformationInput,
    TransformationResult,
)
from dp_types.integrity_types import (
    GhostMindOutput,
    CriticalityLevel,
)
from integrity.circuit_breaker import TransformedOutput


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_profile_dict(profile_id="test") -> dict:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "profile_id":     profile_id,
        "display_name":   "Test Profile",
        "description":    "Unit test persona",
        "base_traits": {
            "formality":        0.50,
            "humor":            0.30,
            "warmth":           0.60,
            "confidence":       0.80,
            "directness":       0.75,
            "professionalism":  0.65,
            "technicality":     0.60,
            "precision":        0.80,
            "curiosity":        0.50,
            "analytical_depth": 0.60,
        },
    }


def _pipeline_with_profile(profile_dict=None) -> tuple:
    """Return (pipeline, rsm) with an in-memory ConfigManager."""
    cm = ConfigManager()
    d = profile_dict or _minimal_profile_dict()
    cm.load_from_dict(d)
    rsm = RuntimeStateManager()
    pipeline = TransformationPipeline(config_manager=cm)
    return pipeline, rsm


# ═══════════════════════════════════════════════════════════════════════════════
# PersonaVector
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersonaVector(unittest.TestCase):

    def test_defaults_in_range(self):
        v = PersonaVector()
        for t in PersonaVector.trait_names():
            val = getattr(v, t)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_clamp_above_1(self):
        v = PersonaVector(humor=1.5, warmth=2.0)
        v.clamp()
        self.assertEqual(v.humor, 1.0)
        self.assertEqual(v.warmth, 1.0)

    def test_clamp_below_0(self):
        v = PersonaVector(humor=-0.5, directness=-1.0)
        v.clamp()
        self.assertEqual(v.humor, 0.0)
        self.assertEqual(v.directness, 0.0)

    def test_scale_halves_all_traits(self):
        v = PersonaVector(humor=0.8, directness=0.6)
        scaled = v.scale(0.5)
        self.assertAlmostEqual(scaled.humor,     0.4, places=5)
        self.assertAlmostEqual(scaled.directness, 0.3, places=5)

    def test_scale_zero_flattens(self):
        v = PersonaVector(humor=0.9, warmth=0.7)
        flat = v.scale(0.0)
        for t in PersonaVector.trait_names():
            self.assertEqual(getattr(flat, t), 0.0)

    def test_blend_zero_weight_returns_self(self):
        a = PersonaVector(humor=0.2)
        b = PersonaVector(humor=0.8)
        result = a.blend(b, 0.0)
        self.assertAlmostEqual(result.humor, 0.2, places=5)

    def test_blend_full_weight_returns_other(self):
        a = PersonaVector(humor=0.2)
        b = PersonaVector(humor=0.8)
        result = a.blend(b, 1.0)
        self.assertAlmostEqual(result.humor, 0.8, places=5)

    def test_blend_midpoint(self):
        a = PersonaVector(humor=0.0)
        b = PersonaVector(humor=1.0)
        result = a.blend(b, 0.5)
        self.assertAlmostEqual(result.humor, 0.5, places=5)

    def test_distance_identical_vectors_is_zero(self):
        v = PersonaVector()
        self.assertAlmostEqual(v.distance(v), 0.0, places=10)

    def test_distance_is_symmetric(self):
        a = PersonaVector(humor=0.2, warmth=0.8)
        b = PersonaVector(humor=0.8, warmth=0.2)
        self.assertAlmostEqual(a.distance(b), b.distance(a), places=10)

    def test_add_delta_applies_change(self):
        v = PersonaVector(humor=0.3)
        result = v.add_delta({"humor": +0.2})
        self.assertAlmostEqual(result.humor, 0.5, places=5)

    def test_add_delta_ignores_unknown_traits(self):
        v = PersonaVector(humor=0.3)
        # Should not raise
        result = v.add_delta({"not_a_trait": +0.9})
        self.assertAlmostEqual(result.humor, 0.3, places=5)

    def test_roundtrip_dict(self):
        v = PersonaVector(humor=0.42, directness=0.77)
        d = v.to_dict()
        restored = PersonaVector.from_dict(d)
        self.assertAlmostEqual(restored.humor,     0.42, places=4)
        self.assertAlmostEqual(restored.directness, 0.77, places=4)

    def test_from_dict_ignores_unknown_keys(self):
        d = {"humor": 0.5, "unknown_field": 99}
        v = PersonaVector.from_dict(d)
        self.assertAlmostEqual(v.humor, 0.5, places=5)

    def test_dominant_traits_sorted_descending(self):
        # Set confidence below humor so the ranking is unambiguous
        v = PersonaVector(
            directness=0.9, humor=0.7, confidence=0.4,
            warmth=0.3, precision=0.3,
        )
        top = v.dominant_traits(2)
        self.assertEqual(top[0], "directness")
        self.assertEqual(top[1], "humor")


# ═══════════════════════════════════════════════════════════════════════════════
# ExpressionBudget
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpressionBudget(unittest.TestCase):

    def test_from_dict(self):
        b = ExpressionBudget.from_dict({"directness": 40, "humor": 30, "warmth": 20})
        self.assertEqual(b.allocations["directness"], 40)

    def test_validate_valid(self):
        b = ExpressionBudget.from_dict({"directness": 50, "humor": 30})
        ok, _ = b.validate()
        self.assertTrue(ok)

    def test_validate_over_budget(self):
        b = ExpressionBudget.from_dict({"directness": 70, "humor": 50})
        ok, reason = b.validate()
        self.assertFalse(ok)
        self.assertIn("120", reason)

    def test_validate_negative(self):
        b = ExpressionBudget.from_dict({"directness": -5})
        ok, reason = b.validate()
        self.assertFalse(ok)

    def test_from_vector_sums_to_100(self):
        v = PersonaVector(directness=0.9, humor=0.7, warmth=0.5, precision=0.8, technicality=0.6)
        b = ExpressionBudget.from_vector(v, top_n=5)
        self.assertEqual(sum(b.allocations.values()), 100)

    def test_from_vector_top_n_respected(self):
        v = PersonaVector()
        b = ExpressionBudget.from_vector(v, top_n=3)
        self.assertEqual(len(b.allocations), 3)

    def test_effective_weight_present(self):
        b = ExpressionBudget.from_dict({"directness": 40})
        self.assertAlmostEqual(b.effective_weight("directness"), 0.40, places=5)

    def test_effective_weight_absent_returns_zero(self):
        b = ExpressionBudget.from_dict({"directness": 40})
        self.assertEqual(b.effective_weight("humor"), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PersonaVectorEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersonaVectorEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PersonaVectorEngine()
        self.base   = PersonaVector()

    def test_build_no_overlay_returns_clamped(self):
        result = self.engine.build(self.base)
        for t in PersonaVector.trait_names():
            val = getattr(result, t)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_emergency_overlay_suppresses_humor(self):
        result = self.engine.build(
            self.base,
            overlay=OverlayType.EMERGENCY,
            overlay_weight=1.0,
        )
        # Emergency strongly reduces humor; result should be well below base default
        self.assertLess(result.humor, self.base.humor)

    def test_emergency_overlay_raises_directness(self):
        base = PersonaVector(directness=0.5)
        result = self.engine.build(
            base,
            overlay=OverlayType.EMERGENCY,
            overlay_weight=1.0,
        )
        self.assertGreater(result.directness, base.directness)

    def test_focused_overlay_raises_technicality(self):
        base = PersonaVector(technicality=0.5)
        result = self.engine.build(
            base,
            overlay=OverlayType.FOCUSED,
            overlay_weight=1.0,
        )
        self.assertGreater(result.technicality, base.technicality)

    def test_relaxed_overlay_raises_humor(self):
        base = PersonaVector(humor=0.3)
        result = self.engine.build(
            base,
            overlay=OverlayType.RELAXED,
            overlay_weight=1.0,
        )
        self.assertGreater(result.humor, base.humor)

    def test_expression_confidence_scales_down(self):
        full    = self.engine.build(self.base, expression_confidence=1.0)
        half    = self.engine.build(self.base, expression_confidence=0.5)
        for t in PersonaVector.trait_names():
            self.assertLessEqual(getattr(half, t), getattr(full, t) + 1e-9)

    def test_zero_confidence_flattens_all(self):
        result = self.engine.build(self.base, expression_confidence=0.0)
        for t in PersonaVector.trait_names():
            self.assertAlmostEqual(getattr(result, t), 0.0, places=5)

    def test_blended_overlays_applies_both(self):
        base = PersonaVector(humor=0.3, technicality=0.5)
        result = self.engine.build_blended(
            base,
            overlays={
                OverlayType.TEACHING: 0.70,
                OverlayType.FOCUSED:  0.30,
            },
        )
        # Teaching pushes up analytical_depth; focused pushes up technicality
        # Both should be above base
        self.assertGreater(result.analytical_depth, base.analytical_depth)
        self.assertGreater(result.technicality,     base.technicality)

    def test_relationship_owner_boosts_warmth(self):
        base   = PersonaVector(warmth=0.5)
        result = self.engine.build(base, relationship=RelationshipContext.OWNER)
        self.assertGreater(result.warmth, base.warmth)

    def test_relationship_friend_reduces_formality(self):
        base   = PersonaVector(formality=0.6)
        result = self.engine.build(base, relationship=RelationshipContext.FRIEND)
        self.assertLess(result.formality, base.formality)

    def test_intent_warn_boosts_directness(self):
        base   = PersonaVector(directness=0.5)
        result = self.engine.build(base, intent=CommunicationIntent.WARN)
        self.assertGreater(result.directness, base.directness)

    def test_result_always_clamped(self):
        # Use extreme base values to force clamping
        base = PersonaVector(
            humor=0.95, directness=0.95, warmth=0.95,
            technicality=0.95, formality=0.95,
        )
        result = self.engine.build(
            base,
            overlay=OverlayType.RELAXED,
            overlay_weight=1.0,
        )
        for t in PersonaVector.trait_names():
            self.assertLessEqual(getattr(result, t), 1.0)
            self.assertGreaterEqual(getattr(result, t), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PersonaProfile
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersonaProfile(unittest.TestCase):

    def test_from_dict_minimal(self):
        d = _minimal_profile_dict()
        p = PersonaProfile.from_dict(d)
        self.assertEqual(p.profile_id, "test")
        self.assertAlmostEqual(p.base_traits.humor, 0.30, places=5)

    def test_roundtrip_dict(self):
        d = _minimal_profile_dict()
        p = PersonaProfile.from_dict(d)
        d2 = p.to_dict()
        p2 = PersonaProfile.from_dict(d2)
        self.assertEqual(p.profile_id,      p2.profile_id)
        self.assertEqual(p.schema_version,  p2.schema_version)
        self.assertAlmostEqual(
            p.base_traits.humor, p2.base_traits.humor, places=5
        )

    def test_missing_required_field_raises(self):
        d = {"profile_id": "x", "display_name": "X"}
        # Missing both schema_version and base_traits
        with self.assertRaises(ValueError):
            PersonaProfile.from_dict(d)


# ═══════════════════════════════════════════════════════════════════════════════
# ConfigManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigManager(unittest.TestCase):

    def test_load_valid_profile(self):
        cm = ConfigManager()
        cm.load_from_dict(_minimal_profile_dict("a"))
        self.assertIn("a", cm.list_profiles())

    def test_validate_invalid_trait_rejected(self):
        d = _minimal_profile_dict()
        d["base_traits"]["humor"] = 1.5   # out of range
        with self.assertRaises(ValidationError):
            ConfigManager().load_from_dict(d)

    def test_validate_budget_over_100_rejected(self):
        d = _minimal_profile_dict()
        d["expression_budget"] = {"directness": 70, "humor": 50}  # 120 total
        with self.assertRaises(ValidationError):
            ConfigManager().load_from_dict(d)

    def test_validate_unknown_trait_rejected(self):
        d = _minimal_profile_dict()
        d["base_traits"]["not_a_trait"] = 0.5
        with self.assertRaises(ValidationError):
            ConfigManager().load_from_dict(d)

    def test_list_profiles_empty(self):
        cm = ConfigManager()
        self.assertEqual(cm.list_profiles(), [])

    def test_set_active_profile(self):
        cm = ConfigManager()
        cm.load_from_dict(_minimal_profile_dict("x"))
        cm.load_from_dict(_minimal_profile_dict("y"))
        cm.set_active("y")
        self.assertEqual(cm.active_profile.profile_id, "y")

    def test_first_loaded_is_active(self):
        cm = ConfigManager()
        cm.load_from_dict(_minimal_profile_dict("first"))
        cm.load_from_dict(_minimal_profile_dict("second"))
        self.assertEqual(cm.active_profile.profile_id, "first")

    def test_active_profile_none_when_empty(self):
        cm = ConfigManager()
        self.assertIsNone(cm.active_profile)

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_persona.json")
            cm = ConfigManager()
            cm.load_from_dict(_minimal_profile_dict("saved"))
            cm.save_file("saved", path)

            cm2 = ConfigManager()
            profile = cm2.load_file(path)
            self.assertEqual(profile.profile_id, "saved")

    def test_load_directory(self):
        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "personas",
        )
        if os.path.isdir(configs_dir):
            cm = ConfigManager(config_dir=configs_dir)
            self.assertGreater(len(cm.list_profiles()), 0)
            self.assertIn("default", cm.list_profiles())

    def test_update_trait(self):
        cm = ConfigManager()
        cm.load_from_dict(_minimal_profile_dict("u"))
        cm.update_trait("u", "humor", 0.99)
        self.assertAlmostEqual(cm.get_profile("u").base_traits.humor, 0.99)

    def test_update_trait_out_of_range_raises(self):
        cm = ConfigManager()
        cm.load_from_dict(_minimal_profile_dict("u"))
        with self.assertRaises(ValueError):
            cm.update_trait("u", "humor", 1.5)

    def test_update_trait_unknown_raises(self):
        cm = ConfigManager()
        cm.load_from_dict(_minimal_profile_dict("u"))
        with self.assertRaises(ValueError):
            cm.update_trait("u", "not_a_trait", 0.5)

    def test_get_nonexistent_profile_raises(self):
        cm = ConfigManager()
        with self.assertRaises(KeyError):
            cm.get_profile("missing")

    def test_validate_dict_valid(self):
        ok, errors = ConfigManager().validate_dict(_minimal_profile_dict())
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_validate_dict_invalid(self):
        d = _minimal_profile_dict()
        d.pop("base_traits")
        ok, errors = ConfigManager().validate_dict(d)
        self.assertFalse(ok)
        self.assertTrue(len(errors) > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# RuntimeState / RuntimeStateManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuntimeStateManager(unittest.TestCase):

    def setUp(self):
        self.rsm = RuntimeStateManager()

    def test_defaults(self):
        s = self.rsm.state
        self.assertIsNone(s.current_topic)
        self.assertEqual(s.current_intent, CommunicationIntent.INFORM)
        self.assertIsNone(s.current_overlay)
        self.assertEqual(s.current_complexity, "medium")
        self.assertAlmostEqual(s.current_expression_confidence, 1.0)
        self.assertEqual(s.active_relationship, RelationshipContext.UNKNOWN)
        self.assertEqual(s.response_index, 0)

    def test_set_topic(self):
        self.rsm.set_topic("python decorators")
        self.assertEqual(self.rsm.state.current_topic, "python decorators")

    def test_set_intent(self):
        self.rsm.set_intent(CommunicationIntent.WARN)
        self.assertEqual(self.rsm.state.current_intent, CommunicationIntent.WARN)

    def test_set_overlay_clears_blends(self):
        self.rsm.set_blended_overlays({OverlayType.TEACHING: 0.7})
        self.rsm.set_overlay(OverlayType.FOCUSED)
        self.assertEqual(self.rsm.state.current_overlay, OverlayType.FOCUSED)
        self.assertEqual(self.rsm.state.current_overlay_blends, {})

    def test_set_blended_overlays_clears_single(self):
        self.rsm.set_overlay(OverlayType.RELAXED)
        self.rsm.set_blended_overlays({OverlayType.TEACHING: 0.7, OverlayType.FOCUSED: 0.3})
        self.assertIsNone(self.rsm.state.current_overlay)
        self.assertIn(OverlayType.TEACHING, self.rsm.state.current_overlay_blends)

    def test_set_complexity_valid(self):
        self.rsm.set_complexity("high")
        self.assertEqual(self.rsm.state.current_complexity, "high")

    def test_set_complexity_invalid_coerced(self):
        self.rsm.set_complexity("extreme")
        self.assertEqual(self.rsm.state.current_complexity, "medium")

    def test_set_expression_confidence_clamped(self):
        self.rsm.set_expression_confidence(1.5)
        self.assertAlmostEqual(self.rsm.state.current_expression_confidence, 1.0)
        self.rsm.set_expression_confidence(-0.5)
        self.assertAlmostEqual(self.rsm.state.current_expression_confidence, 0.0)

    def test_begin_response_increments_counter(self):
        self.rsm.begin_response()
        self.assertEqual(self.rsm.state.response_index, 1)
        self.rsm.begin_response()
        self.assertEqual(self.rsm.state.response_index, 2)

    def test_reset_session_clears_state(self):
        self.rsm.set_topic("test")
        self.rsm.set_overlay(OverlayType.EMERGENCY)
        self.rsm.begin_response()
        self.rsm.reset_session()
        s = self.rsm.state
        self.assertIsNone(s.current_topic)
        self.assertIsNone(s.current_overlay)
        self.assertEqual(s.response_index, 0)

    def test_reset_preserves_relationship(self):
        self.rsm.set_relationship(RelationshipContext.OWNER)
        self.rsm.reset_session()
        self.assertEqual(self.rsm.state.active_relationship, RelationshipContext.OWNER)

    def test_snapshot_is_independent_copy(self):
        snap = self.rsm.snapshot()
        self.rsm.set_topic("new topic")
        self.assertIsNone(snap.current_topic)

    def test_effective_overlay_single(self):
        self.rsm.set_overlay(OverlayType.RELAXED)
        self.assertEqual(self.rsm.effective_overlay(), OverlayType.RELAXED)

    def test_effective_overlay_none_when_blended(self):
        self.rsm.set_blended_overlays({OverlayType.TEACHING: 0.7})
        self.assertIsNone(self.rsm.effective_overlay())

    def test_summary_dict_keys(self):
        summary = self.rsm.summary()
        expected_keys = {
            "topic", "intent", "overlay", "blends",
            "complexity", "expression_confidence", "relationship", "response_index",
        }
        self.assertEqual(set(summary.keys()), expected_keys)


# ═══════════════════════════════════════════════════════════════════════════════
# TransformationPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformationPipeline(unittest.TestCase):

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normal_output(self, content="Request complete.") -> GhostMindOutput:
        return GhostMindOutput.normal(content).finalize()

    def _run(self, ghost_output, pipeline=None, rsm=None):
        if pipeline is None or rsm is None:
            pipeline, rsm = _pipeline_with_profile()
        ti = TransformationInput(
            ghost_output=ghost_output,
            runtime_state_manager=rsm,
        )
        return pipeline.transform(ti)

    # ── Normal passthrough ────────────────────────────────────────────────────

    def test_normal_passthrough_returns_content(self):
        out = self._normal_output("Hello world.")
        result = self._run(out)
        # With stub fingerprint, output is unchanged GhostMind content
        self.assertIn("Hello world.", result.final_output)

    def test_result_contains_persona_vector(self):
        out = self._normal_output()
        result = self._run(out)
        self.assertIsInstance(result.persona_vector, PersonaVector)

    def test_result_contains_expression_budget(self):
        out = self._normal_output()
        result = self._run(out)
        self.assertIsInstance(result.expression_budget, ExpressionBudget)

    def test_result_stages_recorded(self):
        out = self._normal_output()
        result = self._run(out)
        self.assertIn("input_reception",  result.stages_executed)
        self.assertIn("pre_flight",       result.stages_executed)
        self.assertIn("validation_pipeline", result.stages_executed)
        self.assertIn("final_response",   result.stages_executed)

    def test_stub_warnings_present(self):
        out = self._normal_output()
        result = self._run(out)
        # Speech fingerprint stub should produce a warning
        fingerprint_warning = any(
            "speech_fingerprint" in w for w in result.pipeline_warnings
        )
        self.assertTrue(fingerprint_warning)

    # ── Critical Response Override ────────────────────────────────────────────

    def test_emergency_override_fires(self):
        out = GhostMindOutput.emergency("CRITICAL: System halting.").finalize()
        result = self._run(out)
        self.assertTrue(result.override_active)
        self.assertIn("critical_override", result.stages_executed)

    def test_security_override_fires(self):
        out = (
            GhostMindOutput.security_alert("Intrusion detected.")
            .add_security_flag("access_blocked")
            .finalize()
        )
        result = self._run(out)
        self.assertTrue(result.override_active)

    def test_override_output_contains_raw_content(self):
        out = GhostMindOutput.emergency("HALT: memory fault.").finalize()
        result = self._run(out)
        self.assertIn("HALT", result.final_output)

    # ── Pre-flight hard block ─────────────────────────────────────────────────

    def test_unfinalized_with_protected_fields_blocked(self):
        # IC-03: protected fields exist but finalize() not called
        out = GhostMindOutput(content="Latency is 42ms.")
        out.add_number("latency", 42)
        # Deliberately NOT calling finalize()
        result = self._run(out)
        # Should not crash; should return raw content with block warning
        self.assertIsNotNone(result.final_output)
        block_warning = any("block" in w.lower() for w in result.pipeline_warnings)
        self.assertTrue(block_warning)

    # ── Expression confidence applied ─────────────────────────────────────────

    def test_technical_output_caps_expression_confidence(self):
        out = GhostMindOutput.technical("Deploy complete.").finalize()
        result = self._run(out)
        # TECHNICAL criticality caps expression at 0.50
        self.assertLessEqual(result.expression_confidence, 0.50 + 1e-9)

    def test_normal_output_has_higher_confidence_than_technical(self):
        normal_out  = self._normal_output("Hi.")
        tech_out    = GhostMindOutput.technical("Deploy complete.").finalize()
        p, rsm = _pipeline_with_profile()
        r_normal = self._run(normal_out, p, rsm)
        r_tech   = self._run(tech_out,   p, rsm)
        self.assertGreater(r_normal.expression_confidence, r_tech.expression_confidence)

    # ── Circuit breaker integration ───────────────────────────────────────────

    def test_circuit_breaker_catches_dropped_number(self):
        """
        Install a fingerprint hook that drops a protected number, then
        verify the CircuitBreaker catches it.
        """
        out = (
            GhostMindOutput.normal("Query returned 42 results.")
            .add_number("result_count", 42)
            .finalize()
        )

        pipeline, rsm = _pipeline_with_profile()
        # Hook that strips the number from the output
        pipeline.fingerprint_hook = lambda c, v, b, s: "Query returned some results."

        ti = TransformationInput(ghost_output=out, runtime_state_manager=rsm)
        result = pipeline.transform(ti)

        # The CircuitBreaker must reject the transformed text and fall back to raw
        self.assertIn("42", result.final_output)

    # ── Extension hooks ───────────────────────────────────────────────────────

    def test_stability_hook_called_with_vectors(self):
        called_with = {}

        def fake_stability(new_vec, prev_vec, state):
            called_with["new"]  = new_vec
            called_with["prev"] = prev_vec
            return new_vec

        pipeline, rsm = _pipeline_with_profile()
        pipeline.stability_hook = fake_stability
        self._run(self._normal_output(), pipeline, rsm)

        self.assertIn("new",  called_with)
        self.assertIn("prev", called_with)

    def test_conflict_hook_called(self):
        called = []

        def fake_conflict(vec, budget, state):
            called.append(True)
            return vec

        pipeline, rsm = _pipeline_with_profile()
        pipeline.conflict_hook = fake_conflict
        self._run(self._normal_output(), pipeline, rsm)
        self.assertTrue(len(called) > 0)

    def test_fingerprint_hook_called(self):
        called = []

        def fake_fp(content, vec, budget, state):
            called.append(content)
            return content  # return unchanged so CircuitBreaker passes

        pipeline, rsm = _pipeline_with_profile()
        pipeline.fingerprint_hook = fake_fp
        self._run(self._normal_output("Test content."), pipeline, rsm)
        self.assertTrue(len(called) > 0)

    # ── From-config-dir factory ───────────────────────────────────────────────

    def test_from_config_dir_loads_profiles(self):
        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "personas",
        )
        if os.path.isdir(configs_dir):
            pipeline = TransformationPipeline.from_config_dir(
                configs_dir, profile_id="default"
            )
            self.assertIsNotNone(pipeline._resolve_profile())

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def test_response_index_increments_per_call(self):
        pipeline, rsm = _pipeline_with_profile()
        out = self._normal_output()

        for expected_index in range(1, 4):
            ti = TransformationInput(ghost_output=out, runtime_state_manager=rsm)
            pipeline.transform(ti)
            self.assertEqual(rsm.state.response_index, expected_index)

    def test_relationship_override_applied(self):
        """Relationship override should produce a different vector than default."""
        out = self._normal_output()
        pipeline, rsm = _pipeline_with_profile()

        ti_unknown = TransformationInput(
            ghost_output=out,
            runtime_state_manager=rsm,
            relationship_override=RelationshipContext.UNKNOWN,
        )
        ti_friend = TransformationInput(
            ghost_output=out,
            runtime_state_manager=rsm,
            relationship_override=RelationshipContext.FRIEND,
        )

        r_unknown = pipeline.transform(ti_unknown)
        r_friend  = pipeline.transform(ti_friend)

        # FRIEND boosts humor/warmth; vectors should differ
        self.assertNotAlmostEqual(
            r_unknown.persona_vector.humor,
            r_friend.persona_vector.humor,
            places=3,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: load bundled persona profiles
# ═══════════════════════════════════════════════════════════════════════════════

class TestBundledProfiles(unittest.TestCase):
    """Smoke tests against the four bundled .json persona files."""

    @classmethod
    def setUpClass(cls):
        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "personas",
        )
        cls.cm = ConfigManager(config_dir=configs_dir)

    def test_default_loaded(self):
        self.assertIn("default", self.cm.list_profiles())

    def test_professional_loaded(self):
        self.assertIn("professional", self.cm.list_profiles())

    def test_darkpassenger_loaded(self):
        self.assertIn("darkpassenger", self.cm.list_profiles())

    def test_minimalist_loaded(self):
        self.assertIn("minimalist", self.cm.list_profiles())

    def test_darkpassenger_directness_high(self):
        p = self.cm.get_profile("darkpassenger")
        self.assertGreater(p.base_traits.directness, 0.85)

    def test_professional_formality_high(self):
        p = self.cm.get_profile("professional")
        self.assertGreater(p.base_traits.formality, 0.85)

    def test_minimalist_humor_low(self):
        p = self.cm.get_profile("minimalist")
        self.assertLess(p.base_traits.humor, 0.10)

    def test_all_budgets_valid(self):
        for pid in self.cm.list_profiles():
            p = self.cm.get_profile(pid)
            ok, reason = p.expression_budget.validate()
            self.assertTrue(ok, f"{pid}: {reason}")

    def test_all_stability_params_valid(self):
        for pid in self.cm.list_profiles():
            p = self.cm.get_profile(pid)
            ok, reason = p.stability_parameters.validate()
            self.assertTrue(ok, f"{pid}: {reason}")

    def test_all_communication_habits_valid(self):
        for pid in self.cm.list_profiles():
            p = self.cm.get_profile(pid)
            ok, reason = p.communication_habits.validate()
            self.assertTrue(ok, f"{pid}: {reason}")

    def test_full_pipeline_all_profiles(self):
        """Each bundled profile should survive a full pipeline call."""
        rsm = RuntimeStateManager()
        out = GhostMindOutput.normal("Test content.").finalize()

        for pid in self.cm.list_profiles():
            pipeline = TransformationPipeline(
                config_manager=self.cm,
                profile_id=pid,
            )
            ti = TransformationInput(ghost_output=out, runtime_state_manager=rsm)
            result = pipeline.transform(ti)
            self.assertIsNotNone(result.final_output, f"{pid}: final_output is None")
            self.assertIn("final_response", result.stages_executed, f"{pid}: missing stage")


if __name__ == "__main__":
    unittest.main(verbosity=2)
