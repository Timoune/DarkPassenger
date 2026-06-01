"""
tests/test_behavioral_logic.py — DarkPassenger Behavioral Logic Tests

Covers all four modules implemented in Part 5-8:

    Part 5 — CommunicationStabilityLayer
        - Below-threshold pass-through
        - Above-threshold smoothing applied when no context shift
        - Context shift (overlay, intent, relationship, topic) allows transition
        - Emergency overlay always passes through
        - Warm-up period skips smoothing
        - First response (no prev vector) passes through
        - Smoothing formula matches spec example
        - Session reset clears state

    Part 5 (entropy-linked) — ExpressionConfidenceSystem
        - Criticality caps match spec §4 defaults
        - Override levels produce 0.0 confidence
        - Zero uncertainty → full cap
        - Max uncertainty → near-zero result
        - 0.50 uncertainty → ~half the cap
        - Scalar convenience method
        - Requires-override helper

    Part 6 — TraitConflictResolver
        - No conflict when loser < winner
        - Conflict pair attenuates loser toward winner
        - Floor prevents over-suppression
        - Emergency suppresses humor/warmth/curiosity
        - Zero-budget traits are zeroed out
        - Multiple conflict pairs all resolved
        - Output always clamped to [0, 1]
        - Callable interface matches hook signature

    Part 7-8 — SpeechFingerprintEngine
        - Short content is unchanged
        - Long sentences are split at clause boundaries
        - Transitions injected in balanced/expansive pacing
        - No transitions in compact pacing
        - Length gate truncates for "short" preference
        - Dominant trait selects correct transition pool
        - Callable interface matches hook signature
        - FingerprintResult diagnostics are accurate

    Integration — TransformationPipeline with all four wired in
        - Stub warnings are gone (no more "stub active" in warnings)
        - Stability stage name appears in stages_executed
        - Conflict resolution stage appears in stages_executed
        - Fingerprint stage appears in stages_executed
        - Emergency override still bypasses personality
        - High uncertainty attenuates expression_confidence
        - Full pipeline returns TransformationResult.passed = True
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.persona_vector import (
    PersonaVector,
    ExpressionBudget,
    OverlayType,
    RelationshipContext,
    CommunicationIntent,
)
from core.runtime_state import RuntimeState, RuntimeStateManager
from core.config_manager import (
    ConfigManager,
    CommunicationHabits,
    StabilityParameters,
    CURRENT_SCHEMA_VERSION,
)
from core.stability_layer import CommunicationStabilityLayer, WARMUP_RESPONSES
from core.expression_confidence import ExpressionConfidenceSystem
from core.trait_conflict_resolver import TraitConflictResolver
from core.speech_fingerprint import SpeechFingerprintEngine, SINGLE_SENTENCE_THRESHOLD
from core.transformation_pipeline import TransformationPipeline, TransformationInput
from dp_types.integrity_types import GhostMindOutput, CriticalityLevel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state(
    overlay=None,
    intent=CommunicationIntent.INFORM,
    relationship=RelationshipContext.UNKNOWN,
    topic=None,
    complexity="medium",
    response_index=5,   # default: beyond warm-up
    blends=None,
) -> RuntimeState:
    s = RuntimeState(
        current_overlay=overlay,
        current_intent=intent,
        active_relationship=relationship,
        current_topic=topic,
        current_complexity=complexity,
        response_index=response_index,
        current_overlay_blends=blends or {},
    )
    return s


def _vec(**kwargs) -> PersonaVector:
    v = PersonaVector()
    for k, val in kwargs.items():
        setattr(v, k, val)
    return v.clamp()


def _pipeline_with_profile(profile_dict=None) -> tuple:
    cm = ConfigManager()
    d = profile_dict or {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "profile_id":     "test",
        "display_name":   "Test",
        "description":    "",
        "base_traits": {
            "formality": 0.50, "humor": 0.30, "warmth": 0.60,
            "confidence": 0.80, "directness": 0.75, "professionalism": 0.65,
            "technicality": 0.60, "precision": 0.80,
            "curiosity": 0.50, "analytical_depth": 0.60,
        },
    }
    cm.load_from_dict(d)
    rsm = RuntimeStateManager()
    pipeline = TransformationPipeline(config_manager=cm)
    return pipeline, rsm


def _finalized(content="Hello.", criticality=CriticalityLevel.NORMAL,
               uncertainty=0.0, **kwargs) -> GhostMindOutput:
    g = GhostMindOutput(
        content=content,
        criticality=criticality,
        uncertainty_score=uncertainty,
        **kwargs,
    )
    return g.finalize()


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5 — CommunicationStabilityLayer
# ═══════════════════════════════════════════════════════════════════════════════

class TestStabilityLayer(unittest.TestCase):

    def setUp(self):
        self.layer = CommunicationStabilityLayer(
            smoothing_factor=0.50,
            drift_threshold=0.20,
        )

    # ── Constructor validation ────────────────────────────────────────────────

    def test_invalid_smoothing_factor_raises(self):
        with self.assertRaises(ValueError):
            CommunicationStabilityLayer(smoothing_factor=1.5)

    def test_invalid_drift_threshold_raises(self):
        with self.assertRaises(ValueError):
            CommunicationStabilityLayer(drift_threshold=-0.1)

    def test_zero_drift_threshold_raises(self):
        with self.assertRaises(ValueError):
            CommunicationStabilityLayer(drift_threshold=0.0)

    # ── Below threshold ───────────────────────────────────────────────────────

    def test_below_threshold_no_smoothing(self):
        curr = _vec(humor=0.50)
        prev = _vec(humor=0.52)  # tiny delta
        state = _state(response_index=5)
        result = self.layer.check(curr, prev, state)
        self.assertFalse(result.smoothing_applied)
        self.assertEqual(result.output_vector.humor, curr.humor)

    # ── Above threshold, no context shift → smooth ────────────────────────────

    def test_above_threshold_without_context_shift_smooths(self):
        curr  = _vec(humor=0.20, warmth=0.90, directness=0.40)
        prev  = _vec(humor=0.80, warmth=0.60, directness=0.70)
        state = _state(overlay=None, intent=CommunicationIntent.INFORM, response_index=5)
        prev_state = _state(overlay=None, intent=CommunicationIntent.INFORM, response_index=4)
        result = self.layer.check(curr, prev, state, prev_state)
        self.assertTrue(result.smoothing_applied)
        # Smoothed humor should be between prev (0.80) and curr (0.20)
        self.assertGreater(result.output_vector.humor, curr.humor)
        self.assertLess(result.output_vector.humor, prev.humor)

    def test_spec_example_smoothing_values(self):
        """Verify the spec §5 example at smoothing_factor=0.70."""
        layer = CommunicationStabilityLayer(smoothing_factor=0.70, drift_threshold=0.10)
        prev  = _vec(humor=0.80, warmth=0.60, directness=0.70)
        curr  = _vec(humor=0.20, warmth=0.90, directness=0.40)
        state = _state(response_index=5)
        result = layer.check(curr, prev, state, _state(response_index=4))
        # At smoothing_factor=0.70: blend_weight=0.30
        # smoothed = prev*(1-0.30) + curr*0.30 = prev*0.70 + curr*0.30
        expected_humor     = 0.80 * 0.70 + 0.20 * 0.30   # 0.62
        expected_warmth    = 0.60 * 0.70 + 0.90 * 0.30   # 0.69
        expected_directness = 0.70 * 0.70 + 0.40 * 0.30  # 0.61
        self.assertAlmostEqual(result.output_vector.humor,     expected_humor,     places=5)
        self.assertAlmostEqual(result.output_vector.warmth,    expected_warmth,    places=5)
        self.assertAlmostEqual(result.output_vector.directness, expected_directness, places=5)

    # ── Context shifts allow transition ───────────────────────────────────────

    def test_overlay_change_allows_transition(self):
        curr  = _vec(humor=0.10, directness=0.95)
        prev  = _vec(humor=0.70, directness=0.50)
        state      = _state(overlay=OverlayType.FOCUSED,  response_index=5)
        prev_state = _state(overlay=OverlayType.RELAXED,  response_index=4)
        result = self.layer.check(curr, prev, state, prev_state)
        self.assertFalse(result.smoothing_applied)
        self.assertTrue(result.context_shift_detected)

    def test_intent_change_allows_transition(self):
        curr  = _vec(directness=0.90, humor=0.10)
        prev  = _vec(directness=0.40, humor=0.60)
        state      = _state(intent=CommunicationIntent.WARN,     response_index=5)
        prev_state = _state(intent=CommunicationIntent.BRAINSTORM, response_index=4)
        result = self.layer.check(curr, prev, state, prev_state)
        self.assertFalse(result.smoothing_applied)
        self.assertTrue(result.context_shift_detected)

    def test_relationship_change_allows_transition(self):
        curr  = _vec(warmth=0.10, formality=0.90)
        prev  = _vec(warmth=0.70, formality=0.30)
        state      = _state(relationship=RelationshipContext.ADMINISTRATOR, response_index=5)
        prev_state = _state(relationship=RelationshipContext.FRIEND,        response_index=4)
        result = self.layer.check(curr, prev, state, prev_state)
        self.assertFalse(result.smoothing_applied)
        self.assertTrue(result.context_shift_detected)

    def test_topic_change_allows_transition(self):
        curr  = _vec(technicality=0.90, humor=0.05)
        prev  = _vec(technicality=0.10, humor=0.80)
        state      = _state(topic="networking",  response_index=5)
        prev_state = _state(topic="jokes",       response_index=4)
        result = self.layer.check(curr, prev, state, prev_state)
        self.assertFalse(result.smoothing_applied)
        self.assertTrue(result.context_shift_detected)

    # ── Emergency always passes ───────────────────────────────────────────────

    def test_emergency_overlay_always_passes(self):
        curr  = _vec(directness=0.99, humor=0.01)
        prev  = _vec(directness=0.50, humor=0.50)
        state = _state(overlay=OverlayType.EMERGENCY, response_index=5)
        result = self.layer.check(curr, prev, state)
        self.assertFalse(result.smoothing_applied)
        self.assertTrue(result.override_pass_through)

    def test_exit_emergency_also_passes(self):
        """Recovering from EMERGENCY to normal must not be blocked."""
        curr  = _vec(humor=0.50, directness=0.60)
        prev  = _vec(humor=0.01, directness=0.99)
        state      = _state(overlay=None,                  response_index=6)
        prev_state = _state(overlay=OverlayType.EMERGENCY, response_index=5)
        result = self.layer.check(curr, prev, state, prev_state)
        self.assertFalse(result.smoothing_applied)
        self.assertTrue(result.override_pass_through)

    # ── Warm-up period ────────────────────────────────────────────────────────

    def test_warmup_period_skips_smoothing(self):
        curr  = _vec(humor=0.10)
        prev  = _vec(humor=0.90)
        state = _state(response_index=WARMUP_RESPONSES)   # at the boundary
        result = self.layer.check(curr, prev, state)
        self.assertFalse(result.smoothing_applied)
        self.assertTrue(result.override_pass_through)

    def test_post_warmup_smoothing_active(self):
        curr  = _vec(humor=0.10)
        prev  = _vec(humor=0.90)
        state = _state(response_index=WARMUP_RESPONSES + 1)
        result = self.layer.check(curr, prev, state, _state(response_index=WARMUP_RESPONSES))
        self.assertTrue(result.smoothing_applied)

    # ── No previous vector ────────────────────────────────────────────────────

    def test_no_prev_vector_passes_through(self):
        curr  = _vec(humor=0.10)
        state = _state(response_index=10)
        result = self.layer.check(curr, None, state)
        self.assertFalse(result.smoothing_applied)
        self.assertTrue(result.override_pass_through)
        self.assertEqual(result.distance, 0.0)

    # ── Callable interface ────────────────────────────────────────────────────

    def test_callable_interface_returns_vector(self):
        curr  = _vec(humor=0.50)
        prev  = _vec(humor=0.52)
        state = _state(response_index=5)
        result = self.layer(curr, prev, state)
        self.assertIsInstance(result, PersonaVector)

    # ── Session reset ─────────────────────────────────────────────────────────

    def test_session_reset_clears_prev_state(self):
        layer = CommunicationStabilityLayer(smoothing_factor=0.5, drift_threshold=0.1)
        state = _state(response_index=5)
        v1 = _vec(humor=0.90)
        v2 = _vec(humor=0.10)
        # Feed one response so prev is set
        layer(v1, v1, state)
        layer.reset_session()
        # After reset, prev_state is cleared → next call should not smooth
        result = layer.check(v2, None, _state(response_index=5))
        self.assertFalse(result.smoothing_applied)


# ═══════════════════════════════════════════════════════════════════════════════
# Expression Confidence System (Entropy-Linked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpressionConfidenceSystem(unittest.TestCase):

    def setUp(self):
        self.ecs = ExpressionConfidenceSystem()

    def _ghost(self, criticality=CriticalityLevel.NORMAL, uncertainty=0.0):
        return _finalized("content", criticality=criticality, uncertainty=uncertainty)

    # ── Criticality caps ──────────────────────────────────────────────────────

    def test_normal_cap_is_0_90(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.NORMAL, 0.0))
        self.assertAlmostEqual(result.criticality_cap, 0.90)

    def test_technical_cap_is_0_50(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.TECHNICAL, 0.0))
        self.assertAlmostEqual(result.criticality_cap, 0.50)

    def test_high_risk_cap_is_0_50(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.HIGH_RISK, 0.0))
        self.assertAlmostEqual(result.criticality_cap, 0.50)

    def test_emergency_cap_is_0_05(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.EMERGENCY, 0.0))
        self.assertAlmostEqual(result.criticality_cap, 0.05)

    def test_security_cap_is_zero(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.SECURITY, 0.0))
        self.assertEqual(result.final_confidence, 0.0)

    def test_system_fail_cap_is_zero(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.SYSTEM_FAIL, 0.0))
        self.assertEqual(result.final_confidence, 0.0)

    # ── Uncertainty attenuation ───────────────────────────────────────────────

    def test_zero_uncertainty_gives_full_cap(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.NORMAL, 0.0))
        # multiplier should be near 1.0 → final ≈ cap
        self.assertGreater(result.final_confidence, 0.85)
        self.assertFalse(result.capped_by_uncertainty)

    def test_max_uncertainty_gives_near_zero(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.NORMAL, 1.0))
        self.assertLess(result.final_confidence, 0.10)
        self.assertTrue(result.capped_by_uncertainty)

    def test_half_uncertainty_gives_roughly_half_cap(self):
        result = self.ecs.compute(self._ghost(CriticalityLevel.NORMAL, 0.50))
        # uncertainty=0.5 → multiplier≈0.50 → final ≈ 0.90*0.50 = 0.45
        self.assertGreater(result.final_confidence, 0.30)
        self.assertLess(result.final_confidence, 0.60)
        self.assertTrue(result.capped_by_uncertainty)

    def test_confidence_never_exceeds_cap(self):
        for crit in CriticalityLevel:
            for unc in [0.0, 0.3, 0.7, 1.0]:
                result = self.ecs.compute(self._ghost(crit, unc))
                cap = ExpressionConfidenceSystem.criticality_cap(crit)
                self.assertLessEqual(
                    result.final_confidence, cap + 1e-9,
                    f"Confidence {result.final_confidence} exceeds cap {cap} "
                    f"for {crit.value} uncertainty={unc}"
                )

    def test_uncertainty_strictly_decreasing(self):
        """Higher uncertainty → lower confidence (monotone)."""
        prev = float("inf")
        for unc in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            result = self.ecs.compute(self._ghost(CriticalityLevel.NORMAL, unc))
            self.assertLessEqual(result.final_confidence, prev + 1e-9)
            prev = result.final_confidence

    # ── Scalar convenience ────────────────────────────────────────────────────

    def test_scalar_matches_compute(self):
        g = self._ghost(CriticalityLevel.TECHNICAL, 0.3)
        self.assertAlmostEqual(
            self.ecs.scalar(g),
            self.ecs.compute(g).final_confidence,
        )

    # ── Requires override ─────────────────────────────────────────────────────

    def test_requires_override_emergency(self):
        g = GhostMindOutput.emergency("alert").finalize()
        self.assertTrue(ExpressionConfidenceSystem.requires_override(g))

    def test_requires_override_security_flag(self):
        g = GhostMindOutput.normal("content").add_security_flag("breach").finalize()
        self.assertTrue(ExpressionConfidenceSystem.requires_override(g))

    def test_requires_override_normal_false(self):
        g = GhostMindOutput.normal("hello").finalize()
        self.assertFalse(ExpressionConfidenceSystem.requires_override(g))


# ═══════════════════════════════════════════════════════════════════════════════
# Part 6 — TraitConflictResolver
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraitConflictResolver(unittest.TestCase):

    def setUp(self):
        self.resolver = TraitConflictResolver()
        self.budget   = ExpressionBudget()  # empty budget → no zero-budget enforcement
        self.state    = _state()

    # ── No conflict ───────────────────────────────────────────────────────────

    def test_no_conflict_when_loser_below_winner(self):
        v = _vec(formality=0.80, humor=0.30)   # formality > humor: no conflict
        result = self.resolver.resolve(v, self.budget, self.state)
        self.assertEqual(result.conflicts_found, 0)
        self.assertAlmostEqual(result.output_vector.humor, 0.30)

    def test_equal_traits_no_conflict(self):
        v = _vec(formality=0.50, humor=0.50)
        result = self.resolver.resolve(v, self.budget, self.state)
        self.assertEqual(result.conflicts_found, 0)

    # ── Conflict attenuation ──────────────────────────────────────────────────

    def test_humor_attenuated_when_above_formality(self):
        """humor > formality → humor (loser) should be pulled toward formality."""
        v = _vec(formality=0.30, humor=0.90)
        result = self.resolver.resolve(v, self.budget, self.state)
        self.assertGreater(result.conflicts_found, 0)
        self.assertLess(result.output_vector.humor, 0.90)
        self.assertGreater(result.output_vector.humor, 0.30)   # not zeroed

    def test_loser_never_goes_below_floor(self):
        """The floor parameter prevents over-suppression."""
        # formality/humor pair has floor=0.05
        v = _vec(formality=0.01, humor=0.99)   # humor far above formality
        result = self.resolver.resolve(v, self.budget, self.state)
        self.assertGreaterEqual(result.output_vector.humor, 0.05)

    def test_winner_not_raised_by_conflict(self):
        """Conflict resolution never boosts the winning trait."""
        v = _vec(formality=0.40, humor=0.80)
        result = self.resolver.resolve(v, self.budget, self.state)
        self.assertAlmostEqual(result.output_vector.formality, v.formality, places=5)

    # ── Emergency suppression ─────────────────────────────────────────────────

    def test_emergency_suppresses_humor(self):
        v     = _vec(humor=0.80, warmth=0.70, curiosity=0.60)
        state = _state(overlay=OverlayType.EMERGENCY)
        result = self.resolver.resolve(v, self.budget, state)
        self.assertTrue(result.emergency_active)
        self.assertEqual(result.output_vector.humor,    0.0)
        self.assertEqual(result.output_vector.warmth,   0.0)
        self.assertEqual(result.output_vector.curiosity, 0.0)

    def test_emergency_in_blend_suppresses(self):
        v     = _vec(humor=0.70)
        state = _state(blends={OverlayType.EMERGENCY: 0.80, OverlayType.FOCUSED: 0.20})
        result = self.resolver.resolve(v, self.budget, state)
        self.assertTrue(result.emergency_active)
        self.assertEqual(result.output_vector.humor, 0.0)

    def test_non_emergency_no_suppression(self):
        v     = _vec(humor=0.80)
        state = _state(overlay=OverlayType.RELAXED)
        result = self.resolver.resolve(v, self.budget, state)
        self.assertFalse(result.emergency_active)
        self.assertGreater(result.output_vector.humor, 0.0)

    # ── Budget enforcement ────────────────────────────────────────────────────

    def test_zero_budget_zeroes_trait(self):
        v      = _vec(curiosity=0.80)
        budget = ExpressionBudget(allocations={
            "directness": 40, "precision": 30, "technicality": 20,
            "confidence": 10,
            # curiosity NOT in allocations → effective_weight=0
        })
        result = self.resolver.resolve(v, budget, self.state)
        self.assertTrue(result.budget_enforced)
        self.assertEqual(result.output_vector.curiosity, 0.0)

    def test_nonempty_budget_does_not_zero_allocated_trait(self):
        v      = _vec(directness=0.80)
        budget = ExpressionBudget(allocations={"directness": 50})
        result = self.resolver.resolve(v, budget, self.state)
        self.assertGreater(result.output_vector.directness, 0.0)

    # ── Output clamped ────────────────────────────────────────────────────────

    def test_output_always_clamped(self):
        v = PersonaVector()
        for t in PersonaVector.trait_names():
            setattr(v, t, 1.1)   # over-range on purpose
        v = v.clamp()
        result = self.resolver.resolve(v, self.budget, self.state)
        for t in PersonaVector.trait_names():
            val = getattr(result.output_vector, t)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    # ── Multiple conflicts ────────────────────────────────────────────────────

    def test_multiple_conflict_pairs_resolved(self):
        v = _vec(
            formality=0.20,     humor=0.90,       # formality vs humor
            precision=0.20,     curiosity=0.90,   # precision vs curiosity
            professionalism=0.20,                 # professionalism vs humor (also triggers)
        )
        result = self.resolver.resolve(v, self.budget, self.state)
        self.assertGreater(result.conflicts_found, 1)

    # ── Callable interface ────────────────────────────────────────────────────

    def test_callable_returns_persona_vector(self):
        v = _vec(formality=0.30, humor=0.80)
        out = self.resolver(v, self.budget, self.state)
        self.assertIsInstance(out, PersonaVector)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 7-8 — SpeechFingerprintEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeechFingerprintEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SpeechFingerprintEngine()
        self.vector = PersonaVector()
        self.budget = ExpressionBudget()
        self.state  = _state()

    # ── Short-circuit ─────────────────────────────────────────────────────────

    def test_short_content_unchanged(self):
        content = "Hello world."
        result  = self.engine.apply(content, self.vector, self.budget, self.state)
        self.assertFalse(result.modified)
        self.assertEqual(result.output_text, content)
        self.assertEqual(result.pacing_mode, "unchanged_short")

    def test_single_sentence_threshold_boundary(self):
        """Exactly SINGLE_SENTENCE_THRESHOLD words → no-op."""
        content = " ".join(["word"] * SINGLE_SENTENCE_THRESHOLD)
        result  = self.engine.apply(content, self.vector, self.budget, self.state)
        self.assertFalse(result.modified)

    # ── Long sentence splitting ───────────────────────────────────────────────

    def test_long_sentence_split_at_clause(self):
        """A sentence exceeding _LONG_SENTENCE_WORDS with a clause boundary is split."""
        # 40+ words with a clear 'and' clause boundary
        long_sent = (
            "The monitoring system continuously processes all incoming telemetry "
            "requests from distributed nodes, and it then performs rigorous validation "
            "of each data packet against the defined schema rules before carefully "
            "forwarding the normalised payload to the downstream aggregation service "
            "which handles the final persistence and indexing of all records."
        )
        # Repeat to guarantee we are well above SINGLE_SENTENCE_THRESHOLD
        content = (long_sent + " This additional context also matters here greatly.") * 2
        result  = self.engine.apply(content, self.vector, self.budget, self.state)
        self.assertGreaterEqual(result.sentences_split, 1)

    # ── Transition injection ──────────────────────────────────────────────────

    def test_transitions_injected_in_balanced_pacing(self):
        """4+ sentences in balanced pacing should get at least one transition."""
        sentences = [
            "The first point is important for understanding the context of the situation.",
            "The second aspect builds upon what came before and extends the argument.",
            "The third element adds further nuance to the picture we are painting here.",
            "The fourth and final consideration ties everything together into a conclusion.",
            "This conclusion follows naturally from the preceding analysis and evidence.",
            "Therefore we can be confident in the recommendations outlined above.",
        ]
        content = " ".join(sentences)
        vector  = _vec(analytical_depth=0.70, directness=0.40)   # balanced/expansive
        result  = self.engine.apply(content, vector, self.budget, self.state)
        self.assertGreaterEqual(result.transitions_injected, 1)

    def test_no_transitions_in_compact_pacing(self):
        """High directness → compact pacing → no transitions injected."""
        sentences = [
            "The system is down.",
            "All requests are failing.",
            "Restart the service immediately.",
            "This will resolve the issue.",
            "Monitor for 15 minutes after restart.",
        ]
        content = " ".join(sentences)
        vector  = _vec(directness=0.95, precision=0.90, humor=0.01, warmth=0.01)
        result  = self.engine.apply(content, vector, self.budget, self.state)
        self.assertEqual(result.pacing_mode, "compact")
        self.assertEqual(result.transitions_injected, 0)

    # ── Dominant trait selects transition pool ────────────────────────────────

    def test_dominant_trait_identified(self):
        vector = _vec(technicality=0.95)
        result = self.engine.apply(
            " ".join(["A technical explanation." * 5]),
            vector, self.budget, self.state,
        )
        self.assertEqual(result.dominant_trait, "technicality")

    # ── Length gate ───────────────────────────────────────────────────────────

    def test_length_gate_truncates_for_short_preference(self):
        habits  = CommunicationHabits(preferred_response_length="short")
        engine  = SpeechFingerprintEngine(habits=habits)
        # 120 words — above the 80-word short gate
        content = " ".join(["word"] * 120)
        result  = engine.apply(content, self.vector, self.budget, self.state)
        word_count = len(result.output_text.split())
        self.assertLessEqual(word_count, 82)   # +2 tolerance for ellipsis/punctuation
        self.assertTrue(result.length_gate_applied)

    def test_length_gate_not_applied_for_medium(self):
        habits  = CommunicationHabits(preferred_response_length="medium")
        engine  = SpeechFingerprintEngine(habits=habits)
        content = " ".join(["word"] * 200)
        result  = engine.apply(content, self.vector, self.budget, self.state)
        self.assertFalse(result.length_gate_applied)

    # ── Callable interface ────────────────────────────────────────────────────

    def test_callable_returns_string(self):
        content = "Some response text that is long enough to process."
        out = self.engine(content, self.vector, self.budget, self.state)
        self.assertIsInstance(out, str)

    # ── FingerprintResult diagnostics ─────────────────────────────────────────

    def test_result_has_correct_pacing_for_high_analytical(self):
        vector  = _vec(analytical_depth=0.90, warmth=0.80, curiosity=0.80,
                       directness=0.10, precision=0.10)
        # Use enough sentences to clear the short-circuit threshold
        content = " ".join([
            "Some analytical explanation with considerable depth and nuance.",
            "The reasoning here extends across multiple considerations and angles.",
            "Each of these aspects contributes meaningfully to the overall picture.",
        ])
        result  = self.engine.apply(content, vector, self.budget, self.state)
        self.assertEqual(result.pacing_mode, "expansive")

    def test_result_has_correct_pacing_for_high_directness(self):
        vector  = _vec(directness=0.90, precision=0.85,
                       analytical_depth=0.10, warmth=0.10, curiosity=0.10)
        content = " ".join([
            "Direct precise statement about the current system status.",
            "The failure occurred at exactly 14:32 UTC on the primary node.",
            "Immediate action is required to restore service availability now.",
        ])
        result  = self.engine.apply(content, vector, self.budget, self.state)
        self.assertEqual(result.pacing_mode, "compact")

    def test_output_content_preserved(self):
        """Fingerprint must not alter facts — key words must survive."""
        content = (
            "The system error code is 0x4F2A. "
            "Memory usage is at 87 percent. "
            "This is a critical warning that must not be altered. "
            "Please restart the service immediately."
        )
        result = self.engine.apply(content, self.vector, self.budget, self.state)
        # Key facts must appear in the output
        self.assertIn("0x4F2A", result.output_text)
        self.assertIn("87", result.output_text)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration — TransformationPipeline with all four modules wired
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformationPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.pipeline, self.rsm = _pipeline_with_profile()

    def _run(self, content="Hello world.", criticality=CriticalityLevel.NORMAL,
             uncertainty=0.0, **kwargs):
        g = _finalized(content, criticality=criticality,
                       uncertainty=uncertainty, **kwargs)
        ti = TransformationInput(
            ghost_output=g,
            runtime_state_manager=self.rsm,
        )
        return self.pipeline.transform(ti)

    # ── No more stub warnings ─────────────────────────────────────────────────

    def test_no_stub_warnings(self):
        result = self._run()
        stub_warnings = [w for w in result.pipeline_warnings if "stub active" in w]
        self.assertEqual(stub_warnings, [], f"Unexpected stub warnings: {stub_warnings}")

    # ── All key stages execute ────────────────────────────────────────────────

    def test_stability_stage_in_stages(self):
        result = self._run()
        self.assertIn("stability_check", result.stages_executed)

    def test_confidence_stage_in_stages(self):
        result = self._run()
        self.assertIn("expression_confidence_attenuation", result.stages_executed)

    def test_conflict_stage_in_stages(self):
        result = self._run()
        self.assertIn("trait_conflict_resolution", result.stages_executed)

    def test_fingerprint_stage_in_stages(self):
        result = self._run()
        self.assertIn("speech_fingerprint", result.stages_executed)

    # ── Emergency override still bypasses personality ─────────────────────────

    def test_emergency_override_bypasses(self):
        result = self._run(criticality=CriticalityLevel.EMERGENCY)
        self.assertTrue(result.override_active)

    def test_security_override_bypasses(self):
        result = self._run(criticality=CriticalityLevel.SECURITY)
        self.assertTrue(result.override_active)

    # ── High uncertainty attenuates expression_confidence ────────────────────

    def test_high_uncertainty_lowers_confidence(self):
        low_unc  = self._run(uncertainty=0.0)
        high_unc = self._run(uncertainty=0.90)
        self.assertGreater(
            low_unc.expression_confidence,
            high_unc.expression_confidence,
            "Higher uncertainty should yield lower expression confidence.",
        )

    def test_zero_uncertainty_near_cap(self):
        result = self._run(uncertainty=0.0, criticality=CriticalityLevel.NORMAL)
        self.assertGreater(result.expression_confidence, 0.75)

    # ── Pipeline always returns passed result ─────────────────────────────────

    def test_normal_output_passes(self):
        result = self._run("This is a normal response.")
        self.assertTrue(result.passed)

    def test_pipeline_result_has_vector(self):
        result = self._run()
        self.assertIsInstance(result.persona_vector, PersonaVector)

    def test_pipeline_result_has_budget(self):
        result = self._run()
        self.assertIsInstance(result.expression_budget, ExpressionBudget)

    # ── Technical criticality attenuates confidence ───────────────────────────

    def test_technical_criticality_attenuates(self):
        normal    = self._run(criticality=CriticalityLevel.NORMAL,    uncertainty=0.0)
        technical = self._run(criticality=CriticalityLevel.TECHNICAL, uncertainty=0.0)
        self.assertGreater(normal.expression_confidence, technical.expression_confidence)

    # ── External hooks still work ─────────────────────────────────────────────

    def test_external_fingerprint_hook_overrides(self):
        sentinel = {"called": False}

        def my_hook(content, vector, budget, state):
            sentinel["called"] = True
            return "HOOKED: " + content

        self.pipeline.fingerprint_hook = my_hook
        result = self._run("Test content for the hook.")
        self.assertTrue(sentinel["called"])
        self.assertIn("HOOKED:", result.final_output)
        self.pipeline.fingerprint_hook = None   # cleanup

    def test_external_conflict_hook_overrides(self):
        sentinel = {"called": False}

        def my_hook(vector, budget, state):
            sentinel["called"] = True
            return vector

        self.pipeline.conflict_hook = my_hook
        self._run()
        self.assertTrue(sentinel["called"])
        self.pipeline.conflict_hook = None   # cleanup


if __name__ == "__main__":
    unittest.main(verbosity=2)
