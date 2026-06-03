"""
core/transformation_pipeline.py — DarkPassenger Transformation Pipeline  v1.4

The central orchestrator that moves a GhostMindOutput through every stage
of the DarkPassenger system and returns a certified final response.

Pipeline stages (per spec §10):
    1.  Input Reception         — validate and unpack GhostMindOutput
    2.  Pre-flight Check        — CommunicationRulesEngine gate
    3.  Persona Vector Gen      — PersonaVectorEngine.build / build_blended
    4.  Stability Check         — CommunicationStabilityLayer (smoothing)
    5.  Expression Attenuation  — ExpressionConfidenceSystem (entropy-linked)
    6.  Budget Allocation       — ExpressionBudget from vector + profile
    7.  Trait Conflict Res.     — TraitConflictResolver (priority hierarchy)
    8.  Speech Fingerprint      — SpeechFingerprintEngine (structural shaping)
    9.  Validation Pipeline     — CircuitBreaker (3-stage)
    10. Final Response          — certified output

v1.4 additions
──────────────
  • PerformanceManager integration (fast-path + persona/overlay caches)
    - Stages 4-8 are skipped when the PerformanceManager deems context stable.
    - Persona profiles are fetched from PersonaProfileCache on every call.
    - Overlay modifiers are fetched from OverlayConfigCache.
  • AuditLog integration
    - Every completed TransformationResult is recorded to the session AuditLog.
    - The log is accessible via pipeline.audit_log.
  • BehavioralReviewSystem wired to the same AuditLog.
    - Access via pipeline.reviewer.
    - Call pipeline.reviewer.full_review() for a complete diagnostic snapshot.

External hook attributes (unchanged from v1.3):
    pipeline.stability_hook    = custom_stability_fn
    pipeline.conflict_hook     = custom_conflict_fn
    pipeline.fingerprint_hook  = custom_fingerprint_fn

Spec reference: DarkPassenger-Plan.txt §10, §11, §13, §14, §15, §17
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from dp_types.integrity_types import (
    GhostMindOutput,
    ValidationResult,
)
from integrity.circuit_breaker import CircuitBreaker, TransformedOutput
from integrity.communication_rules import CommunicationRulesEngine, PreFlightResult

from core.persona_vector import (
    PersonaVector,
    ExpressionBudget,
    OverlayType,
    RelationshipContext,
    CommunicationIntent,
    PersonaVectorEngine,
)
from core.config_manager import ConfigManager, PersonaProfile
from core.runtime_state import RuntimeState, RuntimeStateManager
from core.stability_layer import CommunicationStabilityLayer
from core.expression_confidence import ExpressionConfidenceSystem
from core.trait_conflict_resolver import TraitConflictResolver
from core.speech_fingerprint import SpeechFingerprintEngine

# v1.4 — Performance & Monitoring Layer
from core.audit_log import AuditLog
from core.behavioral_review import BehavioralReviewSystem
from core.performance import PerformanceManager, FastPathResult


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class TransformationResult:
    """
    The complete output of one TransformationPipeline.transform() call.

    final_output:
        The certified text ready for delivery to the user.
        If an override fired or all validation attempts failed,
        this is raw GhostMind content. It is always safe to send.

    persona_vector:
        The PersonaVector that was active during transformation.
        Includes all context/overlay modifications and confidence scaling.

    expression_budget:
        The ExpressionBudget allocation for this response.

    expression_confidence:
        The final confidence scalar applied to the vector (0.0–1.0).

    override_active:
        True if the Critical Response Override fired (emergency, security, etc.).
        When True, no personality was applied.

    stages_executed:
        Ordered list of stage names that ran. Useful for debugging and audit.

    validation_result:
        The full CircuitBreaker ValidationResult, including any violations.

    pre_flight_result:
        The full CommunicationRulesEngine PreFlightResult.

    pipeline_warnings:
        Non-fatal warnings raised by any stage (e.g. stub stages in use).

    elapsed_ms:
        Approximate wall-clock time for the full pipeline in milliseconds.

    fast_path_active:
        True if stages 4-8 were skipped via PerformanceManager fast-path.
        Informational — does not affect output correctness.

    relationship:
        String label of the RelationshipContext active for this response.
        Stamped by the pipeline for AuditLog consumption (e.g. "owner", "guest").

    intent:
        String label of the CommunicationIntent active for this response.
        Stamped by the pipeline for AuditLog consumption (e.g. "inform", "warn").
    """
    final_output:          str
    persona_vector:        PersonaVector
    expression_budget:     ExpressionBudget
    expression_confidence: float
    override_active:       bool
    stages_executed:       List[str]
    validation_result:     ValidationResult
    pre_flight_result:     PreFlightResult
    pipeline_warnings:     List[str]      = field(default_factory=list)
    elapsed_ms:            float          = 0.0
    fast_path_active:      bool           = False
    relationship:          str            = "unknown"
    intent:                str            = "unknown"

    @property
    def passed(self) -> bool:
        """True if the pipeline completed without integrity failures."""
        return self.validation_result.passed or self.override_active


# ── Pipeline input ────────────────────────────────────────────────────────────

@dataclass
class TransformationInput:
    """
    Everything needed to run one transformation cycle.

    ghost_output:
        The finalized GhostMindOutput to transform.

    runtime_state_manager:
        The active session state.

    relationship_override / intent_override / overlay_override:
        One-shot context overrides for this response only.
    """
    ghost_output:          GhostMindOutput
    runtime_state_manager: RuntimeStateManager
    relationship_override: Optional[RelationshipContext]  = None
    intent_override:       Optional[CommunicationIntent] = None
    overlay_override:      Optional[OverlayType]         = None


# ── Extension hook types ──────────────────────────────────────────────────────

StabilityHookFn   = Callable[[PersonaVector, PersonaVector, RuntimeState], PersonaVector]
ConflictHookFn    = Callable[[PersonaVector, ExpressionBudget, RuntimeState], PersonaVector]
FingerprintHookFn = Callable[[str, PersonaVector, ExpressionBudget, RuntimeState], str]


# ── TransformationPipeline ────────────────────────────────────────────────────

class TransformationPipeline:
    """
    Full DarkPassenger transformation pipeline (v1.4).

    v1.4 adds the Performance & Monitoring Layer transparently — all existing
    call-sites continue to work without modification.

    New attributes:
        pipeline.audit_log   — AuditLog — session record buffer
        pipeline.reviewer    — BehavioralReviewSystem — diagnostic analysis
        pipeline.perf        — PerformanceManager — fast-path + caches

    Quick diagnostics:
        bundle = pipeline.reviewer.full_review()
        print(bundle.stability.health)   # "healthy" | "warning" | "unstable"
        print(bundle.quality.p95_latency_ms)
        print(pipeline.perf.stats())

    Construction:
        pipeline = TransformationPipeline(
            config_manager=...,
            profile_id="default",
            session_id="my-session-42",   # optional; auto-generated if omitted
        )

    Extension hooks (unchanged):
        pipeline.stability_hook   = my_stability_fn
        pipeline.conflict_hook    = my_conflict_fn
        pipeline.fingerprint_hook = my_fingerprint_fn
    """

    MAX_REGEN_ATTEMPTS: int = 3

    def __init__(
        self,
        config_manager: ConfigManager,
        profile_id: Optional[str] = None,
        session_id: Optional[str] = None,
        logger=None,
        # v1.4 performance tuning knobs (all optional)
        profile_cache_ttl: float    = 300.0,
        max_cached_profiles: int    = 16,
        max_cached_overlays: int    = 64,
        baseline_traits: Optional[Dict[str, float]] = None,
    ):
        self._config     = config_manager
        self._profile_id = profile_id
        self._logger     = logger

        self._vector_engine = PersonaVectorEngine()
        self._circuit       = CircuitBreaker(logger=logger)
        self._rules_engine  = CommunicationRulesEngine(logger=logger)

        # ── Behavioral Logic modules ──────────────────────────────────────────
        _profile = self._resolve_profile()
        _sp      = _profile.stability_parameters if _profile else None
        _habits  = _profile.communication_habits if _profile else None

        self._stability_engine   = CommunicationStabilityLayer(
            smoothing_factor=_sp.smoothing_factor if _sp else 0.50,
            drift_threshold=_sp.drift_threshold   if _sp else 0.30,
            logger=logger,
        )
        self._confidence_engine  = ExpressionConfidenceSystem(logger=logger)
        self._conflict_resolver  = TraitConflictResolver(logger=logger)
        self._fingerprint_engine = SpeechFingerprintEngine(
            habits=_habits,
            logger=logger,
        )

        # Extension hooks — external callers may replace with custom fns.
        self.stability_hook:   Optional[StabilityHookFn]   = None
        self.conflict_hook:    Optional[ConflictHookFn]    = None
        self.fingerprint_hook: Optional[FingerprintHookFn] = None

        # Stability tracking
        self._prev_vector: Optional[PersonaVector] = None

        # ── v1.4: Performance & Monitoring Layer ──────────────────────────────

        # AuditLog — ring buffer for this session
        self.audit_log = AuditLog(
            session_id=session_id,
            logger=logger,
        )

        # BehavioralReviewSystem — reads from audit_log; baseline optional
        self.reviewer = BehavioralReviewSystem(
            audit_log=self.audit_log,
            baseline_traits=baseline_traits,
            logger=logger,
        )

        # PerformanceManager — fast-path heuristic + two caches
        self.perf = PerformanceManager(
            profile_ttl=profile_cache_ttl,
            max_profiles=max_cached_profiles,
            max_overlays=max_cached_overlays,
            logger=logger,
        )

        # Pre-warm the profile cache with the active profile (if available)
        if _profile is not None:
            self.perf.profile_cache.set(
                _profile.profile_id if hasattr(_profile, "profile_id") else "active",
                _profile,
            )

    # ── Main entry point ──────────────────────────────────────────────────────

    def transform(self, ti: TransformationInput) -> TransformationResult:
        """
        Run the full 10-stage transformation pipeline.

        In v1.4, stages 4-8 may be skipped via the PerformanceManager
        fast-path when the context is stable.

        Every completed result is automatically recorded to self.audit_log.

        This method never raises. Any unhandled internal error falls through
        to the CircuitBreaker, which returns raw GhostMind content.
        """
        t_start = time.monotonic()
        stages:   List[str] = []
        warnings: List[str] = []
        fast_path_active = False
        relationship_str = "unknown"
        intent_str       = "unknown"

        ti.runtime_state_manager.begin_response()
        state = ti.runtime_state_manager.state

        try:
            # ── Stage 1: Input reception ──────────────────────────────────────
            stages.append("input_reception")
            ghost = ti.ghost_output

            # ── Stage 2: Pre-flight check ─────────────────────────────────────
            stages.append("pre_flight")
            preflight = self._rules_engine.pre_flight(ghost)

            # Critical Response Override
            if preflight.override_active:
                stages.append("critical_override")
                raw = self._circuit.validate(ghost, TransformedOutput(ghost.content))
                elapsed = (time.monotonic() - t_start) * 1000
                ti.runtime_state_manager.end_response()
                result = TransformationResult(
                    final_output=raw.safe_output,
                    persona_vector=PersonaVector(),
                    expression_budget=ExpressionBudget(),
                    expression_confidence=0.0,
                    override_active=True,
                    stages_executed=stages,
                    validation_result=raw,
                    pre_flight_result=preflight,
                    pipeline_warnings=warnings,
                    elapsed_ms=elapsed,
                    fast_path_active=False,
                    relationship=state.active_relationship.value,
                    intent=state.current_intent.value,
                )
                self.audit_log.record(result)
                return result

            # Hard rule block (no override)
            if preflight.blocked:
                stages.append("preflight_hard_block")
                raw_candidate = TransformedOutput(ghost.content)
                validation = self._circuit.validate(ghost, raw_candidate)
                elapsed = (time.monotonic() - t_start) * 1000
                warnings.append(
                    "Pre-flight hard block: "
                    + "; ".join(v.description for v in preflight.violations)
                )
                ti.runtime_state_manager.end_response()
                result = TransformationResult(
                    final_output=validation.safe_output,
                    persona_vector=PersonaVector(),
                    expression_budget=ExpressionBudget(),
                    expression_confidence=0.0,
                    override_active=False,
                    stages_executed=stages,
                    validation_result=validation,
                    pre_flight_result=preflight,
                    pipeline_warnings=warnings,
                    elapsed_ms=elapsed,
                    fast_path_active=False,
                    relationship=state.active_relationship.value,
                    intent=state.current_intent.value,
                )
                self.audit_log.record(result)
                return result

            expression_confidence = preflight.max_expression
            ti.runtime_state_manager.set_expression_confidence(expression_confidence)

            # ── Stage 3: Persona Vector generation ───────────────────────────
            stages.append("persona_vector_generation")
            profile = self._resolve_profile_cached()
            base_vector = profile.base_traits if profile else PersonaVector()

            relationship = (
                ti.relationship_override
                if ti.relationship_override is not None
                else state.active_relationship
            )
            intent = (
                ti.intent_override
                if ti.intent_override is not None
                else state.current_intent
            )
            overlay_override = ti.overlay_override

            # String labels for audit log stamping
            relationship_str = relationship.value if hasattr(relationship, "value") else str(relationship)
            intent_str       = intent.value if hasattr(intent, "value") else str(intent)

            # Resolve overlay label for fast-path query
            overlay_label = self._overlay_label(state, overlay_override)

            # ── v1.4: Fast-path check before stages 4-8 ───────────────────────
            fp: FastPathResult = self.perf.should_fast_path(
                overlay=overlay_label,
                relationship=relationship_str,
                intent=intent_str,
                expression_confidence=expression_confidence,
            )

            if fp.eligible:
                # Fast-path: reuse previous persona state, skip stages 4-8
                fast_path_active = True
                resolved_vector  = fp.persona_vector
                budget           = fp.expression_budget
                expression_confidence = fp.expression_confidence
                stages.extend(["fast_path_taken"] + [f"skipped:{s}" for s in fp.stages_skipped])
                candidate_text = ghost.content  # fingerprint already applied to content
            else:
                # Full pipeline: stages 4-8
                if overlay_override is not None:
                    raw_vector = self._vector_engine.build(
                        base=base_vector,
                        relationship=relationship,
                        intent=intent,
                        overlay=overlay_override,
                        expression_confidence=expression_confidence,
                    )
                elif state.has_blends():
                    raw_vector = self._vector_engine.build_blended(
                        base=base_vector,
                        overlays=state.current_overlay_blends,
                        relationship=relationship,
                        intent=intent,
                        expression_confidence=expression_confidence,
                    )
                else:
                    raw_vector = self._vector_engine.build(
                        base=base_vector,
                        relationship=relationship,
                        intent=intent,
                        overlay=state.current_overlay,
                        expression_confidence=expression_confidence,
                    )

                # Stage 4: Stability Check
                stages.append("stability_check")
                prev_vector = self._prev_vector
                if self.stability_hook is not None:
                    stable_vector = self.stability_hook(raw_vector, prev_vector or raw_vector, state)
                else:
                    stable_vector = self._stability_engine(
                        current_vector=raw_vector,
                        previous_vector=prev_vector or raw_vector,
                        current_state=state,
                    )

                # Stage 5: Expression Confidence Attenuation
                stages.append("expression_confidence_attenuation")
                conf_calc = self._confidence_engine.compute(ghost)
                expression_confidence = conf_calc.final_confidence
                final_vector = stable_vector.scale(expression_confidence).clamp()

                # Stage 6: Expression Budget Allocation
                stages.append("expression_budget_allocation")
                if profile and profile.expression_budget.allocations:
                    budget = profile.expression_budget
                else:
                    budget = ExpressionBudget.from_vector(final_vector, top_n=5)

                # Stage 7: Trait Conflict Resolution
                stages.append("trait_conflict_resolution")
                if self.conflict_hook is not None:
                    resolved_vector = self.conflict_hook(final_vector, budget, state)
                else:
                    resolved_vector = self._conflict_resolver(final_vector, budget, state)

                # Stage 8: Speech Fingerprint Application
                stages.append("speech_fingerprint")
                candidate_text = self._apply_fingerprint(
                    ghost.content, resolved_vector, budget, state
                )

                # Record result snapshot in PerformanceManager for next fast-path check
                self._prev_vector = resolved_vector

            # ── Stage 9: Validation Pipeline (CircuitBreaker) ─────────────────
            stages.append("validation_pipeline")
            candidate = TransformedOutput(candidate_text)
            validation = self._circuit.validate(ghost, candidate)

            attempt = 1
            while not validation.passed and attempt < self.MAX_REGEN_ATTEMPTS:
                attempt += 1
                stages.append(f"regeneration_attempt_{attempt}")
                warnings.append(f"Validation failed (attempt {attempt-1}); regenerating.")
                candidate = TransformedOutput(ghost.content)
                validation = self._circuit.validate(ghost, candidate)

            # ── Stage 10: Final Response ──────────────────────────────────────
            stages.append("final_response")

        except Exception as exc:
            stages.append("pipeline_error_fallback")
            warnings.append(f"Pipeline error: {type(exc).__name__}: {exc}")
            raw_candidate = TransformedOutput(ti.ghost_output.content)
            validation = self._circuit.validate(ti.ghost_output, raw_candidate)
            expression_confidence = 0.0
            resolved_vector = PersonaVector()
            budget = ExpressionBudget()
            preflight = self._rules_engine.pre_flight(ti.ghost_output)
            fast_path_active = False

        elapsed = (time.monotonic() - t_start) * 1000
        ti.runtime_state_manager.end_response()

        result = TransformationResult(
            final_output=validation.safe_output,
            persona_vector=resolved_vector,
            expression_budget=budget,
            expression_confidence=expression_confidence,
            override_active=validation.override_active,
            stages_executed=stages,
            validation_result=validation,
            pre_flight_result=preflight,
            pipeline_warnings=warnings,
            elapsed_ms=elapsed,
            fast_path_active=fast_path_active,
            relationship=relationship_str,
            intent=intent_str,
        )

        # v1.4: Record every result in the audit log.
        # Fast-path results ARE recorded so the reviewer sees accurate throughput.
        self.audit_log.record(result)

        # Update PerformanceManager history (full-pipeline runs only).
        if not fast_path_active:
            self.perf.record_result(result)

        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_profile(self) -> Optional[PersonaProfile]:
        """Resolve the active persona profile (uncached)."""
        try:
            if self._profile_id:
                return self._config.get_profile(self._profile_id)
            return self._config.active_profile
        except KeyError:
            return None

    def _resolve_profile_cached(self) -> Optional[PersonaProfile]:
        """
        Resolve the active persona profile, checking PersonaProfileCache first.

        On a cache miss the profile is loaded from ConfigManager and stored
        for future calls. Adds sub-millisecond profile access in steady state.
        """
        cache_key = self._profile_id or "active"
        cached = self.perf.profile_cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        profile = self._resolve_profile()
        if profile is not None:
            self.perf.profile_cache.set(cache_key, profile)
        return profile

    def _apply_fingerprint(
        self,
        content: str,
        vector: PersonaVector,
        budget: ExpressionBudget,
        state: RuntimeState,
    ) -> str:
        """Stage 8: Speech Fingerprint Application."""
        if self.fingerprint_hook is not None:
            return self.fingerprint_hook(content, vector, budget, state)
        return self._fingerprint_engine(content, vector, budget, state)

    @staticmethod
    def _overlay_label(state: RuntimeState, overlay_override: Optional[OverlayType]) -> str:
        """Produce a string overlay label for PerformanceManager key comparison."""
        if overlay_override is not None:
            return overlay_override.value if hasattr(overlay_override, "value") else str(overlay_override)
        blends = state.current_overlay_blends if state.has_blends() else None
        if blends:
            import json
            return json.dumps({k: round(float(w), 3) for k, w in blends.items()}, sort_keys=True)
        ov = state.current_overlay
        if ov is not None:
            return ov.value if hasattr(ov, "value") else str(ov)
        return "none"

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str,
        profile_id: Optional[str] = None,
        session_id: Optional[str] = None,
        baseline_traits: Optional[Dict[str, float]] = None,
        logger=None,
    ) -> "TransformationPipeline":
        """
        Convenience constructor: build a pipeline from a directory of persona JSON files.

        Args:
            config_dir:      Path to directory containing .json persona profiles.
            profile_id:      Profile to use. If None, uses the first loaded profile.
            session_id:      Optional session UUID for the AuditLog.
            baseline_traits: Optional trait baseline for DriftDetectionReport.
            logger:          Optional logger.

        Example:
            pipeline = TransformationPipeline.from_config_dir(
                "configs/personas",
                profile_id="default",
                baseline_traits={"humor": 0.70, "directness": 0.90},
            )
        """
        cm = ConfigManager(config_dir=config_dir)
        return cls(
            config_manager=cm,
            profile_id=profile_id,
            session_id=session_id,
            baseline_traits=baseline_traits,
            logger=logger,
        )
