"""
core/transformation_pipeline.py — DarkPassenger Transformation Pipeline

The central orchestrator that moves a GhostMindOutput through every stage
of the DarkPassenger system and returns a certified final response.

Pipeline stages (per spec §10):
    1.  Input Reception         — validate and unpack GhostMindOutput
    2.  Pre-flight Check        — CommunicationRulesEngine gate
    3.  Persona Vector Gen      — PersonaVectorEngine.build / build_blended
    4.  Stability Check         — [STUB for Part 5]
    5.  Expression Attenuation  — apply max_expression cap to vector
    6.  Budget Allocation       — ExpressionBudget from vector + profile
    7.  Trait Conflict Res.     — [STUB for Part 6]
    8.  Speech Fingerprint      — [STUB for Parts 7-8]
    9.  Validation Pipeline     — CircuitBreaker (3-stage)
    10. Final Response          — certified output

Stages 4, 7, and 8 are explicit extension points: each has a hook attribute
that can be replaced with a real implementation when those parts are built.

    pipeline.stability_hook    = CommunicationStabilityLayer(...)  # Part 5
    pipeline.conflict_hook     = TraitConflictResolver(...)         # Part 6
    pipeline.fingerprint_hook  = SpeechFingerprintEngine(...)       # Part 7-8

The pipeline is always safe to call. Any internal error falls through to the
CircuitBreaker, which falls back to raw GhostMind content if needed.

Spec reference: DarkPassenger-Plan.txt §10, §11, §13
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
        The finalized GhostMindOutput to transform. Must have had
        .finalize() called before being passed to the pipeline.

    runtime_state_manager:
        The active session state. The pipeline calls begin_response()
        and end_response() on this automatically.

    relationship_override:
        If set, overrides the relationship stored in RuntimeState for this
        one response only. Useful for testing or one-shot calls.

    intent_override:
        If set, overrides the intent inferred from RuntimeState.

    overlay_override:
        If set, overrides the overlay configured in RuntimeState.
    """
    ghost_output:          GhostMindOutput
    runtime_state_manager: RuntimeStateManager
    relationship_override: Optional[RelationshipContext]  = None
    intent_override:       Optional[CommunicationIntent] = None
    overlay_override:      Optional[OverlayType]         = None


# ── Extension hook types ──────────────────────────────────────────────────────
#
# These are the signatures expected by the three stub extension points.
# Replace the corresponding hook attribute on TransformationPipeline with
# any callable that matches.

StabilityHookFn   = Callable[[PersonaVector, PersonaVector, RuntimeState], PersonaVector]
ConflictHookFn    = Callable[[PersonaVector, ExpressionBudget, RuntimeState], PersonaVector]
FingerprintHookFn = Callable[[str, PersonaVector, ExpressionBudget, RuntimeState], str]


# ── TransformationPipeline ────────────────────────────────────────────────────

class TransformationPipeline:
    """
    Full DarkPassenger transformation pipeline.

    Construction:
        pipeline = TransformationPipeline(
            config_manager=...,
            profile_id="default",          # optional; uses active profile
        )

    Extension hooks (assign to replace stubs):
        pipeline.stability_hook   = my_stability_fn    # Part 5
        pipeline.conflict_hook    = my_conflict_fn     # Part 6
        pipeline.fingerprint_hook = my_fingerprint_fn  # Parts 7-8

    Usage:
        result = pipeline.transform(
            TransformationInput(
                ghost_output=finalized_output,
                runtime_state_manager=rsm,
            )
        )
        send_to_user(result.final_output)
    """

    MAX_REGEN_ATTEMPTS: int = 3

    def __init__(
        self,
        config_manager: ConfigManager,
        profile_id: Optional[str] = None,
        logger=None,
    ):
        self._config    = config_manager
        self._profile_id = profile_id  # None → use active profile
        self._logger    = logger

        self._vector_engine = PersonaVectorEngine()
        self._circuit       = CircuitBreaker(logger=logger)
        self._rules_engine  = CommunicationRulesEngine(logger=logger)

        # Extension hooks — replace with real implementations as parts are built
        self.stability_hook:   Optional[StabilityHookFn]   = None  # Part 5
        self.conflict_hook:    Optional[ConflictHookFn]    = None  # Part 6
        self.fingerprint_hook: Optional[FingerprintHookFn] = None  # Parts 7-8

        # Track the last PersonaVector for the stability stub
        self._prev_vector: Optional[PersonaVector] = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def transform(self, ti: TransformationInput) -> TransformationResult:
        """
        Run the full 10-stage transformation pipeline.

        This method never raises. Any unhandled internal error falls through
        to the CircuitBreaker, which returns raw GhostMind content.

        Args:
            ti: TransformationInput with ghost_output and runtime state.

        Returns:
            TransformationResult — always contains a safe final_output.
        """
        t_start = time.monotonic()
        stages: List[str] = []
        warnings: List[str] = []

        ti.runtime_state_manager.begin_response()
        state = ti.runtime_state_manager.state

        try:
            # ── Stage 1: Input reception ──────────────────────────────────────
            stages.append("input_reception")
            ghost = ti.ghost_output

            # ── Stage 2: Pre-flight check ─────────────────────────────────────
            stages.append("pre_flight")
            preflight = self._rules_engine.pre_flight(ghost)

            # Critical Response Override fires here
            if preflight.override_active:
                stages.append("critical_override")
                raw = self._circuit.validate(ghost, TransformedOutput(ghost.content))
                elapsed = (time.monotonic() - t_start) * 1000
                ti.runtime_state_manager.end_response()
                return TransformationResult(
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
                )

            # Hard rule violation (blocked but no override)
            if preflight.blocked:
                stages.append("preflight_hard_block")
                raw_candidate = TransformedOutput(ghost.content)
                validation = self._circuit.validate(ghost, raw_candidate)
                elapsed = (time.monotonic() - t_start) * 1000
                warnings.append(
                    f"Pre-flight hard block: "
                    + "; ".join(v.description for v in preflight.violations)
                )
                ti.runtime_state_manager.end_response()
                return TransformationResult(
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
                )

            # Capture expression confidence ceiling from pre-flight
            expression_confidence = preflight.max_expression
            ti.runtime_state_manager.set_expression_confidence(expression_confidence)

            # ── Stage 3: Persona Vector generation ───────────────────────────
            stages.append("persona_vector_generation")
            profile = self._resolve_profile()
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

            # ── Stage 4: Communication Stability Check ────────────────────────
            stages.append("stability_check")
            if self.stability_hook is not None:
                stable_vector = self.stability_hook(
                    raw_vector,
                    self._prev_vector or raw_vector,
                    state,
                )
            else:
                # STUB — Part 5 will implement smoothing
                stable_vector = raw_vector
                if self._prev_vector is None:
                    warnings.append(
                        "stability_check: stub active (Part 5). "
                        "No smoothing applied."
                    )

            # ── Stage 5: Expression Confidence Attenuation ───────────────────
            # Already applied in PersonaVectorEngine.build() via expression_confidence.
            # This stage records the final confidence and makes it available to
            # downstream stages without recalculating.
            stages.append("expression_confidence_attenuation")
            final_vector = stable_vector

            # ── Stage 6: Expression Budget Allocation ────────────────────────
            stages.append("expression_budget_allocation")
            if profile and profile.expression_budget.allocations:
                budget = profile.expression_budget
            else:
                budget = ExpressionBudget.from_vector(final_vector, top_n=5)

            # ── Stage 7: Trait Conflict Resolution ───────────────────────────
            stages.append("trait_conflict_resolution")
            if self.conflict_hook is not None:
                resolved_vector = self.conflict_hook(final_vector, budget, state)
            else:
                # STUB — Part 6 will resolve directness vs. verbosity, etc.
                resolved_vector = final_vector
                warnings.append(
                    "trait_conflict_resolution: stub active (Part 6). "
                    "Conflicts not resolved."
                )

            # ── Stage 8: Speech Fingerprint Application ───────────────────────
            stages.append("speech_fingerprint")
            candidate_text = self._apply_fingerprint(
                ghost.content, resolved_vector, budget, state
            )
            warnings.append(
                "speech_fingerprint: stub active (Parts 7-8). "
                "Content returned unchanged."
            )

            # ── Stage 9: Validation Pipeline (CircuitBreaker) ─────────────────
            stages.append("validation_pipeline")
            candidate = TransformedOutput(candidate_text)
            validation = self._circuit.validate(ghost, candidate)

            # Regeneration loop if first attempt fails
            attempt = 1
            while not validation.passed and attempt < self.MAX_REGEN_ATTEMPTS:
                attempt += 1
                stages.append(f"regeneration_attempt_{attempt}")
                warnings.append(
                    f"Validation failed (attempt {attempt-1}); regenerating."
                )
                # With stub fingerprint, regeneration won't change the output.
                # Once the real fingerprint is in, regeneration will vary.
                candidate = TransformedOutput(ghost.content)
                validation = self._circuit.validate(ghost, candidate)

            # ── Stage 10: Final Response ──────────────────────────────────────
            stages.append("final_response")
            self._prev_vector = resolved_vector

        except Exception as exc:
            # Fail-safe: any uncaught pipeline error returns raw content
            stages.append("pipeline_error_fallback")
            warnings.append(f"Pipeline error: {type(exc).__name__}: {exc}")
            raw_candidate = TransformedOutput(ti.ghost_output.content)
            validation = self._circuit.validate(ti.ghost_output, raw_candidate)
            expression_confidence = 0.0
            resolved_vector = PersonaVector()
            budget = ExpressionBudget()
            preflight = self._rules_engine.pre_flight(ti.ghost_output)

        elapsed = (time.monotonic() - t_start) * 1000
        ti.runtime_state_manager.end_response()

        return TransformationResult(
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
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_profile(self) -> Optional[PersonaProfile]:
        """
        Resolve the active persona profile.

        Returns the profile specified at construction, or the config
        manager's active profile, or None if no profile is available.
        """
        try:
            if self._profile_id:
                return self._config.get_profile(self._profile_id)
            return self._config.active_profile
        except KeyError:
            return None

    def _apply_fingerprint(
        self,
        content: str,
        vector: PersonaVector,
        budget: ExpressionBudget,
        state: RuntimeState,
    ) -> str:
        """
        Stage 8: Speech Fingerprint Application.

        STUB — Parts 7-8 will implement:
            - Preferred sentence lengths and pacing
            - Preferred transitions and explanation structure
            - Preferred questioning style and analogy usage
            - Organization patterns

        Currently returns content unchanged. This is correct behaviour
        for now: the CircuitBreaker will always pass unmodified GhostMind
        content, and the stub warning in the pipeline makes the gap visible.
        """
        if self.fingerprint_hook is not None:
            return self.fingerprint_hook(content, vector, budget, state)
        return content

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str,
        profile_id: Optional[str] = None,
        logger=None,
    ) -> "TransformationPipeline":
        """
        Convenience constructor: build a pipeline from a directory of persona JSON files.

        Args:
            config_dir:  Path to directory containing .json persona profiles.
            profile_id:  Profile to use. If None, uses the first loaded profile.
            logger:      Optional logger.

        Example:
            pipeline = TransformationPipeline.from_config_dir(
                "configs/personas",
                profile_id="default",
            )
        """
        cm = ConfigManager(config_dir=config_dir)
        return cls(config_manager=cm, profile_id=profile_id, logger=logger)
