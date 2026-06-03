"""
core/__init__.py — DarkPassenger Core Infrastructure  v1.4

Public exports for the core infrastructure layer.

v1.4 additions
──────────────
  • AuditLog, AuditRecord               — Personality Audit Log  (§14)
  • BehavioralReviewSystem              — Diagnostic review engine (§15)
  • StabilityReport, QualityReport      — Report types
  • DriftDetectionReport, AdaptationReport, FullReviewBundle
  • PerformanceManager                  — Fast-path + caches (§17)
  • PersonaProfileCache, OverlayConfigCache, FastPathResult
"""

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
    OverlayPreferences,
    StabilityParameters,
    CommunicationHabits,
    AdaptiveTuning,
    ValidationError,
    CURRENT_SCHEMA_VERSION,
)
from core.runtime_state import (
    RuntimeState,
    RuntimeStateManager,
)
from core.transformation_pipeline import (
    TransformationPipeline,
    TransformationInput,
    TransformationResult,
)
from core.stability_layer import (
    CommunicationStabilityLayer,
    StabilityCheckResult,
    WARMUP_RESPONSES,
)
from core.expression_confidence import (
    ExpressionConfidenceSystem,
    ConfidenceCalculation,
)
from core.trait_conflict_resolver import (
    TraitConflictResolver,
    ConflictResolutionResult,
    TraitAdjustment,
)
from core.speech_fingerprint import (
    SpeechFingerprintEngine,
    FingerprintResult,
    SINGLE_SENTENCE_THRESHOLD,
)

# ── v1.4: Performance & Monitoring Layer ──────────────────────────────────────
from core.audit_log import (
    AuditLog,
    AuditRecord,
)
from core.behavioral_review import (
    BehavioralReviewSystem,
    StabilityReport,
    QualityReport,
    DriftDetectionReport,
    AdaptationReport,
    AdaptationRecommendation,
    FullReviewBundle,
)
from core.performance import (
    PerformanceManager,
    PersonaProfileCache,
    OverlayConfigCache,
    FastPathResult,
    FAST_PATH_CONF_TOLERANCE,
    FAST_PATH_MIN_HISTORY,
    DEFAULT_PROFILE_TTL,
)


__all__ = [
    # ── Persona Vector ────────────────────────────────────────────────────────
    "PersonaVector",
    "ExpressionBudget",
    "OverlayType",
    "RelationshipContext",
    "CommunicationIntent",
    "PersonaVectorEngine",
    # ── Config ────────────────────────────────────────────────────────────────
    "ConfigManager",
    "PersonaProfile",
    "OverlayPreferences",
    "StabilityParameters",
    "CommunicationHabits",
    "AdaptiveTuning",
    "ValidationError",
    "CURRENT_SCHEMA_VERSION",
    # ── Runtime State ─────────────────────────────────────────────────────────
    "RuntimeState",
    "RuntimeStateManager",
    # ── Pipeline ──────────────────────────────────────────────────────────────
    "TransformationPipeline",
    "TransformationInput",
    "TransformationResult",
    # ── Behavioral Logic (Parts 5-8) ──────────────────────────────────────────
    "CommunicationStabilityLayer",
    "StabilityCheckResult",
    "WARMUP_RESPONSES",
    "ExpressionConfidenceSystem",
    "ConfidenceCalculation",
    "TraitConflictResolver",
    "ConflictResolutionResult",
    "TraitAdjustment",
    "SpeechFingerprintEngine",
    "FingerprintResult",
    "SINGLE_SENTENCE_THRESHOLD",
    # ── v1.4: Audit Log (§14) ─────────────────────────────────────────────────
    "AuditLog",
    "AuditRecord",
    # ── v1.4: Behavioral Review System (§15) ──────────────────────────────────
    "BehavioralReviewSystem",
    "StabilityReport",
    "QualityReport",
    "DriftDetectionReport",
    "AdaptationReport",
    "AdaptationRecommendation",
    "FullReviewBundle",
    # ── v1.4: Performance Manager (§17) ───────────────────────────────────────
    "PerformanceManager",
    "PersonaProfileCache",
    "OverlayConfigCache",
    "FastPathResult",
    "FAST_PATH_CONF_TOLERANCE",
    "FAST_PATH_MIN_HISTORY",
    "DEFAULT_PROFILE_TTL",
]
