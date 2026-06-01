"""
core/__init__.py — DarkPassenger Core Infrastructure

Public exports for the core infrastructure layer.
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

__all__ = [
    # Persona Vector
    "PersonaVector",
    "ExpressionBudget",
    "OverlayType",
    "RelationshipContext",
    "CommunicationIntent",
    "PersonaVectorEngine",
    # Config
    "ConfigManager",
    "PersonaProfile",
    "OverlayPreferences",
    "StabilityParameters",
    "CommunicationHabits",
    "AdaptiveTuning",
    "ValidationError",
    "CURRENT_SCHEMA_VERSION",
    # Runtime State
    "RuntimeState",
    "RuntimeStateManager",
    # Pipeline
    "TransformationPipeline",
    "TransformationInput",
    "TransformationResult",
    # Behavioral Logic — The Passenger (Part 5-8)
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
]
