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
]
