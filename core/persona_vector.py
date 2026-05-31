"""
core/persona_vector.py — DarkPassenger Persona Vector Engine

Implements the mathematical representation of personality and the
machinery for composing it from multiple inputs.

Architecture
────────────
PersonaVector is built compositionally from four inputs:

    1. Base Identity       — long-term personality configuration (from ConfigManager)
    2. Relationship Context — how we're addressing this user (owner, guest, etc.)
    3. Communication Intent — what we're trying to accomplish (inform, warn, etc.)
    4. Expression Overlay   — situational modifier (focused, emergency, etc.)

Each layer modifies the base traits; results are blended according to
configured weights. The final vector is passed downstream for Expression
Budget allocation and Trait Conflict Resolution.

Expression Budget
─────────────────
The Expression Budget limits how many "personality points" are active at once.
Total budget = 100 points, distributed among traits by priority.
Dominant traits consume their allocation first; minor traits fill the rest.

This prevents trait overload — you can't simultaneously be maximally direct,
maximally warm, maximally humorous, AND maximally technical.

Spec reference: DarkPassenger-Plan.txt §§2, 3, 6, 7
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class OverlayType(str, Enum):
    """
    Situational expression modifiers.

    Overlays are temporary — they modify the vector for one response,
    not the underlying Base Identity.
    """
    FOCUSED   = "focused"    # Technical, efficient, direct
    RELAXED   = "relaxed"    # Conversational, casual, friendly
    TEACHING  = "teaching"   # Detailed, educational, patient
    CREATIVE  = "creative"   # Analogies, exploration, flexible language
    EMERGENCY = "emergency"  # Minimal words, maximum clarity, no excess personality


class RelationshipContext(str, Enum):
    """How DarkPassenger is addressing this user."""
    OWNER         = "owner"
    TRUSTED_USER  = "trusted_user"
    ADMINISTRATOR = "administrator"
    FRIEND        = "friend"
    NEW_USER      = "new_user"
    GUEST         = "guest"
    UNKNOWN       = "unknown"


class CommunicationIntent(str, Enum):
    """What the communication is trying to accomplish."""
    INFORM     = "inform"
    EXPLAIN    = "explain"
    TEACH      = "teach"
    WARN       = "warn"
    GUIDE      = "guide"
    SUMMARIZE  = "summarize"
    BRAINSTORM = "brainstorm"
    REPORT     = "report"


# ── PersonaVector ─────────────────────────────────────────────────────────────

@dataclass
class PersonaVector:
    """
    A unified numerical representation of an active communication personality.

    All trait values are floats in [0.0, 1.0].

    This is the working personality representation consumed by all downstream
    systems: Expression Budget allocation, Trait Conflict Resolution,
    and Speech Fingerprint application.
    """
    formality:        float = 0.50
    humor:            float = 0.30
    warmth:           float = 0.60
    confidence:       float = 0.80
    directness:       float = 0.75
    professionalism:  float = 0.65
    technicality:     float = 0.60
    precision:        float = 0.80
    curiosity:        float = 0.50
    analytical_depth: float = 0.60

    # ── Math operations ───────────────────────────────────────────────────────

    def clamp(self) -> "PersonaVector":
        """Clamp all traits to [0.0, 1.0] in-place. Returns self for chaining."""
        for trait in self.trait_names():
            setattr(self, trait, max(0.0, min(1.0, getattr(self, trait))))
        return self

    def scale(self, factor: float) -> "PersonaVector":
        """
        Return a new PersonaVector with all traits multiplied by factor.

        Used for Expression Confidence attenuation:
            - factor = 1.0 → full personality
            - factor = 0.5 → half intensity
            - factor = 0.05 → near-flat (emergency level)
        """
        f = max(0.0, min(1.0, factor))
        return PersonaVector(
            **{t: getattr(self, t) * f for t in self.trait_names()}
        )

    def blend(self, other: "PersonaVector", weight: float) -> "PersonaVector":
        """
        Return a new PersonaVector linearly interpolated between self and other.

        weight=0.0 → pure self (no change)
        weight=1.0 → pure other (full replacement)
        weight=0.5 → midpoint

        Used by the Communication Stability Layer to smooth abrupt trait jumps.
        """
        w = max(0.0, min(1.0, weight))
        return PersonaVector(**{
            t: getattr(self, t) * (1.0 - w) + getattr(other, t) * w
            for t in self.trait_names()
        })

    def add_delta(self, deltas: Dict[str, float]) -> "PersonaVector":
        """
        Return a new PersonaVector with deltas applied to named traits.
        Unknown trait names are silently ignored.
        """
        result = PersonaVector(**{t: getattr(self, t) for t in self.trait_names()})
        for trait, delta in deltas.items():
            if hasattr(result, trait):
                setattr(result, trait, getattr(result, trait) + delta)
        return result

    def distance(self, other: "PersonaVector") -> float:
        """
        Euclidean distance between two persona vectors.

        Used by the Communication Stability Layer to detect abrupt shifts.
        A distance > stability_parameters.drift_threshold triggers smoothing.
        """
        return math.sqrt(
            sum(
                (getattr(self, t) - getattr(other, t)) ** 2
                for t in self.trait_names()
            )
        )

    def dominant_traits(self, n: int = 3) -> list[str]:
        """Return the n trait names with the highest values, descending."""
        ranked = sorted(
            self.trait_names(),
            key=lambda t: getattr(self, t),
            reverse=True,
        )
        return ranked[:n]

    # ── Serialisation ─────────────────────────────────────────────────────────

    @staticmethod
    def trait_names() -> list[str]:
        return [
            "formality", "humor", "warmth", "confidence", "directness",
            "professionalism", "technicality", "precision",
            "curiosity", "analytical_depth",
        ]

    def to_dict(self) -> dict:
        return {t: round(getattr(self, t), 6) for t in self.trait_names()}

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaVector":
        known = set(cls.trait_names())
        kwargs = {k: float(v) for k, v in data.items() if k in known}
        return cls(**kwargs).clamp()

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{t}={getattr(self, t):.2f}" for t in self.trait_names()
        )
        return f"PersonaVector({pairs})"


# ── ExpressionBudget ──────────────────────────────────────────────────────────

@dataclass
class ExpressionBudget:
    """
    Finite allocation of personality expression across traits.

    Total budget = 100 points, distributed in priority order.
    Dominant traits consume their allocation first; minor traits fill the rest.

    Purpose:
        - Prevent trait overload (can't be maximally everything at once)
        - Preserve readability
        - Prioritize dominant traits
        - Maintain communication clarity

    Spec reference: DarkPassenger-Plan.txt §6
    """
    allocations: Dict[str, int] = field(default_factory=dict)

    TOTAL: int = 100

    @classmethod
    def from_dict(cls, data: dict) -> "ExpressionBudget":
        """Load budget from a dict of {trait_name: int} allocations."""
        return cls(allocations={k: int(v) for k, v in data.items()})

    @classmethod
    def from_vector(cls, vector: PersonaVector, top_n: int = 5) -> "ExpressionBudget":
        """
        Auto-generate a budget from a PersonaVector.

        Distributes 100 points proportionally among the top_n strongest traits.
        The strongest trait absorbs any rounding remainder.
        """
        traits = PersonaVector.trait_names()
        ranked = sorted(traits, key=lambda t: getattr(vector, t), reverse=True)
        top = ranked[:top_n]

        total_weight = sum(getattr(vector, t) for t in top)

        if total_weight == 0:
            per_trait = cls.TOTAL // top_n
            allocs = {t: per_trait for t in top}
        else:
            raw = {
                t: (getattr(vector, t) / total_weight) * cls.TOTAL
                for t in top
            }
            allocs = {t: int(v) for t, v in raw.items()}
            remainder = cls.TOTAL - sum(allocs.values())
            allocs[top[0]] = allocs.get(top[0], 0) + remainder

        return cls(allocations=allocs)

    def validate(self) -> tuple[bool, str]:
        """
        Return (True, "") if valid, or (False, reason) if not.

        Valid = total allocations do not exceed TOTAL.
        """
        total = sum(self.allocations.values())
        if total > self.TOTAL:
            return False, f"Budget total {total} exceeds maximum {self.TOTAL}"
        if any(v < 0 for v in self.allocations.values()):
            return False, "Budget contains negative allocations"
        return True, ""

    def effective_weight(self, trait: str) -> float:
        """
        Return the effective expression weight for a trait as a fraction of total.

        Returns 0.0 if the trait has no budget allocation.
        """
        return self.allocations.get(trait, 0) / self.TOTAL

    def to_dict(self) -> dict:
        return dict(self.allocations)

    def __repr__(self) -> str:
        return f"ExpressionBudget({self.allocations})"


# ── Overlay delta tables ──────────────────────────────────────────────────────
#
# Each overlay is a dict of trait → signed delta.
# Positive = push trait up; negative = push trait down.
# Applied at overlay_weight (0.0–1.0) on top of the base + context vector.

_OVERLAY_DELTAS: Dict[OverlayType, Dict[str, float]] = {
    OverlayType.FOCUSED: {
        "technicality":     +0.25,
        "directness":       +0.20,
        "precision":        +0.20,
        "humor":            -0.30,
        "warmth":           -0.10,
        "curiosity":        -0.10,
    },
    OverlayType.RELAXED: {
        "humor":            +0.25,
        "warmth":           +0.20,
        "curiosity":        +0.10,
        "formality":        -0.20,
        "technicality":     -0.15,
        "precision":        -0.10,
    },
    OverlayType.TEACHING: {
        "analytical_depth": +0.25,
        "curiosity":        +0.20,
        "warmth":           +0.15,
        "precision":        +0.10,
        "directness":       -0.10,
        "humor":            -0.05,
    },
    OverlayType.CREATIVE: {
        "curiosity":        +0.30,
        "humor":            +0.15,
        "analytical_depth": +0.10,
        "formality":        -0.20,
        "precision":        -0.15,
        "technicality":     -0.05,
    },
    OverlayType.EMERGENCY: {
        # Collapses personality toward bare directness and precision.
        # The spec mandates: minimal words, maximum clarity, no excess personality.
        "directness":       +0.40,
        "precision":        +0.20,
        "humor":            -0.80,
        "warmth":           -0.50,
        "curiosity":        -0.50,
        "formality":        -0.30,
        "technicality":     -0.30,
        "analytical_depth": -0.40,
        "professionalism":  -0.20,
    },
}

# Relationship context modifiers — subtle adjustments to warmth/formality
_RELATIONSHIP_MODIFIERS: Dict[RelationshipContext, Dict[str, float]] = {
    RelationshipContext.OWNER:         {"warmth": +0.10, "directness": +0.10},
    RelationshipContext.TRUSTED_USER:  {"warmth": +0.05, "confidence": +0.05},
    RelationshipContext.ADMINISTRATOR: {"formality": +0.05, "technicality": +0.05},
    RelationshipContext.FRIEND:        {
        "humor": +0.15, "warmth": +0.15, "formality": -0.15, "directness": +0.05,
    },
    RelationshipContext.NEW_USER:      {"warmth": +0.10, "technicality": -0.10},
    RelationshipContext.GUEST:         {"formality": +0.10, "technicality": -0.05},
    RelationshipContext.UNKNOWN:       {},
}

# Communication intent modifiers
_INTENT_MODIFIERS: Dict[CommunicationIntent, Dict[str, float]] = {
    CommunicationIntent.INFORM:     {"directness": +0.10, "precision": +0.10},
    CommunicationIntent.EXPLAIN:    {"analytical_depth": +0.15, "technicality": +0.05},
    CommunicationIntent.TEACH:      {"warmth": +0.10, "analytical_depth": +0.20},
    CommunicationIntent.WARN:       {
        "directness": +0.20, "precision": +0.20, "humor": -0.30, "warmth": -0.05,
    },
    CommunicationIntent.GUIDE:      {"warmth": +0.10, "directness": +0.10},
    CommunicationIntent.SUMMARIZE:  {"directness": +0.15, "precision": +0.15},
    CommunicationIntent.BRAINSTORM: {"curiosity": +0.20, "humor": +0.10},
    CommunicationIntent.REPORT:     {"formality": +0.15, "precision": +0.20},
}


# ── PersonaVectorEngine ───────────────────────────────────────────────────────

class PersonaVectorEngine:
    """
    Builds PersonaVectors from composited inputs.

    This engine is intentionally stateless — it produces a fresh vector
    from inputs on every call. Communication Stability (smoothing against
    the previous vector) is handled in the TransformationPipeline, not here.

    Composition order:
        1. Start with base_vector
        2. Apply relationship context delta
        3. Apply communication intent delta
        4. Apply overlay delta(s) at their weight(s)
        5. Apply expression confidence scaling
        6. Clamp all to [0.0, 1.0]
    """

    def build(
        self,
        base: PersonaVector,
        relationship: RelationshipContext = RelationshipContext.UNKNOWN,
        intent: CommunicationIntent = CommunicationIntent.INFORM,
        overlay: Optional[OverlayType] = None,
        overlay_weight: float = 1.0,
        expression_confidence: float = 1.0,
    ) -> PersonaVector:
        """
        Compose a PersonaVector from base + context + single overlay.

        Args:
            base:                  Starting trait values (from persona profile)
            relationship:          Relationship context for this user
            intent:                Communication intent for this response
            overlay:               Optional situational modifier
            overlay_weight:        How strongly to apply the overlay (0.0–1.0)
            expression_confidence: Global attenuation scalar — applied last.
                                   Derived from CommunicationRulesEngine.max_expression.

        Returns:
            A clamped PersonaVector ready for downstream processing.
        """
        result = PersonaVector(**base.to_dict())

        # Layer 2: relationship
        result = result.add_delta(_RELATIONSHIP_MODIFIERS.get(relationship, {}))

        # Layer 3: intent
        result = result.add_delta(_INTENT_MODIFIERS.get(intent, {}))

        # Layer 4: single overlay
        if overlay is not None:
            w = max(0.0, min(1.0, overlay_weight))
            weighted_delta = {
                t: d * w
                for t, d in _OVERLAY_DELTAS.get(overlay, {}).items()
            }
            result = result.add_delta(weighted_delta)

        # Layer 5: expression confidence attenuation
        if expression_confidence < 1.0:
            result = result.scale(max(0.0, min(1.0, expression_confidence)))

        return result.clamp()

    def build_blended(
        self,
        base: PersonaVector,
        overlays: Dict[OverlayType, float],
        relationship: RelationshipContext = RelationshipContext.UNKNOWN,
        intent: CommunicationIntent = CommunicationIntent.INFORM,
        expression_confidence: float = 1.0,
    ) -> PersonaVector:
        """
        Build a PersonaVector with multiple blended overlays.

        Each overlay in `overlays` is applied independently at its weight.
        Weights do not need to sum to 1.0.

        Example from spec (Teaching: 70%, Focused: 30%):
            overlays = {
                OverlayType.TEACHING: 0.70,
                OverlayType.FOCUSED:  0.30,
            }
        """
        result = PersonaVector(**base.to_dict())

        result = result.add_delta(_RELATIONSHIP_MODIFIERS.get(relationship, {}))
        result = result.add_delta(_INTENT_MODIFIERS.get(intent, {}))

        for overlay, weight in overlays.items():
            w = max(0.0, min(1.0, weight))
            weighted_delta = {
                t: d * w
                for t, d in _OVERLAY_DELTAS.get(overlay, {}).items()
            }
            result = result.add_delta(weighted_delta)

        if expression_confidence < 1.0:
            result = result.scale(max(0.0, min(1.0, expression_confidence)))

        return result.clamp()
