"""
core/trait_conflict_resolver.py — DarkPassenger Trait Conflict Resolver

Resolves trait tensions within a PersonaVector so the resulting personality
is internally consistent rather than a contradictory blend of incompatible styles.

Problem (spec §7)
─────────────────
A PersonaVector can contain trait combinations that work against each other:

    Directness vs. Verbosity        — being terse and expansive simultaneously
    Formality vs. Playfulness       — stiff and irreverent at the same time
    Precision vs. Simplicity        — hyper-exact and approachable simultaneously
    Humor vs. Professionalism       — cracking jokes and acting CEO-serious

Left unresolved, the downstream speech fingerprint would receive contradictory
signals and produce incoherent output.

Solution
────────
A priority hierarchy determines the winner for each conflict pair.
Context modifiers (overlay, criticality level) can elevate a trait's effective
priority before resolution runs.

Priority rules (deterministic, applied in order):
    1. EMERGENCY context → directness and precision always win.
       All opposing traits are suppressed to 0.0.

    2. Conflict pairs — for each pair, the higher-priority trait wins.
       The loser is reduced by a conflict_attenuation factor applied to the
       fraction by which the loser exceeds the winner:

           loser_adjusted = max(floor, loser - attenuation × (loser - winner))

       This never raises the winner and never floors the loser below a minimum.
       It narrows the gap rather than hard-zeroing, which is less jarring.

    3. Expression Budget enforcement — traits with 0-budget allocation are
       floored to 0.0 (they have no room to express).

Conflict pair table (spec §7, with spec examples rendered):
    Pair                            Priority winner     Attenuation
    ──────────────────────────────  ──────────────────  ───────────
    directness ↔ analytical_depth  directness          0.60
    formality  ↔ humor             formality           0.55
    precision  ↔ curiosity         precision           0.60
    professionalism ↔ humor        professionalism     0.50
    technicality ↔ warmth          technicality        0.40

The priority winner is the trait that "wins" when both are high.  The loser
is attenuated — not zeroed — so some character remains.

Spec reference: DarkPassenger-Plan.txt §7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.persona_vector import ExpressionBudget, OverlayType, PersonaVector
from core.runtime_state import RuntimeState


# ── Conflict pair definitions ─────────────────────────────────────────────────
#
# Each entry: (winner_trait, loser_trait, attenuation_factor, floor)
#
#   winner_trait:      The trait that takes priority when both are elevated.
#   loser_trait:       The trait that is attenuated when it exceeds the winner.
#   attenuation:       0.0–1.0; how aggressively the loser is pulled down.
#                      0.5 = close gap by half; 1.0 = set loser = winner (hard cap)
#   floor:             Minimum value the loser may be reduced to (0.0–1.0).
#                      Prevents over-suppression in mild conflicts.

_CONFLICT_PAIRS: List[Tuple[str, str, float, float]] = [
    # winner               loser                  attenuation  floor
    ("directness",         "analytical_depth",    0.60,        0.10),
    ("formality",          "humor",               0.55,        0.05),
    ("precision",          "curiosity",           0.60,        0.10),
    ("professionalism",    "humor",               0.50,        0.05),
    ("technicality",       "warmth",              0.40,        0.15),
]

# Traits that receive maximum boost in emergency context
_EMERGENCY_PRIORITY_TRAITS = frozenset({"directness", "precision"})

# Traits suppressed in emergency context (set to 0.0)
_EMERGENCY_SUPPRESSED_TRAITS = frozenset({
    "humor", "warmth", "curiosity", "formality",
    "technicality", "analytical_depth", "professionalism",
})

# Minimum trait value regardless of conflict outcome
_GLOBAL_FLOOR: float = 0.0


# ── ConflictResolutionRecord ──────────────────────────────────────────────────

@dataclass
class TraitAdjustment:
    """Records a single trait modification during conflict resolution."""
    trait:      str
    before:     float
    after:      float
    reason:     str


@dataclass
class ConflictResolutionResult:
    """
    Full diagnostics from one TraitConflictResolver.resolve() call.

    Attributes
    ──────────
    output_vector:
        The resolved PersonaVector with all conflicts addressed.

    adjustments:
        List of every trait modification made, in application order.

    emergency_active:
        True if Emergency context suppression was triggered.

    conflicts_found:
        Number of conflict pairs where the loser exceeded the winner.

    budget_enforced:
        True if any traits were zeroed out due to zero budget allocation.
    """
    output_vector:    PersonaVector
    adjustments:      List[TraitAdjustment] = field(default_factory=list)
    emergency_active: bool                  = False
    conflicts_found:  int                   = 0
    budget_enforced:  bool                  = False


# ── TraitConflictResolver ─────────────────────────────────────────────────────

class TraitConflictResolver:
    """
    Resolves trait conflicts in a PersonaVector using a deterministic
    priority hierarchy.

    Construction:
        resolver = TraitConflictResolver()

    Usage (called as conflict_hook in TransformationPipeline):
        resolved = resolver(vector, budget, state)

    Or with full diagnostics:
        result = resolver.resolve(vector, budget, state)
        resolved_vector = result.output_vector
    """

    def __init__(self, logger=None):
        self._logger = logger

    # ── Public callable interface (matches ConflictHookFn signature) ──────────

    def __call__(
        self,
        vector: PersonaVector,
        budget: ExpressionBudget,
        state:  RuntimeState,
    ) -> PersonaVector:
        """
        Minimal hook-compatible interface.

        Returns the conflict-resolved PersonaVector.
        For full diagnostics use resolve().
        """
        return self.resolve(vector, budget, state).output_vector

    # ── Full resolution interface ─────────────────────────────────────────────

    def resolve(
        self,
        vector: PersonaVector,
        budget: ExpressionBudget,
        state:  RuntimeState,
    ) -> ConflictResolutionResult:
        """
        Full conflict resolution with diagnostics.

        Stages (in order):
            1. Emergency suppression  — if overlay is EMERGENCY, hard-suppress
                                        conflicting traits before pair resolution.
            2. Pair resolution        — for each conflict pair, attenuate the
                                        loser if it exceeds the winner.
            3. Budget enforcement     — zero out traits with 0-budget allocation.
            4. Clamp                  — ensure all values are in [0.0, 1.0].

        Args:
            vector: The PersonaVector to resolve (not mutated — a copy is made).
            budget: The active ExpressionBudget (used for zero-budget zeroing).
            state:  The current RuntimeState (used to detect emergency context).

        Returns:
            ConflictResolutionResult with the resolved vector and full diagnostics.
        """
        adjustments:      List[TraitAdjustment] = []
        emergency_active: bool                  = False
        budget_enforced:  bool                  = False
        conflicts_found:  int                   = 0

        # Work on a mutable copy of trait values
        traits = {t: getattr(vector, t) for t in PersonaVector.trait_names()}

        is_emergency = self._is_emergency(state)

        # ── Stage 1: Emergency suppression ───────────────────────────────────
        if is_emergency:
            emergency_active = True
            for trait in _EMERGENCY_SUPPRESSED_TRAITS:
                if trait in traits and traits[trait] > 0.0:
                    adjustments.append(TraitAdjustment(
                        trait=trait,
                        before=traits[trait],
                        after=0.0,
                        reason="emergency_suppression",
                    ))
                    traits[trait] = 0.0

        # ── Stage 2: Conflict pair resolution ────────────────────────────────
        for winner_trait, loser_trait, attenuation, floor in _CONFLICT_PAIRS:
            if winner_trait not in traits or loser_trait not in traits:
                continue  # trait not in this vector; skip

            winner_val = traits[winner_trait]
            loser_val  = traits[loser_trait]

            # Conflict only applies when the loser is meaningfully above the winner
            if loser_val <= winner_val:
                continue  # no conflict — loser is already below winner

            conflicts_found += 1

            # Attenuation: close the gap by `attenuation` fraction
            gap         = loser_val - winner_val
            reduction   = attenuation * gap
            new_loser   = max(floor, loser_val - reduction)

            if abs(new_loser - loser_val) > 1e-9:
                adjustments.append(TraitAdjustment(
                    trait=loser_trait,
                    before=loser_val,
                    after=new_loser,
                    reason=(
                        f"conflict:{winner_trait}(winner={winner_val:.3f})"
                        f">{loser_trait}(loser={loser_val:.3f})"
                        f":attenuation={attenuation}"
                    ),
                ))
                traits[loser_trait] = new_loser

        # ── Stage 3: Budget enforcement ───────────────────────────────────────
        for trait in PersonaVector.trait_names():
            if budget.allocations and budget.effective_weight(trait) == 0.0:
                if traits.get(trait, 0.0) > 0.0:
                    adjustments.append(TraitAdjustment(
                        trait=trait,
                        before=traits[trait],
                        after=0.0,
                        reason="zero_budget_allocation",
                    ))
                    traits[trait] = 0.0
                    budget_enforced = True

        # ── Stage 4: Clamp ────────────────────────────────────────────────────
        for trait in traits:
            traits[trait] = max(_GLOBAL_FLOOR, min(1.0, traits[trait]))

        resolved = PersonaVector(**{
            t: traits.get(t, getattr(vector, t))
            for t in PersonaVector.trait_names()
        })

        if adjustments:
            self._log(
                f"trait_conflict_resolver: {len(adjustments)} adjustment(s), "
                f"{conflicts_found} conflict(s) found, "
                f"emergency={emergency_active}"
            )

        return ConflictResolutionResult(
            output_vector=resolved,
            adjustments=adjustments,
            emergency_active=emergency_active,
            conflicts_found=conflicts_found,
            budget_enforced=budget_enforced,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_emergency(state: RuntimeState) -> bool:
        """
        Return True if the current state warrants emergency-level conflict rules.

        Triggers:
            - current_overlay is EMERGENCY
            - EMERGENCY is present in any blended overlay
        """
        if state.current_overlay == OverlayType.EMERGENCY:
            return True
        if OverlayType.EMERGENCY in state.current_overlay_blends:
            return True
        return False

    def _log(self, message: str) -> None:
        if self._logger is not None:
            try:
                self._logger.debug(message)
            except Exception:
                pass
