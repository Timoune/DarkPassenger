"""
core/performance.py — DarkPassenger Performance & Latency Manager

Implements the two latency optimisation strategies described in spec §17:

    1. Fast-Path Execution
       For responses whose context has not meaningfully changed since the
       last pipeline run, skip expensive stages (stability check, conflict
       resolution, fingerprint application) and return a cached or lightly
       re-applied result. The fast-path is ONLY taken when it is safe to do
       so — integrity validation always runs.

    2. Persona Profile Cache
       Resolved PersonaProfiles are expensive to reload from disk on every
       response (JSON parse + schema validation). The PersonaProfileCache
       holds a fixed number of recently-used profiles in memory with a
       configurable TTL, ensuring sub-millisecond profile access after the
       first load.

    3. Overlay Configuration Cache
       Computed overlay modifier dicts (used by PersonaVectorEngine) are
       cached by (overlay_type, expression_confidence) key. This avoids
       re-computing the same modifier math for every response in a
       steady-state context.

Architecture
────────────
PerformanceManager is constructed once and injected into
TransformationPipeline. The pipeline queries should_fast_path() before
running stages 4-8, and calls record_result() after every run to keep
the fast-path heuristic accurate.

Fast-path eligibility criteria (all must hold):
    ✓ Same overlay as the previous response
    ✓ Same relationship context as the previous response
    ✓ Same intent as the previous response
    ✓ Expression confidence within FAST_PATH_CONF_TOLERANCE of previous
    ✓ No pipeline_warnings in the previous result
    ✓ override_active was False in the previous result
    ✓ At least FAST_PATH_MIN_HISTORY successful results in the history

The fast-path produces a FastPathResult (not a full TransformationResult).
The caller is responsible for wrapping it in a minimal TransformationResult
before returning.

Spec reference: DarkPassenger-Plan.txt §17
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.persona_vector import (
    PersonaVector,
    ExpressionBudget,
    OverlayType,
    RelationshipContext,
    CommunicationIntent,
)


# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum absolute difference in expression_confidence before fast-path is
# ineligible. A change of ≥0.05 means something upstream changed meaningfully.
FAST_PATH_CONF_TOLERANCE: float = 0.05

# Minimum successful results in history before fast-path is ever attempted.
FAST_PATH_MIN_HISTORY: int = 2

# Default persona profile TTL in seconds.
DEFAULT_PROFILE_TTL: float = 300.0  # 5 minutes

# Maximum profiles held in memory at once.
DEFAULT_PROFILE_CACHE_SIZE: int = 16

# Maximum overlay configs held in memory.
DEFAULT_OVERLAY_CACHE_SIZE: int = 64


# ── FastPathResult ────────────────────────────────────────────────────────────

@dataclass
class FastPathResult:
    """
    The output of a fast-path pipeline execution.

    This is a lightweight substitute for a full TransformationResult.
    The final_output is re-used from the previous response verbatim — the
    caller applies any minimal text delta (e.g. swapping in the new content
    string) before returning it.

    Fields
    ──────
    eligible:
        True if the fast-path was taken. False means the caller must run
        the full pipeline.

    persona_vector:
        Re-used PersonaVector from the last full-pipeline run.

    expression_budget:
        Re-used ExpressionBudget from the last full-pipeline run.

    expression_confidence:
        Re-used confidence scalar.

    stages_skipped:
        Names of pipeline stages that were bypassed.

    reason:
        Human-readable explanation of why the fast-path was (or was not) taken.
    """
    eligible:             bool
    persona_vector:       Optional[PersonaVector]   = None
    expression_budget:    Optional[ExpressionBudget] = None
    expression_confidence: float                    = 0.0
    stages_skipped:       List[str]                 = field(default_factory=list)
    reason:               str                       = ""


# ── _ContextSnapshot ─────────────────────────────────────────────────────────
# Internal — captures the key context fields from a result for comparison.

@dataclass
class _ContextSnapshot:
    overlay:              str
    relationship:         str
    intent:               str
    expression_confidence: float
    had_warnings:         bool
    override_active:      bool
    persona_vector:       PersonaVector
    expression_budget:    ExpressionBudget


# ── PersonaProfileCache ───────────────────────────────────────────────────────

class PersonaProfileCache:
    """
    LRU-TTL cache for resolved PersonaProfile objects.

    When the TransformationPipeline resolves a profile by ID, it first
    checks this cache. On a miss it loads from disk (via ConfigManager),
    stores the result, and returns it. On a hit it returns instantly.

    TTL expiry ensures that on-disk edits are eventually picked up without
    requiring a restart.

    Usage:
        cache = PersonaProfileCache(ttl_seconds=300, max_size=16)
        profile = cache.get("default")         # returns None on miss
        cache.set("default", resolved_profile)
        cache.invalidate("default")            # force reload on next get
        cache.clear()                          # wipe entire cache
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_PROFILE_TTL,
        max_size: int = DEFAULT_PROFILE_CACHE_SIZE,
        logger=None,
    ):
        self._ttl     = ttl_seconds
        self._max     = max_size
        self._logger  = logger

        # profile_id → (profile_object, expiry_timestamp)
        self._store: Dict[str, Tuple[object, float]] = {}
        # Insertion-order list for LRU eviction
        self._order: List[str] = []

        # Statistics
        self.hits:   int = 0
        self.misses: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, profile_id: str) -> Optional[object]:
        """
        Return the cached profile for profile_id, or None on miss/expiry.
        """
        entry = self._store.get(profile_id)
        if entry is None:
            self.misses += 1
            return None

        profile, expiry = entry
        if time.monotonic() > expiry:
            # Expired
            self._evict(profile_id)
            self.misses += 1
            if self._logger:
                self._logger.debug("profile_cache_expired", profile_id=profile_id)
            return None

        # Hit — move to most-recently-used position
        self._touch(profile_id)
        self.hits += 1

        if self._logger:
            self._logger.debug("profile_cache_hit", profile_id=profile_id)
        return profile

    def set(self, profile_id: str, profile: object) -> None:
        """
        Store a profile. Evicts the LRU entry if the cache is at capacity.
        """
        if profile_id in self._store:
            self._touch(profile_id)
        else:
            if len(self._store) >= self._max:
                self._evict_lru()
            self._order.append(profile_id)

        expiry = time.monotonic() + self._ttl
        self._store[profile_id] = (profile, expiry)

        if self._logger:
            self._logger.debug(
                "profile_cache_set",
                profile_id=profile_id,
                ttl=self._ttl,
                cache_size=len(self._store),
            )

    def invalidate(self, profile_id: str) -> None:
        """Force a reload on next get() for profile_id."""
        self._evict(profile_id)

    def clear(self) -> None:
        """Wipe all cached entries."""
        self._store.clear()
        self._order.clear()
        if self._logger:
            self._logger.debug("profile_cache_cleared")

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _touch(self, profile_id: str) -> None:
        """Move profile_id to end of LRU order (most recently used)."""
        try:
            self._order.remove(profile_id)
        except ValueError:
            pass
        self._order.append(profile_id)

    def _evict(self, profile_id: str) -> None:
        self._store.pop(profile_id, None)
        try:
            self._order.remove(profile_id)
        except ValueError:
            pass

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry."""
        if self._order:
            lru_id = self._order[0]
            self._evict(lru_id)
            if self._logger:
                self._logger.debug("profile_cache_lru_evicted", evicted=lru_id)


# ── OverlayConfigCache ────────────────────────────────────────────────────────

class OverlayConfigCache:
    """
    Cache for computed overlay modifier configurations.

    The PersonaVectorEngine computes modifier dicts from (overlay_type,
    expression_confidence) pairs. For any stable conversational context —
    same overlay, same confidence tier — these computations are identical.
    Caching them saves repeated floating-point math.

    Key: (overlay_type_value: str, confidence_bucket: int)
         where confidence_bucket = int(confidence * 20)  [5% buckets]

    Value: the modifier dict returned by the overlay computation.

    The 5% confidence bucket ensures that minor fluctuations in confidence
    (< 0.05) reuse the cached modifier, while genuine shifts (>= 0.05) get
    a fresh computation.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_OVERLAY_CACHE_SIZE,
        logger=None,
    ):
        self._max    = max_size
        self._logger = logger
        self._store: Dict[Tuple[str, int], dict] = {}
        self._order: List[Tuple[str, int]]       = []

        self.hits:   int = 0
        self.misses: int = 0

    @staticmethod
    def _key(overlay: str, confidence: float) -> Tuple[str, int]:
        """Bucket confidence into 5% intervals for cache key."""
        bucket = int(round(confidence * 20))  # 0..20
        return (overlay, bucket)

    def get(self, overlay: str, confidence: float) -> Optional[dict]:
        k = self._key(overlay, confidence)
        val = self._store.get(k)
        if val is None:
            self.misses += 1
            return None
        self._touch(k)
        self.hits += 1
        return val

    def set(self, overlay: str, confidence: float, modifier: dict) -> None:
        k = self._key(overlay, confidence)
        if k not in self._store:
            if len(self._store) >= self._max:
                self._evict_lru()
            self._order.append(k)
        self._store[k] = modifier
        self._touch(k)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def _touch(self, k) -> None:
        try:
            self._order.remove(k)
        except ValueError:
            pass
        self._order.append(k)

    def _evict_lru(self) -> None:
        if self._order:
            lru = self._order[0]
            self._store.pop(lru, None)
            self._order.pop(0)


# ── PerformanceManager ────────────────────────────────────────────────────────

class PerformanceManager:
    """
    Central performance coordinator for the TransformationPipeline.

    Owns:
        - PersonaProfileCache     (profile resolution fast-path)
        - OverlayConfigCache      (overlay modifier fast-path)
        - Fast-path heuristic     (skip stages 4-8 when context is stable)

    The pipeline calls:
        pm.should_fast_path(overlay, relationship, intent, confidence)
            → FastPathResult (eligible=True/False + cached state if eligible)

        pm.record_result(result)
            → updates heuristic history with the latest pipeline output

    Statistics are available via pm.stats() for monitoring and the
    Personality Audit Log.

    Usage in TransformationPipeline.__init__:
        self._perf = PerformanceManager(logger=logger)
        self._perf.profile_cache.set(profile_id, resolved_profile)

    Usage in TransformationPipeline.transform:
        fp = self._perf.should_fast_path(overlay, relationship, intent, conf)
        if fp.eligible:
            # skip stages 4-8; use fp.persona_vector, fp.expression_budget
            ...
        else:
            # run full pipeline
            ...
        self._perf.record_result(result)
    """

    def __init__(
        self,
        profile_ttl: float   = DEFAULT_PROFILE_TTL,
        max_profiles: int    = DEFAULT_PROFILE_CACHE_SIZE,
        max_overlays: int    = DEFAULT_OVERLAY_CACHE_SIZE,
        min_history: int     = FAST_PATH_MIN_HISTORY,
        conf_tolerance: float = FAST_PATH_CONF_TOLERANCE,
        logger=None,
    ):
        self._logger          = logger
        self._min_history     = min_history
        self._conf_tolerance  = conf_tolerance

        self.profile_cache = PersonaProfileCache(
            ttl_seconds=profile_ttl,
            max_size=max_profiles,
            logger=logger,
        )
        self.overlay_cache = OverlayConfigCache(
            max_size=max_overlays,
            logger=logger,
        )

        # Ring buffer of the last N context snapshots from full pipeline runs
        self._history: List[_ContextSnapshot] = []
        self._history_max: int = 20

        # Counters
        self._fast_path_taken:  int = 0
        self._fast_path_skipped: int = 0

    # ── Fast-path API ─────────────────────────────────────────────────────────

    def should_fast_path(
        self,
        overlay:              str,
        relationship:         str,
        intent:               str,
        expression_confidence: float,
    ) -> FastPathResult:
        """
        Determine whether stages 4-8 can be skipped for this response.

        Returns a FastPathResult. If eligible=True, the caller can use
        the bundled persona_vector and expression_budget directly.

        Args:
            overlay:               Current overlay label (OverlayType.value or blend JSON)
            relationship:          Current RelationshipContext.value
            intent:                Current CommunicationIntent.value
            expression_confidence: Current expression confidence scalar

        Returns:
            FastPathResult with eligible=True iff all fast-path criteria are met.
        """
        if len(self._history) < self._min_history:
            return FastPathResult(
                eligible=False,
                reason=f"Insufficient history ({len(self._history)}/{self._min_history})",
            )

        prev = self._history[-1]

        # Criterion 1: same overlay
        if overlay != prev.overlay:
            self._fast_path_skipped += 1
            return FastPathResult(
                eligible=False,
                reason=f"Overlay changed: '{prev.overlay}' → '{overlay}'",
            )

        # Criterion 2: same relationship
        if relationship != prev.relationship:
            self._fast_path_skipped += 1
            return FastPathResult(
                eligible=False,
                reason=f"Relationship changed: '{prev.relationship}' → '{relationship}'",
            )

        # Criterion 3: same intent
        if intent != prev.intent:
            self._fast_path_skipped += 1
            return FastPathResult(
                eligible=False,
                reason=f"Intent changed: '{prev.intent}' → '{intent}'",
            )

        # Criterion 4: confidence within tolerance
        conf_delta = abs(expression_confidence - prev.expression_confidence)
        if conf_delta > self._conf_tolerance:
            self._fast_path_skipped += 1
            return FastPathResult(
                eligible=False,
                reason=(
                    f"Confidence shifted by {conf_delta:.3f} "
                    f"(tolerance {self._conf_tolerance})"
                ),
            )

        # Criterion 5: previous run had no warnings
        if prev.had_warnings:
            self._fast_path_skipped += 1
            return FastPathResult(
                eligible=False,
                reason="Previous response had pipeline warnings; running full pipeline.",
            )

        # Criterion 6: no override active in previous run
        if prev.override_active:
            self._fast_path_skipped += 1
            return FastPathResult(
                eligible=False,
                reason="Previous response had override_active; running full pipeline.",
            )

        # All criteria met — fast-path eligible
        self._fast_path_taken += 1

        skipped_stages = [
            "stability_check",
            "expression_confidence_attenuation",
            "expression_budget_allocation",
            "trait_conflict_resolution",
            "speech_fingerprint",
        ]

        if self._logger:
            self._logger.debug(
                "fast_path_taken",
                overlay=overlay,
                relationship=relationship,
                intent=intent,
                confidence=expression_confidence,
            )

        return FastPathResult(
            eligible=True,
            persona_vector=prev.persona_vector,
            expression_budget=prev.expression_budget,
            expression_confidence=prev.expression_confidence,
            stages_skipped=skipped_stages,
            reason="Context stable; reusing previous persona state.",
        )

    def record_result(self, result: "TransformationResult") -> None:  # type: ignore[name-defined]
        """
        Update the fast-path history with the output of a full pipeline run.

        Should be called after every non-fast-path transform() invocation.
        Fast-path results should NOT be recorded here (they produce no new state).
        """
        overlay_label = _overlay_from_vector(result.persona_vector)
        # Use the audit-ready fields stamped by the pipeline first; fall back to
        # getattr for custom integrations that store context on the vector.
        relationship = (
            getattr(result, "relationship", None)
            or getattr(result.persona_vector, "relationship_context", None)
            or "unknown"
        )
        intent = (
            getattr(result, "intent", None)
            or getattr(result.persona_vector, "communication_intent", None)
            or "unknown"
        )

        snap = _ContextSnapshot(
            overlay=overlay_label,
            relationship=relationship,
            intent=intent,
            expression_confidence=result.expression_confidence,
            had_warnings=bool(result.pipeline_warnings),
            override_active=result.override_active,
            persona_vector=result.persona_vector,
            expression_budget=result.expression_budget,
        )

        self._history.append(snap)
        if len(self._history) > self._history_max:
            self._history.pop(0)

    def invalidate_fast_path(self) -> None:
        """
        Flush the fast-path history.

        Call this when the persona profile changes at runtime so the next
        response always runs the full pipeline.
        """
        self._history.clear()
        if self._logger:
            self._logger.debug("fast_path_history_invalidated")

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return a snapshot of all performance counters.

        Suitable for inclusion in AuditLog records or the Personality
        Audit Log's session summary.
        """
        fp_total = self._fast_path_taken + self._fast_path_skipped
        return {
            "fast_path_taken":       self._fast_path_taken,
            "fast_path_skipped":     self._fast_path_skipped,
            "fast_path_rate":        round(
                self._fast_path_taken / fp_total if fp_total else 0.0, 4
            ),
            "profile_cache_size":    self.profile_cache.size,
            "profile_cache_hits":    self.profile_cache.hits,
            "profile_cache_misses":  self.profile_cache.misses,
            "profile_cache_hit_rate": round(self.profile_cache.hit_rate, 4),
            "overlay_cache_size":    self.overlay_cache.size,
            "overlay_cache_hits":    self.overlay_cache.hits,
            "overlay_cache_misses":  self.overlay_cache.misses,
            "overlay_cache_hit_rate": round(self.overlay_cache.hit_rate, 4),
            "history_depth":         len(self._history),
        }

    def reset_stats(self) -> None:
        """Reset all counters (does not clear caches or history)."""
        self._fast_path_taken   = 0
        self._fast_path_skipped = 0
        self.profile_cache.hits   = 0
        self.profile_cache.misses = 0
        self.overlay_cache.hits   = 0
        self.overlay_cache.misses = 0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _overlay_from_vector(vector: "PersonaVector") -> str:  # type: ignore[name-defined]
    """Extract a string overlay label from a PersonaVector."""
    try:
        blends = getattr(vector, "overlay_blends", None)
        if blends:
            import json
            return json.dumps(
                {k: round(float(w), 3) for k, w in blends.items()},
                sort_keys=True,
            )
        overlay = getattr(vector, "overlay", None)
        if overlay is not None:
            return overlay.value if hasattr(overlay, "value") else str(overlay)
    except Exception:
        pass
    return "none"
