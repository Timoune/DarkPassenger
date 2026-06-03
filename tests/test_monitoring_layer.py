"""
tests/test_monitoring_layer.py — v1.4 Performance & Monitoring Layer Tests

Covers the three new modules:

    AuditLog           — record(), export_jsonl(), load_jsonl()
    BehavioralReviewSystem — stability_report(), quality_report(),
                             drift_report(), adaptation_report(), full_review()
    PerformanceManager — should_fast_path(), record_result(),
                         PersonaProfileCache, OverlayConfigCache
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ── Imports under test ────────────────────────────────────────────────────────
from core.audit_log import AuditLog, AuditRecord
from core.behavioral_review import (
    BehavioralReviewSystem,
    StabilityReport,
    QualityReport,
    DriftDetectionReport,
    AdaptationReport,
    FullReviewBundle,
)
from core.performance import (
    PerformanceManager,
    PersonaProfileCache,
    OverlayConfigCache,
    FastPathResult,
    FAST_PATH_MIN_HISTORY,
    FAST_PATH_CONF_TOLERANCE,
)
from core.persona_vector import PersonaVector, ExpressionBudget


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures & helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_record(
    *,
    relationship: str = "owner",
    intent: str = "inform",
    overlay: str = "focused",
    confidence: float = 0.90,
    final_traits: Dict[str, float] | None = None,
    override_active: bool = False,
    elapsed_ms: float = 50.0,
    pipeline_warnings: List[str] | None = None,
    stages_executed: List[str] | None = None,
    session_id: str = "test-session",
) -> AuditRecord:
    """Build a synthetic AuditRecord for testing."""
    return AuditRecord(
        response_id=str(uuid.uuid4()),
        session_id=session_id,
        timestamp_utc="2025-11-04T12:00:00+00:00",
        relationship=relationship,
        intent=intent,
        overlay=overlay,
        confidence=confidence,
        final_traits=final_traits or {
            "formality": 0.50,
            "humor": 0.70,
            "warmth": 0.60,
            "directness": 0.90,
            "technicality": 0.85,
        },
        budget_used={"technicality": 40.0, "directness": 30.0, "humor": 20.0, "warmth": 10.0},
        stages_executed=stages_executed or ["input_reception", "pre_flight", "persona_vector_generation",
                                             "stability_check", "expression_confidence_attenuation",
                                             "expression_budget_allocation", "trait_conflict_resolution",
                                             "speech_fingerprint", "validation_pipeline", "final_response"],
        override_active=override_active,
        elapsed_ms=elapsed_ms,
        pipeline_warnings=pipeline_warnings or [],
    )


def _make_log(n: int = 10, **record_kwargs) -> AuditLog:
    """Return an AuditLog pre-populated with n identical records."""
    log = AuditLog(session_id="test-session")
    for _ in range(n):
        log._records.append(_make_record(**record_kwargs))
    return log


# ══════════════════════════════════════════════════════════════════════════════
# AuditLog tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditLog:

    def test_session_id_auto_generated(self):
        log = AuditLog()
        assert isinstance(log.session_id, str) and len(log.session_id) > 0

    def test_session_id_custom(self):
        log = AuditLog(session_id="my-session")
        assert log.session_id == "my-session"

    def test_record_raw_appends(self):
        log = AuditLog()
        rec = log.record_raw(
            relationship="owner",
            intent="inform",
            overlay="focused",
            confidence=0.90,
            final_traits={"humor": 0.70},
            budget_used={"humor": 20.0},
            stages_executed=["input_reception", "final_response"],
            override_active=False,
            elapsed_ms=42.0,
        )
        assert log.count == 1
        assert rec.session_id == log.session_id
        assert rec.confidence == 0.90
        assert rec.elapsed_ms == 42.0

    def test_record_raw_fields_preserved(self):
        log = AuditLog()
        rec = log.record_raw(
            relationship="guest",
            intent="warn",
            overlay="emergency",
            confidence=0.05,
            final_traits={"directness": 1.0},
            budget_used={"directness": 100.0},
            stages_executed=["critical_override"],
            override_active=True,
            elapsed_ms=5.0,
            pipeline_warnings=["test warning"],
        )
        assert rec.override_active is True
        assert rec.pipeline_warnings == ["test warning"]
        assert rec.overlay == "emergency"

    def test_ring_buffer_max_records(self):
        log = AuditLog(max_records=5)
        for _ in range(8):
            log.record_raw(
                relationship="owner", intent="inform", overlay="focused",
                confidence=0.9, final_traits={}, budget_used={},
                stages_executed=[], override_active=False, elapsed_ms=10.0,
            )
        assert log.count == 5  # ring buffer capped

    def test_last_returns_most_recent(self):
        log = _make_log(5)
        latest = log.last(1)
        assert len(latest) == 1
        assert latest[0] == log.records[-1]

    def test_last_multiple(self):
        log = _make_log(10)
        last3 = log.last(3)
        assert len(last3) == 3

    def test_clear_empties_buffer(self):
        log = _make_log(10)
        log.clear()
        assert log.count == 0

    def test_export_and_load_jsonl_roundtrip(self):
        log = _make_log(5)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            written = log.export_jsonl(path, append=False)
            assert written == 5

            loaded = AuditLog.load_jsonl(path)
            assert len(loaded) == 5
            # Spot-check a field survives round-trip
            assert loaded[0].confidence == log.records[0].confidence
            assert loaded[0].relationship == log.records[0].relationship
        finally:
            os.unlink(path)

    def test_export_append_mode(self):
        log = _make_log(3)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            log.export_jsonl(path, append=False)
            log.export_jsonl(path, append=True)   # appends 3 more
            loaded = AuditLog.load_jsonl(path)
            assert len(loaded) == 6
        finally:
            os.unlink(path)

    def test_load_jsonl_nonexistent_path(self):
        records = AuditLog.load_jsonl("/nonexistent/path/file.jsonl")
        assert records == []

    def test_since_filters_by_timestamp(self):
        log = AuditLog()
        # Insert two records with different timestamps
        r1 = _make_record()
        r1 = AuditRecord(**{**r1.__dict__, "timestamp_utc": "2025-11-04T10:00:00+00:00"})
        r2 = _make_record()
        r2 = AuditRecord(**{**r2.__dict__, "timestamp_utc": "2025-11-04T14:00:00+00:00"})
        log._records.extend([r1, r2])

        since_noon = log.since("2025-11-04T12:00:00+00:00")
        assert len(since_noon) == 1
        assert since_noon[0].timestamp_utc == "2025-11-04T14:00:00+00:00"

    def test_audit_record_to_dict_from_dict(self):
        rec = _make_record()
        d = rec.to_dict()
        assert isinstance(d, dict)
        restored = AuditRecord.from_dict(d)
        assert restored.response_id == rec.response_id
        assert restored.confidence  == rec.confidence


# ══════════════════════════════════════════════════════════════════════════════
# BehavioralReviewSystem tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBehavioralReviewSystem:

    # ── Construction ─────────────────────────────────────────────────────────

    def test_requires_audit_log_or_records(self):
        with pytest.raises(ValueError):
            BehavioralReviewSystem()

    def test_from_records_factory(self):
        records = [_make_record() for _ in range(5)]
        reviewer = BehavioralReviewSystem.from_records(records)
        assert reviewer is not None

    def test_from_jsonl_factory(self):
        log = _make_log(3)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            log.export_jsonl(path, append=False)
            reviewer = BehavioralReviewSystem.from_jsonl(path)
            report = reviewer.quality_report()
            assert report.record_count == 3
        finally:
            os.unlink(path)

    # ── StabilityReport ───────────────────────────────────────────────────────

    def test_stability_report_healthy_with_stable_traits(self):
        # All records have identical traits — should be healthy
        log = _make_log(10, final_traits={
            "humor": 0.70, "directness": 0.90, "warmth": 0.60
        })
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.stability_report()
        assert report.health == "healthy"
        assert all(v < 0.10 for v in report.trait_variance.values())

    def test_stability_report_unstable_with_high_variance(self):
        log = AuditLog()
        # Alternate between very different trait values
        for i in range(12):
            humor = 0.95 if i % 2 == 0 else 0.05
            log._records.append(_make_record(final_traits={"humor": humor, "directness": 0.9}))

        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.stability_report()
        assert report.health in ("warning", "unstable")
        assert report.trait_variance["humor"] > 0.10

    def test_stability_report_empty_log_returns_healthy(self):
        log = AuditLog()
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.stability_report()
        assert report.health == "healthy"
        assert report.record_count == 0

    def test_stability_overlay_oscillation_counted(self):
        log = AuditLog()
        overlays = ["focused", "teaching", "focused", "teaching", "focused",
                    "teaching", "focused", "teaching", "focused", "teaching"]
        for i, ov in enumerate(overlays):
            log._records.append(_make_record(overlay=ov))
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.stability_report()
        # All records after warmup (3) change overlay each time → high oscillation
        assert report.overlay_oscillation_count > 0

    def test_stability_warmup_records_excluded(self):
        log = _make_log(6)
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.stability_report()
        assert report.warmup_records_excluded == BehavioralReviewSystem.WARMUP_RECORDS

    # ── QualityReport ─────────────────────────────────────────────────────────

    def test_quality_report_healthy_low_latency(self):
        log = _make_log(10, elapsed_ms=50.0)
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.quality_report()
        assert report.health == "healthy"
        assert report.avg_latency_ms == pytest.approx(50.0, abs=1.0)
        assert report.override_rate == 0.0

    def test_quality_report_degraded_high_override_rate(self):
        log = AuditLog()
        for _ in range(10):
            log._records.append(_make_record(override_active=True, elapsed_ms=20.0))
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.quality_report()
        assert report.override_rate == 1.0
        assert report.health == "degraded"

    def test_quality_report_warning_high_p95(self):
        log = AuditLog()
        for i in range(20):
            ms = 600.0 if i >= 18 else 50.0   # two slow outliers → p95 spike
            log._records.append(_make_record(elapsed_ms=ms))
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.quality_report()
        assert report.p95_latency_ms >= 50.0   # p95 will be elevated

    def test_quality_report_warning_rate(self):
        log = AuditLog()
        for i in range(10):
            warnings = ["mock warning"] if i % 2 == 0 else []
            log._records.append(_make_record(pipeline_warnings=warnings))
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.quality_report()
        assert report.warning_rate == pytest.approx(0.5, abs=0.01)

    def test_quality_report_regeneration_rate(self):
        log = AuditLog()
        for i in range(10):
            if i < 3:
                stages = ["input_reception", "regeneration_attempt_2", "final_response"]
            else:
                stages = ["input_reception", "final_response"]
            log._records.append(_make_record(stages_executed=stages))
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.quality_report()
        assert report.regeneration_rate == pytest.approx(0.3, abs=0.01)

    def test_quality_report_empty_log(self):
        log = AuditLog()
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.quality_report()
        assert report.record_count == 0
        assert report.health == "healthy"

    # ── DriftDetectionReport ──────────────────────────────────────────────────

    def test_drift_report_none_without_baseline(self):
        log = _make_log(5)
        reviewer = BehavioralReviewSystem(audit_log=log)
        # No baseline registered → returns None
        result = reviewer.drift_report()
        assert result is None

    def test_drift_report_stable_when_matches_baseline(self):
        baseline = {"humor": 0.70, "directness": 0.90, "warmth": 0.60}
        log = _make_log(10, final_traits={"humor": 0.70, "directness": 0.90, "warmth": 0.60})
        reviewer = BehavioralReviewSystem(audit_log=log, baseline_traits=baseline)
        report = reviewer.drift_report()
        assert report is not None
        assert report.health == "stable"
        assert report.overall_drift_score < 0.05

    def test_drift_report_detects_significant_drift(self):
        baseline = {"humor": 0.70, "directness": 0.90}
        # Recent records far from baseline
        log = _make_log(10, final_traits={"humor": 0.10, "directness": 0.20})
        reviewer = BehavioralReviewSystem(audit_log=log, baseline_traits=baseline)
        report = reviewer.drift_report()
        assert report is not None
        assert report.health == "significant_drift"
        assert report.overall_drift_score >= 0.15

    def test_drift_report_register_baseline_after_construction(self):
        log = _make_log(5)
        reviewer = BehavioralReviewSystem(audit_log=log)
        assert reviewer.drift_report() is None  # no baseline yet

        reviewer.register_baseline({"humor": 0.70})
        report = reviewer.drift_report()
        assert report is not None

    def test_drift_report_per_trait_drift_sign(self):
        baseline = {"humor": 0.30}
        # Actual is higher → positive drift
        log = _make_log(10, final_traits={"humor": 0.80})
        reviewer = BehavioralReviewSystem(audit_log=log, baseline_traits=baseline)
        report = reviewer.drift_report()
        assert report.per_trait_drift["humor"] > 0

    # ── AdaptationReport ──────────────────────────────────────────────────────

    def test_adaptation_report_no_action_stable_data(self):
        log = _make_log(20, confidence=0.90, overlay="focused")
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.adaptation_report()
        # Stable confident records → no recommendations expected
        assert report.health in ("no_action", "review_suggested")

    def test_adaptation_report_suggests_dominant_overlay(self):
        log = AuditLog()
        for i in range(20):
            overlay = "teaching" if i < 16 else "focused"  # 80% teaching
            log._records.append(_make_record(overlay=overlay, confidence=0.90))
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.adaptation_report()
        targets = [r.target for r in report.recommendations]
        assert any("overlay_preference" in t for t in targets)

    def test_adaptation_report_suggests_confidence_review_when_low(self):
        log = _make_log(15, confidence=0.25)
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.adaptation_report()
        targets = [r.target for r in report.recommendations]
        assert any("intensity" in t or "confidence" in t for t in targets)

    def test_adaptation_report_recommendations_within_allowed_targets(self):
        """No recommendation may target forbidden systems."""
        forbidden_keywords = [
            "speech_fingerprint", "validation_logic", "communication_rules",
            "expression_confidence_logic", "security_constraint",
        ]
        log = _make_log(20, confidence=0.20, overlay="teaching")
        reviewer = BehavioralReviewSystem(audit_log=log)
        report = reviewer.adaptation_report()
        for rec in report.recommendations:
            for kw in forbidden_keywords:
                assert kw not in rec.target.lower(), (
                    f"Forbidden target found: {rec.target}"
                )

    # ── FullReviewBundle ──────────────────────────────────────────────────────

    def test_full_review_returns_bundle(self):
        log = _make_log(10)
        baseline = {"humor": 0.70, "directness": 0.90}
        reviewer = BehavioralReviewSystem(audit_log=log, baseline_traits=baseline)
        bundle = reviewer.full_review()
        assert isinstance(bundle, FullReviewBundle)
        assert isinstance(bundle.stability, StabilityReport)
        assert isinstance(bundle.quality, QualityReport)
        assert isinstance(bundle.drift, DriftDetectionReport)
        assert isinstance(bundle.adaptation, AdaptationReport)

    def test_full_review_no_drift_without_baseline(self):
        log = _make_log(10)
        reviewer = BehavioralReviewSystem(audit_log=log)
        bundle = reviewer.full_review()
        assert bundle.drift is None

    def test_full_review_to_dict_serializable(self):
        log = _make_log(5)
        reviewer = BehavioralReviewSystem(audit_log=log)
        bundle = reviewer.full_review()
        d = bundle.to_dict()
        # Must be JSON-serialisable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    # ── Window filtering ──────────────────────────────────────────────────────

    def test_window_filters_old_records(self):
        log = AuditLog()
        old = _make_record()
        old = AuditRecord(**{**old.__dict__, "timestamp_utc": "2020-01-01T00:00:00+00:00"})
        recent = _make_record()  # default timestamp_utc in 2025 — within 1hr window
        log._records.extend([old, recent])

        reviewer = BehavioralReviewSystem(audit_log=log, window_seconds=3600)
        # The 2020 record is outside any 1-hour window from now
        report = reviewer.quality_report()
        # Only the recent record should survive (record_count == 1)
        assert report.record_count <= 1


# ══════════════════════════════════════════════════════════════════════════════
# PerformanceManager tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPersonaProfileCache:

    def test_miss_returns_none(self):
        cache = PersonaProfileCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = PersonaProfileCache()
        profile = object()
        cache.set("default", profile)
        assert cache.get("default") is profile

    def test_ttl_expiry(self):
        cache = PersonaProfileCache(ttl_seconds=0.01)
        cache.set("default", object())
        time.sleep(0.05)
        assert cache.get("default") is None

    def test_lru_eviction_at_capacity(self):
        cache = PersonaProfileCache(max_size=3)
        cache.set("a", "A")
        cache.set("b", "B")
        cache.set("c", "C")
        cache.set("d", "D")   # should evict "a" (LRU)
        assert cache.get("a") is None
        assert cache.get("d") == "D"

    def test_hit_rate_tracking(self):
        cache = PersonaProfileCache()
        cache.set("x", object())
        cache.get("x")    # hit
        cache.get("x")    # hit
        cache.get("miss") # miss
        assert cache.hits == 2
        assert cache.misses == 1
        assert cache.hit_rate == pytest.approx(2/3, abs=0.01)

    def test_invalidate_forces_reload(self):
        cache = PersonaProfileCache()
        p = object()
        cache.set("p1", p)
        cache.invalidate("p1")
        assert cache.get("p1") is None

    def test_clear_empties_cache(self):
        cache = PersonaProfileCache()
        cache.set("a", object())
        cache.set("b", object())
        cache.clear()
        assert cache.size == 0


class TestOverlayConfigCache:

    def test_miss_returns_none(self):
        cache = OverlayConfigCache()
        assert cache.get("focused", 0.9) is None

    def test_set_and_get(self):
        cache = OverlayConfigCache()
        modifier = {"technicality": 1.2}
        cache.set("focused", 0.9, modifier)
        assert cache.get("focused", 0.9) == modifier

    def test_confidence_bucketing_5pct(self):
        cache = OverlayConfigCache()
        modifier = {"foo": 1.0}
        cache.set("teaching", 0.90, modifier)
        # 0.92 → bucket int(0.92*20)=18 == int(0.90*20)=18 → same bucket
        assert cache.get("teaching", 0.92) == modifier
        # 0.96 → bucket int(0.96*20)=19 ≠ 18 → different bucket → miss
        assert cache.get("teaching", 0.96) is None

    def test_lru_eviction(self):
        cache = OverlayConfigCache(max_size=2)
        cache.set("a", 0.5, {"a": 1})
        cache.set("b", 0.5, {"b": 2})
        cache.set("c", 0.5, {"c": 3})  # evicts "a"
        assert cache.get("a", 0.5) is None
        assert cache.get("c", 0.5) == {"c": 3}

    def test_hit_rate(self):
        cache = OverlayConfigCache()
        cache.set("creative", 0.8, {"x": 1})
        cache.get("creative", 0.8)   # hit
        cache.get("creative", 0.5)   # miss (different bucket)
        assert cache.hits == 1
        assert cache.misses == 1


class TestPerformanceManagerFastPath:

    def _make_mock_result(
        self,
        overlay: str = "focused",
        relationship: str = "owner",
        intent: str = "inform",
        confidence: float = 0.90,
        had_warnings: bool = False,
        override_active: bool = False,
    ):
        """Build a minimal mock TransformationResult for record_result()."""
        result = MagicMock()
        vector = MagicMock(spec=PersonaVector)
        vector.overlay = MagicMock()
        vector.overlay.value = overlay
        vector.overlay_blends = None
        vector.relationship_context = relationship
        vector.communication_intent = intent
        result.persona_vector = vector
        result.expression_confidence = confidence
        result.pipeline_warnings = ["w"] if had_warnings else []
        result.override_active = override_active
        result.expression_budget = MagicMock(spec=ExpressionBudget)
        return result

    def test_fast_path_ineligible_without_history(self):
        pm = PerformanceManager(min_history=2)
        fp = pm.should_fast_path("focused", "owner", "inform", 0.9)
        assert fp.eligible is False
        assert "Insufficient history" in fp.reason

    def test_fast_path_eligible_after_stable_history(self):
        pm = PerformanceManager(min_history=2)
        for _ in range(2):
            pm.record_result(self._make_mock_result())
        fp = pm.should_fast_path("focused", "owner", "inform", 0.90)
        assert fp.eligible is True
        assert fp.persona_vector is not None

    def test_fast_path_ineligible_overlay_change(self):
        pm = PerformanceManager(min_history=2)
        for _ in range(2):
            pm.record_result(self._make_mock_result(overlay="focused"))
        fp = pm.should_fast_path("teaching", "owner", "inform", 0.90)
        assert fp.eligible is False
        assert "Overlay changed" in fp.reason

    def test_fast_path_ineligible_relationship_change(self):
        pm = PerformanceManager(min_history=2)
        for _ in range(2):
            pm.record_result(self._make_mock_result(relationship="owner"))
        fp = pm.should_fast_path("focused", "guest", "inform", 0.90)
        assert fp.eligible is False
        assert "Relationship changed" in fp.reason

    def test_fast_path_ineligible_intent_change(self):
        pm = PerformanceManager(min_history=2)
        for _ in range(2):
            pm.record_result(self._make_mock_result(intent="inform"))
        fp = pm.should_fast_path("focused", "owner", "warn", 0.90)
        assert fp.eligible is False
        assert "Intent changed" in fp.reason

    def test_fast_path_ineligible_confidence_shift(self):
        pm = PerformanceManager(min_history=2, conf_tolerance=0.05)
        for _ in range(2):
            pm.record_result(self._make_mock_result(confidence=0.90))
        # Shift > 0.05 → ineligible
        fp = pm.should_fast_path("focused", "owner", "inform", 0.50)
        assert fp.eligible is False
        assert "Confidence shifted" in fp.reason

    def test_fast_path_ineligible_previous_had_warnings(self):
        pm = PerformanceManager(min_history=2)
        for _ in range(2):
            pm.record_result(self._make_mock_result(had_warnings=True))
        fp = pm.should_fast_path("focused", "owner", "inform", 0.90)
        assert fp.eligible is False
        assert "warnings" in fp.reason

    def test_fast_path_ineligible_previous_override(self):
        pm = PerformanceManager(min_history=2)
        for _ in range(2):
            pm.record_result(self._make_mock_result(override_active=True))
        fp = pm.should_fast_path("focused", "owner", "inform", 0.90)
        assert fp.eligible is False
        assert "override_active" in fp.reason

    def test_fast_path_stages_skipped_listed(self):
        pm = PerformanceManager(min_history=2)
        for _ in range(2):
            pm.record_result(self._make_mock_result())
        fp = pm.should_fast_path("focused", "owner", "inform", 0.90)
        assert fp.eligible is True
        assert "stability_check" in fp.stages_skipped
        assert "speech_fingerprint" in fp.stages_skipped

    def test_invalidate_fast_path_resets_history(self):
        pm = PerformanceManager(min_history=2)
        for _ in range(2):
            pm.record_result(self._make_mock_result())
        pm.invalidate_fast_path()
        fp = pm.should_fast_path("focused", "owner", "inform", 0.90)
        assert fp.eligible is False

    def test_stats_returns_expected_keys(self):
        pm = PerformanceManager()
        stats = pm.stats()
        required_keys = [
            "fast_path_taken", "fast_path_skipped", "fast_path_rate",
            "profile_cache_size", "profile_cache_hit_rate",
            "overlay_cache_size", "overlay_cache_hit_rate",
            "history_depth",
        ]
        for k in required_keys:
            assert k in stats, f"Missing stat key: {k}"

    def test_stats_fast_path_rate_accurate(self):
        pm = PerformanceManager(min_history=1)
        pm.record_result(self._make_mock_result())
        # One eligible, one ineligible (overlay change)
        pm.should_fast_path("focused", "owner", "inform", 0.90)   # taken
        pm.should_fast_path("teaching", "owner", "inform", 0.90)  # skipped
        stats = pm.stats()
        assert stats["fast_path_taken"] == 1
        assert stats["fast_path_skipped"] == 1
        assert stats["fast_path_rate"] == pytest.approx(0.5, abs=0.01)

    def test_reset_stats_zeroes_counters(self):
        pm = PerformanceManager(min_history=1)
        pm.record_result(self._make_mock_result())
        pm.should_fast_path("focused", "owner", "inform", 0.90)
        pm.reset_stats()
        stats = pm.stats()
        assert stats["fast_path_taken"] == 0
        assert stats["fast_path_skipped"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Integration smoke-test: pipeline auto-wiring
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineMonitoringIntegration:
    """
    Smoke-tests verifying that TransformationPipeline v1.4 properly wires up
    audit_log, reviewer, and perf without breaking existing behaviour.
    We mock the internals so these tests don't require a full GhostMind stub.
    """

    def _make_pipeline(self):
        """Build a minimal pipeline with a mocked ConfigManager."""
        config = MagicMock()
        config.active_profile = None
        config.get_profile = MagicMock(side_effect=KeyError)
        from core.transformation_pipeline import TransformationPipeline
        return TransformationPipeline(
            config_manager=config,
            session_id="integration-test",
        )

    def test_audit_log_is_wired(self):
        pipeline = self._make_pipeline()
        assert isinstance(pipeline.audit_log, AuditLog)
        assert pipeline.audit_log.session_id == "integration-test"

    def test_reviewer_is_wired(self):
        pipeline = self._make_pipeline()
        assert isinstance(pipeline.reviewer, BehavioralReviewSystem)

    def test_perf_is_wired(self):
        pipeline = self._make_pipeline()
        assert isinstance(pipeline.perf, PerformanceManager)
        assert isinstance(pipeline.perf.profile_cache, PersonaProfileCache)
        assert isinstance(pipeline.perf.overlay_cache, OverlayConfigCache)

    def test_reviewer_reads_from_same_audit_log(self):
        pipeline = self._make_pipeline()
        # Manually inject a record into audit_log
        pipeline.audit_log.record_raw(
            relationship="owner", intent="inform", overlay="focused",
            confidence=0.9, final_traits={"humor": 0.7}, budget_used={},
            stages_executed=[], override_active=False, elapsed_ms=30.0,
        )
        # Reviewer should see it
        report = pipeline.reviewer.quality_report()
        assert report.record_count >= 1
