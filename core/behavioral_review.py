"""
core/behavioral_review.py — DarkPassenger Behavioral Review System

Consumes AuditLog records and produces structured diagnostic reports:

    StabilityReport      — consistency of traits and confidence over time
    QualityReport        — response latency health and warning rates
    DriftDetectionReport — long-term personality drift vs a baseline
    AdaptationReport     — recommended tuning adjustments

Architecture (spec §15)
────────────────────────
The BehavioralReviewSystem is a pure analysis layer. It reads AuditRecords
but NEVER modifies them, and it can ONLY recommend changes — it cannot
directly modify protected personality systems (Speech Fingerprint, Validation
Logic, Communication Rules, Expression Confidence Logic, Security Constraints).

Inputs:
    AuditLog (in-memory) or a JSONL file path (offline)

Outputs:
    - StabilityReport     — trait variance, overlay oscillation, identity jitter
    - QualityReport       — latency, override rate, warning rate, pipeline health
    - DriftDetectionReport — drift from a registered baseline
    - AdaptationReport    — data-backed tuning suggestions (style weights, habits)

All reports carry:
    generated_at   : ISO-8601 UTC timestamp
    record_count   : how many records were analysed
    window_seconds : the time window analysed (or None for all-time)

Usage
─────
    reviewer = BehavioralReviewSystem(audit_log=log, logger=logger)

    stability = reviewer.stability_report()
    quality   = reviewer.quality_report()
    drift     = reviewer.drift_report(baseline_traits={"humor": 0.70, ...})
    adapt     = reviewer.adaptation_report()

    # Or generate all four at once:
    bundle = reviewer.full_review()

Spec reference: DarkPassenger-Plan.txt §15, §16
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from core.audit_log import AuditLog, AuditRecord


# ── Report types ──────────────────────────────────────────────────────────────

@dataclass
class StabilityReport:
    """
    Measures personality consistency across the analysed records.

    Trait variance:
        Per-trait standard deviation. High variance for a trait means that
        trait is oscillating unexpectedly — it should feel stable unless the
        context genuinely changed.

    Overlay oscillation count:
        Number of times the overlay changed between consecutive records.
        Frequent oscillation without matching context shifts indicates jitter.

    Confidence variance:
        Standard deviation of expression_confidence across records.
        Stable confidence means the pipeline is behaving predictably.

    warmup_records_excluded:
        Number of early records excluded from analysis (pipeline warm-up
        records often show higher variance and are not representative).

    health:
        "healthy" | "warning" | "unstable" — derived from the highest trait
        variance observed. Thresholds: warning ≥ 0.10, unstable ≥ 0.20.
    """
    generated_at:              str
    record_count:              int
    window_seconds:            Optional[float]
    trait_variance:            Dict[str, float]  # trait → stddev
    overlay_oscillation_count: int
    confidence_variance:       float
    warmup_records_excluded:   int
    health:                    str              # "healthy" | "warning" | "unstable"
    notes:                     List[str]        = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityReport:
    """
    Measures response quality and pipeline health.

    avg_latency_ms:
        Mean elapsed_ms across analysed records.

    p95_latency_ms:
        95th percentile latency. Outliers here indicate fast-path
        cache misses or expensive conflict resolution.

    max_latency_ms:
        Worst-case observed latency.

    override_rate:
        Fraction of responses where override_active=True.
        High override rates may indicate upstream criticality misclassification.

    warning_rate:
        Fraction of responses that had pipeline_warnings.

    regeneration_rate:
        Fraction of responses that required at least one regeneration attempt.
        Detected by counting "regeneration_attempt_" prefixed stages.

    health:
        "healthy" | "warning" | "degraded" based on override_rate and p95_latency.
    """
    generated_at:       str
    record_count:       int
    window_seconds:     Optional[float]
    avg_latency_ms:     float
    p95_latency_ms:     float
    max_latency_ms:     float
    override_rate:      float
    warning_rate:       float
    regeneration_rate:  float
    health:             str   # "healthy" | "warning" | "degraded"
    notes:              List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftDetectionReport:
    """
    Detects long-term personality drift from a registered baseline.

    A baseline is a dict of expected trait values, e.g. the original
    persona profile's base_traits at deployment time.

    per_trait_drift:
        Signed difference between baseline and the mean of recent records.
        Positive = trait intensified, negative = trait diminished.

    max_drift_trait:
        The trait with the largest absolute drift.

    max_drift_magnitude:
        Absolute drift of the worst-drifted trait.

    overall_drift_score:
        Root-mean-square of all per-trait drifts. A single number for
        alerting: 0.0 = no drift, 1.0 = extreme drift.

    health:
        "stable" | "drifting" | "significant_drift"
        Thresholds: drifting ≥ 0.05 rms, significant_drift ≥ 0.15 rms.
    """
    generated_at:         str
    record_count:         int
    window_seconds:       Optional[float]
    baseline_traits:      Dict[str, float]
    per_trait_drift:      Dict[str, float]   # trait → signed delta (recent_mean − baseline)
    max_drift_trait:      Optional[str]
    max_drift_magnitude:  float
    overall_drift_score:  float              # rms of per_trait_drift values
    health:               str               # "stable" | "drifting" | "significant_drift"
    notes:                List[str]         = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdaptationRecommendation:
    """A single data-backed tuning suggestion."""
    target:      str    # e.g. "style_weight:humor", "response_length", "overlay_preference"
    current:     str    # human-readable current value or range
    suggested:   str    # human-readable suggested change
    rationale:   str    # brief evidence-backed explanation
    confidence:  float  # 0.0–1.0 — how confident we are in this suggestion


@dataclass
class AdaptationReport:
    """
    Data-backed tuning suggestions derived from audit records.

    Allowed tuning targets (spec §16):
        - Style weights (trait intensities)
        - Overlay preferences
        - Communication habits (response length, example frequency)
        - Response length preferences

    Forbidden targets:
        - Speech Fingerprint
        - Validation Logic
        - Communication Rules
        - Expression Confidence Logic
        - Security Constraints

    All suggestions here fall within the Allowed category only.
    """
    generated_at:     str
    record_count:     int
    window_seconds:   Optional[float]
    recommendations:  List[AdaptationRecommendation]
    health:           str    # "no_action" | "review_suggested" | "tuning_recommended"
    notes:            List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class FullReviewBundle:
    """All four reports produced in a single reviewer.full_review() call."""
    generated_at:  str
    record_count:  int
    stability:     StabilityReport
    quality:       QualityReport
    drift:         Optional[DriftDetectionReport]  # None if no baseline was registered
    adaptation:    AdaptationReport

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "record_count": self.record_count,
            "stability":    self.stability.to_dict(),
            "quality":      self.quality.to_dict(),
            "drift":        self.drift.to_dict() if self.drift else None,
            "adaptation":   self.adaptation.to_dict(),
        }


# ── BehavioralReviewSystem ────────────────────────────────────────────────────

class BehavioralReviewSystem:
    """
    Diagnostic analysis engine for the DarkPassenger personality layer.

    Construction:
        reviewer = BehavioralReviewSystem(audit_log=log)

        # Optionally register a baseline for drift detection:
        reviewer.register_baseline({"humor": 0.70, "directness": 0.90, ...})

        # Optionally restrict analysis to a rolling time window:
        reviewer = BehavioralReviewSystem(audit_log=log, window_seconds=3600)

    Generating reports:
        stability   = reviewer.stability_report()
        quality     = reviewer.quality_report()
        drift       = reviewer.drift_report()         # requires registered baseline
        adaptation  = reviewer.adaptation_report()
        bundle      = reviewer.full_review()

    Loading from a JSONL file (offline analysis):
        records  = AuditLog.load_jsonl("data/audit/session.jsonl")
        reviewer = BehavioralReviewSystem.from_records(records)
    """

    # Health thresholds
    STABILITY_WARNING_VARIANCE:    float = 0.10
    STABILITY_UNSTABLE_VARIANCE:   float = 0.20
    DRIFT_DRIFTING_RMS:            float = 0.05
    DRIFT_SIGNIFICANT_RMS:         float = 0.15
    QUALITY_WARNING_OVERRIDE_RATE: float = 0.10  # >10% overrides → warning
    QUALITY_DEGRADED_OVERRIDE_RATE:float = 0.25  # >25% overrides → degraded
    QUALITY_WARNING_P95_MS:        float = 500.0
    QUALITY_DEGRADED_P95_MS:       float = 2000.0
    WARMUP_RECORDS:                int   = 3     # exclude from stability

    def __init__(
        self,
        audit_log: Optional[AuditLog] = None,
        records: Optional[List[AuditRecord]] = None,
        window_seconds: Optional[float] = None,
        baseline_traits: Optional[Dict[str, float]] = None,
        logger=None,
    ):
        """
        Args:
            audit_log:       Live AuditLog to read from. Either this or records.
            records:         Pre-loaded list of AuditRecords (offline mode).
            window_seconds:  If set, only analyse records from the last N seconds.
            baseline_traits: Optional trait baseline for drift detection.
            logger:          Optional structured logger.
        """
        if audit_log is None and records is None:
            raise ValueError("Either audit_log or records must be provided.")

        self._audit_log     = audit_log
        self._static_records = records
        self._window        = window_seconds
        self._baseline      = baseline_traits or {}
        self._logger        = logger

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_records(
        cls,
        records: List[AuditRecord],
        window_seconds: Optional[float] = None,
        baseline_traits: Optional[Dict[str, float]] = None,
        logger=None,
    ) -> "BehavioralReviewSystem":
        """Construct from a pre-loaded list of AuditRecords."""
        return cls(
            records=records,
            window_seconds=window_seconds,
            baseline_traits=baseline_traits,
            logger=logger,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        window_seconds: Optional[float] = None,
        baseline_traits: Optional[Dict[str, float]] = None,
        logger=None,
    ) -> "BehavioralReviewSystem":
        """Construct from a JSONL file previously written by AuditLog.export_jsonl()."""
        loaded = AuditLog.load_jsonl(path)
        return cls.from_records(
            loaded,
            window_seconds=window_seconds,
            baseline_traits=baseline_traits,
            logger=logger,
        )

    # ── Baseline management ───────────────────────────────────────────────────

    def register_baseline(self, traits: Dict[str, float]) -> None:
        """
        Register a trait baseline for drift detection.

        Typically called once at startup with the persona profile's base_traits.
        Example:
            reviewer.register_baseline({"humor": 0.70, "directness": 0.90})
        """
        self._baseline = dict(traits)

    # ── Report generators ─────────────────────────────────────────────────────

    def stability_report(self) -> StabilityReport:
        """
        Analyse trait consistency and overlay oscillation.

        Records are windowed then warmup records excluded before analysis.
        """
        now_str = self._now_utc()
        recs = self._windowed_records()

        # Exclude warmup records from stability analysis
        warmup_excluded = min(self.WARMUP_RECORDS, len(recs))
        analysis_recs = recs[warmup_excluded:]

        if not analysis_recs:
            return StabilityReport(
                generated_at=now_str,
                record_count=0,
                window_seconds=self._window,
                trait_variance={},
                overlay_oscillation_count=0,
                confidence_variance=0.0,
                warmup_records_excluded=warmup_excluded,
                health="healthy",
                notes=["Insufficient records for stability analysis."],
            )

        # Per-trait standard deviation
        all_traits = set()
        for r in analysis_recs:
            all_traits.update(r.final_traits.keys())

        trait_variance: Dict[str, float] = {}
        for trait in sorted(all_traits):
            vals = [r.final_traits.get(trait, 0.0) for r in analysis_recs]
            trait_variance[trait] = round(_stddev(vals), 6)

        # Overlay oscillation
        oscillations = 0
        for i in range(1, len(analysis_recs)):
            if analysis_recs[i].overlay != analysis_recs[i - 1].overlay:
                oscillations += 1

        # Confidence variance
        confs = [r.confidence for r in analysis_recs]
        conf_variance = round(_stddev(confs), 6)

        # Health classification
        max_variance = max(trait_variance.values()) if trait_variance else 0.0
        if max_variance >= self.STABILITY_UNSTABLE_VARIANCE:
            health = "unstable"
        elif max_variance >= self.STABILITY_WARNING_VARIANCE:
            health = "warning"
        else:
            health = "healthy"

        notes: List[str] = []
        if health == "unstable":
            worst = max(trait_variance, key=trait_variance.get)  # type: ignore[arg-type]
            notes.append(
                f"Trait '{worst}' has high variance ({trait_variance[worst]:.3f}). "
                "Consider tightening stability_parameters.smoothing_factor."
            )
        if oscillations > len(analysis_recs) * 0.4:
            notes.append(
                f"Overlay changed {oscillations}/{len(analysis_recs)} times — "
                "possible overlay oscillation. Check for context-shift triggers."
            )

        return StabilityReport(
            generated_at=now_str,
            record_count=len(analysis_recs),
            window_seconds=self._window,
            trait_variance=trait_variance,
            overlay_oscillation_count=oscillations,
            confidence_variance=conf_variance,
            warmup_records_excluded=warmup_excluded,
            health=health,
            notes=notes,
        )

    def quality_report(self) -> QualityReport:
        """
        Analyse latency, override rates, and warning rates.
        """
        now_str = self._now_utc()
        recs = self._windowed_records()

        if not recs:
            return QualityReport(
                generated_at=now_str,
                record_count=0,
                window_seconds=self._window,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                max_latency_ms=0.0,
                override_rate=0.0,
                warning_rate=0.0,
                regeneration_rate=0.0,
                health="healthy",
                notes=["No records to analyse."],
            )

        latencies = sorted(r.elapsed_ms for r in recs)
        avg_lat = round(statistics.mean(latencies), 3)
        p95_lat = round(_percentile(latencies, 95), 3)
        max_lat = round(max(latencies), 3)

        override_rate = round(sum(1 for r in recs if r.override_active) / len(recs), 6)
        warning_rate  = round(sum(1 for r in recs if r.pipeline_warnings) / len(recs), 6)
        regen_rate    = round(
            sum(
                1 for r in recs
                if any("regeneration_attempt_" in s for s in r.stages_executed)
            ) / len(recs),
            6,
        )

        # Health
        if (
            override_rate >= self.QUALITY_DEGRADED_OVERRIDE_RATE
            or p95_lat >= self.QUALITY_DEGRADED_P95_MS
        ):
            health = "degraded"
        elif (
            override_rate >= self.QUALITY_WARNING_OVERRIDE_RATE
            or p95_lat >= self.QUALITY_WARNING_P95_MS
        ):
            health = "warning"
        else:
            health = "healthy"

        notes: List[str] = []
        if override_rate >= self.QUALITY_WARNING_OVERRIDE_RATE:
            notes.append(
                f"Override rate {override_rate:.1%} is elevated. "
                "Check GhostMind criticality classification."
            )
        if p95_lat >= self.QUALITY_WARNING_P95_MS:
            notes.append(
                f"P95 latency {p95_lat:.0f}ms exceeds threshold. "
                "Consider enabling fast-path overlays for simple contexts."
            )
        if regen_rate > 0.05:
            notes.append(
                f"Regeneration triggered for {regen_rate:.1%} of responses. "
                "Review CircuitBreaker validation thresholds."
            )

        return QualityReport(
            generated_at=now_str,
            record_count=len(recs),
            window_seconds=self._window,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            max_latency_ms=max_lat,
            override_rate=override_rate,
            warning_rate=warning_rate,
            regeneration_rate=regen_rate,
            health=health,
            notes=notes,
        )

    def drift_report(
        self,
        baseline_traits: Optional[Dict[str, float]] = None,
    ) -> Optional[DriftDetectionReport]:
        """
        Compare recent trait means against a baseline.

        Args:
            baseline_traits: Override the registered baseline for this call only.

        Returns None if no baseline is available.
        """
        baseline = baseline_traits or self._baseline
        if not baseline:
            return None

        now_str = self._now_utc()
        recs = self._windowed_records()

        if not recs:
            return DriftDetectionReport(
                generated_at=now_str,
                record_count=0,
                window_seconds=self._window,
                baseline_traits=baseline,
                per_trait_drift={},
                max_drift_trait=None,
                max_drift_magnitude=0.0,
                overall_drift_score=0.0,
                health="stable",
                notes=["No records to analyse."],
            )

        # Compute recent mean per trait
        recent_means: Dict[str, float] = {}
        all_traits = set(baseline.keys())
        for r in recs:
            all_traits.update(r.final_traits.keys())

        for trait in all_traits:
            vals = [r.final_traits.get(trait) for r in recs if trait in r.final_traits]
            if vals:
                recent_means[trait] = statistics.mean(vals)

        # Signed drift (positive = more intense than baseline)
        per_trait_drift: Dict[str, float] = {}
        for trait, base_val in baseline.items():
            recent = recent_means.get(trait)
            if recent is not None:
                per_trait_drift[trait] = round(recent - base_val, 6)

        if not per_trait_drift:
            max_drift_trait = None
            max_drift_magnitude = 0.0
            overall_drift_score = 0.0
        else:
            max_drift_trait = max(
                per_trait_drift, key=lambda t: abs(per_trait_drift[t])
            )
            max_drift_magnitude = round(abs(per_trait_drift[max_drift_trait]), 6)
            rms = math.sqrt(
                sum(v ** 2 for v in per_trait_drift.values()) / len(per_trait_drift)
            )
            overall_drift_score = round(rms, 6)

        # Health
        if overall_drift_score >= self.DRIFT_SIGNIFICANT_RMS:
            health = "significant_drift"
        elif overall_drift_score >= self.DRIFT_DRIFTING_RMS:
            health = "drifting"
        else:
            health = "stable"

        notes: List[str] = []
        if max_drift_trait and max_drift_magnitude >= self.DRIFT_DRIFTING_RMS:
            direction = "increased" if per_trait_drift[max_drift_trait] > 0 else "decreased"
            notes.append(
                f"Trait '{max_drift_trait}' has {direction} by "
                f"{max_drift_magnitude:.3f} from baseline. "
                "Review AdaptiveTuning or persona profile."
            )

        return DriftDetectionReport(
            generated_at=now_str,
            record_count=len(recs),
            window_seconds=self._window,
            baseline_traits=dict(baseline),
            per_trait_drift=per_trait_drift,
            max_drift_trait=max_drift_trait,
            max_drift_magnitude=max_drift_magnitude,
            overall_drift_score=overall_drift_score,
            health=health,
            notes=notes,
        )

    def adaptation_report(self) -> AdaptationReport:
        """
        Produce data-backed tuning suggestions.

        All recommendations stay within the Allowed tuning targets:
            - Style weights (trait intensities in the persona profile)
            - Overlay preferences
            - Communication habits

        This method NEVER recommends changes to:
            - Speech Fingerprint
            - Validation Logic / CircuitBreaker
            - Communication Rules
            - Expression Confidence Logic
            - Security Constraints
        """
        now_str = self._now_utc()
        recs = self._windowed_records()

        recs = recs[self.WARMUP_RECORDS:]  # skip warmup
        recommendations: List[AdaptationRecommendation] = []

        if not recs:
            return AdaptationReport(
                generated_at=now_str,
                record_count=0,
                window_seconds=self._window,
                recommendations=[],
                health="no_action",
                notes=["Insufficient records for adaptation analysis."],
            )

        # ── Overlay preference analysis ───────────────────────────────────────
        overlay_counts: Dict[str, int] = {}
        for r in recs:
            overlay_counts[r.overlay] = overlay_counts.get(r.overlay, 0) + 1

        if overlay_counts:
            dominant_overlay, dominant_count = max(
                overlay_counts.items(), key=lambda x: x[1]
            )
            dominant_frac = dominant_count / len(recs)
            if dominant_frac >= 0.70 and dominant_overlay != "none":
                # Strongly dominant overlay — suggest making it the default
                recommendations.append(AdaptationRecommendation(
                    target="overlay_preference.default_overlay",
                    current=f"(varies; '{dominant_overlay}' used {dominant_frac:.0%})",
                    suggested=f"Set default_overlay = '{dominant_overlay}'",
                    rationale=(
                        f"'{dominant_overlay}' was active in {dominant_frac:.0%} of "
                        "responses. Setting it as the default reduces runtime overlay "
                        "resolution overhead and improves consistency."
                    ),
                    confidence=round(dominant_frac, 3),
                ))

        # ── Confidence pattern analysis ───────────────────────────────────────
        mean_conf = statistics.mean(r.confidence for r in recs)
        if mean_conf < 0.40:
            recommendations.append(AdaptationRecommendation(
                target="style_weights.global_intensity",
                current=f"mean confidence {mean_conf:.2f}",
                suggested="Review upstream criticality classification; "
                          "many responses are being heavily attenuated.",
                rationale=(
                    f"Average expression confidence is {mean_conf:.2f}, significantly "
                    "below the normal-conversation baseline of 0.90. This reduces "
                    "personality expression across the board and may make responses "
                    "feel generic."
                ),
                confidence=0.80,
            ))

        # ── Trait intensity suggestions ───────────────────────────────────────
        # For each trait, if the mean value is significantly below 0.5 across
        # all non-override responses, it may be under-weighted in the profile.
        non_override_recs = [r for r in recs if not r.override_active]
        if non_override_recs:
            all_traits = set()
            for r in non_override_recs:
                all_traits.update(r.final_traits.keys())

            for trait in sorted(all_traits):
                vals = [r.final_traits.get(trait, 0.0) for r in non_override_recs]
                mean_val = statistics.mean(vals)
                stddev   = _stddev(vals)

                # High-variance low-mean trait → candidate for smoothing
                if stddev >= self.STABILITY_WARNING_VARIANCE and mean_val < 0.50:
                    recommendations.append(AdaptationRecommendation(
                        target=f"style_weights.{trait}",
                        current=f"mean={mean_val:.2f}, stddev={stddev:.2f}",
                        suggested=(
                            f"Increase base_traits.{trait} slightly "
                            f"(current mean {mean_val:.2f} with high variance)."
                        ),
                        rationale=(
                            f"'{trait}' shows low mean ({mean_val:.2f}) with high "
                            f"variance ({stddev:.2f}), suggesting the stability "
                            "layer is frequently overriding it. A higher base value "
                            "gives the smoothing algorithm a more stable target."
                        ),
                        confidence=round(min(stddev * 3, 0.90), 3),
                    ))

        health = (
            "no_action"          if not recommendations
            else "review_suggested" if len(recommendations) <= 2
            else "tuning_recommended"
        )

        return AdaptationReport(
            generated_at=now_str,
            record_count=len(recs),
            window_seconds=self._window,
            recommendations=recommendations,
            health=health,
        )

    def full_review(
        self,
        baseline_traits: Optional[Dict[str, float]] = None,
    ) -> FullReviewBundle:
        """
        Generate all four reports in a single call.

        Args:
            baseline_traits: Optional baseline override for drift detection.

        Returns:
            FullReviewBundle with stability, quality, drift, and adaptation.
        """
        now_str = self._now_utc()
        recs    = self._windowed_records()

        return FullReviewBundle(
            generated_at=now_str,
            record_count=len(recs),
            stability=self.stability_report(),
            quality=self.quality_report(),
            drift=self.drift_report(baseline_traits=baseline_traits),
            adaptation=self.adaptation_report(),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _windowed_records(self) -> List[AuditRecord]:
        """
        Return records, filtered to the time window if one is set.
        Sorted oldest-first.
        """
        if self._audit_log is not None:
            recs = self._audit_log.records
        else:
            recs = list(self._static_records or [])

        if not recs or self._window is None:
            return recs

        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=self._window)
        ).isoformat()

        return [r for r in recs if r.timestamp_utc >= cutoff]

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()


# ── Utility functions ─────────────────────────────────────────────────────────

def _stddev(values: List[float]) -> float:
    """Population standard deviation; returns 0.0 for < 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _percentile(sorted_values: List[float], pct: int) -> float:
    """Nearest-rank percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(math.ceil(len(sorted_values) * pct / 100)) - 1))
    return sorted_values[k]
