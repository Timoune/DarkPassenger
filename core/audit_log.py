"""
core/audit_log.py — DarkPassenger Personality Audit Log

Records structured metadata for every response that passes through the
transformation pipeline. These records are the raw material consumed by
the BehavioralReviewSystem to detect drift, instability, and quality issues.

What is logged (spec §14)
──────────────────────────
  response_id    : unique UUID per response
  timestamp_utc  : ISO-8601 wall-clock time
  session_id     : groups records within a single conversation session
  relationship   : RelationshipContext active for this response
  intent         : CommunicationIntent active for this response
  overlay        : OverlayType (or blend description) active
  confidence     : final expression confidence scalar (0.0–1.0)
  final_traits   : PersonaVector after conflict resolution + scaling
  budget_used    : ExpressionBudget allocations
  stages_executed: pipeline stages that ran
  override_active: True if Critical Response Override fired
  elapsed_ms     : pipeline wall-clock time in milliseconds
  pipeline_warnings: any non-fatal pipeline warnings

What is NOT logged
──────────────────
  - Response content / user messages (privacy)
  - Protected fields / GhostMind output content (integrity)
  - Security decision internals

Storage
───────
AuditLog is the in-memory ring buffer. Records are stored in a deque
capped at max_records. Each record is an AuditRecord dataclass.

For persistence, call AuditLog.export_jsonl(path) which appends records
to a newline-delimited JSON file. The BehavioralReviewSystem consumes
either the in-memory buffer or the exported file.

Spec reference: DarkPassenger-Plan.txt §14, §15
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

from core.persona_vector import (
    PersonaVector,
    ExpressionBudget,
    OverlayType,
    RelationshipContext,
    CommunicationIntent,
)


# ── AuditRecord ───────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """
    A single pipeline response record.

    Immutable after creation — the audit trail must not be retroactively
    altered. All values are captured at the moment transform() returns.

    Attributes
    ──────────
    response_id:
        UUID identifying this specific response. Primary key for log queries.

    session_id:
        UUID identifying the conversation session. Groups records for per-session
        stability and drift analysis.

    timestamp_utc:
        ISO-8601 timestamp in UTC (e.g. "2025-11-04T14:32:01.123456+00:00").

    relationship:
        RelationshipContext.value string active for this response.

    intent:
        CommunicationIntent.value string active for this response.

    overlay:
        OverlayType.value string, or a JSON-serialised blend dict
        (e.g. '{"teaching": 0.7, "focused": 0.3}') when blends are active.

    confidence:
        Final expression confidence scalar applied to the PersonaVector.
        Range 0.0–1.0. Values near 0.0 indicate override or high uncertainty.

    final_traits:
        Trait values from the PersonaVector after conflict resolution and
        confidence scaling. Dict[trait_name → float].

    budget_used:
        Expression budget allocations actually applied.
        Dict[trait_name → allocated_points].

    stages_executed:
        Ordered list of pipeline stage names that ran (for replay diagnostics).

    override_active:
        True when the Critical Response Override fired. When True, confidence
        is 0.0 and final_traits reflect a zeroed vector.

    elapsed_ms:
        Approximate wall-clock duration for the full transformation pipeline.

    pipeline_warnings:
        Non-fatal diagnostic messages raised during the pipeline run.
    """
    response_id:       str
    session_id:        str
    timestamp_utc:     str
    relationship:      str
    intent:            str
    overlay:           str
    confidence:        float
    final_traits:      Dict[str, float]
    budget_used:       Dict[str, float]
    stages_executed:   List[str]
    override_active:   bool
    elapsed_ms:        float
    pipeline_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialisable dict for JSON export."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditRecord":
        """Reconstruct from a dict (e.g. parsed from JSONL)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── AuditLog ──────────────────────────────────────────────────────────────────

class AuditLog:
    """
    In-memory ring buffer of AuditRecords.

    Thread safety
    ─────────────
    AuditLog is NOT thread-safe. It is designed for single-threaded use
    within the DarkPassenger pipeline. If the pipeline is called from
    multiple threads, callers are responsible for external locking.

    Persistence
    ───────────
    Call export_jsonl(path) to append all current records to a JSONL file.
    The BehavioralReviewSystem can consume this file offline.

    Usage
    ─────
        # Typically constructed once per session by TransformationPipeline
        log = AuditLog(session_id="...", max_records=1000)

        # Called automatically by the patched pipeline; also callable manually:
        log.record(result)    # TransformationResult → AuditRecord appended

        # Access records
        for rec in log.records:
            print(rec.response_id, rec.confidence)

        # Export
        log.export_jsonl("data/audit/session_2025-11-04.jsonl")
    """

    DEFAULT_MAX_RECORDS = 2000

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_records: int = DEFAULT_MAX_RECORDS,
        logger=None,
    ):
        self.session_id  = session_id or str(uuid.uuid4())
        self.max_records = max_records
        self._logger     = logger

        # Deque acts as a ring buffer: oldest records drop off when full
        self._records: Deque[AuditRecord] = deque(maxlen=max_records)

    # ── Core API ──────────────────────────────────────────────────────────────

    def record(self, result: "TransformationResult") -> AuditRecord:  # type: ignore[name-defined]
        """
        Capture an AuditRecord from a TransformationResult.

        Imports are inline to avoid circular dependency with transformation_pipeline.

        Args:
            result: The TransformationResult returned by pipeline.transform().

        Returns:
            The AuditRecord that was appended to the log.
        """
        # Resolve overlay label — blend or single
        overlay_label = self._overlay_label(result)

        # Trait dict from the resolved PersonaVector
        final_traits = self._vector_to_dict(result.persona_vector)

        # Budget allocation dict
        budget_used = {
            k: float(v)
            for k, v in result.expression_budget.allocations.items()
        }

        # relationship and intent: prefer explicit fields stamped onto result by
        # the pipeline (result.relationship / result.intent), then fall back to
        # PersonaVector attributes (for custom integrations that store them there),
        # then default to "unknown".
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

        rec = AuditRecord(
            response_id=str(uuid.uuid4()),
            session_id=self.session_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            relationship=relationship,
            intent=intent,
            overlay=overlay_label,
            confidence=round(result.expression_confidence, 6),
            final_traits=final_traits,
            budget_used=budget_used,
            stages_executed=list(result.stages_executed),
            override_active=result.override_active,
            elapsed_ms=round(result.elapsed_ms, 3),
            pipeline_warnings=list(result.pipeline_warnings),
        )

        self._records.append(rec)

        if self._logger:
            self._logger.debug(
                "audit_record_appended",
                response_id=rec.response_id,
                confidence=rec.confidence,
                override_active=rec.override_active,
                elapsed_ms=rec.elapsed_ms,
            )

        return rec

    def record_raw(
        self,
        *,
        relationship: str,
        intent: str,
        overlay: str,
        confidence: float,
        final_traits: Dict[str, float],
        budget_used: Dict[str, float],
        stages_executed: List[str],
        override_active: bool,
        elapsed_ms: float,
        pipeline_warnings: Optional[List[str]] = None,
    ) -> AuditRecord:
        """
        Append a manually-constructed record (for testing or custom integrations).

        All fields are caller-supplied; no TransformationResult is required.
        """
        rec = AuditRecord(
            response_id=str(uuid.uuid4()),
            session_id=self.session_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            relationship=relationship,
            intent=intent,
            overlay=overlay,
            confidence=round(confidence, 6),
            final_traits=final_traits,
            budget_used=budget_used,
            stages_executed=list(stages_executed),
            override_active=override_active,
            elapsed_ms=round(elapsed_ms, 3),
            pipeline_warnings=pipeline_warnings or [],
        )
        self._records.append(rec)
        return rec

    # ── Access ────────────────────────────────────────────────────────────────

    @property
    def records(self) -> List[AuditRecord]:
        """All records currently in the buffer, oldest-first."""
        return list(self._records)

    @property
    def count(self) -> int:
        """Number of records currently in the buffer."""
        return len(self._records)

    def last(self, n: int = 1) -> List[AuditRecord]:
        """Return the n most recent records (most recent last)."""
        return list(self._records)[-n:]

    def since(self, timestamp_utc: str) -> List[AuditRecord]:
        """
        Return records with timestamp_utc >= the given ISO-8601 string.
        Useful for windowed analysis without loading from disk.
        """
        return [r for r in self._records if r.timestamp_utc >= timestamp_utc]

    def clear(self) -> None:
        """Discard all records from the in-memory buffer."""
        self._records.clear()

    # ── Persistence ───────────────────────────────────────────────────────────

    def export_jsonl(self, path: str, append: bool = True) -> int:
        """
        Write all in-memory records to a newline-delimited JSON file.

        Args:
            path:   File path to write. Parent directories are created.
            append: If True (default), append to the file. If False, overwrite.

        Returns:
            Number of records written.

        Each line is a self-contained JSON object (one AuditRecord).
        The BehavioralReviewSystem can load this with load_jsonl().
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        written = 0
        with out.open(mode, encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                written += 1

        if self._logger:
            self._logger.info(
                "audit_log_exported",
                path=str(out),
                records_written=written,
                mode=mode,
            )
        return written

    @staticmethod
    def load_jsonl(path: str) -> List[AuditRecord]:
        """
        Load records from a JSONL file written by export_jsonl().

        Skips malformed lines with a warning to stdout (no logger available
        in a static context). Returns records sorted by timestamp_utc.
        """
        records: List[AuditRecord] = []
        p = Path(path)
        if not p.exists():
            return records

        with p.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    records.append(AuditRecord.from_dict(d))
                except Exception as exc:
                    print(f"[AuditLog] Skipping malformed line {lineno}: {exc}")

        records.sort(key=lambda r: r.timestamp_utc)
        return records

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _overlay_label(result: "TransformationResult") -> str:  # type: ignore[name-defined]
        """
        Produce a human-readable overlay label.

        Single overlay  → OverlayType.value string (e.g. "teaching")
        Blended overlay → JSON-serialised dict (e.g. '{"teaching": 0.7, "focused": 0.3}')
        No overlay      → "none"
        """
        try:
            v = result.persona_vector
            # Blended overlay: PersonaVector may store blend weights as a dict attribute
            blends = getattr(v, "overlay_blends", None)
            if blends:
                return json.dumps(
                    {k: round(float(w), 3) for k, w in blends.items()},
                    sort_keys=True,
                )
            overlay = getattr(v, "overlay", None)
            if overlay is not None:
                return overlay.value if hasattr(overlay, "value") else str(overlay)
        except Exception:
            pass
        return "none"

    @staticmethod
    def _vector_to_dict(vector: "PersonaVector") -> Dict[str, float]:  # type: ignore[name-defined]
        """
        Extract trait values from a PersonaVector into a plain dict.

        PersonaVector stores traits as individual float attributes (formality,
        humor, warmth, directness, technicality, confidence, precision).
        Falls back gracefully if the vector has an unexpected structure.
        """
        trait_names = [
            "formality", "humor", "warmth", "directness",
            "technicality", "confidence", "precision",
            "curiosity", "analytical_depth",
        ]
        result: Dict[str, float] = {}
        for name in trait_names:
            val = getattr(vector, name, None)
            if val is not None:
                try:
                    result[name] = round(float(val), 6)
                except (TypeError, ValueError):
                    pass
        return result
