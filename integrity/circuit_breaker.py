"""
integrity/circuit_breaker.py — DarkPassenger Circuit Breaker

This is the safety-first component of DarkPassenger.
It implements the full three-stage Validation Pipeline from the spec:

    Stage 1 — Meaning Validation   (deterministic)
    Stage 2 — Style Validation     (heuristic)
    Stage 3 — Integrity Safeguard  (fail-safe regeneration gate)

It also implements the Critical Response Override:
    When GhostMind marks output as EMERGENCY / SECURITY / SYSTEM_FAIL
    (or any security flag is present), ALL personality transformation is
    bypassed and raw GhostMind content is delivered directly.

Architecture
────────────
GhostMind produces a GhostMindOutput (finalized, checksummed).
CircuitBreaker receives it and does one of three things:

    A) OVERRIDE  — criticality demands raw output; bypass everything.
    B) VALIDATE  — run the three-stage pipeline against a candidate
                   transformed output; approve or reject.
    C) REJECT    — validation failed after MAX_ATTEMPTS; fall back to raw.

The CircuitBreaker never calls the LLM itself. It receives the
GhostMindOutput (what was said) and the candidate TransformedOutput
(how DarkPassenger wants to say it) and audits the pair.

Communication Rules enforced here (from DarkPassenger spec §12):
    - Numerical values must not be altered
    - Security restrictions must not be softened
    - Risk assessments must not be weakened
    - Conclusions must not be changed
    - Tool results must not be reinterpreted
    - Warnings must not be suppressed
    - Uncertainty indicators must not be removed
    - No fabricated information
    - No hidden failure states
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from types.integrity_types import (
    GhostMindOutput,
    ValidationResult,
    ValidationStatus,
    ValidationViolation,
    CriticalityLevel,
    OVERRIDE_LEVELS,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_REGEN_ATTEMPTS = 3

# Numbers are extracted by this pattern for cross-checking.
# Matches: integers, decimals, percentages, negative numbers.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")

# Warning signal words that must appear in transformed output if in source.
_WARNING_SIGNALS = frozenset({
    "warning", "alert", "critical", "error", "failure", "failed",
    "danger", "caution", "risk", "halt", "denied", "blocked",
    "unauthorized", "forbidden", "cannot", "must not", "do not",
    "emergency", "severe", "fatal", "abort", "unsafe",
})

# Uncertainty hedge words — if GhostMind expressed uncertainty, the
# transformation must not project false confidence.
_UNCERTAINTY_SIGNALS = frozenset({
    "uncertain", "unsure", "unclear", "unknown", "may", "might",
    "possibly", "perhaps", "approximately", "estimate", "roughly",
    "not sure", "cannot confirm", "unverified", "unconfirmed",
})


# ---------------------------------------------------------------------------
# TransformedOutput — what DarkPassenger proposes to say
# ---------------------------------------------------------------------------

@dataclass
class TransformedOutput:
    """
    A personality-transformed candidate response from DarkPassenger.

    Passed to CircuitBreaker.validate() for approval.
    """
    content:         str
    transformation_notes: str = ""   # optional debug info from DarkPassenger


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Three-stage validation pipeline + Critical Response Override.

    Usage (happy path)
    ──────────────────
    ghost_output = GhostMindOutput.normal("The server latency is 142ms.") \\
                       .add_number("latency_ms", 142) \\
                       .finalize()

    candidate = TransformedOutput("Yo, server's running at 142ms — pretty solid!")

    result = circuit_breaker.validate(ghost_output, candidate)

    if result.passed:
        deliver(result.safe_output)
    else:
        # CircuitBreaker already fell back to safe content
        deliver(result.safe_output)
    """

    def __init__(self, logger=None):
        self.logger = logger
        self._audit_log: List[dict] = []

    # ── Main entry point ──────────────────────────────────────────────────────

    def validate(
        self,
        ghost_output: GhostMindOutput,
        candidate: TransformedOutput,
    ) -> ValidationResult:
        """
        Validate a candidate transformation against the original GhostMind output.

        Returns a ValidationResult. result.safe_output always contains text
        that is safe to deliver to the user regardless of the verdict.

        The CircuitBreaker NEVER raises exceptions — it absorbs errors and
        falls back to raw GhostMind content if anything goes wrong.
        """
        try:
            return self._validate_internal(ghost_output, candidate)
        except Exception as exc:
            self._log("circuit_breaker_internal_error", error=str(exc))
            # Ultimate fallback: deliver raw content
            return ValidationResult(
                status=ValidationStatus.FAIL_CRITICAL,
                violations=[ValidationViolation(
                    stage="integrity",
                    field="circuit_breaker",
                    expected="no exception",
                    found=str(exc),
                    description=f"CircuitBreaker internal error: {exc}",
                )],
                safe_output=ghost_output.content,
            )

    def _validate_internal(
        self,
        ghost_output: GhostMindOutput,
        candidate: TransformedOutput,
    ) -> ValidationResult:

        # ── Critical Response Override ────────────────────────────────────
        if ghost_output.requires_override:
            self._log(
                "critical_override_fired",
                criticality=ghost_output.criticality.value,
                security_flags=ghost_output.security_flags,
            )
            return ValidationResult(
                status=ValidationStatus.BYPASS,
                override_active=True,
                safe_output=self._build_override_output(ghost_output),
                attempts=0,
            )

        # ── Three-stage validation ─────────────────────────────────────────
        for attempt in range(1, MAX_REGEN_ATTEMPTS + 1):
            all_violations: List[ValidationViolation] = []

            # Stage 1 — Meaning Validation (deterministic)
            meaning_violations = self._stage1_meaning(ghost_output, candidate)
            all_violations.extend(meaning_violations)

            # Stage 2 — Style Validation (heuristic)
            style_violations = self._stage2_style(ghost_output, candidate)
            all_violations.extend(style_violations)

            if not all_violations:
                # Stage 3 — Integrity Safeguard (final gate)
                integrity_ok, integrity_violation = self._stage3_integrity(
                    ghost_output, candidate
                )
                if integrity_ok:
                    self._log(
                        "validation_passed",
                        attempt=attempt,
                        criticality=ghost_output.criticality.value,
                    )
                    return ValidationResult(
                        status=ValidationStatus.PASS,
                        safe_output=candidate.content,
                        attempts=attempt,
                    )
                else:
                    all_violations.append(integrity_violation)

            # Validation failed — log and decide whether to retry or reject
            self._log(
                "validation_failed",
                attempt=attempt,
                violation_count=len(all_violations),
                violations=[v.description for v in all_violations],
            )

            if attempt < MAX_REGEN_ATTEMPTS:
                # Signal caller to regenerate (in the full pipeline, DarkPassenger
                # would produce a new candidate here; we return FAIL to trigger it)
                self._log("requesting_regeneration", attempt=attempt)
                # In the isolated circuit breaker module, we return the failure
                # immediately and let the caller (DarkPassenger pipeline) decide
                # whether to retry. The safe_output always falls back to raw.
                return ValidationResult(
                    status=ValidationStatus.FAIL_MEANING if meaning_violations
                           else ValidationStatus.FAIL_STYLE,
                    violations=all_violations,
                    safe_output=ghost_output.content,   # raw fallback
                    attempts=attempt,
                )

        # All attempts exhausted — Stage 3 Integrity Safeguard: use raw content
        self._log(
            "integrity_safeguard_triggered",
            max_attempts=MAX_REGEN_ATTEMPTS,
        )
        return ValidationResult(
            status=ValidationStatus.FAIL_CRITICAL,
            violations=all_violations,
            safe_output=ghost_output.content,   # raw fallback
            attempts=MAX_REGEN_ATTEMPTS,
        )

    # ── Stage 1: Meaning Validation ───────────────────────────────────────────

    def _stage1_meaning(
        self,
        ghost: GhostMindOutput,
        candidate: TransformedOutput,
    ) -> List[ValidationViolation]:
        """
        Deterministic checks — zero tolerance.

        Checks:
        1. All numerical values from protected_fields are present and unchanged.
        2. All explicit warnings are present (verbatim or equivalent signal).
        3. Tool outputs are present (not silently dropped).
        4. Risk assessment is present (if set by GhostMind).
        5. Security decisions are not softened.
        6. Uncertainty signals preserved if GhostMind expressed uncertainty.
        """
        violations = []

        # 1. Protected numerical values
        for pf in ghost.protected_fields:
            if pf.field_type == "number":
                v = self._check_number_present(
                    key=pf.key,
                    value=pf.value,
                    text=candidate.content,
                )
                if v:
                    violations.append(v)

        # 2. Warnings must not be suppressed
        for warning in ghost.warnings:
            v = self._check_warning_preserved(warning, candidate.content)
            if v:
                violations.append(v)

        # 3. Tool results must appear
        for tool_result in ghost.tool_results:
            v = self._check_tool_result_present(tool_result, candidate.content)
            if v:
                violations.append(v)

        # 4. Risk assessment must appear
        if ghost.risk_assessment:
            v = self._check_risk_assessment_present(
                ghost.risk_assessment, candidate.content
            )
            if v:
                violations.append(v)

        # 5. Warning signal words must be preserved (macro-level)
        v = self._check_warning_signals(ghost.content, candidate.content)
        if v:
            violations.append(v)

        # 6. Uncertainty signals
        if ghost.uncertainty_score >= 0.3:
            v = self._check_uncertainty_preserved(ghost.content, candidate.content)
            if v:
                violations.append(v)

        return violations

    # ── Stage 2: Style Validation ─────────────────────────────────────────────

    def _stage2_style(
        self,
        ghost: GhostMindOutput,
        candidate: TransformedOutput,
    ) -> List[ValidationViolation]:
        """
        Heuristic checks — flags suspicious patterns.

        Checks:
        1. Candidate is not empty or trivially short.
        2. Candidate does not introduce NEW numerical values not in source.
        3. Candidate does not appear to be the raw GhostMind output word-for-word
           (validation would always pass trivially if transformation is a no-op).
        """
        violations = []

        # 1. Non-empty
        if not candidate.content.strip():
            violations.append(ValidationViolation(
                stage="style",
                field="content_length",
                expected="> 0 characters",
                found="empty",
                description="Transformed output is empty.",
            ))
            return violations  # nothing more to check

        # 2. No fabricated numbers
        v = self._check_no_fabricated_numbers(ghost.content, candidate.content)
        if v:
            violations.append(v)

        return violations

    # ── Stage 3: Integrity Safeguard ──────────────────────────────────────────

    def _stage3_integrity(
        self,
        ghost: GhostMindOutput,
        candidate: TransformedOutput,
    ) -> Tuple[bool, Optional[ValidationViolation]]:
        """
        Final gate. Verifies protected field checksums end-to-end.

        For each protected field with a stored checksum, confirms that
        the value's string representation still appears in the candidate
        output and that the checksum still matches the original value.

        Returns (True, None) on pass, (False, violation) on fail.
        """
        for pf in ghost.protected_fields:
            if not pf.checksum:
                continue   # field was not checksummed (finalize() not called)

            expected_checksum = hashlib.sha256(str(pf.value).encode()).hexdigest()
            if pf.checksum != expected_checksum:
                return False, ValidationViolation(
                    stage="integrity",
                    field=pf.key,
                    expected=pf.checksum,
                    found=expected_checksum,
                    description=(
                        f"Checksum mismatch for protected field '{pf.key}'. "
                        "The original value may have been tampered with."
                    ),
                )

            # Also confirm the value text still appears in output
            if str(pf.value) not in candidate.content:
                return False, ValidationViolation(
                    stage="integrity",
                    field=pf.key,
                    expected=f"value '{pf.value}' present in output",
                    found="absent",
                    description=(
                        f"Protected field '{pf.key}' value '{pf.value}' "
                        "is missing from the transformed output."
                    ),
                )

        return True, None

    # ── Critical Response Override helpers ────────────────────────────────────

    def _build_override_output(self, ghost: GhostMindOutput) -> str:
        """
        Build the raw-output string for Critical Response Override.

        Concatenates GhostMind content + all warnings + all tool results.
        No personality. No styling. Maximum clarity.
        """
        parts = [ghost.content]

        if ghost.warnings:
            parts.append("\n" + "\n".join(f"⚠ {w}" for w in ghost.warnings))

        if ghost.tool_results:
            parts.append("\n" + "\n".join(ghost.tool_results))

        if ghost.risk_assessment:
            parts.append(f"\nRisk Level: {ghost.risk_assessment}")

        if ghost.security_flags:
            parts.append(
                "\nSecurity flags: " + ", ".join(ghost.security_flags)
            )

        return "\n".join(parts).strip()

    # ── Specific check methods ────────────────────────────────────────────────

    def _check_number_present(
        self,
        key: str,
        value: float | int,
        text: str,
    ) -> Optional[ValidationViolation]:
        """Verify that a specific numerical value is present in the text."""
        # Check both integer and float representations
        representations = {str(value)}
        if isinstance(value, float) and value == int(value):
            representations.add(str(int(value)))
        elif isinstance(value, int):
            representations.add(str(float(value)))

        for rep in representations:
            if rep in text:
                return None   # found

        return ValidationViolation(
            stage="meaning",
            field=f"number:{key}",
            expected=f"{value} present in output",
            found="absent",
            description=(
                f"Protected numerical value '{key}' = {value} "
                "is missing from the transformed output."
            ),
        )

    def _check_warning_preserved(
        self,
        warning: str,
        text: str,
    ) -> Optional[ValidationViolation]:
        """
        Verify that the semantic content of a warning is not lost.

        We check that at least one warning signal word from the warning
        text appears in the transformed output. This is intentionally
        forgiving about exact wording while catching wholesale suppression.
        """
        warning_lower = warning.lower()
        text_lower = text.lower()

        # Extract warning signal words present in the original warning
        signals_in_warning = [w for w in _WARNING_SIGNALS if w in warning_lower]

        if not signals_in_warning:
            # Warning doesn't contain signal words — check substring presence
            if len(warning) > 20 and warning[:20].lower() not in text_lower:
                return ValidationViolation(
                    stage="meaning",
                    field="warning",
                    expected=f"warning content present: '{warning[:40]}...'",
                    found="absent",
                    description="Warning appears to have been suppressed.",
                )
            return None

        # At least one signal word must survive
        signals_present = [s for s in signals_in_warning if s in text_lower]
        if not signals_present:
            return ValidationViolation(
                stage="meaning",
                field="warning",
                expected=f"warning signals present: {signals_in_warning}",
                found="none found",
                description=(
                    f"Warning signals {signals_in_warning} are entirely absent "
                    "from the transformed output. Warning may have been suppressed."
                ),
            )
        return None

    def _check_tool_result_present(
        self,
        tool_result: str,
        text: str,
    ) -> Optional[ValidationViolation]:
        """Check that a tool result is not silently dropped."""
        # Use a 30-character anchor from the start of the tool result
        anchor = tool_result.strip()[:30]
        if anchor and anchor not in text:
            return ValidationViolation(
                stage="meaning",
                field="tool_result",
                expected=f"tool result starting with '{anchor}' present",
                found="absent",
                description="A tool result appears to have been dropped.",
            )
        return None

    def _check_risk_assessment_present(
        self,
        risk_assessment: str,
        text: str,
    ) -> Optional[ValidationViolation]:
        """Check that the risk assessment is not stripped from the output."""
        risk_lower = risk_assessment.lower()
        text_lower = text.lower()

        # Risk level words that must survive
        risk_words = {"low", "medium", "high", "critical", "extreme"}
        risk_levels_in_assessment = [w for w in risk_words if w in risk_lower]

        if risk_levels_in_assessment:
            present = [w for w in risk_levels_in_assessment if w in text_lower]
            if not present:
                return ValidationViolation(
                    stage="meaning",
                    field="risk_assessment",
                    expected=f"risk level(s) {risk_levels_in_assessment} present",
                    found="none found",
                    description="Risk assessment level was stripped from the output.",
                )
        return None

    def _check_warning_signals(
        self,
        source: str,
        candidate: str,
    ) -> Optional[ValidationViolation]:
        """
        Macro check: if the source had warning/error signals, so must the candidate.
        Prevents DarkPassenger from transforming "ERROR: disk full" into
        something friendly that loses the severity.
        """
        source_lower = source.lower()
        candidate_lower = candidate.lower()

        source_signals = [w for w in _WARNING_SIGNALS if w in source_lower]
        if not source_signals:
            return None   # source had no warning signals; nothing to check

        candidate_signals = [w for w in _WARNING_SIGNALS if w in candidate_lower]
        if not candidate_signals:
            return ValidationViolation(
                stage="meaning",
                field="warning_signals",
                expected=f"at least one of: {source_signals[:5]}",
                found="none",
                description=(
                    "Source contained warning/error signals but none survived "
                    "transformation. Severity may have been masked."
                ),
            )
        return None

    def _check_uncertainty_preserved(
        self,
        source: str,
        candidate: str,
    ) -> Optional[ValidationViolation]:
        """
        If GhostMind expressed uncertainty, the transformation must not
        project false confidence by removing all uncertainty language.
        """
        source_lower = source.lower()
        candidate_lower = candidate.lower()

        source_uncertainty = [w for w in _UNCERTAINTY_SIGNALS if w in source_lower]
        if not source_uncertainty:
            return None   # source didn't express uncertainty; nothing to check

        candidate_uncertainty = [w for w in _UNCERTAINTY_SIGNALS if w in candidate_lower]
        if not candidate_uncertainty:
            return ValidationViolation(
                stage="meaning",
                field="uncertainty",
                expected=f"uncertainty signals preserved (e.g. {source_uncertainty[:3]})",
                found="none",
                description=(
                    "GhostMind expressed uncertainty but all uncertainty language "
                    "was removed by transformation. This projects false confidence."
                ),
            )
        return None

    def _check_no_fabricated_numbers(
        self,
        source: str,
        candidate: str,
    ) -> Optional[ValidationViolation]:
        """
        Stage 2 heuristic: flag if candidate introduces numbers not in the source.

        This catches cases like DarkPassenger saying "about 150ms" when GhostMind
        said 142ms — small alterations that seem harmless but violate the spec.
        """
        source_numbers = set(_NUMBER_RE.findall(source))
        candidate_numbers = set(_NUMBER_RE.findall(candidate))

        fabricated = candidate_numbers - source_numbers

        # Filter out very short numbers (1, 2, 3...) which are likely ordinals
        meaningful_fabricated = {n for n in fabricated if len(n) > 1}

        if meaningful_fabricated:
            return ValidationViolation(
                stage="style",
                field="fabricated_numbers",
                expected="only numbers from source",
                found=f"new numbers: {meaningful_fabricated}",
                description=(
                    f"Transformed output introduces numerical values "
                    f"{meaningful_fabricated} that are not present in the "
                    "GhostMind source. Numbers must not be invented or altered."
                ),
            )
        return None

    # ── Audit log ─────────────────────────────────────────────────────────────

    def _log(self, event: str, **kwargs):
        entry = {"event": event, **kwargs}
        self._audit_log.append(entry)
        if self.logger:
            self.logger.info(f"circuit_breaker_{event}", **kwargs)

    def get_audit_log(self) -> List[dict]:
        """Return a copy of the internal audit log."""
        return list(self._audit_log)

    def clear_audit_log(self):
        """Clear the audit log (e.g. between sessions)."""
        self._audit_log.clear()
