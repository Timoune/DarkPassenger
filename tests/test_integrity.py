"""
tests/test_integrity.py — DarkPassenger Integrity Layer Test Suite

Tests the complete safety-first architecture:
    - GhostMindOutput construction and finalization
    - CircuitBreaker three-stage validation pipeline
    - Critical Response Override (all triggering conditions)
    - CommunicationRulesEngine pre-flight checks
    - Immutable Core rules
    - Integrity Rules
    - Edge cases and adversarial inputs

Run from the DarkPassenger/ directory:
    python -m pytest tests/test_integrity.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from dp_types.integrity_types import (
    GhostMindOutput,
    CriticalityLevel,
    ProtectedField,
    ValidationStatus,
    ValidationResult,
)
from integrity.circuit_breaker import CircuitBreaker, TransformedOutput
from integrity.communication_rules import (
    CommunicationRulesEngine,
    RuleCategory,
    check_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cb() -> CircuitBreaker:
    return CircuitBreaker()


def _rules() -> CommunicationRulesEngine:
    return CommunicationRulesEngine()


def _output(content: str, **kwargs) -> GhostMindOutput:
    return GhostMindOutput(content=content, **kwargs).finalize()


def _candidate(text: str) -> TransformedOutput:
    return TransformedOutput(content=text)


# ===========================================================================
# GhostMindOutput — construction and finalization
# ===========================================================================

class TestGhostMindOutput(unittest.TestCase):

    def test_finalize_sets_checksums(self):
        out = GhostMindOutput(content="Latency is 142ms")
        out.add_number("latency_ms", 142)
        out.finalize()
        self.assertTrue(out.protected_fields[0].checksum)

    def test_finalized_blocks_mutation(self):
        out = GhostMindOutput(content="hello").finalize()
        with self.assertRaises(RuntimeError):
            out.content = "tampered"

    def test_requires_override_emergency(self):
        out = GhostMindOutput.emergency("system down").finalize()
        self.assertTrue(out.requires_override)

    def test_requires_override_security(self):
        out = GhostMindOutput.security_alert("auth breach").finalize()
        self.assertTrue(out.requires_override)

    def test_requires_override_system_fail(self):
        out = GhostMindOutput.system_failure("disk full").finalize()
        self.assertTrue(out.requires_override)

    def test_requires_override_security_flag(self):
        out = GhostMindOutput(content="ok")
        out.add_security_flag("unauthorized_access")
        out.finalize()
        self.assertTrue(out.requires_override)

    def test_normal_does_not_require_override(self):
        out = GhostMindOutput.normal("Hello!").finalize()
        self.assertFalse(out.requires_override)

    def test_expression_confidence_cap_levels(self):
        cases = {
            CriticalityLevel.NORMAL:      0.90,
            CriticalityLevel.TECHNICAL:   0.50,
            CriticalityLevel.HIGH_RISK:   0.50,
            CriticalityLevel.EMERGENCY:   0.05,
            CriticalityLevel.SECURITY:    0.00,
            CriticalityLevel.SYSTEM_FAIL: 0.00,
        }
        for level, expected_cap in cases.items():
            out = GhostMindOutput(content="x", criticality=level).finalize()
            self.assertAlmostEqual(
                out.expression_confidence_cap, expected_cap, places=3,
                msg=f"Cap mismatch for {level}"
            )


# ===========================================================================
# CircuitBreaker — Critical Response Override
# ===========================================================================

class TestCriticalResponseOverride(unittest.TestCase):

    def _assert_override(self, out: GhostMindOutput):
        cb = _cb()
        result = cb.validate(out, _candidate("some personality-transformed text"))
        self.assertEqual(result.status, ValidationStatus.BYPASS)
        self.assertTrue(result.override_active)
        # Override output should contain original GhostMind content
        self.assertIn(out.content, result.safe_output)

    def test_override_on_emergency(self):
        out = GhostMindOutput.emergency("Power failure detected.").finalize()
        self._assert_override(out)

    def test_override_on_security_alert(self):
        out = GhostMindOutput.security_alert("Unauthorized access.").finalize()
        self._assert_override(out)

    def test_override_on_system_failure(self):
        out = GhostMindOutput.system_failure("Database unreachable.").finalize()
        self._assert_override(out)

    def test_override_includes_warnings(self):
        out = GhostMindOutput.emergency("Core dump.").finalize()
        # Re-create with warning (can't mutate finalized)
        out2 = GhostMindOutput(
            content="Core dump.",
            criticality=CriticalityLevel.EMERGENCY,
            warnings=["CRITICAL: memory overflow"]
        ).finalize()
        cb = _cb()
        result = cb.validate(out2, _candidate("everything is fine"))
        self.assertIn("memory overflow", result.safe_output)

    def test_override_includes_security_flags(self):
        out = GhostMindOutput(
            content="Access denied.",
            security_flags=["unauthorized_access"],
        ).finalize()
        cb = _cb()
        result = cb.validate(out, _candidate("no worries"))
        self.assertIn("unauthorized_access", result.safe_output)

    def test_override_ignores_candidate_content(self):
        """Whatever DarkPassenger proposes is ignored on override."""
        out = GhostMindOutput.security_alert("Breach detected.").finalize()
        cb = _cb()
        candidate_text = "Hey there! Everything seems totally fine!"
        result = cb.validate(out, _candidate(candidate_text))
        self.assertNotIn("totally fine", result.safe_output)

    def test_normal_criticality_does_not_override(self):
        out = GhostMindOutput.normal("The weather is nice.").finalize()
        cb = _cb()
        result = cb.validate(out, _candidate("The weather is nice."))
        self.assertNotEqual(result.status, ValidationStatus.BYPASS)


# ===========================================================================
# CircuitBreaker — Stage 1: Meaning Validation
# ===========================================================================

class TestStage1MeaningValidation(unittest.TestCase):

    def test_passes_when_number_present(self):
        out = GhostMindOutput(content="Latency is 142ms").add_number("latency", 142).finalize()
        candidate = _candidate("Server latency is running at 142ms — solid.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.passed)

    def test_fails_when_number_absent(self):
        out = GhostMindOutput(content="Latency is 142ms").add_number("latency", 142).finalize()
        candidate = _candidate("Server latency is running at about 140ms.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)
        self.assertTrue(any("latency" in v.field for v in result.violations))

    def test_fails_when_number_altered(self):
        out = GhostMindOutput(content="Error rate is 3.7%").add_number("error_rate", 3.7).finalize()
        candidate = _candidate("Error rate is around 4% right now.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)

    def test_passes_with_integer_float_equivalence(self):
        """142 and 142.0 should both count as the same number."""
        out = GhostMindOutput(content="Count: 100").add_number("count", 100).finalize()
        candidate = _candidate("The count is 100.0 items.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.passed)

    def test_fails_when_warning_suppressed(self):
        out = GhostMindOutput(
            content="Task complete.",
            warnings=["WARNING: disk usage at 94%"]
        ).finalize()
        candidate = _candidate("Your task is complete! Great job!")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)
        self.assertTrue(any("warning" in v.field for v in result.violations))

    def test_passes_when_warning_preserved(self):
        out = GhostMindOutput(
            content="Task complete.",
            warnings=["WARNING: disk usage at 94%"]
        ).add_number("disk_pct", 94).finalize()
        candidate = _candidate("Task complete — but warning: disk usage at 94%. Please clean up.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.passed)

    def test_fails_when_tool_result_dropped(self):
        out = GhostMindOutput(
            content="Here are your results.",
            tool_results=["[QUERY RESULT] 42 records found"]
        ).finalize()
        candidate = _candidate("I found some records for you!")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)

    def test_passes_when_tool_result_preserved(self):
        out = GhostMindOutput(
            content="Here are your results.",
            tool_results=["[QUERY RESULT] 42 records found"]
        ).finalize()
        candidate = _candidate("Here's what I found: [QUERY RESULT] 42 records found")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.passed)

    def test_fails_when_risk_assessment_stripped(self):
        out = GhostMindOutput(
            content="This action has risk.",
            risk_assessment="HIGH risk — irreversible action"
        ).finalize()
        candidate = _candidate("Sure, I can do that for you!")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)

    def test_passes_when_risk_assessment_preserved(self):
        out = GhostMindOutput(
            content="This action has risk.",
            risk_assessment="HIGH risk — irreversible action"
        ).finalize()
        candidate = _candidate("Note: this is a HIGH risk action that is irreversible. Proceed?")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.passed)

    def test_fails_when_uncertainty_removed(self):
        out = GhostMindOutput(
            content="The value is approximately 50, though I'm uncertain.",
            uncertainty_score=0.6
        ).finalize()
        candidate = _candidate("The value is definitely 50.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)

    def test_passes_when_uncertainty_preserved(self):
        out = GhostMindOutput(
            content="Possibly around 50.",
            uncertainty_score=0.5
        ).finalize()
        candidate = _candidate("It's possibly around 50 — not fully certain.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.passed)

    def test_warning_signal_words_must_survive(self):
        """'error' in source → some warning signal must appear in candidate."""
        out = GhostMindOutput(content="error: connection refused").finalize()
        candidate = _candidate("Looks like we have a small hiccup — please try again!")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)

    def test_warning_signal_words_pass_when_present(self):
        out = GhostMindOutput(content="error: connection refused").finalize()
        candidate = _candidate("There was an error: the connection was refused.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.passed)


# ===========================================================================
# CircuitBreaker — Stage 2: Style Validation
# ===========================================================================

class TestStage2StyleValidation(unittest.TestCase):

    def test_fails_on_empty_candidate(self):
        out = GhostMindOutput.normal("Hello world.").finalize()
        candidate = _candidate("")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)

    def test_fails_on_whitespace_only_candidate(self):
        out = GhostMindOutput.normal("Hello world.").finalize()
        candidate = _candidate("   \n\t  ")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)

    def test_fails_when_fabricated_numbers_introduced(self):
        out = GhostMindOutput(content="The queue has items in it.").finalize()
        # Candidate invents "47 items" — not in source
        candidate = _candidate("The queue has 47 items in it.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)
        self.assertTrue(
            any("fabricated" in v.field for v in result.violations)
        )

    def test_passes_when_numbers_match_source(self):
        out = GhostMindOutput(content="Processing 47 items.").finalize()
        candidate = _candidate("I'm currently processing 47 items.")
        result = _cb().validate(out, candidate)
        self.assertTrue(result.passed)

    def test_single_digit_ordinals_not_flagged(self):
        """Numbers like 1, 2, 3 in lists/steps should not trigger fabrication check."""
        out = GhostMindOutput(content="Follow these steps to complete it.").finalize()
        candidate = _candidate("1. First do this. 2. Then do that. 3. Done!")
        result = _cb().validate(out, candidate)
        # Should pass — single digits are filtered
        self.assertTrue(result.passed)


# ===========================================================================
# CircuitBreaker — Stage 3: Integrity Safeguard
# ===========================================================================

class TestStage3IntegritySafeguard(unittest.TestCase):

    def test_checksum_validates_correctly(self):
        out = GhostMindOutput(content="Score: 98.6").add_number("score", 98.6).finalize()
        candidate = _candidate("Your score is 98.6 — excellent!")
        result = _cb().validate(out, candidate)
        self.assertEqual(result.status, ValidationStatus.PASS)

    def test_tampered_checksum_detected(self):
        """Manually corrupt a checksum after finalization bypass."""
        out = GhostMindOutput(content="Score: 98.6").add_number("score", 98.6).finalize()

        # Bypass the immutability guard to simulate tampering
        object.__setattr__(out, "_finalized", False)
        out.protected_fields[0].checksum = "deadbeef" * 8
        object.__setattr__(out, "_finalized", True)

        candidate = _candidate("Your score is 98.6 — excellent!")
        result = _cb().validate(out, candidate)
        # Should fail because checksum no longer matches the value
        self.assertTrue(result.failed)

    def test_fallback_to_raw_on_failure(self):
        """safe_output must always be safe, even on validation failure."""
        out = GhostMindOutput(
            content="Error rate is 3.7%"
        ).add_number("error_rate", 3.7).finalize()
        candidate = _candidate("Things seem mostly fine!")   # drops the number

        result = _cb().validate(out, candidate)
        self.assertTrue(result.failed)
        # The raw GhostMind content is the fallback
        self.assertEqual(result.safe_output, out.content)


# ===========================================================================
# CommunicationRulesEngine — Pre-flight
# ===========================================================================

class TestCommunicationRulesEngine(unittest.TestCase):

    def test_override_blocks_transformation(self):
        out = GhostMindOutput.emergency("Power failure.").finalize()
        result = _rules().pre_flight(out)
        self.assertTrue(result.blocked)
        self.assertTrue(result.override_active)
        self.assertEqual(result.max_expression, 0.0)

    def test_security_flag_blocks_transformation(self):
        out = GhostMindOutput(
            content="All good.",
            security_flags=["sql_injection_attempt"]
        ).finalize()
        result = _rules().pre_flight(out)
        self.assertTrue(result.blocked)

    def test_unfinalized_output_with_protected_fields_blocked(self):
        out = GhostMindOutput(content="Value: 42")
        out.add_number("val", 42)
        # deliberately NOT calling .finalize()
        result = _rules().pre_flight(out)
        self.assertTrue(result.blocked)
        rule_ids = [v.rule_id for v in result.violations]
        self.assertIn("IC-03", rule_ids)

    def test_normal_output_passes_preflight(self):
        out = GhostMindOutput.normal("Hello, how can I help?").finalize()
        result = _rules().pre_flight(out)
        self.assertTrue(result.allowed)
        self.assertFalse(result.override_active)

    def test_max_expression_attenuated_by_uncertainty(self):
        """High uncertainty should reduce expression confidence cap."""
        out = GhostMindOutput(
            content="Possibly around 50.",
            criticality=CriticalityLevel.NORMAL,
            uncertainty_score=0.5
        ).finalize()
        result = _rules().pre_flight(out)
        # normal cap is 0.90, uncertainty 0.5 → 0.90 × 0.5 = 0.45
        self.assertAlmostEqual(result.max_expression, 0.45, places=3)

    def test_max_expression_full_certainty(self):
        out = GhostMindOutput(
            content="The answer is 42.",
            uncertainty_score=0.0
        ).finalize()
        result = _rules().pre_flight(out)
        self.assertAlmostEqual(result.max_expression, 0.90, places=3)

    def test_max_expression_technical_context(self):
        out = GhostMindOutput.technical("Deploy complete.").finalize()
        result = _rules().pre_flight(out)
        self.assertAlmostEqual(result.max_expression, 0.50, places=3)

    def test_ir01_warning_issued_for_failure_state_as_normal(self):
        """Failure content marked NORMAL should get a warning (not a hard block)."""
        out = GhostMindOutput(
            content="error: database connection failed",
            criticality=CriticalityLevel.NORMAL
        ).finalize()
        result = _rules().pre_flight(out)
        # Should pass (just a warning), not hard block
        self.assertTrue(result.allowed)
        rule_ids = [v.rule_id for v in result.violations]
        self.assertIn("IR-01", rule_ids)

    def test_ir02_warning_for_tool_result_without_protected_fields(self):
        out = GhostMindOutput(
            content="Here are your results.",
            tool_results=["[RESULT] 10 rows"]
        ).finalize()
        result = _rules().pre_flight(out)
        self.assertTrue(result.allowed)   # just a warning, not hard block
        rule_ids = [v.rule_id for v in result.violations]
        self.assertIn("IR-02", rule_ids)

    def test_convenience_check_rules_function(self):
        from integrity.communication_rules import check_rules
        out = GhostMindOutput.normal("Hi there!").finalize()
        result = check_rules(out)
        self.assertTrue(result.allowed)


# ===========================================================================
# Integration — full flow
# ===========================================================================

class TestFullIntegrityFlow(unittest.TestCase):

    def test_normal_conversation_full_flow(self):
        """
        Happy path: normal conversation, numbers present, no warnings.
        Pre-flight passes → CircuitBreaker passes → safe output delivered.
        """
        ghost = (
            GhostMindOutput(content="Your request processed in 87ms. 3 results found.")
            .add_number("latency_ms", 87)
            .add_number("result_count", 3)
            .finalize()
        )

        # Pre-flight
        preflight = _rules().pre_flight(ghost)
        self.assertTrue(preflight.allowed)
        self.assertFalse(preflight.override_active)

        # DarkPassenger transforms (simulated)
        candidate = _candidate(
            "Done! Processed in 87ms and found 3 results — pretty quick!"
        )

        # Validation
        result = _cb().validate(ghost, candidate)
        self.assertTrue(result.passed)
        self.assertEqual(result.safe_output, candidate.content)

    def test_emergency_bypasses_personality_entirely(self):
        """
        Emergency path: override fires, personality never applied.
        """
        ghost = GhostMindOutput.emergency(
            "CRITICAL: Memory exhausted. System halting."
        ).finalize()

        # Pre-flight
        preflight = _rules().pre_flight(ghost)
        self.assertTrue(preflight.blocked)
        self.assertTrue(preflight.override_active)

        # CircuitBreaker
        candidate = _candidate("Oh no, looks like we're having some trouble! 😅")
        result = _cb().validate(ghost, candidate)
        self.assertEqual(result.status, ValidationStatus.BYPASS)
        self.assertIn("Memory exhausted", result.safe_output)
        self.assertNotIn("😅", result.safe_output)

    def test_personality_cannot_hide_risk_assessment(self):
        """
        Personality transformation that strips risk level is rejected.
        """
        ghost = (
            GhostMindOutput(
                content="This will delete all user data.",
                criticality=CriticalityLevel.HIGH_RISK,
                risk_assessment="CRITICAL risk — irreversible data deletion",
                warnings=["This action cannot be undone."]
            )
            .finalize()
        )

        # Pre-flight should warn about HIGH_RISK
        preflight = _rules().pre_flight(ghost)
        # HIGH_RISK without security flags does not hard-block pre-flight
        # (it allows transformation at 50% expression confidence)
        self.assertTrue(preflight.allowed)
        self.assertAlmostEqual(preflight.max_expression, 0.50, places=2)

        # But a candidate that strips the risk level must fail
        friendly_candidate = _candidate(
            "Sure! I'll take care of that right away!"
        )
        result = _cb().validate(ghost, friendly_candidate)
        self.assertTrue(result.failed)

        # And the safe fallback is the raw GhostMind output
        self.assertEqual(result.safe_output, ghost.content)

    def test_circuit_breaker_never_raises(self):
        """
        Even if the GhostMindOutput is malformed, circuit breaker
        must not raise — it must always return a usable ValidationResult.
        """
        cb = CircuitBreaker()
        # Completely empty output
        ghost = GhostMindOutput(content="")
        ghost._finalized = True  # manually set to bypass guard
        candidate = _candidate("Hello!")
        result = cb.validate(ghost, candidate)
        # Must not raise; safe_output must be a string
        self.assertIsInstance(result.safe_output, str)

    def test_uncertainty_score_affects_expression_confidence(self):
        """
        High uncertainty + technical context = very low expression cap.
        """
        ghost = GhostMindOutput(
            content="The value might be around 50.",
            criticality=CriticalityLevel.TECHNICAL,
            uncertainty_score=0.8,
        ).finalize()
        preflight = _rules().pre_flight(ghost)
        # technical cap 0.50 × (1.0 - 0.8) = 0.10
        self.assertAlmostEqual(preflight.max_expression, 0.10, places=3)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_multiple_protected_fields_all_must_be_present(self):
        ghost = (
            GhostMindOutput(content="CPU: 87%, RAM: 4.2GB, Uptime: 99.9%")
            .add_number("cpu_pct", 87)
            .add_number("ram_gb", 4.2)
            .add_number("uptime_pct", 99.9)
            .finalize()
        )
        # Missing RAM in candidate
        candidate = _candidate("CPU usage is 87%, uptime is 99.9%.")
        result = _cb().validate(ghost, candidate)
        self.assertTrue(result.failed)
        self.assertTrue(any("ram_gb" in v.field for v in result.violations))

    def test_zero_is_a_valid_protected_number(self):
        ghost = (
            GhostMindOutput(content="There are 0 errors.")
            .add_number("error_count", 0)
            .finalize()
        )
        candidate = _candidate("Great news — there are 0 errors!")
        result = _cb().validate(ghost, candidate)
        self.assertTrue(result.passed)

    def test_negative_number_protected(self):
        ghost = (
            GhostMindOutput(content="Temperature is -12.5°C.")
            .add_number("temp_c", -12.5)
            .finalize()
        )
        candidate = _candidate("It's chilly out there — currently -12.5°C.")
        result = _cb().validate(ghost, candidate)
        self.assertTrue(result.passed)

    def test_output_with_no_protected_fields_passes_stage3(self):
        """If no protected fields are set, stage 3 trivially passes."""
        ghost = GhostMindOutput.normal("Have a great day!").finalize()
        candidate = _candidate("Hope your day goes brilliantly!")
        result = _cb().validate(ghost, candidate)
        self.assertTrue(result.passed)

    def test_audit_log_records_events(self):
        cb = CircuitBreaker()
        ghost = GhostMindOutput.normal("Hello!").finalize()
        candidate = _candidate("Hey there!")
        cb.validate(ghost, candidate)
        log = cb.get_audit_log()
        self.assertTrue(len(log) > 0)
        event_names = [entry["event"] for entry in log]
        self.assertIn("validation_passed", event_names)

    def test_audit_log_clears(self):
        cb = CircuitBreaker()
        ghost = GhostMindOutput.normal("Hello!").finalize()
        cb.validate(ghost, _candidate("Hey!"))
        cb.clear_audit_log()
        self.assertEqual(len(cb.get_audit_log()), 0)

    def test_override_output_concatenates_all_signals(self):
        ghost = GhostMindOutput(
            content="System failure.",
            criticality=CriticalityLevel.SYSTEM_FAIL,
            warnings=["Disk is full", "Swap exhausted"],
            tool_results=["[DISK] /dev/sda1: 100%"],
            risk_assessment="CRITICAL",
            security_flags=["anomaly_detected"],
        ).finalize()

        cb = CircuitBreaker()
        result = cb.validate(ghost, _candidate("Everything's fine!"))

        self.assertIn("System failure.", result.safe_output)
        self.assertIn("Disk is full", result.safe_output)
        self.assertIn("Swap exhausted", result.safe_output)
        self.assertIn("[DISK] /dev/sda1: 100%", result.safe_output)
        self.assertIn("CRITICAL", result.safe_output)
        self.assertIn("anomaly_detected", result.safe_output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
