"""
core/speech_fingerprint.py — DarkPassenger Speech Fingerprint Engine

The Speech Fingerprint is DarkPassenger's permanent voice — the structural
communication patterns that remain consistent regardless of which personality
overlay is active.

Personality determines expression (what traits are expressed).
Speech Fingerprint determines voice (how the expression is structured).

Spec §8 — The Speech Fingerprint governs:
    • Preferred sentence lengths and pacing
    • Preferred transitions between ideas
    • Preferred explanation structure
    • Preferred questioning style
    • Preferred organisation patterns
    • Preferred analogy usage frequency

Architecture
────────────
The SpeechFingerprintEngine does NOT rewrite content from scratch.
It applies a light structural pass to the content received from the
TransformationPipeline, adjusting:

    1. Sentence length normalisation
       Long run-ons are split at natural clause boundaries.
       Very short choppy sentences are joined when coherent.

    2. Pacing markers
       High-directness / technical vectors: compact paragraphs, no padding.
       Teaching / analytical vectors: space for examples and elaboration.

    3. Transition injection
       Injects preferred transition phrases at logical break points
       (calibrated to the vector's dominant traits).

    4. Response length gating
       Applies CommunicationHabits.preferred_response_length to clip or
       confirm the response stays within the configured range.

Design constraints
──────────────────
• The fingerprint is NOT configurable per response.  It is fixed in the
  PersonaProfile and must be unchanged by Adaptive Tuning (spec §16).

• The fingerprint NEVER alters facts, numbers, warnings, or tool results.
  It works only on prose structure around protected content.

• If the content is a single short sentence (≤ SINGLE_SENTENCE_TOKENS),
  the fingerprint is a no-op — there is nothing to structure.

• The engine is intentionally stateless.  It does not store previous outputs.

Spec reference: DarkPassenger-Plan.txt §§8, 16
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from core.config_manager import CommunicationHabits
from core.persona_vector import ExpressionBudget, PersonaVector
from core.runtime_state import RuntimeState


# ── Constants ─────────────────────────────────────────────────────────────────

# Responses shorter than this (approx token count) are fingerprint no-ops
SINGLE_SENTENCE_THRESHOLD: int = 20

# Approximate tokens per word (crude estimate for threshold comparisons)
_TOKENS_PER_WORD: float = 1.3

# Maximum number of transitions injected per response
_MAX_TRANSITIONS: int = 3

# Trait threshold above which a trait is considered "high"
_HIGH_TRAIT: float = 0.65

# Sentence boundaries for splitting
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Clause boundaries suitable for splitting long sentences
_CLAUSE_SPLIT_RE = re.compile(
    r',\s+(and|but|yet|so|however|therefore|which|while|although|because)\s+',
    re.IGNORECASE,
)

# Long-sentence word threshold (above this → candidate for splitting)
_LONG_SENTENCE_WORDS: int = 35

# Short-sentence word threshold (below this → candidate for joining)
_SHORT_SENTENCE_WORDS: int = 5


# ── Transition phrase sets keyed by dominant trait ────────────────────────────
#
# Each set contains phrases appropriate when the named trait is dominant.
# The engine picks from the appropriate set based on the vector's top traits.

_TRANSITIONS: dict[str, List[str]] = {
    "analytical_depth": [
        "To break this down further,",
        "Looking at this more closely,",
        "The key distinction here is",
        "It's worth noting that",
    ],
    "directness": [
        "In short:",
        "The bottom line:",
        "Put simply:",
        "Concretely:",
    ],
    "teaching": [
        "Here's why that matters:",
        "Think of it this way:",
        "A useful way to see this:",
        "To put this in context:",
    ],
    "warmth": [
        "The good news is",
        "What this means for you:",
        "Keep in mind:",
        "Worth remembering:",
    ],
    "precision": [
        "More precisely:",
        "To be exact:",
        "The specific detail here is",
        "Note that",
    ],
    "technicality": [
        "Under the hood,",
        "Technically speaking,",
        "At the implementation level,",
        "From a technical standpoint,",
    ],
}

# Default transitions used when no dominant trait matches
_DEFAULT_TRANSITIONS: List[str] = [
    "Additionally,",
    "On the other hand,",
    "It follows that",
    "As a result,",
]


# ── FingerprintResult ─────────────────────────────────────────────────────────

@dataclass
class FingerprintResult:
    """
    Output from one SpeechFingerprintEngine.apply() call.

    Attributes
    ──────────
    output_text:
        The content after fingerprint processing. Structurally shaped but
        factually identical to the input.

    modified:
        True if any structural changes were made.

    transitions_injected:
        Number of transition phrases inserted.

    sentences_split:
        Number of long sentences that were split.

    length_gate_applied:
        True if the response was truncated or confirmed within the length gate.

    dominant_trait:
        The highest-value trait in the vector (drives transition selection).

    pacing_mode:
        The pacing mode chosen: "compact", "balanced", or "expansive".
    """
    output_text:          str
    modified:             bool
    transitions_injected: int
    sentences_split:      int
    length_gate_applied:  bool
    dominant_trait:       str
    pacing_mode:          str


# ── SpeechFingerprintEngine ───────────────────────────────────────────────────

class SpeechFingerprintEngine:
    """
    Applies structural communication patterns to response content.

    Construction:
        engine = SpeechFingerprintEngine(habits=CommunicationHabits())

    Usage (called as fingerprint_hook in TransformationPipeline):
        text = engine(content, vector, budget, state)

    Or with full diagnostics:
        result = engine.apply(content, vector, budget, state)
        text = result.output_text
    """

    def __init__(
        self,
        habits: Optional[CommunicationHabits] = None,
        logger=None,
    ):
        """
        Args:
            habits:
                CommunicationHabits from the active PersonaProfile.
                Controls response length gating and pacing defaults.
                Defaults to a standard CommunicationHabits() if None.

            logger:
                Optional logger.
        """
        self._habits = habits or CommunicationHabits()
        self._logger  = logger

    # ── Public callable interface (matches FingerprintHookFn signature) ───────

    def __call__(
        self,
        content: str,
        vector:  PersonaVector,
        budget:  ExpressionBudget,
        state:   RuntimeState,
    ) -> str:
        """
        Minimal hook-compatible interface. Returns the shaped text.
        For full diagnostics use apply().
        """
        return self.apply(content, vector, budget, state).output_text

    # ── Full interface ────────────────────────────────────────────────────────

    def apply(
        self,
        content: str,
        vector:  PersonaVector,
        budget:  ExpressionBudget,
        state:   RuntimeState,
    ) -> FingerprintResult:
        """
        Apply the speech fingerprint to content.

        Stages:
            1. Short-circuit check  — tiny responses are returned unchanged.
            2. Pacing mode          — choose "compact" / "balanced" / "expansive".
            3. Sentence normalisation — split long, join lone short sentences.
            4. Transition injection  — inject 0–_MAX_TRANSITIONS phrase markers.
            5. Length gating        — confirm/clip to preferred_response_length.

        The content is NEVER changed semantically. Facts, numbers, warnings,
        and tool results pass through unchanged.

        Args:
            content: The raw text from GhostMind (or prior pipeline stages).
            vector:  The resolved PersonaVector after conflict resolution.
            budget:  The active ExpressionBudget.
            state:   The current RuntimeState.

        Returns:
            FingerprintResult with shaped output and diagnostics.
        """
        # ── 1. Short-circuit for tiny responses ──────────────────────────────
        word_count = len(content.split())
        if word_count < SINGLE_SENTENCE_THRESHOLD:
            return FingerprintResult(
                output_text=content,
                modified=False,
                transitions_injected=0,
                sentences_split=0,
                length_gate_applied=False,
                dominant_trait=vector.dominant_traits(1)[0],
                pacing_mode="unchanged_short",
            )

        dominant  = vector.dominant_traits(1)[0]
        pacing    = self._choose_pacing(vector)
        sentences = self._split_into_sentences(content)

        # ── 2. Sentence normalisation ─────────────────────────────────────────
        sentences, splits_made = self._normalise_sentences(sentences, pacing)

        # ── 3. Transition injection ───────────────────────────────────────────
        sentences, transitions_added = self._inject_transitions(
            sentences, vector, pacing
        )

        # ── 4. Re-assemble ────────────────────────────────────────────────────
        text = self._join_sentences(sentences, pacing)

        # ── 5. Length gating ──────────────────────────────────────────────────
        text, length_gated = self._apply_length_gate(text, state)

        modified = splits_made > 0 or transitions_added > 0 or length_gated

        if modified:
            self._log(
                f"fingerprint: pacing={pacing}, splits={splits_made}, "
                f"transitions={transitions_added}, dominant={dominant}"
            )

        return FingerprintResult(
            output_text=text,
            modified=modified,
            transitions_injected=transitions_added,
            sentences_split=splits_made,
            length_gate_applied=length_gated,
            dominant_trait=dominant,
            pacing_mode=pacing,
        )

    # ── Pacing selection ──────────────────────────────────────────────────────

    def _choose_pacing(self, vector: PersonaVector) -> str:
        """
        Choose a pacing mode from the PersonaVector's dominant traits.

        compact    — high directness or precision: tight paragraphs, no padding
        expansive  — high analytical_depth or teaching orientation: room to breathe
        balanced   — default

        Returns one of: "compact", "balanced", "expansive"
        """
        directness  = vector.directness
        precision   = vector.precision
        analytical  = vector.analytical_depth
        warmth      = vector.warmth
        curiosity   = vector.curiosity

        compact_score   = (directness + precision) / 2.0
        expansive_score = (analytical + warmth + curiosity) / 3.0

        if compact_score > _HIGH_TRAIT and compact_score > expansive_score:
            return "compact"
        if expansive_score > _HIGH_TRAIT and expansive_score > compact_score:
            return "expansive"
        return "balanced"

    # ── Sentence utilities ────────────────────────────────────────────────────

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into a list of sentence strings, preserving content."""
        parts = _SENTENCE_END_RE.split(text)
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _join_sentences(sentences: List[str], pacing: str) -> str:
        """
        Re-join sentences, using single-space or paragraph breaks based on pacing.
        """
        if pacing == "expansive":
            # Expansive: group into small paragraphs of 2-3 sentences
            paragraphs: List[List[str]] = []
            for i, s in enumerate(sentences):
                if i % 3 == 0:
                    paragraphs.append([])
                paragraphs[-1].append(s)
            return "\n\n".join(" ".join(p) for p in paragraphs if p)
        else:
            return " ".join(sentences)

    def _normalise_sentences(
        self,
        sentences: List[str],
        pacing: str,
    ) -> Tuple[List[str], int]:
        """
        Split long sentences; optionally join lone short ones.

        Returns (modified_sentences, count_of_splits).
        """
        result: List[str] = []
        splits: int        = 0

        for sentence in sentences:
            words = sentence.split()
            if len(words) > _LONG_SENTENCE_WORDS:
                parts = self._split_at_clause(sentence)
                if len(parts) > 1:
                    result.extend(parts)
                    splits += 1
                    continue
            result.append(sentence)

        # In compact mode, join orphaned very-short final sentences
        if pacing == "compact" and len(result) >= 2:
            final = result[-1].split()
            if len(final) <= _SHORT_SENTENCE_WORDS and len(result) > 1:
                result[-2] = result[-2].rstrip(".") + " — " + result[-1]
                result.pop()

        return result, splits

    @staticmethod
    def _split_at_clause(sentence: str) -> List[str]:
        """
        Attempt to split a long sentence at the first strong clause boundary.
        Returns [sentence] unchanged if no suitable boundary is found.
        """
        match = _CLAUSE_SPLIT_RE.search(sentence)
        if match:
            pivot      = match.start()
            conjunction = match.group(1).capitalize()
            first_part  = sentence[:pivot].strip()
            second_part = conjunction + " " + sentence[match.end():].strip()
            # Ensure first part ends with a period
            if first_part and not first_part[-1] in ".!?":
                first_part += "."
            return [first_part, second_part]
        return [sentence]

    # ── Transition injection ──────────────────────────────────────────────────

    def _inject_transitions(
        self,
        sentences: List[str],
        vector:    PersonaVector,
        pacing:    str,
    ) -> Tuple[List[str], int]:
        """
        Insert transition phrases at appropriate sentence boundaries.

        Transitions are injected only:
            - Between complete sentences (not before the first or last)
            - Up to _MAX_TRANSITIONS times per response
            - Only when pacing is "balanced" or "expansive" (compact skips them)
            - Only at paragraph-like boundaries (every 3rd sentence)

        Returns (modified_sentences, count_inserted).
        """
        if pacing == "compact" or len(sentences) < 3:
            return sentences, 0

        phrase_pool = self._select_transition_pool(vector)
        result:     List[str] = list(sentences)
        inserted:   int        = 0
        pool_idx:   int        = 0

        # Insert at positions 2, 5, 8 ... (every 3rd sentence, after the first)
        insert_positions = list(range(2, len(sentences), 3))[:_MAX_TRANSITIONS]

        # Known transition starters — avoid double-injecting
        _KNOWN_STARTS = frozenset({
            "additionally,", "on the other hand,", "it follows that",
            "as a result,", "to break this down", "looking at this",
            "the key distinction", "it's worth noting", "in short:",
            "the bottom line:", "put simply:", "concretely:", "here's why",
            "think of it", "a useful way", "to put this", "the good news",
            "what this means", "keep in mind:", "worth remembering:",
            "more precisely:", "to be exact:", "the specific detail",
            "note that", "under the hood,", "technically speaking,",
            "at the implementation", "from a technical",
        })

        # Offset tracks cumulative shift as we insert
        for offset, pos in enumerate(insert_positions):
            actual_pos = pos + offset
            if actual_pos >= len(result):
                break
            phrase = phrase_pool[pool_idx % len(phrase_pool)]
            pool_idx += 1

            # Skip if this sentence already appears to start with a transition
            target_lower = result[actual_pos].lower()
            already_transitioned = any(
                target_lower.startswith(t) for t in _KNOWN_STARTS
            )
            if already_transitioned:
                continue

            target = result[actual_pos]
            result[actual_pos] = phrase + " " + target[0].lower() + target[1:]
            inserted += 1

        return result, inserted

    def _select_transition_pool(self, vector: PersonaVector) -> List[str]:
        """
        Choose the transition phrase set that best fits the dominant trait.
        Falls back to the default pool if no trait exceeds _HIGH_TRAIT.
        """
        scored = sorted(
            PersonaVector.trait_names(),
            key=lambda t: getattr(vector, t),
            reverse=True,
        )
        for trait in scored:
            if getattr(vector, trait) >= _HIGH_TRAIT and trait in _TRANSITIONS:
                return _TRANSITIONS[trait]
        return _DEFAULT_TRANSITIONS

    # ── Length gating ─────────────────────────────────────────────────────────

    def _apply_length_gate(
        self,
        text:  str,
        state: RuntimeState,
    ) -> Tuple[str, bool]:
        """
        Apply the preferred_response_length gate from CommunicationHabits.

        short:  ≤ 80 words  — truncate with "…" if over
        medium: no upper truncation (default)
        long:   no truncation

        Returns (gated_text, was_truncated).
        """
        limit_words: Optional[int] = None
        if self._habits.preferred_response_length == "short":
            limit_words = 80

        if limit_words is None:
            return text, False

        words = text.split()
        if len(words) <= limit_words:
            return text, False

        truncated = " ".join(words[:limit_words])
        # Ensure the truncated text ends at a sentence boundary if possible
        last_sentence_end = max(
            truncated.rfind(". "),
            truncated.rfind("! "),
            truncated.rfind("? "),
        )
        if last_sentence_end > len(truncated) * 0.6:
            truncated = truncated[:last_sentence_end + 1]
        else:
            truncated = truncated.rstrip(",;:") + "…"

        return truncated, True

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self._logger is not None:
            try:
                self._logger.debug(message)
            except Exception:
                pass
