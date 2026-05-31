"""
core/config_manager.py — DarkPassenger Configuration Manager

Responsible for loading, validating, saving, and managing persona profiles.

A Persona Profile packages a complete communication identity:
    - Base personality traits (PersonaVector)
    - Expression budget allocations
    - Overlay preferences
    - Stability parameters
    - Communication habits (response length, example frequency)
    - Adaptive tuning flags

Profiles are stored as JSON files. The ConfigManager handles:
    - Loading from file path or raw dict
    - Schema validation (required fields, value ranges, version checks)
    - Version compatibility and migration stubs
    - Runtime in-memory updates
    - Import/Export

What is NOT configurable
─────────────────────────
ConfigManager deals only with personality preferences.
It cannot modify:
    - Validation logic (CircuitBreaker)
    - Communication rules (CommunicationRulesEngine)
    - Expression Confidence calculation
    - Security constraints
    - GhostMind outputs

These are enforced structurally in the integrity layer and cannot
be reached through the configuration system.

Spec reference: DarkPassenger-Plan.txt §§16, 18, 19
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from core.persona_vector import (
    PersonaVector,
    ExpressionBudget,
    OverlayType,
)


# ── Schema constants ──────────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION = "1.0"

# Fields that must be present in every persona profile JSON
REQUIRED_FIELDS = {"schema_version", "profile_id", "display_name", "base_traits"}

# Valid string values for communication_habits fields
VALID_RESPONSE_LENGTHS = {"short", "medium", "long"}
VALID_EXAMPLE_FREQS    = {"rarely", "moderate", "often"}
VALID_TECH_DEPTHS      = {"low", "medium", "high", "adaptive"}


# ── Sub-profile dataclasses ───────────────────────────────────────────────────

@dataclass
class OverlayPreferences:
    """
    Configures how overlays are applied by default.

    default_overlay: applied when no explicit overlay is passed at runtime.
    blends: named blend presets, e.g. {"teaching_focused": {"teaching": 0.7, "focused": 0.3}}
    """
    default_overlay: Optional[str] = None      # OverlayType value or None
    blends: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "default_overlay": self.default_overlay,
            "blends": self.blends,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OverlayPreferences":
        return cls(
            default_overlay=data.get("default_overlay"),
            blends=data.get("blends", {}),
        )

    def get_default(self) -> Optional[OverlayType]:
        if self.default_overlay is None:
            return None
        try:
            return OverlayType(self.default_overlay)
        except ValueError:
            return None


@dataclass
class StabilityParameters:
    """
    Controls the Communication Stability Layer (Part 5).

    smoothing_factor: 0.0 = no smoothing (abrupt changes allowed),
                      1.0 = fully locked (changes never apply).
                      Recommended range: 0.3–0.7.

    drift_threshold:  Euclidean distance between PersonaVectors above which
                      smoothing activates. Higher = more permissive.
    """
    smoothing_factor: float = 0.50
    drift_threshold:  float = 0.30

    def to_dict(self) -> dict:
        return {
            "smoothing_factor": self.smoothing_factor,
            "drift_threshold":  self.drift_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StabilityParameters":
        return cls(
            smoothing_factor=float(data.get("smoothing_factor", 0.50)),
            drift_threshold=float(data.get("drift_threshold", 0.30)),
        )

    def validate(self) -> tuple[bool, str]:
        if not 0.0 <= self.smoothing_factor <= 1.0:
            return False, f"smoothing_factor {self.smoothing_factor} not in [0, 1]"
        if self.drift_threshold <= 0.0:
            return False, f"drift_threshold {self.drift_threshold} must be > 0"
        return True, ""


@dataclass
class CommunicationHabits:
    """
    Style preferences that influence the Speech Fingerprint (Part 7-8).

    These are learned over time by Adaptive Tuning (Part 10).
    """
    preferred_response_length: str = "medium"   # short / medium / long
    example_frequency:         str = "moderate"  # rarely / moderate / often
    technical_depth:           str = "adaptive"  # low / medium / high / adaptive

    def to_dict(self) -> dict:
        return {
            "preferred_response_length": self.preferred_response_length,
            "example_frequency":         self.example_frequency,
            "technical_depth":           self.technical_depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommunicationHabits":
        return cls(
            preferred_response_length=data.get("preferred_response_length", "medium"),
            example_frequency=data.get("example_frequency", "moderate"),
            technical_depth=data.get("technical_depth", "adaptive"),
        )

    def validate(self) -> tuple[bool, str]:
        if self.preferred_response_length not in VALID_RESPONSE_LENGTHS:
            return False, (
                f"preferred_response_length '{self.preferred_response_length}' "
                f"must be one of {VALID_RESPONSE_LENGTHS}"
            )
        if self.example_frequency not in VALID_EXAMPLE_FREQS:
            return False, (
                f"example_frequency '{self.example_frequency}' "
                f"must be one of {VALID_EXAMPLE_FREQS}"
            )
        if self.technical_depth not in VALID_TECH_DEPTHS:
            return False, (
                f"technical_depth '{self.technical_depth}' "
                f"must be one of {VALID_TECH_DEPTHS}"
            )
        return True, ""


@dataclass
class AdaptiveTuning:
    """
    Flags that control what the Adaptive Tuning layer (Part 10) is allowed to learn.

    What may be learned (per spec §16):
        - Preferred answer length
        - Technical depth
        - Humor tolerance
        - Explanation density
        - Example frequency

    What may NOT be learned (forbidden per spec §16):
        - Speech Fingerprint
        - Validation logic
        - Communication rules
        - Expression Confidence logic
        - Security constraints
        - GhostMind outputs
    """
    learn_response_length: bool = True
    learn_technical_depth: bool = True
    learn_humor_tolerance: bool = True
    learn_example_frequency: bool = True

    def to_dict(self) -> dict:
        return {
            "learn_response_length":  self.learn_response_length,
            "learn_technical_depth":  self.learn_technical_depth,
            "learn_humor_tolerance":  self.learn_humor_tolerance,
            "learn_example_frequency": self.learn_example_frequency,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptiveTuning":
        return cls(
            learn_response_length=bool(data.get("learn_response_length", True)),
            learn_technical_depth=bool(data.get("learn_technical_depth", True)),
            learn_humor_tolerance=bool(data.get("learn_humor_tolerance", True)),
            learn_example_frequency=bool(data.get("learn_example_frequency", True)),
        )


# ── PersonaProfile ────────────────────────────────────────────────────────────

@dataclass
class PersonaProfile:
    """
    A complete, packaged communication identity.

    Contains everything DarkPassenger needs to produce a PersonaVector
    and govern its communication style for a session.

    Spec reference: DarkPassenger-Plan.txt §19
    """
    schema_version:        str
    profile_id:            str
    display_name:          str
    description:           str
    base_traits:           PersonaVector
    expression_budget:     ExpressionBudget
    overlay_preferences:   OverlayPreferences
    stability_parameters:  StabilityParameters
    communication_habits:  CommunicationHabits
    adaptive_tuning:       AdaptiveTuning

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version":       self.schema_version,
            "profile_id":           self.profile_id,
            "display_name":         self.display_name,
            "description":          self.description,
            "base_traits":          self.base_traits.to_dict(),
            "expression_budget":    self.expression_budget.to_dict(),
            "overlay_preferences":  self.overlay_preferences.to_dict(),
            "stability_parameters": self.stability_parameters.to_dict(),
            "communication_habits": self.communication_habits.to_dict(),
            "adaptive_tuning":      self.adaptive_tuning.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaProfile":
        missing = REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(f"PersonaProfile missing required fields: {missing}")

        return cls(
            schema_version=str(data.get("schema_version", CURRENT_SCHEMA_VERSION)),
            profile_id=str(data["profile_id"]),
            display_name=str(data.get("display_name", data["profile_id"])),
            description=str(data.get("description", "")),
            base_traits=PersonaVector.from_dict(data["base_traits"]),
            expression_budget=ExpressionBudget.from_dict(
                data.get("expression_budget", {})
            ),
            overlay_preferences=OverlayPreferences.from_dict(
                data.get("overlay_preferences", {})
            ),
            stability_parameters=StabilityParameters.from_dict(
                data.get("stability_parameters", {})
            ),
            communication_habits=CommunicationHabits.from_dict(
                data.get("communication_habits", {})
            ),
            adaptive_tuning=AdaptiveTuning.from_dict(
                data.get("adaptive_tuning", {})
            ),
        )


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when a persona profile fails schema validation."""
    pass


def _validate_profile_dict(data: dict) -> list[str]:
    """
    Validate a raw persona profile dict. Returns a list of error strings.
    An empty list means the profile is valid.
    """
    errors = []

    # Required fields
    for field_name in REQUIRED_FIELDS:
        if field_name not in data:
            errors.append(f"Missing required field: '{field_name}'")

    if errors:
        return errors  # can't continue without required fields

    # profile_id must not be empty
    if not str(data.get("profile_id", "")).strip():
        errors.append("'profile_id' must not be empty")

    # base_traits: all values in [0.0, 1.0]
    base_traits = data.get("base_traits", {})
    if not isinstance(base_traits, dict):
        errors.append("'base_traits' must be a dict")
    else:
        valid_names = set(PersonaVector.trait_names())
        for k, v in base_traits.items():
            if k not in valid_names:
                errors.append(f"Unknown trait in base_traits: '{k}'")
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                errors.append(f"base_traits.{k} must be a number, got {v!r}")
                continue
            if not 0.0 <= fv <= 1.0:
                errors.append(
                    f"base_traits.{k} = {fv} out of range [0.0, 1.0]"
                )

    # expression_budget: non-negative integers summing to <= 100
    budget = data.get("expression_budget", {})
    if isinstance(budget, dict) and budget:
        total = 0
        for k, v in budget.items():
            try:
                iv = int(v)
            except (TypeError, ValueError):
                errors.append(f"expression_budget.{k} must be an integer, got {v!r}")
                continue
            if iv < 0:
                errors.append(f"expression_budget.{k} = {iv} must be >= 0")
            total += iv
        if total > ExpressionBudget.TOTAL:
            errors.append(
                f"expression_budget total {total} exceeds maximum {ExpressionBudget.TOTAL}"
            )

    # stability_parameters
    sp = data.get("stability_parameters", {})
    if isinstance(sp, dict):
        sf = sp.get("smoothing_factor")
        if sf is not None:
            try:
                sf = float(sf)
                if not 0.0 <= sf <= 1.0:
                    errors.append(f"stability_parameters.smoothing_factor {sf} not in [0, 1]")
            except (TypeError, ValueError):
                errors.append("stability_parameters.smoothing_factor must be a number")
        dt = sp.get("drift_threshold")
        if dt is not None:
            try:
                dt = float(dt)
                if dt <= 0.0:
                    errors.append(f"stability_parameters.drift_threshold {dt} must be > 0")
            except (TypeError, ValueError):
                errors.append("stability_parameters.drift_threshold must be a number")

    # communication_habits
    habits = data.get("communication_habits", {})
    if isinstance(habits, dict):
        rl = habits.get("preferred_response_length")
        if rl and rl not in VALID_RESPONSE_LENGTHS:
            errors.append(
                f"communication_habits.preferred_response_length '{rl}' "
                f"must be one of {sorted(VALID_RESPONSE_LENGTHS)}"
            )
        ef = habits.get("example_frequency")
        if ef and ef not in VALID_EXAMPLE_FREQS:
            errors.append(
                f"communication_habits.example_frequency '{ef}' "
                f"must be one of {sorted(VALID_EXAMPLE_FREQS)}"
            )
        td = habits.get("technical_depth")
        if td and td not in VALID_TECH_DEPTHS:
            errors.append(
                f"communication_habits.technical_depth '{td}' "
                f"must be one of {sorted(VALID_TECH_DEPTHS)}"
            )

    return errors


# ── ConfigManager ─────────────────────────────────────────────────────────────

class ConfigManager:
    """
    Central manager for DarkPassenger persona profiles.

    Responsibilities:
        - Load profiles from JSON files or dicts
        - Validate against schema
        - Version compatibility checking
        - Runtime in-memory profile store
        - Active profile management
        - Import / export

    Thread safety: This implementation is single-threaded.
    For concurrent use, the caller is responsible for synchronisation.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: Optional path to a directory of .json persona files.
                        If provided, all .json files are loaded on construction.
        """
        self._profiles:      Dict[str, PersonaProfile] = {}
        self._active_id:     Optional[str] = None
        self._config_dir:    Optional[Path] = Path(config_dir) if config_dir else None

        if self._config_dir and self._config_dir.is_dir():
            self.load_directory(str(self._config_dir))

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_from_dict(self, data: dict) -> PersonaProfile:
        """
        Validate and load a persona profile from a raw dict.

        Raises:
            ValidationError: if the profile fails schema validation.
        """
        errors = _validate_profile_dict(data)
        if errors:
            raise ValidationError(
                f"Profile validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        profile = PersonaProfile.from_dict(data)
        self._profiles[profile.profile_id] = profile

        # Set as active if it's the first profile loaded
        if self._active_id is None:
            self._active_id = profile.profile_id

        return profile

    def load_file(self, path: str) -> PersonaProfile:
        """
        Load a persona profile from a JSON file.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValidationError:   if the profile fails schema validation.
            json.JSONDecodeError: if the file is not valid JSON.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Persona profile not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.load_from_dict(data)

    def load_directory(self, directory: str) -> list[PersonaProfile]:
        """
        Load all .json files in a directory as persona profiles.

        Invalid files are skipped with a warning; they do not halt loading.
        Returns the list of successfully loaded profiles.
        """
        loaded = []
        dir_path = Path(directory)

        if not dir_path.is_dir():
            return loaded

        for json_file in sorted(dir_path.glob("*.json")):
            try:
                profile = self.load_file(str(json_file))
                loaded.append(profile)
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                # Log but don't crash — bad profile files are skipped
                print(f"[ConfigManager] Skipping {json_file.name}: {e}")

        return loaded

    # ── Saving / exporting ────────────────────────────────────────────────────

    def save_file(self, profile_id: str, path: str) -> None:
        """
        Save a loaded profile to a JSON file.

        Raises:
            KeyError: if profile_id is not loaded.
        """
        profile = self.get_profile(profile_id)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)

    def export_dict(self, profile_id: str) -> dict:
        """Return a profile as a plain dict (for serialisation or transport)."""
        return self.get_profile(profile_id).to_dict()

    # ── Profile management ────────────────────────────────────────────────────

    def get_profile(self, profile_id: str) -> PersonaProfile:
        """
        Retrieve a loaded profile by ID.

        Raises:
            KeyError: if the profile is not loaded.
        """
        if profile_id not in self._profiles:
            available = list(self._profiles.keys())
            raise KeyError(
                f"Profile '{profile_id}' not found. "
                f"Available: {available or '(none loaded)'}"
            )
        return self._profiles[profile_id]

    def set_active(self, profile_id: str) -> None:
        """
        Set the active profile by ID.

        Raises:
            KeyError: if the profile is not loaded.
        """
        self.get_profile(profile_id)  # validates existence
        self._active_id = profile_id

    @property
    def active_profile(self) -> Optional[PersonaProfile]:
        """The currently active persona profile, or None if none is loaded."""
        if self._active_id is None:
            return None
        return self._profiles.get(self._active_id)

    @property
    def active_vector(self) -> Optional[PersonaVector]:
        """Shortcut: base_traits PersonaVector of the active profile."""
        p = self.active_profile
        return p.base_traits if p else None

    def list_profiles(self) -> list[str]:
        """Return all loaded profile IDs, sorted alphabetically."""
        return sorted(self._profiles.keys())

    def remove_profile(self, profile_id: str) -> None:
        """
        Remove a profile from the in-memory store.

        If the removed profile was active, active_profile becomes None.
        """
        self._profiles.pop(profile_id, None)
        if self._active_id == profile_id:
            self._active_id = None

    # ── Runtime update ────────────────────────────────────────────────────────

    def update_trait(
        self,
        profile_id: str,
        trait: str,
        value: float,
    ) -> None:
        """
        Update a single base trait on a loaded profile.

        This is an in-memory update only — call save_file() to persist.

        Raises:
            KeyError:   if the profile is not loaded.
            ValueError: if the trait name is invalid or value out of range.
        """
        if trait not in PersonaVector.trait_names():
            raise ValueError(f"Unknown trait: '{trait}'")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Trait value {value} out of range [0.0, 1.0]")

        profile = self.get_profile(profile_id)
        setattr(profile.base_traits, trait, value)

    def validate_dict(self, data: dict) -> tuple[bool, list[str]]:
        """
        Validate a raw profile dict without loading it.

        Returns (True, []) if valid, or (False, [errors]) if invalid.
        """
        errors = _validate_profile_dict(data)
        return (len(errors) == 0), errors

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._profiles)

    def __repr__(self) -> str:
        return (
            f"ConfigManager("
            f"profiles={self.list_profiles()}, "
            f"active={self._active_id!r})"
        )
