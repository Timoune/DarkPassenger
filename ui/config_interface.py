"""
ui/config_interface.py — DarkPassenger Real-Time Configuration Interface

Provides the platform-agnostic UI control model that allows real-time
persona modification via sliders, toggles, dropdowns, and preset selectors.

Architecture (DarkPassenger-Plan.txt §18)
──────────────────────────────────────────
The configuration interface sits between the user (desktop/web/mobile/voice)
and the DarkPassenger system. It does NOT directly mutate pipeline internals.
Every change is emitted as a dp.config.update.v1 BusEvent through the gate.

                ┌─────────────────────────────┐
                │    PersonaConfigInterface    │
                │                             │
                │  Sliders:  trait values     │
                │  Toggles:  overlay on/off   │
                │  Dropdowns: preset / rel.   │
                │  Blends:   multi-overlay    │
                │                             │
                │  apply() → BusEvent emit   │
                └─────────────┬───────────────┘
                              │  dp.config.update.v1
                    ┌─────────▼──────────────┐
                    │  DarkPassengerGate      │
                    │  (BloodyHeart §middleware)│
                    └─────────────────────────┘

Components
──────────
  PersonaConfigInterface   — the main UI model; platform-agnostic
  UIControlState           — snapshot of the current control values
  SliderControl            — a single bounded float control
  ToggleControl            — a boolean on/off control
  DropdownControl          — a string selection from a fixed set
  BlendControl             — multi-overlay blend weight manager
  PresetManager            — named preset save/load/delete

Platform adapters
─────────────────
  DesktopConfigAdapter     — Qt/Tkinter-compatible dict serialiser
  WebConfigAdapter         — JSON/REST-compatible serialiser
  MobileConfigAdapter      — compact mobile-optimised serialiser
  DiscordConfigAdapter     — slash-command compatible serialiser

These adapters produce serialised representations of UIControlState
suitable for each platform's rendering layer. Rendering itself is
outside DarkPassenger's scope — the adapter output is consumed by
the platform UI layer.

Spec reference: DarkPassenger-Plan.txt §18 (Configuration System)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from bloodyheart.events import (
    BusEvent,
    DPEventType,
    EventPriority,
    make_persona_config_update_payload,
)
from core.persona_vector import OverlayType, RelationshipContext

# CoreBus emit callable type
EmitFn = Callable[[BusEvent], None]


# ── Control primitives ────────────────────────────────────────────────────────

@dataclass
class SliderControl:
    """
    A bounded float control — maps to a single personality trait.

    Used for: formality, humor, warmth, directness, technicality,
              confidence, precision, curiosity, analytical_depth.

    Attributes
    ──────────
    name:        Trait name (must match PersonaVector attribute).
    value:       Current value (clamped to [min_val, max_val]).
    min_val:     Minimum allowed value (default 0.0).
    max_val:     Maximum allowed value (default 1.0).
    step:        Smallest meaningful increment (default 0.05).
    label:       Human-readable display label.
    description: Tooltip / accessibility description.
    """
    name:        str
    value:       float
    min_val:     float = 0.0
    max_val:     float = 1.0
    step:        float = 0.05
    label:       str   = ""
    description: str   = ""

    def __post_init__(self):
        self.label = self.label or self.name.replace("_", " ").title()
        self._clamp()

    def set(self, value: float) -> "SliderControl":
        """Return a new SliderControl with the updated value (immutable update)."""
        return SliderControl(
            name=self.name, value=value,
            min_val=self.min_val, max_val=self.max_val,
            step=self.step, label=self.label, description=self.description,
        )

    def _clamp(self) -> None:
        self.value = max(self.min_val, min(self.max_val, self.value))

    def to_dict(self) -> dict:
        return {
            "type": "slider", "name": self.name, "value": round(self.value, 3),
            "min": self.min_val, "max": self.max_val, "step": self.step,
            "label": self.label, "description": self.description,
        }


@dataclass
class ToggleControl:
    """
    A boolean on/off control.

    Used for: overlay activation, adaptive tuning enable/disable,
              personality audit log enable/disable.
    """
    name:        str
    value:       bool
    label:       str  = ""
    description: str  = ""

    def __post_init__(self):
        self.label = self.label or self.name.replace("_", " ").title()

    def set(self, value: bool) -> "ToggleControl":
        return ToggleControl(name=self.name, value=value,
                             label=self.label, description=self.description)

    def to_dict(self) -> dict:
        return {"type": "toggle", "name": self.name, "value": self.value,
                "label": self.label, "description": self.description}


@dataclass
class DropdownControl:
    """
    A single-choice selection from a fixed set of string options.

    Used for: active overlay, relationship context, active persona profile.
    """
    name:        str
    value:       str
    options:     List[str]  = field(default_factory=list)
    label:       str        = ""
    description: str        = ""

    def __post_init__(self):
        self.label = self.label or self.name.replace("_", " ").title()
        if self.options and self.value not in self.options:
            self.value = self.options[0] if self.options else self.value

    def set(self, value: str) -> "DropdownControl":
        if self.options and value not in self.options:
            raise ValueError(f"'{value}' is not a valid option for '{self.name}'")
        return DropdownControl(name=self.name, value=value, options=self.options,
                               label=self.label, description=self.description)

    def to_dict(self) -> dict:
        return {"type": "dropdown", "name": self.name, "value": self.value,
                "options": self.options, "label": self.label,
                "description": self.description}


@dataclass
class BlendControl:
    """
    Multi-overlay blend weight manager.

    Maps each OverlayType to a weight slider (0.0–1.0).
    When blend_active is False, the single_overlay dropdown takes precedence.
    When blend_active is True, all non-zero weights are sent as a blend.

    Weights do not need to sum to 1.0 — the PersonaVectorEngine normalises them.
    """
    weights:      Dict[str, float] = field(default_factory=dict)
    blend_active: bool             = False

    def __post_init__(self):
        # Initialise missing overlays to 0.0
        for ov in OverlayType:
            self.weights.setdefault(ov.value, 0.0)

    def set_weight(self, overlay: str, weight: float) -> "BlendControl":
        new_weights = dict(self.weights)
        new_weights[overlay] = max(0.0, min(1.0, weight))
        return BlendControl(weights=new_weights, blend_active=self.blend_active)

    def set_blend_active(self, active: bool) -> "BlendControl":
        return BlendControl(weights=dict(self.weights), blend_active=active)

    def active_blends(self) -> Dict[str, float]:
        """Return only non-zero weights when blend is active."""
        if not self.blend_active:
            return {}
        return {k: v for k, v in self.weights.items() if v > 0.0}

    def to_dict(self) -> dict:
        return {
            "type": "blend",
            "blend_active": self.blend_active,
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
        }


# ── UIControlState ────────────────────────────────────────────────────────────

@dataclass
class UIControlState:
    """
    Complete snapshot of all UI control values.

    This is what the platform rendering layer reads to draw the UI,
    and what the adapters serialise for their respective platforms.

    Attributes
    ──────────
    trait_sliders:    One SliderControl per personality trait.
    overlay_dropdown: Active single overlay selection.
    relationship_dropdown: Active relationship context.
    profile_dropdown: Active persona profile ID.
    blend_control:    Multi-overlay blend weight state.
    adaptive_toggle:  Whether adaptive tuning is enabled.
    audit_toggle:     Whether audit logging is enabled (always True in prod).
    """
    trait_sliders:         Dict[str, SliderControl]  = field(default_factory=dict)
    overlay_dropdown:      DropdownControl            = field(default_factory=lambda:
        DropdownControl("overlay", "none",
                        options=["none"] + [ov.value for ov in OverlayType],
                        label="Expression Overlay",
                        description="Active expression modifier. 'none' uses base identity."))
    relationship_dropdown: DropdownControl            = field(default_factory=lambda:
        DropdownControl("relationship", RelationshipContext.UNKNOWN.value,
                        options=[r.value for r in RelationshipContext],
                        label="Relationship Context",
                        description="Who DarkPassenger is speaking to."))
    profile_dropdown:      DropdownControl            = field(default_factory=lambda:
        DropdownControl("profile", "default",
                        options=["default"],
                        label="Persona Profile",
                        description="The active persona profile preset."))
    blend_control:         BlendControl               = field(default_factory=BlendControl)
    adaptive_toggle:       ToggleControl              = field(default_factory=lambda:
        ToggleControl("adaptive_tuning", True,
                      label="Adaptive Tuning",
                      description="Allow DarkPassenger to learn communication preferences."))
    audit_toggle:          ToggleControl              = field(default_factory=lambda:
        ToggleControl("audit_log", True,
                      label="Personality Audit Log",
                      description="Record metadata for offline behavioral analysis."))

    def to_dict(self) -> dict:
        return {
            "traits":       {k: v.to_dict() for k, v in self.trait_sliders.items()},
            "overlay":      self.overlay_dropdown.to_dict(),
            "relationship": self.relationship_dropdown.to_dict(),
            "profile":      self.profile_dropdown.to_dict(),
            "blend":        self.blend_control.to_dict(),
            "adaptive":     self.adaptive_toggle.to_dict(),
            "audit":        self.audit_toggle.to_dict(),
        }


# ── Named Presets ─────────────────────────────────────────────────────────────

@dataclass
class PersonaPreset:
    """
    A named snapshot of trait slider values and overlay selection.

    Presets allow one-click persona switching. They do NOT capture
    relationship context (that's session-specific) or security settings.

    Attributes
    ──────────
    name:          Display name (e.g. "Professional", "Casual Friday").
    profile_id:    The persona profile to activate (e.g. "default").
    trait_values:  Dict of trait_name → float overrides.
    overlay:       OverlayType.value or "none".
    description:   Human-readable description shown in the preset selector.
    """
    name:         str
    profile_id:   str
    trait_values: Dict[str, float]  = field(default_factory=dict)
    overlay:      str               = "none"
    description:  str               = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "profile_id": self.profile_id,
            "trait_values": self.trait_values, "overlay": self.overlay,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersonaPreset":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PresetManager:
    """
    Saves, loads, and deletes named PersonaPresets.

    Presets are stored in memory during the session. Call export_json() /
    import_json() to persist them to a file between sessions.

    Built-in presets (read-only, cannot be deleted):
        "DarkPassenger" — the signature high-precision, high-humor persona
        "Professional"  — high formality, low humor, high precision
        "Mentor"        — high warmth, high analytical_depth, medium directness
        "Minimalist"    — maximum directness, minimum expressiveness
    """

    _BUILTIN_PRESETS: List[PersonaPreset] = [
        PersonaPreset(
            name="DarkPassenger",
            profile_id="darkpassenger",
            trait_values={
                "formality": 0.40, "humor": 0.70, "warmth": 0.50,
                "directness": 0.90, "technicality": 0.85,
                "confidence": 0.80, "precision": 0.95,
            },
            overlay="focused",
            description="The signature DarkPassenger voice — precise, confident, with understated wit.",
        ),
        PersonaPreset(
            name="Professional",
            profile_id="professional",
            trait_values={
                "formality": 0.85, "humor": 0.15, "warmth": 0.40,
                "directness": 0.75, "technicality": 0.70,
                "confidence": 0.80, "precision": 0.90,
            },
            overlay="focused",
            description="Polished, formal, and precise. Ideal for business or client-facing contexts.",
        ),
        PersonaPreset(
            name="Mentor",
            profile_id="default",
            trait_values={
                "formality": 0.45, "humor": 0.40, "warmth": 0.80,
                "directness": 0.60, "technicality": 0.60,
                "confidence": 0.70, "analytical_depth": 0.85,
            },
            overlay="teaching",
            description="Patient, warm, and educational. Best for explanations and onboarding.",
        ),
        PersonaPreset(
            name="Minimalist",
            profile_id="minimalist",
            trait_values={
                "formality": 0.60, "humor": 0.10, "warmth": 0.30,
                "directness": 0.95, "technicality": 0.70,
                "confidence": 0.90, "precision": 0.95,
            },
            overlay="focused",
            description="Maximum signal, minimum noise. Short, direct, efficient.",
        ),
    ]

    def __init__(self):
        # user_presets are mutable; builtins are read-only
        self._user_presets: Dict[str, PersonaPreset] = {}
        self._builtins: Dict[str, PersonaPreset] = {
            p.name: p for p in self._BUILTIN_PRESETS
        }

    @property
    def all_presets(self) -> Dict[str, PersonaPreset]:
        """All presets — builtins first, then user-defined."""
        combined = dict(self._builtins)
        combined.update(self._user_presets)
        return combined

    @property
    def preset_names(self) -> List[str]:
        return list(self.all_presets.keys())

    def get(self, name: str) -> Optional[PersonaPreset]:
        """Return the preset with the given name, or None."""
        return self.all_presets.get(name)

    def save(self, preset: PersonaPreset) -> None:
        """
        Save a user-defined preset.

        Raises ValueError if name clashes with a built-in preset.
        """
        if preset.name in self._builtins:
            raise ValueError(
                f"Cannot overwrite built-in preset '{preset.name}'. "
                "Choose a different name."
            )
        self._user_presets[preset.name] = preset

    def delete(self, name: str) -> None:
        """
        Delete a user-defined preset.

        Raises KeyError if the preset does not exist.
        Raises ValueError if the preset is a built-in.
        """
        if name in self._builtins:
            raise ValueError(f"Cannot delete built-in preset '{name}'.")
        if name not in self._user_presets:
            raise KeyError(f"Preset '{name}' not found.")
        del self._user_presets[name]

    def export_json(self) -> str:
        """Export user-defined presets to a JSON string."""
        return json.dumps(
            [p.to_dict() for p in self._user_presets.values()],
            indent=2
        )

    def import_json(self, json_str: str) -> int:
        """
        Import presets from a JSON string (produced by export_json).

        Returns the number of presets imported.
        Skips any that clash with built-in names.
        """
        loaded = json.loads(json_str)
        count = 0
        for d in loaded:
            preset = PersonaPreset.from_dict(d)
            if preset.name not in self._builtins:
                self._user_presets[preset.name] = preset
                count += 1
        return count


# ── PersonaConfigInterface ────────────────────────────────────────────────────

# Standard trait definitions with display labels
_TRAIT_DEFINITIONS: List[Tuple[str, str, str]] = [
    ("formality",       "Formality",        "How formal or casual the language is."),
    ("humor",           "Humor",            "Frequency and weight of wit and levity."),
    ("warmth",          "Warmth",           "Emotional closeness and personal care."),
    ("directness",      "Directness",       "How blunt vs. diplomatic the communication is."),
    ("technicality",    "Technicality",     "Assumed technical depth of the audience."),
    ("confidence",      "Confidence",       "Assertiveness in statements and conclusions."),
    ("precision",       "Precision",        "Accuracy and specificity of language."),
    ("curiosity",       "Curiosity",        "Expressed interest and exploratory questioning."),
    ("analytical_depth","Analytical Depth", "Depth of reasoning and multi-step explanation."),
]


class PersonaConfigInterface:
    """
    Platform-agnostic configuration interface for DarkPassenger.

    This class manages the complete UI control state and translates user
    interactions into dp.config.update.v1 BusEvents emitted through the gate.

    It does NOT directly mutate pipeline internals — every change goes
    through the gate, which enforces safe-mode rules and schema validation.

    Usage
    ─────
        interface = PersonaConfigInterface(
            emit=gate.handle,          # or core_bus.emit
            available_profiles=["default", "professional", "darkpassenger"],
            initial_trait_values={"humor": 0.70, "directness": 0.90, ...},
        )

        # User moves the humor slider to 0.85:
        interface.set_trait("humor", 0.85)
        interface.apply()              # emits dp.config.update.v1

        # User switches to the Professional preset:
        interface.load_preset("Professional")
        interface.apply()

        # Get current state for rendering:
        state = interface.control_state
        # → UIControlState with all current control values

    Real-time updates (no buffering):
        interface.set_trait("humor", 0.85, immediate=True)
        # emits immediately without calling apply()

    Platform adapters:
        desktop_dict = DesktopConfigAdapter.serialise(interface.control_state)
        web_json     = WebConfigAdapter.serialise(interface.control_state)
        mobile_dict  = MobileConfigAdapter.serialise(interface.control_state)
    """

    SOURCE_NAME = "ui"

    def __init__(
        self,
        emit: EmitFn,
        available_profiles: Optional[List[str]] = None,
        initial_trait_values: Optional[Dict[str, float]] = None,
        preset_manager: Optional[PresetManager] = None,
        requested_by: str = "ui",
        logger=None,
    ):
        """
        Args:
            emit:                 Callable to emit BusEvents (gate.handle or bus.emit).
            available_profiles:   Profile IDs to populate the profile dropdown.
            initial_trait_values: Starting trait values (from active persona profile).
            preset_manager:       Shared PresetManager (creates one if not provided).
            requested_by:         Attribution for config-update events ("ui"/"admin"/"api").
            logger:               Optional structured logger.
        """
        self._emit         = emit
        self._requested_by = requested_by
        self._logger       = logger

        self._preset_manager = preset_manager or PresetManager()

        # Build the initial control state
        profiles = available_profiles or ["default"]
        self._state = self._build_initial_state(
            initial_trait_values or {},
            profiles,
        )

        # Pending changes buffer — flushed on apply()
        self._pending: Dict[str, object] = {}

    # ── Control update methods ────────────────────────────────────────────────

    def set_trait(self, trait: str, value: float, immediate: bool = False) -> None:
        """
        Update a trait slider value.

        Args:
            trait:     Trait name (e.g. "humor", "directness").
            value:     New value, clamped to [0.0, 1.0].
            immediate: If True, emit dp.config.update.v1 immediately without
                       buffering. If False (default), buffer until apply().
        """
        if trait not in self._state.trait_sliders:
            if self._logger:
                self._logger.warning("ui_unknown_trait", trait=trait)
            return

        slider = self._state.trait_sliders[trait]
        self._state.trait_sliders[trait] = slider.set(value)

        if immediate:
            self._emit_update(
                trait_overrides={trait: self._state.trait_sliders[trait].value}
            )
        else:
            pending_traits = dict(self._pending.get("trait_overrides") or {})
            pending_traits[trait] = self._state.trait_sliders[trait].value
            self._pending["trait_overrides"] = pending_traits

    def set_overlay(self, overlay: str, immediate: bool = False) -> None:
        """
        Update the active overlay selection.

        Args:
            overlay:   OverlayType.value or "none".
            immediate: Emit immediately if True.
        """
        try:
            self._state.overlay_dropdown = self._state.overlay_dropdown.set(overlay)
        except ValueError:
            if self._logger:
                self._logger.warning("ui_unknown_overlay", overlay=overlay)
            return

        effective_overlay = None if overlay == "none" else overlay

        if immediate:
            self._emit_update(overlay=effective_overlay)
        else:
            self._pending["overlay"] = effective_overlay

    def set_relationship(self, relationship: str, immediate: bool = False) -> None:
        """
        Update the active relationship context.

        Args:
            relationship: RelationshipContext.value.
            immediate:    Emit immediately if True.
        """
        try:
            self._state.relationship_dropdown = self._state.relationship_dropdown.set(relationship)
        except ValueError:
            if self._logger:
                self._logger.warning("ui_unknown_relationship", relationship=relationship)
            return

        if immediate:
            self._emit_update(relationship=relationship)
        else:
            self._pending["relationship"] = relationship

    def set_profile(self, profile_id: str, immediate: bool = False) -> None:
        """
        Switch to a different persona profile.

        Args:
            profile_id: Profile ID string (must be in available_profiles).
            immediate:  Emit immediately if True.
        """
        try:
            self._state.profile_dropdown = self._state.profile_dropdown.set(profile_id)
        except ValueError:
            if self._logger:
                self._logger.warning("ui_unknown_profile", profile_id=profile_id)
            return

        if immediate:
            self._emit_update(profile_id=profile_id)
        else:
            self._pending["profile_id"] = profile_id

    def set_blend_weight(self, overlay: str, weight: float, immediate: bool = False) -> None:
        """
        Update a single blend weight for multi-overlay mode.

        Also activates blend mode if not already active.
        """
        self._state.blend_control = self._state.blend_control.set_weight(overlay, weight)
        if not self._state.blend_control.blend_active:
            self._state.blend_control = self._state.blend_control.set_blend_active(True)

        if immediate:
            self.apply()

    def set_blend_active(self, active: bool, immediate: bool = False) -> None:
        """Enable or disable multi-overlay blend mode."""
        self._state.blend_control = self._state.blend_control.set_blend_active(active)
        if not active:
            # Revert to the single overlay dropdown value
            overlay_val = self._state.overlay_dropdown.value
            self._state.blend_control = BlendControl()
            if immediate:
                effective = None if overlay_val == "none" else overlay_val
                self._emit_update(overlay=effective)

    def toggle(self, name: str, immediate: bool = False) -> None:
        """Toggle a boolean control by name ("adaptive_tuning" or "audit_log")."""
        if name == "adaptive_tuning":
            self._state.adaptive_toggle = self._state.adaptive_toggle.set(
                not self._state.adaptive_toggle.value
            )
        elif name == "audit_log":
            self._state.audit_toggle = self._state.audit_toggle.set(
                not self._state.audit_toggle.value
            )
        if immediate:
            self.apply()

    # ── Preset operations ─────────────────────────────────────────────────────

    def load_preset(self, name: str, immediate: bool = False) -> bool:
        """
        Load a named preset into the control state.

        Updates trait sliders, overlay, and profile dropdown to match the preset.
        Does NOT change the relationship context (session-specific).

        Returns True if the preset was found and loaded, False if not found.
        """
        preset = self._preset_manager.get(name)
        if preset is None:
            if self._logger:
                self._logger.warning("ui_preset_not_found", name=name)
            return False

        # Apply trait values to sliders
        for trait, value in preset.trait_values.items():
            if trait in self._state.trait_sliders:
                self._state.trait_sliders[trait] = self._state.trait_sliders[trait].set(value)

        # Apply overlay
        overlay_val = preset.overlay
        try:
            self._state.overlay_dropdown = self._state.overlay_dropdown.set(overlay_val)
        except ValueError:
            pass

        # Apply profile
        try:
            self._state.profile_dropdown = self._state.profile_dropdown.set(preset.profile_id)
        except ValueError:
            pass

        # Buffer all changes
        self._pending["trait_overrides"] = dict(preset.trait_values)
        self._pending["profile_id"]      = preset.profile_id
        self._pending["overlay"]         = None if overlay_val == "none" else overlay_val

        if immediate:
            self.apply()

        if self._logger:
            self._logger.info("ui_preset_loaded", preset=name)

        return True

    def save_preset(self, name: str, description: str = "") -> PersonaPreset:
        """
        Save the current trait slider state as a named preset.

        Returns the saved PersonaPreset.
        """
        trait_values = {
            k: v.value for k, v in self._state.trait_sliders.items()
        }
        overlay = self._state.overlay_dropdown.value
        profile_id = self._state.profile_dropdown.value

        preset = PersonaPreset(
            name=name,
            profile_id=profile_id,
            trait_values=trait_values,
            overlay=overlay,
            description=description,
        )
        self._preset_manager.save(preset)

        if self._logger:
            self._logger.info("ui_preset_saved", name=name)

        return preset

    # ── Apply — flush pending changes ─────────────────────────────────────────

    def apply(self) -> bool:
        """
        Flush all pending changes as a single dp.config.update.v1 BusEvent.

        Returns True if any changes were applied, False if nothing was pending.

        For blend mode: if blend is active, overlay is derived from the active
        blend weights. The gate receives the full overlay string which the
        pipeline resolves via set_blended_overlays().
        """
        if not self._pending and not self._state.blend_control.blend_active:
            return False

        # Resolve overlay from blend or single selection
        if self._state.blend_control.blend_active:
            active = self._state.blend_control.active_blends()
            if active:
                # Encode blend as JSON overlay string — gate/pipeline decode this
                overlay_str = json.dumps(
                    {k: round(v, 3) for k, v in active.items()},
                    sort_keys=True,
                )
                self._pending["overlay"] = overlay_str

        trait_overrides = dict(self._pending.get("trait_overrides") or {})
        profile_id      = self._pending.get("profile_id")
        overlay         = self._pending.get("overlay")
        relationship    = self._pending.get("relationship")

        self._emit_update(
            trait_overrides=trait_overrides or None,
            profile_id=profile_id,
            overlay=overlay,
            relationship=relationship,
        )
        self._pending.clear()
        return True

    def reset_to_defaults(self, immediate: bool = True) -> None:
        """
        Reset all trait sliders to 0.5 (neutral) and clear overlay.

        Useful for the UI's "Reset" button.
        """
        for trait in self._state.trait_sliders:
            self._state.trait_sliders[trait] = self._state.trait_sliders[trait].set(0.5)
        self._state.overlay_dropdown = self._state.overlay_dropdown.set("none")
        self._state.blend_control = BlendControl()

        self._pending = {
            "trait_overrides": {t: 0.5 for t in self._state.trait_sliders},
            "overlay": None,
        }

        if immediate:
            self.apply()

    # ── State access ──────────────────────────────────────────────────────────

    @property
    def control_state(self) -> UIControlState:
        """Current control state — read-only view for platform adapters."""
        return self._state

    @property
    def preset_names(self) -> List[str]:
        """All available preset names (built-in + user-defined)."""
        return self._preset_manager.preset_names

    @property
    def has_pending_changes(self) -> bool:
        """True if there are buffered changes waiting for apply()."""
        return bool(self._pending)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit_update(
        self,
        trait_overrides: Optional[Dict[str, float]] = None,
        profile_id: Optional[str] = None,
        overlay: Optional[str] = None,
        relationship: Optional[str] = None,
    ) -> None:
        """Build and emit a dp.config.update.v1 BusEvent."""
        payload = make_persona_config_update_payload(
            profile_id=profile_id,
            trait_overrides=trait_overrides,
            overlay=overlay,
            relationship=relationship,
            requested_by=self._requested_by,
        )
        event = BusEvent.create(
            event_type=DPEventType.DP_CONFIG_UPDATE,
            source=self.SOURCE_NAME,
            destination="darkpassenger",
            priority=EventPriority.HUMAN,
            payload=payload,
        )
        self._emit(event)

        if self._logger:
            self._logger.debug(
                "ui_config_update_emitted",
                trait_overrides=list(trait_overrides.keys()) if trait_overrides else [],
                profile_id=profile_id,
                overlay=overlay,
                relationship=relationship,
            )

    @staticmethod
    def _build_initial_state(
        initial_traits: Dict[str, float],
        available_profiles: List[str],
    ) -> UIControlState:
        """Construct the initial UIControlState from defaults + overrides."""
        sliders: Dict[str, SliderControl] = {}
        for name, label, description in _TRAIT_DEFINITIONS:
            sliders[name] = SliderControl(
                name=name,
                value=initial_traits.get(name, 0.5),
                label=label,
                description=description,
            )

        profiles = available_profiles or ["default"]
        profile_dd = DropdownControl(
            name="profile",
            value=profiles[0],
            options=profiles,
            label="Persona Profile",
            description="The active persona profile preset.",
        )

        return UIControlState(
            trait_sliders=sliders,
            profile_dropdown=profile_dd,
        )


# ── Platform Adapters ─────────────────────────────────────────────────────────

class DesktopConfigAdapter:
    """
    Serialises UIControlState for Desktop UI frameworks (Qt, Tkinter, wxPython).

    Produces a flat dict with all controls listed by name, suitable for
    binding to widget variables in a desktop GUI event loop.

    Output format:
        {
          "controls": [
            {"type": "slider", "name": "humor", "value": 0.70, ...},
            {"type": "dropdown", "name": "overlay", "value": "focused", ...},
            ...
          ],
          "preset_names": [...],
          "platform": "desktop"
        }
    """

    @staticmethod
    def serialise(state: UIControlState, preset_names: Optional[List[str]] = None) -> dict:
        controls = []
        for slider in state.trait_sliders.values():
            controls.append(slider.to_dict())
        controls.append(state.overlay_dropdown.to_dict())
        controls.append(state.relationship_dropdown.to_dict())
        controls.append(state.profile_dropdown.to_dict())
        controls.append(state.blend_control.to_dict())
        controls.append(state.adaptive_toggle.to_dict())
        controls.append(state.audit_toggle.to_dict())
        return {
            "controls":     controls,
            "preset_names": preset_names or [],
            "platform":     "desktop",
        }


class WebConfigAdapter:
    """
    Serialises UIControlState for Web UIs (React, Vue, plain HTML/JS).

    Produces a JSON string with controls grouped by category, suitable for
    a REST endpoint or WebSocket push payload. Controls are grouped for
    easier rendering in a web panel layout.

    Groups: "personality", "context", "advanced", "presets"
    """

    @staticmethod
    def serialise(state: UIControlState, preset_names: Optional[List[str]] = None) -> str:
        payload = {
            "platform": "web",
            "groups": {
                "personality": {
                    "label": "Personality Traits",
                    "controls": [s.to_dict() for s in state.trait_sliders.values()],
                },
                "context": {
                    "label": "Context",
                    "controls": [
                        state.overlay_dropdown.to_dict(),
                        state.relationship_dropdown.to_dict(),
                        state.profile_dropdown.to_dict(),
                    ],
                },
                "advanced": {
                    "label": "Advanced",
                    "controls": [
                        state.blend_control.to_dict(),
                        state.adaptive_toggle.to_dict(),
                        state.audit_toggle.to_dict(),
                    ],
                },
                "presets": {
                    "label": "Presets",
                    "names": preset_names or [],
                },
            },
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def parse_update(json_str: str) -> dict:
        """
        Parse an incoming JSON update from the web UI.

        Expected format from a web form submission:
            {"control": "humor", "value": 0.85}
            {"control": "overlay", "value": "teaching"}
            {"preset": "DarkPassenger"}
        """
        return json.loads(json_str)


class MobileConfigAdapter:
    """
    Serialises UIControlState for Mobile UIs (iOS, Android, React Native).

    Produces a compact dict with abbreviated control representations
    optimised for small-screen display. Only the most essential controls
    are included by default; advanced controls are in a collapsible group.

    Priority traits shown by default (most impactful on UX):
        humor, warmth, directness, technicality
    """

    PRIMARY_TRAITS = ["humor", "warmth", "directness", "technicality"]

    @staticmethod
    def serialise(state: UIControlState, preset_names: Optional[List[str]] = None) -> dict:
        primary = [
            state.trait_sliders[t].to_dict()
            for t in MobileConfigAdapter.PRIMARY_TRAITS
            if t in state.trait_sliders
        ]
        secondary = [
            s.to_dict()
            for name, s in state.trait_sliders.items()
            if name not in MobileConfigAdapter.PRIMARY_TRAITS
        ]
        return {
            "platform":  "mobile",
            "primary":   primary,
            "secondary": secondary,
            "context": [
                state.overlay_dropdown.to_dict(),
                state.profile_dropdown.to_dict(),
            ],
            "preset_names": preset_names or [],
        }


class DiscordConfigAdapter:
    """
    Serialises UIControlState for Discord slash-command configuration.

    Discord slash commands take simple key=value pairs. This adapter
    produces a flat dict of all configurable parameters in a format
    suitable for Discord's ApplicationCommandOption system.

    Also provides slash_command_help() for the /persona command manifest.
    """

    @staticmethod
    def serialise(state: UIControlState) -> dict:
        """Flat dict of all current configurable values."""
        d: dict = {}
        for name, slider in state.trait_sliders.items():
            d[name] = round(slider.value, 2)
        d["overlay"]      = state.overlay_dropdown.value
        d["relationship"] = state.relationship_dropdown.value
        d["profile"]      = state.profile_dropdown.value
        d["adaptive"]     = state.adaptive_toggle.value
        return d

    @staticmethod
    def slash_command_help() -> dict:
        """
        Returns a Discord ApplicationCommand-compatible option manifest
        for the /persona command.

        Callers register this with the Discord gateway. Each option maps
        to a PersonaConfigInterface method call in the bot handler.
        """
        return {
            "name":        "persona",
            "description": "Configure Mini Von's DarkPassenger persona in real-time.",
            "options": [
                {"name": "humor",       "description": "Set humor level (0–100)",        "type": 10, "min_value": 0, "max_value": 100},
                {"name": "warmth",      "description": "Set warmth level (0–100)",        "type": 10, "min_value": 0, "max_value": 100},
                {"name": "directness",  "description": "Set directness level (0–100)",    "type": 10, "min_value": 0, "max_value": 100},
                {"name": "technicality","description": "Set technicality level (0–100)",  "type": 10, "min_value": 0, "max_value": 100},
                {"name": "overlay",     "description": "Set expression overlay",           "type": 3,
                 "choices": [{"name": ov.value, "value": ov.value} for ov in OverlayType] + [{"name": "none", "value": "none"}]},
                {"name": "profile",     "description": "Switch persona profile",           "type": 3},
                {"name": "preset",      "description": "Load a named persona preset",      "type": 3},
            ],
        }
