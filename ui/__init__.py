"""
ui — DarkPassenger Real-Time Configuration Interface  v1.5

Platform-agnostic UI controls, preset management, and platform adapters
for Desktop / Web / Mobile / Discord configuration of the DarkPassenger
persona system.

Public API
──────────
  PersonaConfigInterface    — main UI model; apply() emits dp.config.update.v1
  UIControlState            — snapshot of all current control values
  SliderControl             — bounded float trait control
  ToggleControl             — boolean on/off control
  DropdownControl           — single-choice string selection
  BlendControl              — multi-overlay blend weight manager
  PersonaPreset             — named snapshot of trait + overlay state
  PresetManager             — save / load / delete presets

Platform adapters (serialise UIControlState for each platform)
──────────────────
  DesktopConfigAdapter      — Qt / Tkinter / wxPython compatible dict
  WebConfigAdapter          — JSON / REST / WebSocket payload
  MobileConfigAdapter       — compact mobile-optimised dict
  DiscordConfigAdapter      — slash-command compatible dict + manifest

Spec reference: DarkPassenger-Plan.txt §18, §19
"""

from ui.config_interface import (
    PersonaConfigInterface,
    UIControlState,
    SliderControl,
    ToggleControl,
    DropdownControl,
    BlendControl,
    PersonaPreset,
    PresetManager,
    DesktopConfigAdapter,
    WebConfigAdapter,
    MobileConfigAdapter,
    DiscordConfigAdapter,
)

__all__ = [
    "PersonaConfigInterface",
    "UIControlState",
    "SliderControl",
    "ToggleControl",
    "DropdownControl",
    "BlendControl",
    "PersonaPreset",
    "PresetManager",
    "DesktopConfigAdapter",
    "WebConfigAdapter",
    "MobileConfigAdapter",
    "DiscordConfigAdapter",
]
