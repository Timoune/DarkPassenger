"""
bloodyheart — BloodyHeart Middleware for DarkPassenger  v1.5

Provides the CoreBus event contracts and gate middleware that enforce
the GhostMind → DarkPassenger → Voicy/EchoLink routing guarantee.

Public API
──────────
  BusEvent              — CoreBus message envelope
  EventPriority         — P0–P4 priority queue levels
  DPEventType           — versioned event type string registry
  DPEventSchemas        — schema validation registry
  DarkPassengerGate     — the gate middleware class
  GateSafeModeLevel     — safe-mode level enum
  GateHealthStatus      — health report dataclass
  RoutingTable          — channel → module name resolver

Payload factories
─────────────────
  make_ghostmind_output_payload()
  make_dp_output_ready_payload()
  make_persona_config_update_payload()

Spec reference: BloodyHeart-Plan.txt §1, §2, §4, §20
                DarkPassenger-Plan.txt §20
"""

from bloodyheart.events import (
    BusEvent,
    EventPriority,
    DPEventType,
    DPEventSchemas,
    make_ghostmind_output_payload,
    make_dp_output_ready_payload,
    make_persona_config_update_payload,
)
from bloodyheart.middleware import (
    DarkPassengerGate,
    GateSafeModeLevel,
    GateHealthStatus,
    RoutingTable,
)

__all__ = [
    # Events
    "BusEvent",
    "EventPriority",
    "DPEventType",
    "DPEventSchemas",
    # Payload factories
    "make_ghostmind_output_payload",
    "make_dp_output_ready_payload",
    "make_persona_config_update_payload",
    # Gate
    "DarkPassengerGate",
    "GateSafeModeLevel",
    "GateHealthStatus",
    "RoutingTable",
]
