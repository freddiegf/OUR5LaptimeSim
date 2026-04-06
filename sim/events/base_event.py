"""
base_event.py
=============
Abstract base class for all FS events and shared EventResult container.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from sim.solver.vehicle_state import VehicleState


@dataclass
class EventResult:
    """Summary and telemetry for a single event run."""
    event_name:         str
    total_time:         float              # s
    states:             List[VehicleState] # full telemetry, one per solver step
    energy_used_kWh:    float
    final_SOC:          float              # 0–1
    final_battery_temp: float              # °C
    peak_ax:            float              # m/s²
    peak_ay:            float              # m/s² (absolute)
    peak_speed_ms:      float              # m/s
    total_distance_m:   float              # m

    @property
    def peak_speed_kph(self) -> float:
        return self.peak_speed_ms * 3.6

    @property
    def peak_ax_g(self) -> float:
        return self.peak_ax / 9.81

    @property
    def peak_ay_g(self) -> float:
        return self.peak_ay / 9.81


class Event(ABC):
    """Abstract base class for Formula Student events."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self) -> EventResult:
        """Execute the event and return results."""
        ...

    def _make_result(self, states: List[VehicleState]) -> EventResult:
        """
        Build an EventResult from a flat list of VehicleState objects.
        Assumes battery state is current (not reset) at end of states list.
        """
        if not states:
            raise ValueError("No states returned from solver.")

        total_time  = states[-1].t - states[0].t + states[-1].dt
        energy_J    = sum(s.power_demand * s.dt for s in states)
        final_soc   = states[-1].SOC
        final_temp  = states[-1].battery_temp
        peak_ax     = max(s.ax for s in states)
        peak_ay     = max(abs(s.ay) for s in states)
        peak_speed  = max(s.v for s in states)
        distance    = states[-1].s - states[0].s

        return EventResult(
            event_name=self.name,
            total_time=total_time,
            states=states,
            energy_used_kWh=energy_J / 3.6e6,
            final_SOC=final_soc,
            final_battery_temp=final_temp,
            peak_ax=peak_ax,
            peak_ay=peak_ay,
            peak_speed_ms=peak_speed,
            total_distance_m=distance,
        )
