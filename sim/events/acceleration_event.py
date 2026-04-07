"""
acceleration_event.py
=====================
FS Acceleration Event: 75 m dash from rest.

The car starts stationary (v=0) and is timed over 75 m.
Battery thermal effects are included but the pack is not a limiting factor
for a 75 m run, so battery tracking is enabled for completeness.
"""

from __future__ import annotations

from sim.events.base_event import Event, EventResult
from sim.solver.lap_solver import LapSolver
from sim.track.track_builder import TrackProfile
from sim.vehicle.battery import BatteryModel


class AccelerationEvent(Event):
    def __init__(
        self,
        solver: LapSolver,
        track: TrackProfile,
        battery: BatteryModel,
    ) -> None:
        super().__init__("Acceleration")
        self.solver  = solver
        self.track   = track
        self.battery = battery

    def run(self) -> EventResult:
        self.battery.reset()
        power_limit_W = min(
            self.solver.pt.p.power_limit_kW * 1000.0,
            self.battery.max_power(),
        )
        states = self.solver.solve(
            self.track,
            v_initial=0.0,
            v_final=1000.0,       # unconstrained exit — backward pass is non-binding
            enable_battery=True,
            power_limit_W=power_limit_W,
        )
        return self._make_result(states)
