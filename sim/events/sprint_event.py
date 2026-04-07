"""
sprint_event.py
===============
FS Sprint Event: one timed lap of the autocross/sprint track.

The car starts from rest and crosses the finish line at speed
(v_final is unconstrained — the backward pass does not limit exit speed).

Battery thermal tracking is enabled. The battery is reset before the event.
"""

from __future__ import annotations

from sim.events.base_event import Event, EventResult
from sim.solver.lap_solver import LapSolver
from sim.track.track_builder import TrackProfile
from sim.vehicle.battery import BatteryModel


class SprintEvent(Event):
    def __init__(
        self,
        solver: LapSolver,
        track: TrackProfile,
        battery: BatteryModel,
    ) -> None:
        super().__init__("Sprint")
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
            v_final=1000.0,      # unconstrained — car crosses finish at speed
            enable_battery=True,
            power_limit_W=power_limit_W,
        )
        return self._make_result(states)
