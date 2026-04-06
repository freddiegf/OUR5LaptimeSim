"""
sprint_event.py
===============
FS Sprint Event: one timed lap of the autocross/sprint track.

The car starts from rest and must also be at rest at the finish
(or the solver allows any exit speed — we use v_final=0 to be conservative
and realistic for a standing-start timed lap).

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
        states = self.solver.solve(
            self.track,
            v_initial=0.0,
            v_final=0.0,
            enable_battery=True,
        )
        return self._make_result(states)
