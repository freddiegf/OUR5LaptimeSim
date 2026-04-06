"""
skidpad_event.py
================
FS Skidpad Event: figure-8 with two circles of radius 7.625 m.

FS rules:
  - Two left-hand (CCW) laps followed by two right-hand (CW) laps.
  - Only the inner lap of each direction is timed.
  - The official time is the average of the two timed laps.

Implementation:
  The track YAML defines all four circles in sequence.
  The solver runs the full four-circle path from rest.
  The timed portion is extracted from states corresponding to
  circles 2 (left, timed) and 3 (right, timed).
"""

from __future__ import annotations

import numpy as np

from sim.events.base_event import Event, EventResult
from sim.solver.lap_solver import LapSolver
from sim.track.track_builder import TrackProfile
from sim.vehicle.battery import BatteryModel


class SkidpadEvent(Event):
    def __init__(
        self,
        solver: LapSolver,
        track: TrackProfile,
        battery: BatteryModel,
    ) -> None:
        super().__init__("Skidpad")
        self.solver  = solver
        self.track   = track
        self.battery = battery

    def run(self) -> EventResult:
        self.battery.reset()
        # Run the full 4-circle path; car starts and finishes at a crawl
        states = self.solver.solve(
            self.track,
            v_initial=0.1,
            v_final=0.1,
            enable_battery=True,
        )

        # --- Extract timed portion ---
        # Each of the 4 circles has equal length (total_length / 4)
        total_s = self.track.total_length
        circle_length = total_s / 4.0

        # Timed: circles 2 (left) and 3 (right)
        s_start_left  = circle_length          # start of circle 2
        s_end_left    = 2.0 * circle_length    # end of circle 2
        s_start_right = 2.0 * circle_length    # start of circle 3
        s_end_right   = 3.0 * circle_length    # end of circle 3

        s_arr = np.array([st.s for st in states])

        def lap_time(s_start: float, s_end: float) -> float:
            mask = (s_arr >= s_start) & (s_arr < s_end)
            return float(np.sum([states[i].dt for i in range(len(states)) if mask[i]]))

        t_left  = lap_time(s_start_left,  s_end_left)
        t_right = lap_time(s_start_right, s_end_right)
        t_timed = t_left + t_right   # FS rules: sum of one left + one right lap

        result = self._make_result(states)
        # Override total_time with the officially timed portion
        result.total_time = t_timed
        return result
