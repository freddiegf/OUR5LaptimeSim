"""
skidpad_event.py
================
FS Skidpad Event: figure-8 with two circles of radius 7.625 m.

FS rules:
  - Two left-hand (CCW) laps followed by two right-hand (CW) laps.
  - Only the inner lap of each direction is timed.
  - The official time is the average of the two timed laps.

Track layout (from YAML):
  approach straight → left entry (untimed) → left timed →
  crossing straight → right timed → right exit (untimed) →
  exit straight

The timed regions are determined from the track segment structure:
  segments[0]: approach straight
  segments[1]: left circle (untimed)
  segments[2]: left circle (timed)
  segments[3]: crossing straight (untimed transition)
  segments[4]: right circle (timed)
  segments[5]: right circle (untimed)
  segments[6]: exit straight
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
        power_limit_W = min(
            self.solver.pt.p.power_limit_kW * 1000.0,
            self.battery.max_power(),
        )
        # Rolling start and finish — car approaches at moderate speed
        states = self.solver.solve(
            self.track,
            v_initial=10.0,
            v_final=10.0,
            enable_battery=True,
            power_limit_W=power_limit_W,
        )

        # --- Extract timed portion from segment boundaries ---
        seg_lengths = [seg.length for seg in self.track.segments]
        seg_ends = np.cumsum(seg_lengths)

        # Timed left: segment[2] → from seg_ends[1] to seg_ends[2]
        s_left_start  = seg_ends[1]
        s_left_end    = seg_ends[2]

        # Timed right: segment[4] → from seg_ends[3] to seg_ends[4]
        s_right_start = seg_ends[3]
        s_right_end   = seg_ends[4]

        s_arr = np.array([st.s for st in states])

        def lap_time(s_start: float, s_end: float) -> float:
            mask = (s_arr >= s_start) & (s_arr < s_end)
            return float(np.sum([states[i].dt for i in range(len(states)) if mask[i]]))

        t_left  = lap_time(s_left_start,  s_left_end)
        t_right = lap_time(s_right_start, s_right_end)
        t_timed = t_left + t_right   # FS: sum of one left + one right lap

        result = self._make_result(states)
        # Override total_time with the officially timed portion
        result.total_time = t_timed
        return result
