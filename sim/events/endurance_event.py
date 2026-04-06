"""
endurance_event.py
==================
FS Endurance Event: drive laps until cumulative distance ≥ 22 km.

Key rules modelled:
  - Battery state (SOC + temperature) persists across laps — no reset.
  - The car starts from rest; between laps we assume a rolling restart
    (v_initial = v of last state of previous lap, capped to a sensible value).
  - Telemetry from all laps is concatenated with correct s and t offsets.
  - The event ends when cumulative arc-length reaches or exceeds 22,000 m.
"""

from __future__ import annotations

from typing import List

from sim.events.base_event import Event, EventResult
from sim.solver.lap_solver import LapSolver
from sim.solver.vehicle_state import VehicleState
from sim.track.track_builder import TrackProfile
from sim.vehicle.battery import BatteryModel

_ENDURANCE_DISTANCE_M = 22_000.0
_MAX_LAPS = 300   # safety cap


class EnduranceEvent(Event):
    def __init__(
        self,
        solver: LapSolver,
        track: TrackProfile,
        battery: BatteryModel,
    ) -> None:
        super().__init__("Endurance")
        self.solver  = solver
        self.track   = track
        self.battery = battery

    def run(self) -> EventResult:
        self.battery.reset()

        all_states: List[VehicleState] = []
        cumulative_distance = 0.0
        cumulative_time     = 0.0
        lap_num             = 0
        v_entry             = 0.0   # rolling entry speed (0 for first lap)

        while cumulative_distance < _ENDURANCE_DISTANCE_M and lap_num < _MAX_LAPS:
            lap_num += 1
            lap_states = self.solver.solve(
                self.track,
                v_initial=v_entry,
                v_final=v_entry,          # rolling finish → carry speed to next lap
                enable_battery=True,
                t_start=cumulative_time,
                s_offset=cumulative_distance,
            )

            if not lap_states:
                break

            all_states.extend(lap_states)
            lap_distance = self.track.total_length
            lap_time     = lap_states[-1].t - lap_states[0].t + lap_states[-1].dt

            cumulative_distance += lap_distance
            cumulative_time     += lap_time

            # Rolling restart: use the exit speed of this lap as entry for next
            v_entry = min(lap_states[-1].v, 15.0)   # cap at 15 m/s for realism

            # Early exit if battery is depleted
            if self.battery.SOC <= 0.01:
                print(f"  [Endurance] Battery depleted at lap {lap_num}, "
                      f"distance {cumulative_distance/1000:.2f} km")
                break

        return self._make_result(all_states)
