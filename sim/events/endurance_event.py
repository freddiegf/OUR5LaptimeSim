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

        lap_times: list[float] = []
        lap_socs:  list[float] = []
        lap_temps: list[float] = []

        rules_power_W = self.solver.pt.p.power_limit_kW * 1000.0
        rules_regen_W = self.solver.pt.p.regen_power_limit_kW * 1000.0

        while cumulative_distance < _ENDURANCE_DISTANCE_M and lap_num < _MAX_LAPS:
            lap_num += 1

            # Thermal derate based on lap-start battery temperature. Mirrored
            # onto regen so charging current is throttled alongside discharge.
            derate_frac = self.battery.derate_fraction()

            # Battery deliverable power may be less than the rules limit
            battery_power_W = self.battery.max_power()
            effective_power_W = min(rules_power_W * derate_frac, battery_power_W)
            effective_regen_W = rules_regen_W * derate_frac

            if derate_frac < 1.0:
                print(f"  [Endurance] Lap {lap_num}: T_bat="
                      f"{self.battery.temperature:.1f} C, power derated to "
                      f"{effective_power_W/1000:.1f} kW "
                      f"({derate_frac*100:.0f}%)")

            lap_states = self.solver.solve(
                self.track,
                v_initial=v_entry,
                v_final=1000.0,           # unconstrained exit — car carries speed
                enable_battery=True,
                t_start=cumulative_time,
                s_offset=cumulative_distance,
                power_limit_W=effective_power_W,
                regen_power_limit_W=effective_regen_W,
            )

            if not lap_states:
                break

            all_states.extend(lap_states)
            lap_distance = self.track.total_length
            lap_time     = lap_states[-1].t - lap_states[0].t + lap_states[-1].dt

            cumulative_distance += lap_distance
            cumulative_time     += lap_time

            # Track per-lap metrics
            lap_times.append(lap_time)
            lap_socs.append(self.battery.SOC)
            lap_temps.append(self.battery.temperature)

            # Rolling restart: use the exit speed of this lap as entry for next
            # Track is a closed loop, so start/finish speed should match
            v_entry = lap_states[-1].v

            # Early exit if battery is depleted
            if self.battery.SOC <= 0.01:
                print(f"  [Endurance] Battery depleted at lap {lap_num}, "
                      f"distance {cumulative_distance/1000:.2f} km")
                break

        result = self._make_result(all_states)
        result.lap_times = lap_times
        result.lap_socs  = lap_socs
        result.lap_temps = lap_temps
        return result
