"""
lap_solver.py
=============
Quasi-static forward-backward velocity profile solver.

Algorithm
---------
1. Forward pass  — starting from v_initial, accelerate as hard as the GGV
                   allows at each arc-length station.
2. Backward pass — starting from v_final, "brake" as hard as the GGV allows
                   going backwards along the track.
3. Combine       — take the element-wise minimum of the two profiles, then
                   enforce the curvature-derived cornering speed limit.
4. Build states  — walk the final speed profile and populate a VehicleState
                   at every station (forces, battery, timing).

The solver is deliberately separated from the event logic. Events control
which track is used, how many laps to run, and whether the battery persists
between laps.

Extension notes
---------------
To convert to a dynamic simulation, replace _forward_pass/_backward_pass with
an ODE integrator (e.g., scipy.integrate.solve_ivp) that uses the same
force-computation helpers exposed here.
"""

from __future__ import annotations

from typing import List

import numpy as np

from sim.ggv.ggv_builder import GGVBuilder
from sim.track.track_builder import TrackProfile
from sim.vehicle.car_params import CarParams
from sim.vehicle.aero import AeroModel
from sim.vehicle.powertrain import Powertrain
from sim.vehicle.battery import BatteryModel
from sim.solver.vehicle_state import VehicleState, WheelState


_MIN_SPEED = 0.5   # m/s — numerical floor to avoid division by zero


class LapSolver:
    def __init__(
        self,
        ggv: GGVBuilder,
        car: CarParams,
        aero: AeroModel,
        powertrain: Powertrain,
        battery: BatteryModel,
    ) -> None:
        self.ggv = ggv
        self.car = car
        self.aero = aero
        self.pt = powertrain
        self.battery = battery

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        track: TrackProfile,
        v_initial: float = 0.0,
        v_final: float = 0.0,
        enable_battery: bool = True,
        t_start: float = 0.0,
        s_offset: float = 0.0,
    ) -> List[VehicleState]:
        """
        Run the forward-backward solver on the given track.

        Parameters
        ----------
        track          : discretised track profile
        v_initial      : speed at the start of the track, m/s
        v_final        : target speed at the end (for backward pass), m/s
        enable_battery : if False, battery.step() is skipped (battery state is
                         still logged as current values)
        t_start        : time offset for the returned states (for multi-lap runs)
        s_offset       : arc-length offset for returned states

        Returns
        -------
        List[VehicleState] — one entry per arc-length station
        """
        v_fwd = self._forward_pass(track, v_initial)
        v_bwd = self._backward_pass(track, v_final)
        v     = self._combine(track, v_fwd, v_bwd)
        return self._build_states(track, v, enable_battery, t_start, s_offset)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_pass(self, track: TrackProfile, v_initial: float) -> np.ndarray:
        N  = len(track.s)
        ds = track.ds
        v  = np.zeros(N)
        v[0] = max(v_initial, 0.0)

        for i in range(N - 1):
            vi  = max(v[i], _MIN_SPEED)
            ay  = float(np.clip(vi ** 2 * track.kappa[i],
                                -self.ggv.ay_abs_max, self.ggv.ay_abs_max))
            ax  = self.ggv.query_ax_max(vi, ay)
            ax  = max(0.0, ax)   # forward pass: only accelerate
            v_next_sq = vi ** 2 + 2.0 * ax * ds
            v[i + 1] = max(0.0, v_next_sq) ** 0.5

        return v

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward_pass(self, track: TrackProfile, v_final: float) -> np.ndarray:
        N  = len(track.s)
        ds = track.ds
        v  = np.zeros(N)
        v[-1] = max(v_final, 0.0)

        for i in range(N - 1, 0, -1):
            vi  = max(v[i], _MIN_SPEED)
            ay  = float(np.clip(vi ** 2 * track.kappa[i],
                                -self.ggv.ay_abs_max, self.ggv.ay_abs_max))
            ax_min   = self.ggv.query_ax_min(vi, ay)   # negative
            ax_brake = abs(ax_min)
            v_prev_sq = vi ** 2 + 2.0 * ax_brake * ds
            v[i - 1] = max(0.0, v_prev_sq) ** 0.5

        return v

    # ------------------------------------------------------------------
    # Combine and apply cornering cap
    # ------------------------------------------------------------------

    def _combine(
        self,
        track: TrackProfile,
        v_fwd: np.ndarray,
        v_bwd: np.ndarray,
    ) -> np.ndarray:
        v = np.minimum(v_fwd, v_bwd)

        # Hard cornering cap: v ≤ sqrt(ay_max / |kappa|)
        ay_max = self.ggv.ay_abs_max
        for i, kap in enumerate(track.kappa):
            if abs(kap) > 1e-6:
                v_corner = (ay_max / abs(kap)) ** 0.5
                v[i] = min(v[i], v_corner)

        return np.maximum(v, 0.0)

    # ------------------------------------------------------------------
    # Build full VehicleState list
    # ------------------------------------------------------------------

    def _build_states(
        self,
        track: TrackProfile,
        v: np.ndarray,
        enable_battery: bool,
        t_start: float,
        s_offset: float,
    ) -> List[VehicleState]:
        N  = len(track.s)
        ds = track.ds
        states: List[VehicleState] = []
        t = t_start

        front_frac, rear_frac = self.pt.driven_axle_fractions()
        r_rear   = self.car.rear_tyre.radius
        r_front  = self.car.front_tyre.radius
        # Weighted effective driven radius; fall back to rear if no driven axle
        r_driven = (rear_frac * r_rear + front_frac * r_front
                    if (front_frac > 0 or rear_frac > 0) else r_rear)

        for i in range(N):
            vi    = max(v[i], _MIN_SPEED)
            kap_i = track.kappa[i]

            # Longitudinal acceleration from finite difference of v²
            # Use raw v[i] (not vi) so the floor doesn't skew ax at low speed
            if i < N - 1:
                ax_i = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds)
            else:
                ax_i = (v[i] ** 2 - v[i - 1] ** 2) / (2.0 * ds)

            ay_i = vi ** 2 * kap_i

            # Clamp ax to GGV envelope — prevents numerical spikes at
            # speed-profile transitions (forward↔backward crossover points)
            ay_clipped = float(np.clip(ay_i, -self.ggv.ay_abs_max,
                                       self.ggv.ay_abs_max))
            ax_i = float(np.clip(ax_i,
                                 self.ggv.query_ax_min(vi, ay_clipped),
                                 self.ggv.query_ax_max(vi, ay_clipped)))

            dt_i = ds / vi

            # --- Normal loads ---
            Fz_FL, Fz_FR, Fz_RL, Fz_RR = self.ggv.normal_loads(vi, ax_i, ay_i)

            # --- Fy demand (proportional to Fz) ---
            Fy_total = self.car.mass * ay_i
            Fz_sum   = Fz_FL + Fz_FR + Fz_RL + Fz_RR
            if Fz_sum > 0:
                k = Fy_total / Fz_sum
                Fy_FL, Fy_FR, Fy_RL, Fy_RR = (k * Fz_FL, k * Fz_FR,
                                                k * Fz_RL, k * Fz_RR)
            else:
                Fy_FL = Fy_FR = Fy_RL = Fy_RR = 0.0

            # --- Fx: drive or brake ---
            Fxmax_FL = self.ggv.tyre_f.Fxmax(Fz_FL)
            Fxmax_FR = self.ggv.tyre_f.Fxmax(Fz_FR)
            Fxmax_RL = self.ggv.tyre_r.Fxmax(Fz_RL)
            Fxmax_RR = self.ggv.tyre_r.Fxmax(Fz_RR)
            Fymax_FL = self.ggv.tyre_f.Fymax(Fz_FL)
            Fymax_FR = self.ggv.tyre_f.Fymax(Fz_FR)
            Fymax_RL = self.ggv.tyre_r.Fymax(Fz_RL)
            Fymax_RR = self.ggv.tyre_r.Fymax(Fz_RR)

            Fx_avail_FL = self.ggv.tyre_f.Fx_available(Fz_FL, abs(Fy_FL))
            Fx_avail_FR = self.ggv.tyre_f.Fx_available(Fz_FR, abs(Fy_FR))
            Fx_avail_RL = self.ggv.tyre_r.Fx_available(Fz_RL, abs(Fy_RL))
            Fx_avail_RR = self.ggv.tyre_r.Fx_available(Fz_RR, abs(Fy_RR))

            if ax_i >= 0.0:
                # Accelerating: driven wheels only
                F_drive_tyre = (front_frac * (Fx_avail_FL + Fx_avail_FR)
                                + rear_frac  * (Fx_avail_RL + Fx_avail_RR))
                F_motor = self.pt.max_drive_force(vi, r_driven)
                drive_force = min(F_drive_tyre, F_motor)
                brake_force = 0.0
                # Distribute Fx proportionally to driven axle
                Fx_FL = front_frac * min(Fx_avail_FL, drive_force / max(1e-3, front_frac * 2 + rear_frac * 2))
                Fx_FR = Fx_FL
                Fx_RL = rear_frac  * min(Fx_avail_RL, drive_force / max(1e-3, front_frac * 2 + rear_frac * 2))
                Fx_RR = Fx_RL
            else:
                # Braking: all four wheels
                brake_force = Fx_avail_FL + Fx_avail_FR + Fx_avail_RL + Fx_avail_RR
                drive_force = 0.0
                Fx_FL = -Fx_avail_FL
                Fx_FR = -Fx_avail_FR
                Fx_RL = -Fx_avail_RL
                Fx_RR = -Fx_avail_RR

            # --- Aero ---
            df = self.aero.downforce(vi)
            drag = self.aero.drag_force(vi)

            # --- Motor state ---
            rpm = self.pt.wheel_speed_to_motor_rpm(vi, r_driven)
            torque = self.pt.motor_torque_at_rpm(rpm)

            # --- Battery ---
            P_demand = max(0.0, ax_i * self.car.mass * vi + drag * vi)
            if enable_battery:
                bat_state = self.battery.step(P_demand, dt_i)
                soc  = bat_state.SOC
                temp = bat_state.temperature
                curr = bat_state.current
            else:
                soc  = self.battery.SOC
                temp = self.battery.temperature
                curr = 0.0

            # --- Populate state ---
            state = VehicleState(
                s=track.s[i] + s_offset,
                x=track.x[i],
                y=track.y[i],
                heading=track.heading[i],
                v=v[i],
                ax=ax_i,
                ay=ay_i,
                kappa=kap_i,
                wheel_FL=WheelState(Fz_FL, Fx_FL, Fy_FL, Fxmax_FL, Fymax_FL),
                wheel_FR=WheelState(Fz_FR, Fx_FR, Fy_FR, Fxmax_FR, Fymax_FR),
                wheel_RL=WheelState(Fz_RL, Fx_RL, Fy_RL, Fxmax_RL, Fymax_RL),
                wheel_RR=WheelState(Fz_RR, Fx_RR, Fy_RR, Fxmax_RR, Fymax_RR),
                downforce=df,
                drag=drag,
                motor_rpm=rpm,
                motor_torque=torque,
                drive_force=drive_force,
                brake_force=brake_force,
                SOC=soc,
                battery_temp=temp,
                battery_current=curr,
                power_demand=P_demand,
                t=t,
                dt=dt_i,
            )
            states.append(state)
            t += dt_i

        return states
