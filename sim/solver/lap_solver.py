"""
lap_solver.py
=============
Quasi-static forward-backward velocity profile solver.

Algorithm
---------
1. Corner caps   — pre-compute max cornering speed at each station via
                   binary search on the GGV lateral limit.
2. Forward pass  — starting from v_initial, accelerate as hard as the GGV
                   allows, clamped to corner speed caps at every station.
3. Backward pass — starting from v_final, "brake" as hard as the GGV allows
                   going backwards, clamped to corner speed caps.
4. Combine       — take the element-wise minimum of the two profiles.
5. Build states  — walk the final speed profile and populate a VehicleState
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
        power_limit_W: float | None = None,
        regen_power_limit_W: float | None = None,
    ) -> List[VehicleState]:
        """
        Run the forward-backward solver on the given track.

        Parameters
        ----------
        track               : discretised track profile
        v_initial           : speed at the start of the track, m/s
        v_final             : target speed at the end (for backward pass), m/s
        enable_battery      : if False, battery.step() is skipped (battery
                              state is still logged as current values)
        t_start             : time offset for returned states (multi-lap runs)
        s_offset            : arc-length offset for returned states
        power_limit_W       : if given, overrides the powertrain power_limit_kW
                              for this solve (used to enforce battery
                              deliverable power and thermal derate)
        regen_power_limit_W : if given, overrides the powertrain
                              regen_power_limit_kW for this solve (used to
                              mirror a thermal derate on regen charging)

        Returns
        -------
        List[VehicleState] — one entry per arc-length station
        """
        v_cap = self._corner_speed_caps(track)
        v_fwd = self._forward_pass(track, v_initial, v_cap, power_limit_W)
        v_bwd = self._backward_pass(track, v_final, v_cap)
        v     = self._combine(v_fwd, v_bwd)
        return self._build_states(track, v, enable_battery, t_start, s_offset,
                                  power_limit_W, regen_power_limit_W)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_pass(
        self, track: TrackProfile, v_initial: float, v_cap: np.ndarray,
        power_limit_W: float | None = None,
    ) -> np.ndarray:
        N  = len(track.s)
        ds = track.ds
        v  = np.zeros(N)
        v[0] = min(max(v_initial, 0.0), v_cap[0])

        for i in range(N - 1):
            vi  = max(v[i], _MIN_SPEED)
            ay  = float(np.clip(vi ** 2 * track.kappa[i],
                                -self.ggv.ay_abs_max, self.ggv.ay_abs_max))
            ax  = self.ggv.query_ax_max(vi, ay)

            # If a power limit is supplied (e.g. battery max power), cap ax.
            # power_limit_W is terminal power; multiply by η to get wheel force.
            if power_limit_W is not None:
                eta  = self.pt.p.drivetrain_efficiency
                drag = self.aero.drag_force(vi)
                F_power = power_limit_W * eta / vi
                ax_power = (F_power - drag) / self.car.mass
                ax = min(ax, ax_power)

            # Use v[i] (actual speed) in kinematics; vi is only for GGV lookup.
            v_next_sq = v[i] ** 2 + 2.0 * ax * ds
            v[i + 1] = min(max(0.0, v_next_sq) ** 0.5, v_cap[i + 1])

        return v

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward_pass(
        self, track: TrackProfile, v_final: float, v_cap: np.ndarray,
    ) -> np.ndarray:
        N  = len(track.s)
        ds = track.ds
        v  = np.zeros(N)
        v[-1] = min(max(v_final, 0.0), v_cap[-1])

        for i in range(N - 1, 0, -1):
            vi  = max(v[i], _MIN_SPEED)
            ay  = float(np.clip(vi ** 2 * track.kappa[i],
                                -self.ggv.ay_abs_max, self.ggv.ay_abs_max))
            ax_min   = self.ggv.query_ax_min(vi, ay)   # negative
            ax_brake = abs(ax_min)
            # Use v[i] (actual speed) in kinematics; vi is only for GGV lookup.
            v_prev_sq = v[i] ** 2 + 2.0 * ax_brake * ds
            v[i - 1] = min(max(0.0, v_prev_sq) ** 0.5, v_cap[i - 1])

        return v

    # ------------------------------------------------------------------
    # Corner speed caps
    # ------------------------------------------------------------------

    def _corner_speed_caps(self, track: TrackProfile) -> np.ndarray:
        """
        Pre-compute maximum cornering speed at each station.

        Uses binary search on the GGV to find the speed at which the
        required lateral acceleration v²×|kappa| just saturates the
        tyre lateral limit (ax_max drops to zero).

        Returns np.inf at straight stations (no cap).
        """
        N = len(track.s)
        v_cap = np.full(N, np.inf)
        for i, kap in enumerate(track.kappa):
            if abs(kap) > 1e-6:
                v_hi = (self.ggv.ay_abs_max / abs(kap)) ** 0.5
                v_lo = 0.0
                for _ in range(20):
                    v_mid = 0.5 * (v_lo + v_hi)
                    ay_req = v_mid ** 2 * abs(kap)
                    ax_at_ay = self.ggv.query_ax_max(
                        max(v_mid, _MIN_SPEED), ay_req)
                    if ax_at_ay > 0.0:
                        v_lo = v_mid
                    else:
                        v_hi = v_mid
                v_cap[i] = v_hi
        return v_cap

    # ------------------------------------------------------------------
    # Combine forward and backward profiles
    # ------------------------------------------------------------------

    def _combine(
        self,
        v_fwd: np.ndarray,
        v_bwd: np.ndarray,
    ) -> np.ndarray:
        """Element-wise minimum of forward and backward profiles."""
        return np.maximum(np.minimum(v_fwd, v_bwd), 0.0)

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
        power_limit_W: float | None = None,
        regen_power_limit_W: float | None = None,
    ) -> List[VehicleState]:
        N  = len(track.s)
        ds = track.ds
        states: List[VehicleState] = []
        t = t_start

        eta = self.pt.p.drivetrain_efficiency
        rules_power_W = self.pt.p.power_limit_kW * 1000.0

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

            # --- Aero ---
            df = self.aero.downforce(vi)
            drag = self.aero.drag_force(vi)

            # --- Drive / brake forces from Newton's second law ---
            # F_net = m*ax = drive_force - drag  (accel)
            #       = m*ax = -brake_force - drag (braking)
            F_net_plus_drag = self.car.mass * ax_i + drag
            if F_net_plus_drag >= 0.0:
                # Throttle regime (including coasting where drag > decel demand)
                drive_force = F_net_plus_drag
                brake_force = 0.0
                # Distribute Fx to driven wheels proportional to capacity
                Fx_driven_max = (front_frac * (Fx_avail_FL + Fx_avail_FR)
                                 + rear_frac * (Fx_avail_RL + Fx_avail_RR))
                scale = drive_force / max(1e-6, Fx_driven_max)
                scale = min(scale, 1.0)
                Fx_FL = front_frac * Fx_avail_FL * scale
                Fx_FR = front_frac * Fx_avail_FR * scale
                Fx_RL = rear_frac  * Fx_avail_RL * scale
                Fx_RR = rear_frac  * Fx_avail_RR * scale
            else:
                # Braking regime
                drive_force = 0.0
                brake_force = -F_net_plus_drag
                Fx_brake_max = Fx_avail_FL + Fx_avail_FR + Fx_avail_RL + Fx_avail_RR
                scale = brake_force / max(1e-6, Fx_brake_max)
                scale = min(scale, 1.0)
                Fx_FL = -Fx_avail_FL * scale
                Fx_FR = -Fx_avail_FR * scale
                Fx_RL = -Fx_avail_RL * scale
                Fx_RR = -Fx_avail_RR * scale

            # --- Motor state ---
            rpm = self.pt.wheel_speed_to_motor_rpm(vi, r_driven)
            torque = self.pt.motor_torque_at_rpm(rpm)

            # --- Limiting factor (acceleration regime only) ---
            if brake_force > 1.0:
                limiting_factor = "Braking"
            else:
                # Compute each force limit at the wheel level
                F_tyre_lim    = (front_frac * (Fx_avail_FL + Fx_avail_FR)
                                 + rear_frac * (Fx_avail_RL + Fx_avail_RR))
                F_motor_lim   = self.pt.max_drive_force(vi, r_driven)
                F_rules_lim   = rules_power_W * eta / vi
                F_battery_lim = self.battery.max_power() * eta / vi

                limits = {
                    "Tyre":        F_tyre_lim,
                    "Motor":       F_motor_lim,
                    "Power limit": F_rules_lim,
                    "Battery":     F_battery_lim,
                }
                limiting_factor = min(limits, key=limits.get)

            # --- Battery ---
            # P_demand is terminal power: wheel power / drivetrain efficiency
            # Positive = discharging, Negative = regen charging
            if regen_power_limit_W is not None:
                regen_limit_W = regen_power_limit_W
            else:
                regen_limit_W = self.pt.p.regen_power_limit_kW * 1000.0
            if drive_force > 0.0:
                # Driving: battery discharges
                P_wheel = drive_force * vi
                P_terminal = P_wheel / eta
                if power_limit_W is not None:
                    P_terminal = min(P_terminal, power_limit_W)
            elif brake_force > 0.0 and regen_limit_W > 0.0:
                # Braking with regen: only driven wheels recover energy
                # Compute braking force on driven wheels only
                Fx_brake_driven = (rear_frac * (abs(Fx_RL) + abs(Fx_RR))
                                   + front_frac * (abs(Fx_FL) + abs(Fx_FR)))
                P_regen_wheel = Fx_brake_driven * vi
                # Apply drivetrain losses (wheel→motor→battery)
                P_regen_electrical = P_regen_wheel * eta
                # Cap at regen power limit
                P_regen_electrical = min(P_regen_electrical, regen_limit_W)
                P_terminal = -P_regen_electrical   # negative = charging
            else:
                P_terminal = 0.0

            if enable_battery:
                bat_state = self.battery.step(P_terminal, dt_i)
                soc  = bat_state.SOC
                temp = bat_state.temperature
                curr = bat_state.current_pack
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
                power_demand=P_terminal,
                limiting_factor=limiting_factor,
                t=t,
                dt=dt_i,
            )
            states.append(state)
            t += dt_i

        return states
