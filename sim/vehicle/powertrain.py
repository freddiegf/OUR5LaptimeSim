"""
powertrain.py
=============
Electric powertrain model.

Torque curve is stored as a lookup table (RPM, Nm) and interpolated.
Provides maximum drive force at a given wheel speed, accounting for
gear ratio, drivetrain efficiency, and drivetrain type (RWD/FWD/AWD).
"""

from __future__ import annotations

import numpy as np
from sim.vehicle.car_params import PowertrainParams


class Powertrain:
    def __init__(self, params: PowertrainParams) -> None:
        self.p = params
        pairs = sorted(params.torque_curve, key=lambda x: x[0])
        self._rpm_arr = np.array([pt[0] for pt in pairs], dtype=float)
        self._trq_arr = np.array([pt[1] for pt in pairs], dtype=float)

    # ------------------------------------------------------------------
    # Motor torque
    # ------------------------------------------------------------------

    def motor_torque_at_rpm(self, rpm: float) -> float:
        """Interpolated motor torque (Nm) at given RPM. Zero beyond max RPM."""
        return float(np.interp(rpm, self._rpm_arr, self._trq_arr,
                               left=self._trq_arr[0], right=0.0))

    def wheel_speed_to_motor_rpm(self, v: float, tyre_radius: float) -> float:
        """Convert vehicle speed (m/s) to motor shaft RPM."""
        omega_wheel = v / tyre_radius          # rad/s
        omega_motor = omega_wheel * self.p.gear_ratio
        return omega_motor * 60.0 / (2.0 * np.pi)

    # ------------------------------------------------------------------
    # Drive force
    # ------------------------------------------------------------------

    def max_drive_force(self, v: float, tyre_radius: float) -> float:
        """
        Maximum total wheel-level traction force from motor torque limit, N.

        F_wheel = T_motor × gear_ratio × η / r_tyre
        """
        rpm = self.wheel_speed_to_motor_rpm(v, tyre_radius)
        T_motor = self.motor_torque_at_rpm(rpm)
        return T_motor * self.p.gear_ratio * self.p.drivetrain_efficiency / tyre_radius

    def driven_axle_fractions(self) -> tuple[float, float]:
        """
        Returns (front_fraction, rear_fraction) of total drive force.

        For RWD: (0, 1), FWD: (1, 0), AWD: split by awd_front_bias.
        """
        dt = self.p.drivetrain_type.upper()
        if dt == "RWD":
            return 0.0, 1.0
        elif dt == "FWD":
            return 1.0, 0.0
        else:  # AWD
            f = self.p.awd_front_bias
            return f, 1.0 - f

    def motor_torque_at_speed(self, v: float, tyre_radius: float) -> float:
        """Convenience: motor torque (Nm) at vehicle speed v."""
        rpm = self.wheel_speed_to_motor_rpm(v, tyre_radius)
        return self.motor_torque_at_rpm(rpm)

    @property
    def max_rpm(self) -> float:
        return float(self._rpm_arr[-1])
