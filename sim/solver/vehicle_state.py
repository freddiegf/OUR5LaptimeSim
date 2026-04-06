"""
vehicle_state.py
================
Pure data containers for the vehicle state at a single solver step.

VehicleState is intentionally a flat, serialisable dataclass with no methods.
This makes it straightforward to:
  - Log telemetry (CSV, HDF5, JSON).
  - Extend to a dynamic simulation by adding velocity derivatives, slip angles, etc.
  - Feed into post-processing and plotting without coupling to solver logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WheelState:
    """
    State of one wheel at a single integration step.

    Forces are in the vehicle frame:
      Fz  — normal load (positive = compressed), N
      Fx  — longitudinal force (positive = drive, negative = brake), N
      Fy  — lateral force (positive = left), N
      Fxmax — longitudinal capacity from traction ellipse at this Fz, N
      Fymax — lateral capacity, N
    """
    Fz:    float
    Fx:    float
    Fy:    float
    Fxmax: float
    Fymax: float

    @property
    def ellipse_utilisation(self) -> float:
        """Traction ellipse usage fraction (0–1+)."""
        if self.Fxmax <= 0 or self.Fymax <= 0:
            return 0.0
        return ((self.Fx / self.Fxmax) ** 2 + (self.Fy / self.Fymax) ** 2) ** 0.5


@dataclass
class VehicleState:
    """
    Complete vehicle state at one solver step (one arc-length station).

    Designed to be the single source of telemetry truth. All quantities
    that matter for performance analysis or debugging are captured here.

    Extension notes
    ---------------
    For a future dynamic simulation, add:
      - vx, vy     : body-frame velocity components
      - yaw_rate   : r (rad/s)
      - slip_FL … slip_RR : tyre slip angles / longitudinal slip ratios
      - roll_angle, pitch_angle
    """
    # ---- Track position ----
    s:       float   # arc-length along track, m
    x:       float   # Cartesian x, m
    y:       float   # Cartesian y, m
    heading: float   # track tangent angle, rad

    # ---- Vehicle motion ----
    v:     float   # speed, m/s
    ax:    float   # longitudinal acceleration, m/s²  (+ = forward)
    ay:    float   # lateral acceleration, m/s²       (+ = left turn)
    kappa: float   # track curvature, 1/m

    # ---- Tyre states (FL=Front Left, FR=Front Right, RL=Rear Left, RR=Rear Right) ----
    wheel_FL: WheelState
    wheel_FR: WheelState
    wheel_RL: WheelState
    wheel_RR: WheelState

    # ---- Aerodynamics ----
    downforce: float   # total aero downforce, N
    drag:      float   # aero drag force, N

    # ---- Powertrain ----
    motor_rpm:    float   # motor shaft speed, RPM
    motor_torque: float   # motor shaft torque, Nm
    drive_force:  float   # total wheel-level drive force, N
    brake_force:  float   # total wheel-level braking force, N (positive magnitude)

    # ---- Battery ----
    SOC:             float   # state of charge, 0–1
    battery_temp:    float   # pack temperature, °C
    battery_current: float   # pack current, A
    power_demand:    float   # mechanical power demand, W

    # ---- Timing ----
    t:  float   # elapsed time from start of event, s
    dt: float   # duration of this integration step, s
