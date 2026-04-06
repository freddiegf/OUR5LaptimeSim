"""
battery.py
==========
Lumped electrical + thermal battery model.

Electrical model
----------------
  V_oc(SOC) = V_nominal × SOC   (linear OCV curve)

  Power balance: P = V_oc × I - I² × R_int
  Solving for current I (quadratic formula, lower root chosen):
      R_int × I² - V_oc × I + P = 0
      I = (V_oc - sqrt(V_oc² - 4×R_int×P)) / (2×R_int)

Thermal model (Ohmic heating, no cooling)
------------------------------------------
  Q_heat = I² × R_int   (W)
  dT/dt  = Q_heat / pack_thermal_mass
  T_new  = T + Q_heat × dt / pack_thermal_mass
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.vehicle.car_params import BatteryParams


@dataclass
class BatteryState:
    SOC: float           # 0–1
    temperature: float   # °C
    voltage_oc: float    # V
    current: float       # A (this step)
    power_loss: float    # W (heat generated)
    energy_used_J: float # J consumed this step


class BatteryModel:
    def __init__(self, params: BatteryParams) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        """Restore battery to initial conditions."""
        self.SOC = self.p.initial_SOC
        self.temperature = self.p.initial_temperature
        self._capacity_J = self.p.capacity_kWh * 3.6e6

    # ------------------------------------------------------------------
    # OCV model
    # ------------------------------------------------------------------

    def V_oc(self, SOC: float) -> float:
        """Open-circuit voltage (V) as a linear function of SOC."""
        return self.p.nominal_voltage * max(0.0, SOC)

    # ------------------------------------------------------------------
    # Current solver
    # ------------------------------------------------------------------

    def solve_current(self, P_demand: float) -> tuple[float, bool]:
        """
        Solve for battery current given power demand P_demand (W).

        Quadratic: R×I² - V_oc×I + P = 0
        Takes the lower root (physically correct for normal operation).

        Returns (I, feasible). feasible=False means P_demand exceeds the
        maximum power the pack can deliver; current is clamped in that case.
        """
        if P_demand <= 0.0:
            # Regeneration or idle — simplified: treat as zero current
            return 0.0, True

        V = self.V_oc(self.SOC)
        R = self.p.internal_resistance
        discriminant = V ** 2 - 4.0 * R * P_demand

        if discriminant < 0.0:
            # Demand exceeds maximum deliverable power — clamp to peak current
            I_peak = V / (2.0 * R)
            return I_peak, False

        I = (V - math.sqrt(discriminant)) / (2.0 * R)
        return I, True

    # ------------------------------------------------------------------
    # Step update
    # ------------------------------------------------------------------

    def step(self, P_demand: float, dt: float) -> BatteryState:
        """
        Advance battery state by dt seconds under power demand P_demand (W).

        Updates SOC and temperature in-place; returns a BatteryState snapshot.
        """
        I, feasible = self.solve_current(P_demand)
        R = self.p.internal_resistance

        Q_heat      = I ** 2 * R * dt               # J — Ohmic heating
        energy_used = self.V_oc(self.SOC) * I * dt  # J — energy drawn from pack

        self.SOC = max(0.0, self.SOC - energy_used / self._capacity_J)
        self.temperature += Q_heat / self.p.pack_thermal_mass

        return BatteryState(
            SOC=self.SOC,
            temperature=self.temperature,
            voltage_oc=self.V_oc(self.SOC),
            current=I,
            power_loss=Q_heat / dt if dt > 0 else 0.0,
            energy_used_J=energy_used,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def energy_remaining_kWh(self) -> float:
        return self.SOC * self.p.capacity_kWh

    def energy_used_kWh(self) -> float:
        return (self.p.initial_SOC - self.SOC) * self.p.capacity_kWh
