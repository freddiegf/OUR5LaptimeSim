"""
battery.py
==========
Per-cell electrical + lumped thermal battery model.

Cells: Molicel P42A 21700 (or any cell configured in the YAML).
Pack: n_series cells in series, n_parallel strings in parallel.

Electrical model (pack level)
------------------------------
  V_oc(SOC) = n_series × V_cell(SOC)
  where V_cell(SOC) is interpolated from a realistic OCV lookup table.

  Quadratic current solve (P = terminal power = V_terminal × I):
    R_pack × I² - V_oc × I + P_demand = 0
    I = (V_oc - sqrt(V_oc² - 4·R_pack·P)) / (2·R_pack)
    lower root → normal operating regime

  Cell current = I_pack / n_parallel

Thermal model (Ohmic heating, no cooling)
------------------------------------------
  Heat generated per cell = (I_cell)² × R_cell
  Total heat = n_cells × (I_pack/n_parallel)² × R_cell
             = I_pack² × n_series × R_cell / n_parallel
             = I_pack² × R_pack

  dT/dt = Q_heat_total / (n_cells × m_cell × Cp)
        = I_pack² × R_pack / pack_thermal_mass
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.vehicle.car_params import BatteryParams


# Molicel P42A 21700 — per-cell OCV vs SOC lookup table
_CELL_OCV_SOC = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_CELL_OCV_V   = [2.5, 3.2, 3.6, 3.8, 3.95, 4.2]


@dataclass
class BatteryState:
    SOC: float              # 0–1
    temperature: float      # °C (cell/pack temperature, lumped)
    voltage_oc: float       # pack OCV, V
    current_pack: float     # pack current, A
    current_cell: float     # per-cell current, A
    power_loss: float       # Ohmic heat rate, W
    energy_used_J: float    # energy drawn this step, J


class BatteryModel:
    def __init__(self, params: BatteryParams) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        """Restore battery to initial conditions."""
        self.SOC         = self.p.initial_SOC
        self.temperature = self.p.initial_temperature
        self._capacity_J = self.p.pack_capacity_kWh * 3.6e6

    # ------------------------------------------------------------------
    # OCV model
    # ------------------------------------------------------------------

    def V_oc(self, SOC: float) -> float:
        """Pack open-circuit voltage (V) from per-cell OCV lookup table."""
        SOC_clamped = max(0.0, min(1.0, SOC))
        cell_V = float(np.interp(SOC_clamped, _CELL_OCV_SOC, _CELL_OCV_V))
        return cell_V * self.p.n_series

    # ------------------------------------------------------------------
    # Current solver
    # ------------------------------------------------------------------

    def solve_current(self, P_demand: float) -> tuple[float, bool]:
        """
        Solve for pack current given terminal power demand P_demand (W).

        Positive P_demand = discharging (driving).
        Negative P_demand = charging (regenerative braking).

        Quadratic: R_pack×I² - V_oc×I + P = 0
        Returns (I_pack, feasible).
        I_pack > 0 = discharge, I_pack < 0 = charge.
        feasible=False if demand exceeds maximum deliverable power.
        """
        if P_demand == 0.0:
            return 0.0, True

        V = self.V_oc(self.SOC)
        R = self.p.pack_R_int
        discriminant = V ** 2 - 4.0 * R * P_demand

        if discriminant < 0.0:
            # Discharge exceeds max power — clamp to peak current
            return V / (2.0 * R), False

        # For discharge (P>0): lower root gives smaller current (normal regime)
        # For charge (P<0): discriminant > V², so sqrt > V, giving I < 0
        I = (V - math.sqrt(discriminant)) / (2.0 * R)
        return I, True

    # ------------------------------------------------------------------
    # Step update
    # ------------------------------------------------------------------

    def step(self, P_demand: float, dt: float) -> BatteryState:
        """
        Advance battery state by dt seconds under power demand P_demand (W).

        P_demand > 0: discharging (driving) — SOC decreases.
        P_demand < 0: charging (regen braking) — SOC increases.

        Updates SOC and temperature in-place.
        """
        I_pack, _feasible = self.solve_current(P_demand)
        R = self.p.pack_R_int

        Q_heat      = I_pack ** 2 * R * dt              # J — total Ohmic heat
        energy_used = self.V_oc(self.SOC) * I_pack * dt  # J — energy from pack
        # energy_used > 0 → discharge, < 0 → charge

        self.SOC = max(0.0, min(1.0, self.SOC - energy_used / self._capacity_J))
        self.temperature += Q_heat / self.p.pack_thermal_mass

        return BatteryState(
            SOC=self.SOC,
            temperature=self.temperature,
            voltage_oc=self.V_oc(self.SOC),
            current_pack=I_pack,
            current_cell=I_pack / self.p.n_parallel,
            power_loss=Q_heat / dt if dt > 0 else 0.0,
            energy_used_J=energy_used,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def energy_remaining_kWh(self) -> float:
        return self.SOC * self.p.pack_capacity_kWh

    def max_power(self) -> float:
        """
        Maximum deliverable electrical power at current SOC, in Watts.

        Derived from the matched-load condition on the battery's equivalent
        circuit: P_max = V_oc² / (4 × R_pack).
        """
        V = self.V_oc(self.SOC)
        R = self.p.pack_R_int
        if R <= 0.0:
            return float("inf")
        return V ** 2 / (4.0 * R)

    def energy_used_kWh(self) -> float:
        return (self.p.initial_SOC - self.SOC) * self.p.pack_capacity_kWh
