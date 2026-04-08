"""
car_params.py
=============
Dataclass hierarchy for all vehicle parameters.
Loaded from a YAML file via load_car_params().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import yaml


# ---------------------------------------------------------------------------
# Sub-parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TyreParams:
    mu_x_nominal: float
    mu_y_nominal: float
    mu_x_load_sensitivity: float   # d(mu_x)/dFz  [1/N]  — typically negative
    mu_y_load_sensitivity: float   # d(mu_y)/dFz  [1/N]
    radius: float                  # rolling radius, m
    Fz_nominal: float              # reference normal load for mu calibration, N


@dataclass
class AeroParams:
    Cl: float           # lift coefficient (negative = downforce)
    Cd: float           # drag coefficient
    area: float         # reference area, m²
    cop_x: float        # Centre of Pressure from front axle, m
    rho_air: float = 1.225


@dataclass
class PowertrainParams:
    torque_curve: List[Tuple[float, float]]   # [(rpm, Nm), ...]
    gear_ratio: float
    drivetrain_efficiency: float
    drivetrain_type: str          # "RWD" | "FWD" | "AWD"
    awd_front_bias: float = 0.5  # fraction of torque to front (AWD only)
    power_limit_kW: float = 80.0  # system power limit (e.g. FS rules), kW


@dataclass
class BatteryParams:
    """
    Per-cell battery parameters (e.g. Molicel P42A 21700).
    Pack quantities (voltage, resistance, capacity, thermal mass) are derived
    from the cell parameters and the series/parallel configuration.
    """
    # Cell electrical
    cell_capacity_Ah: float       # Ah per cell
    cell_V_nominal: float         # open-circuit voltage at SOC=1 per cell, V
    cell_R_int_Ohm: float         # DC internal resistance per cell, Ω
    # Cell thermal
    cell_mass_kg: float           # mass per cell, kg
    cell_Cp_J_per_kgK: float      # specific heat per cell, J/(kg·K)
    # Pack configuration
    n_series: int                 # number of cells in series
    n_parallel: int               # number of parallel strings
    # Initial conditions
    initial_temperature: float    # °C (uniform cell temperature)
    initial_SOC: float = 1.0

    # --- Derived pack quantities (computed in __post_init__) ---
    pack_V_nominal: float = field(init=False)        # V
    pack_R_int: float = field(init=False)            # Ω
    pack_capacity_kWh: float = field(init=False)     # kWh
    pack_thermal_mass: float = field(init=False)     # J/K (all cells)
    n_cells_total: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_cells_total    = self.n_series * self.n_parallel
        self.pack_V_nominal   = self.n_series * self.cell_V_nominal
        # Series stack R_cell/n_parallel in each series layer, times n_series layers
        self.pack_R_int       = self.n_series * self.cell_R_int_Ohm / self.n_parallel
        # Total capacity in kWh: pack_Ah × pack_V gives Wh, ÷ 1000 → kWh
        pack_Ah               = self.cell_capacity_Ah * self.n_parallel
        self.pack_capacity_kWh = (pack_Ah * self.pack_V_nominal) / 1000.0
        # Thermal: all cells lumped together
        self.pack_thermal_mass = (self.n_cells_total
                                  * self.cell_mass_kg
                                  * self.cell_Cp_J_per_kgK)


# ---------------------------------------------------------------------------
# Top-level CarParams
# ---------------------------------------------------------------------------

@dataclass
class CarParams:
    # Geometry & mass
    mass: float        # kg (including driver)
    cog_x: float       # longitudinal CoG from front axle, m
    cog_y: float       # lateral CoG from centreline, m
    cog_z: float       # CoG height, m
    wheelbase: float   # m
    front_track: float # m
    rear_track: float  # m

    # Sub-models
    front_tyre: TyreParams
    rear_tyre: TyreParams
    aero: AeroParams
    powertrain: PowertrainParams
    battery: BatteryParams

    # Derived — computed in __post_init__
    lr: float = field(init=False)   # CoG to rear axle, m
    lf: float = field(init=False)   # CoG to front axle, m
    weight_front: float = field(init=False)   # static front load, N
    weight_rear: float = field(init=False)    # static rear load, N

    def __post_init__(self) -> None:
        g = 9.81
        self.lf = self.cog_x                        # front axle to CoG
        self.lr = self.wheelbase - self.cog_x        # CoG to rear axle
        self.weight_rear  = self.mass * g * self.lf / self.wheelbase
        self.weight_front = self.mass * g * self.lr / self.wheelbase


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def build_car_params(raw: dict) -> CarParams:
    """Build a CarParams from a raw dict (e.g. parsed YAML)."""
    front_tyre = TyreParams(**raw["front_tyre"])
    rear_tyre  = TyreParams(**raw["rear_tyre"])

    aero_raw = raw["aero"]
    aero = AeroParams(**aero_raw)

    pt_raw = raw["powertrain"]
    torque_curve = [tuple(pair) for pair in pt_raw["torque_curve"]]
    powertrain = PowertrainParams(
        torque_curve=torque_curve,
        gear_ratio=pt_raw["gear_ratio"],
        drivetrain_efficiency=pt_raw["drivetrain_efficiency"],
        drivetrain_type=pt_raw["drivetrain_type"],
        awd_front_bias=pt_raw.get("awd_front_bias", 0.5),
        power_limit_kW=pt_raw.get("power_limit_kW", 80.0),
    )

    bat_raw = raw["battery"]
    battery = BatteryParams(
        cell_capacity_Ah    = bat_raw["cell_capacity_Ah"],
        cell_V_nominal      = bat_raw["cell_V_nominal"],
        cell_R_int_Ohm      = bat_raw["cell_R_int_Ohm"],
        cell_mass_kg        = bat_raw["cell_mass_kg"],
        cell_Cp_J_per_kgK   = bat_raw["cell_Cp_J_per_kgK"],
        n_series            = int(bat_raw["n_series"]),
        n_parallel          = int(bat_raw["n_parallel"]),
        initial_temperature = bat_raw["initial_temperature"],
        initial_SOC         = bat_raw.get("initial_SOC", 1.0),
    )

    return CarParams(
        mass=raw["mass"],
        cog_x=raw["cog_x"],
        cog_y=raw["cog_y"],
        cog_z=raw["cog_z"],
        wheelbase=raw["wheelbase"],
        front_track=raw["front_track"],
        rear_track=raw["rear_track"],
        front_tyre=front_tyre,
        rear_tyre=rear_tyre,
        aero=aero,
        powertrain=powertrain,
        battery=battery,
    )


def load_car_params(yaml_path: str) -> CarParams:
    """Parse a car parameter YAML file and return a validated CarParams."""
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)
    return build_car_params(raw)
