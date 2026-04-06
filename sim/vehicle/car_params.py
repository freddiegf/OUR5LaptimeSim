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


@dataclass
class BatteryParams:
    capacity_kWh: float
    nominal_voltage: float        # V_oc at SOC=1.0, V
    internal_resistance: float    # total pack resistance, Ω
    pack_thermal_mass: float      # effective J/K  (m_pack × Cp)
    initial_temperature: float    # °C
    initial_SOC: float = 1.0


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

def load_car_params(yaml_path: str) -> CarParams:
    """Parse a car parameter YAML file and return a validated CarParams."""
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)

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
    )

    bat_raw = raw["battery"]
    battery = BatteryParams(
        capacity_kWh=bat_raw["capacity_kWh"],
        nominal_voltage=bat_raw["nominal_voltage"],
        internal_resistance=bat_raw["internal_resistance"],
        pack_thermal_mass=bat_raw["pack_thermal_mass"],
        initial_temperature=bat_raw["initial_temperature"],
        initial_SOC=bat_raw.get("initial_SOC", 1.0),
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
