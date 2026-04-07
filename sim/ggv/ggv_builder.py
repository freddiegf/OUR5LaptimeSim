"""
ggv_builder.py
==============
Builds and queries the GGV (speed-acceleration) performance envelope.

The GGV surface maps (v, ay) → (ax_max, ax_min):
  - ax_max: maximum longitudinal acceleration at speed v with lateral demand ay
  - ax_min: maximum braking deceleration (most negative) at (v, ay)

Construction uses the two-track load transfer model:
  1. Distribute lateral load transfer between front/rear axles.
  2. Compute normal loads on all four wheels.
  3. Distribute lateral force demand proportional to Fz.
  4. Apply traction ellipse to get remaining longitudinal capacity.
  5. Check motor torque limit (ax_max) or braking limit (ax_min).
  6. Iterate 3 times to resolve the ax ↔ longitudinal load transfer coupling.

After building, use query_ax_max / query_ax_min for fast interpolated lookup
via scipy.interpolate.RegularGridInterpolator.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from sim.vehicle.car_params import CarParams
from sim.vehicle.tyre_model import TyreModel
from sim.vehicle.aero import AeroModel
from sim.vehicle.powertrain import Powertrain


class GGVBuilder:
    def __init__(
        self,
        car: CarParams,
        tyre_front: TyreModel,
        tyre_rear: TyreModel,
        aero: AeroModel,
        powertrain: Powertrain,
    ) -> None:
        self.car = car
        self.tyre_f = tyre_front
        self.tyre_r = tyre_rear
        self.aero = aero
        self.pt = powertrain

        self._v_range:  Optional[np.ndarray] = None
        self._ay_range: Optional[np.ndarray] = None
        self._ax_max_grid: Optional[np.ndarray] = None
        self._ax_min_grid: Optional[np.ndarray] = None
        self._interp_max: Optional[RegularGridInterpolator] = None
        self._interp_min: Optional[RegularGridInterpolator] = None
        self._ay_abs_max: float = 30.0   # updated when build() is called

    # ------------------------------------------------------------------
    # Internal: four-wheel load transfer
    # ------------------------------------------------------------------

    def _normal_loads(
        self, v: float, ax: float, ay: float
    ) -> tuple[float, float, float, float]:
        """
        Compute per-wheel normal loads (Fz_FL, Fz_FR, Fz_RL, Fz_RR) in N.

        Sign convention:
          ax > 0 = acceleration → rear loads up, front unloads
          ay > 0 = left turn    → right wheels (inside) unload, left wheels load

        Load transfer derivation (rigid vehicle, no suspension compliance):

          Axle loads with longitudinal transfer + aero:
            Fz_front = m*g*lr/L + Fz_aero_front - m*ax*h/L
            Fz_rear  = m*g*lf/L + Fz_aero_rear  + m*ax*h/L

          Lateral transfer split proportional to axle load:
            Fz_total = Fz_front + Fz_rear
            ΔFz_f = m*|ay|*h * (Fz_front / Fz_total) / t_front
            ΔFz_r = m*|ay|*h * (Fz_rear  / Fz_total) / t_rear

          Per wheel (left = outside for ay > 0):
            Fz_FL = 0.5*Fz_front + ΔFz_f * sign(ay)
            Fz_FR = 0.5*Fz_front - ΔFz_f * sign(ay)
            etc.
        """
        car = self.car
        g   = 9.81
        m   = car.mass
        h   = car.cog_z
        L   = car.wheelbase
        lf  = car.lf
        lr  = car.lr
        tf  = car.front_track
        tr  = car.rear_track

        Fz_aero_f, Fz_aero_r = self.aero.downforce_distribution(v)

        # Axle normal loads (including longitudinal transfer and aero)
        Fz_front = m * g * lr / L + Fz_aero_f - m * ax * h / L
        Fz_rear  = m * g * lf / L + Fz_aero_r + m * ax * h / L

        # Clamp axle loads (cannot go negative in rigid model)
        Fz_front = max(0.0, Fz_front)
        Fz_rear  = max(0.0, Fz_rear)

        # Lateral load transfer magnitude per axle
        Fz_total = Fz_front + Fz_rear
        if Fz_total > 0.0:
            dFz_f = m * abs(ay) * h * (Fz_front / Fz_total) / tf
            dFz_r = m * abs(ay) * h * (Fz_rear  / Fz_total) / tr
        else:
            dFz_f = 0.0
            dFz_r = 0.0

        # Distribute: ay > 0 → left turn → left is outside (more load)
        sign = 1.0 if ay >= 0.0 else -1.0
        Fz_FL = max(0.0, 0.5 * Fz_front + dFz_f * sign)
        Fz_FR = max(0.0, 0.5 * Fz_front - dFz_f * sign)
        Fz_RL = max(0.0, 0.5 * Fz_rear  + dFz_r * sign)
        Fz_RR = max(0.0, 0.5 * Fz_rear  - dFz_r * sign)

        return Fz_FL, Fz_FR, Fz_RL, Fz_RR

    def _distribute_Fy(
        self,
        ay: float,
        Fz_FL: float, Fz_FR: float, Fz_RL: float, Fz_RR: float,
    ) -> tuple[float, float, float, float]:
        """
        Distribute total lateral force m*ay across wheels proportional to Fz.
        This assumes equal slip angles (no front/rear balance effect).
        """
        Fy_total = self.car.mass * ay
        Fz_total = Fz_FL + Fz_FR + Fz_RL + Fz_RR
        if Fz_total <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        k = Fy_total / Fz_total
        return k * Fz_FL, k * Fz_FR, k * Fz_RL, k * Fz_RR

    # ------------------------------------------------------------------
    # Internal: ax limits at a single (v, ay) operating point
    # ------------------------------------------------------------------

    def _ax_max_at(self, v: float, ay: float, n_iter: int = 3) -> float:
        """Maximum longitudinal acceleration at (v, ay), iterating on ax."""
        ax = 0.0
        front_frac, rear_frac = self.pt.driven_axle_fractions()
        r_rear  = self.car.rear_tyre.radius
        r_front = self.car.front_tyre.radius
        # Use average tyre radius for motor RPM calculation
        r_driven = rear_frac * r_rear + front_frac * r_front

        for _ in range(n_iter):
            Fz_FL, Fz_FR, Fz_RL, Fz_RR = self._normal_loads(v, ax, ay)
            Fy_FL, Fy_FR, Fy_RL, Fy_RR = self._distribute_Fy(
                ay, Fz_FL, Fz_FR, Fz_RL, Fz_RR)

            # Traction ellipse: available Fx per wheel
            Fx_FL = self.tyre_f.Fx_available(Fz_FL, abs(Fy_FL))
            Fx_FR = self.tyre_f.Fx_available(Fz_FR, abs(Fy_FR))
            Fx_RL = self.tyre_r.Fx_available(Fz_RL, abs(Fy_RL))
            Fx_RR = self.tyre_r.Fx_available(Fz_RR, abs(Fy_RR))

            # Only driven wheels contribute to drive force
            Fx_tyre = (front_frac * (Fx_FL + Fx_FR)
                       + rear_frac  * (Fx_RL + Fx_RR))

            # Motor torque limit
            F_motor = self.pt.max_drive_force(max(v, 0.1), r_driven)

            # System power limit (e.g. FS rules): terminal power × η → wheel force
            eta = self.pt.p.drivetrain_efficiency
            F_power_limit = (self.pt.p.power_limit_kW * 1000.0 * eta) / max(v, 0.1)

            F_drive  = min(Fx_tyre, F_motor, F_power_limit)
            F_drag   = self.aero.drag_force(v)
            ax = (F_drive - F_drag) / self.car.mass

        return ax

    def _ax_min_at(self, v: float, ay: float, n_iter: int = 3) -> float:
        """Maximum braking deceleration (negative) at (v, ay)."""
        ax = 0.0

        for _ in range(n_iter):
            Fz_FL, Fz_FR, Fz_RL, Fz_RR = self._normal_loads(v, ax, ay)
            Fy_FL, Fy_FR, Fy_RL, Fy_RR = self._distribute_Fy(
                ay, Fz_FL, Fz_FR, Fz_RL, Fz_RR)

            # All four wheels contribute to braking
            Fx_FL = self.tyre_f.Fx_available(Fz_FL, abs(Fy_FL))
            Fx_FR = self.tyre_f.Fx_available(Fz_FR, abs(Fy_FR))
            Fx_RL = self.tyre_r.Fx_available(Fz_RL, abs(Fy_RL))
            Fx_RR = self.tyre_r.Fx_available(Fz_RR, abs(Fy_RR))

            Fx_brake = Fx_FL + Fx_FR + Fx_RL + Fx_RR
            F_drag   = self.aero.drag_force(v)

            ax = -(Fx_brake + F_drag) / self.car.mass

        return ax

    # ------------------------------------------------------------------
    # Public: build GGV surface
    # ------------------------------------------------------------------

    def build(
        self,
        v_range: np.ndarray,
        ay_range: np.ndarray,
        n_iter: int = 3,
    ) -> None:
        """
        Build the full GGV surface.

        Parameters
        ----------
        v_range  : 1-D array of speeds, m/s
        ay_range : 1-D array of lateral accelerations, m/s² (symmetric about 0)
        n_iter   : fixed-point iterations for ax ↔ load-transfer coupling
        """
        self._v_range  = v_range
        self._ay_range = ay_range
        self._ay_abs_max = float(np.max(np.abs(ay_range)))

        nv  = len(v_range)
        nay = len(ay_range)

        ax_max_grid = np.zeros((nv, nay))
        ax_min_grid = np.zeros((nv, nay))

        for i, v in enumerate(v_range):
            for j, ay in enumerate(ay_range):
                ax_max_grid[i, j] = self._ax_max_at(v, ay, n_iter)
                ax_min_grid[i, j] = self._ax_min_at(v, ay, n_iter)

        self._ax_max_grid = ax_max_grid
        self._ax_min_grid = ax_min_grid

        self._interp_max = RegularGridInterpolator(
            (v_range, ay_range), ax_max_grid,
            method="linear", bounds_error=False, fill_value=None,
        )
        self._interp_min = RegularGridInterpolator(
            (v_range, ay_range), ax_min_grid,
            method="linear", bounds_error=False, fill_value=None,
        )

    # ------------------------------------------------------------------
    # Public: query
    # ------------------------------------------------------------------

    def _clip_query(self, v: float, ay: float) -> tuple[float, float]:
        """Clip (v, ay) to the GGV grid bounds for safe interpolation."""
        v_c  = float(np.clip(v,  self._v_range[0],  self._v_range[-1]))
        ay_c = float(np.clip(ay, self._ay_range[0], self._ay_range[-1]))
        return v_c, ay_c

    def query_ax_max(self, v: float, ay: float) -> float:
        """Interpolated maximum longitudinal acceleration at (v, ay)."""
        v_c, ay_c = self._clip_query(v, ay)
        return float(self._interp_max([[v_c, ay_c]]))

    def query_ax_min(self, v: float, ay: float) -> float:
        """Interpolated maximum braking deceleration (negative) at (v, ay)."""
        v_c, ay_c = self._clip_query(v, ay)
        return float(self._interp_min([[v_c, ay_c]]))

    @property
    def ay_abs_max(self) -> float:
        """Maximum absolute lateral acceleration in the GGV grid."""
        return self._ay_abs_max

    @property
    def v_range(self) -> np.ndarray:
        return self._v_range

    @property
    def ay_range(self) -> np.ndarray:
        return self._ay_range

    @property
    def ax_max_grid(self) -> np.ndarray:
        return self._ax_max_grid

    @property
    def ax_min_grid(self) -> np.ndarray:
        return self._ax_min_grid

    def normal_loads(
        self, v: float, ax: float, ay: float
    ) -> tuple[float, float, float, float]:
        """Public access to four-wheel normal load computation."""
        return self._normal_loads(v, ax, ay)
