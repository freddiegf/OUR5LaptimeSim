"""
aero.py
=======
Aerodynamic model: downforce and drag as functions of speed.

Downforce is distributed front/rear using the Centre of Pressure (CoP) location.
"""

from __future__ import annotations

from sim.vehicle.car_params import AeroParams, CarParams


class AeroModel:
    def __init__(self, params: AeroParams, car: CarParams) -> None:
        self.p = params
        self.car = car
        # Front/rear downforce fractions (constant for a rigid aero package)
        self._rear_frac  = self.p.cop_x / self.car.wheelbase
        self._front_frac = 1.0 - self._rear_frac

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def downforce(self, v: float) -> float:
        """Total aero downforce (positive = pushing car down), N."""
        return 0.5 * self.p.rho_air * v ** 2 * abs(self.p.Cl) * self.p.area

    def drag_force(self, v: float) -> float:
        """Aerodynamic drag (opposing vehicle motion), N."""
        return 0.5 * self.p.rho_air * v ** 2 * self.p.Cd * self.p.area

    def downforce_distribution(self, v: float) -> tuple[float, float]:
        """
        Returns (Fz_aero_front, Fz_aero_rear) in N.

        Distribution is fixed by the CoP position relative to the wheelbase:
            rear_fraction  = cop_x / wheelbase
            front_fraction = 1 - rear_fraction
        """
        df = self.downforce(v)
        return df * self._front_frac, df * self._rear_frac
