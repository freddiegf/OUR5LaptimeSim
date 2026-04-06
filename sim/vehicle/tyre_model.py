"""
tyre_model.py
=============
Traction-ellipse tyre model.

Each tyre has load-dependent friction coefficients:
    mu(Fz) = mu_nominal + k_load * (Fz - Fz_nominal)

The traction ellipse couples longitudinal and lateral capacity:
    (Fx / Fxmax)² + (Fy / Fymax)² ≤ 1
"""

from __future__ import annotations

import numpy as np
from sim.vehicle.car_params import TyreParams


class TyreModel:
    def __init__(self, params: TyreParams) -> None:
        self.p = params

    # ------------------------------------------------------------------
    # Friction coefficients (load-dependent)
    # ------------------------------------------------------------------

    def mu_x(self, Fz: float) -> float:
        """Longitudinal friction coefficient at normal load Fz."""
        return max(0.01, self.p.mu_x_nominal
                   + self.p.mu_x_load_sensitivity * (Fz - self.p.Fz_nominal))

    def mu_y(self, Fz: float) -> float:
        """Lateral friction coefficient at normal load Fz."""
        return max(0.01, self.p.mu_y_nominal
                   + self.p.mu_y_load_sensitivity * (Fz - self.p.Fz_nominal))

    # ------------------------------------------------------------------
    # Force capacities
    # ------------------------------------------------------------------

    def Fxmax(self, Fz: float) -> float:
        """Maximum longitudinal force capacity, N."""
        return self.mu_x(Fz) * max(0.0, Fz)

    def Fymax(self, Fz: float) -> float:
        """Maximum lateral force capacity, N."""
        return self.mu_y(Fz) * max(0.0, Fz)

    # ------------------------------------------------------------------
    # Traction ellipse
    # ------------------------------------------------------------------

    def Fx_available(self, Fz: float, Fy_demanded: float) -> float:
        """
        Maximum longitudinal force available given a demanded lateral force Fy.

        From the traction ellipse:
            Fx_avail = Fxmax * sqrt(1 - (Fy / Fymax)²)

        Returns 0 if the lateral demand already saturates the tyre.
        """
        fxmax = self.Fxmax(Fz)
        fymax = self.Fymax(Fz)
        if fymax <= 0.0:
            return 0.0
        ratio_sq = min(1.0, (Fy_demanded / fymax) ** 2)
        return fxmax * np.sqrt(max(0.0, 1.0 - ratio_sq))

    def combined_limit(self, Fz: float, Fx: float, Fy: float) -> float:
        """
        Returns the traction ellipse utilisation (0–1+).
        Values > 1 indicate the tyre is over-limit.
        """
        fxmax = self.Fxmax(Fz)
        fymax = self.Fymax(Fz)
        if fxmax <= 0 or fymax <= 0:
            return float("inf")
        return np.sqrt((Fx / fxmax) ** 2 + (Fy / fymax) ** 2)
