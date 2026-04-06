"""
track_segment.py
================
Dataclass for a single track segment (straight or constant-radius corner).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TrackSegment:
    """
    A segment of constant curvature.

    Attributes
    ----------
    segment_type : "straight" | "corner"
    length       : Arc length of the segment, m.
    radius       : Signed radius for corners, m.
                   Positive = left turn (CCW), negative = right turn (CW).
                   Zero / unused for straights.
    """
    segment_type: Literal["straight", "corner"]
    length: float
    radius: float = 0.0

    @property
    def curvature(self) -> float:
        """Signed curvature κ = 1/radius (0 for straights)."""
        if self.segment_type == "straight" or self.radius == 0.0:
            return 0.0
        return 1.0 / self.radius
