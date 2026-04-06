"""
track_builder.py
================
Converts a list of TrackSegment objects into a discretised TrackProfile.

The track is parameterised by arc length s. At each station:
    heading[i+1] = heading[i] + kappa[i] * ds
    x[i+1]       = x[i]       + ds * cos(heading[i])
    y[i+1]       = y[i]       + ds * sin(heading[i])

This gives a smooth, integration-exact path for constant-curvature segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import yaml

from sim.track.track_segment import TrackSegment


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class TrackProfile:
    """Discretised track representation."""
    s:            np.ndarray   # arc-length stations, m  — shape (N,)
    kappa:        np.ndarray   # curvature at each station, 1/m
    x:            np.ndarray   # Cartesian x, m
    y:            np.ndarray   # Cartesian y, m
    heading:      np.ndarray   # heading angle, rad
    total_length: float        # m
    ds:           float        # spatial step size, m
    segments:     List[TrackSegment]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_track(segments: List[TrackSegment], ds: float = 0.5) -> TrackProfile:
    """
    Discretise a list of TrackSegments into a uniform arc-length grid.

    Parameters
    ----------
    segments : list of TrackSegment
    ds       : spatial resolution, m (default 0.5 m)

    Returns
    -------
    TrackProfile
    """
    total_length = sum(seg.length for seg in segments)
    N = max(2, int(round(total_length / ds)) + 1)
    s_arr       = np.linspace(0.0, total_length, N)
    ds_actual   = s_arr[1] - s_arr[0]

    kappa_arr   = np.zeros(N)
    x_arr       = np.zeros(N)
    y_arr       = np.zeros(N)
    heading_arr = np.zeros(N)

    # Build curvature profile from segment list
    seg_ends = np.cumsum([seg.length for seg in segments])

    for i, s_val in enumerate(s_arr):
        # Find which segment this station belongs to
        seg_idx = int(np.searchsorted(seg_ends, s_val, side="right"))
        seg_idx = min(seg_idx, len(segments) - 1)
        kappa_arr[i] = segments[seg_idx].curvature

    # Integrate heading and position
    for i in range(N - 1):
        heading_arr[i + 1] = heading_arr[i] + kappa_arr[i] * ds_actual
        x_arr[i + 1]       = x_arr[i] + ds_actual * np.cos(heading_arr[i])
        y_arr[i + 1]       = y_arr[i] + ds_actual * np.sin(heading_arr[i])

    return TrackProfile(
        s=s_arr,
        kappa=kappa_arr,
        x=x_arr,
        y=y_arr,
        heading=heading_arr,
        total_length=total_length,
        ds=ds_actual,
        segments=segments,
    )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_track_from_yaml(yaml_path: str) -> List[TrackSegment]:
    """
    Parse a track YAML file and return a list of TrackSegment objects.

    Expected YAML format:
        segments:
          - {type: straight, length: 75.0}
          - {type: corner, radius: -5.0, length: 15.71}
    """
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)

    segments = []
    for item in raw["segments"]:
        seg_type = item.get("type", "straight")
        length   = float(item["length"])
        radius   = float(item.get("radius", 0.0))
        segments.append(TrackSegment(
            segment_type=seg_type,
            length=length,
            radius=radius,
        ))
    return segments
