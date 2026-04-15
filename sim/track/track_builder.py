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


# ---------------------------------------------------------------------------
# SVG loader
# ---------------------------------------------------------------------------

def load_track_from_svg(
    svg_path: str,
    total_length: float,
    ds: float = 0.5,
    smooth: bool = True,
) -> TrackProfile:
    """
    Build a TrackProfile from the first <path> element of an SVG file.

    The SVG path is parsed, densely sampled along each sub-segment, uniformly
    resampled in arc length, scaled so its real-world length equals
    ``total_length`` metres, and differentiated to produce a curvature profile.
    A Savitzky-Golay smoother (~2.5 m window) removes node-junction noise.
    Heading and Cartesian position are then re-integrated from the smoothed
    curvature, matching ``build_track`` so the returned profile is
    self-consistent. s=0 is placed at the first node of the SVG path.

    SVG y is flipped to give a standard map frame (y-up).
    """
    import re
    from svgpathtools import parse_path
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d

    with open(svg_path, "r") as fh:
        content = fh.read()
    match = re.search(r'<path\b[^>]*\bd="([^"]*)"', content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No <path d='...'> element found in {svg_path}")
    path = parse_path(match.group(1))

    # Densely sample each sub-segment, building (x, y) with cumulative arc length.
    n_sub = 256
    raw_x: list[float] = []
    raw_y: list[float] = []
    raw_s: list[float] = []
    running = 0.0
    for seg_idx, seg in enumerate(path):
        ts  = np.linspace(0.0, 1.0, n_sub)
        pts = [seg.point(t) for t in ts]
        xs  = np.array([p.real for p in pts])
        ys  = np.array([p.imag for p in pts])
        seg_len = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        cum     = np.concatenate([[0.0], np.cumsum(seg_len)]) + running
        if seg_idx == 0:
            raw_x.extend(xs.tolist())
            raw_y.extend(ys.tolist())
            raw_s.extend(cum.tolist())
        else:
            raw_x.extend(xs[1:].tolist())
            raw_y.extend(ys[1:].tolist())
            raw_s.extend(cum[1:].tolist())
        running = cum[-1]

    raw_x_arr = np.asarray(raw_x)
    raw_y_arr = np.asarray(raw_y)
    raw_s_arr = np.asarray(raw_s)

    total_svg = float(raw_s_arr[-1])
    if total_svg <= 0.0:
        raise ValueError(f"SVG path in {svg_path} has zero length")
    scale = total_length / total_svg

    # Uniform arc-length grid in metres.
    N         = max(2, int(round(total_length / ds)) + 1)
    s_arr     = np.linspace(0.0, total_length, N)
    ds_actual = float(s_arr[1] - s_arr[0])

    target_svg_s = s_arr / scale
    x_raw =  np.interp(target_svg_s, raw_s_arr, raw_x_arr) * scale
    y_raw = -np.interp(target_svg_s, raw_s_arr, raw_y_arr) * scale  # SVG y-down → map y-up

    # Pre-smooth x/y to round out corner-node tangent discontinuities from
    # the hand-drawn SVG. sigma ≈ 1.5 m preserves true corners down to ~4 m
    # radius while killing node-junction spikes. Wrap mode since track is closed.
    if smooth:
        sigma_pts = 1.5 / ds_actual
        x_raw = gaussian_filter1d(x_raw, sigma_pts, mode="wrap")
        y_raw = gaussian_filter1d(y_raw, sigma_pts, mode="wrap")

    # Heading by finite difference, unwrapped so curvature is continuous.
    dx = np.gradient(x_raw, ds_actual)
    dy = np.gradient(y_raw, ds_actual)
    heading_raw = np.unwrap(np.arctan2(dy, dx))
    kappa_arr   = np.gradient(heading_raw, ds_actual)

    if smooth:
        window = max(5, int(round(2.5 / ds_actual)))
        if window % 2 == 0:
            window += 1
        if window < len(kappa_arr):
            kappa_arr = savgol_filter(
                kappa_arr, window_length=window, polyorder=3, mode="wrap",
            )

    # Re-integrate so x/y/heading are consistent with kappa (same scheme as build_track).
    # Start at origin with heading matching the SVG's first tangent, so plan-view
    # plots align with the synthetic track convention.
    heading_arr = np.zeros(N)
    x_arr       = np.zeros(N)
    y_arr       = np.zeros(N)
    heading_arr[0] = float(heading_raw[0])
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
        segments=[],
    )


# ---------------------------------------------------------------------------
# Sprint / endurance track layout dispatch
# ---------------------------------------------------------------------------

SPRINT_LAYOUTS = ("fsuk2025", "synthetic")
DEFAULT_SPRINT_LAYOUT = "fsuk2025"


def build_sprint_track(layout: str, ds: float) -> TrackProfile:
    """
    Return the sprint/endurance lap for the chosen layout.

    'fsuk2025' — real FSUK 2025 centerline from FSUK2025.svg, scaled to 1 km/lap.
    'synthetic' — hand-authored track in config/track_sprint.yaml (~891 m).
    """
    if layout == "fsuk2025":
        return load_track_from_svg("FSUK2025.svg", total_length=1000.0, ds=ds)
    if layout == "synthetic":
        segs = load_track_from_yaml("config/track_sprint.yaml")
        return build_track(segs, ds=ds)
    raise ValueError(f"Unknown track layout: {layout!r}")
