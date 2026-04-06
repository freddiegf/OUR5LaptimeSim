"""
plotter.py
==========
All matplotlib figure generation for event results.

Each plot is saved as a PNG file in an output directory.
Figures are also returned so callers can display them interactively.

Plots generated per event:
  1. Track map with speed colour overlay
  2. g-g diagram with GGV envelope overlay
  3. Speed vs distance
  4. Battery temperature and SOC vs distance (if battery data present)
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from sim.events.base_event import EventResult
from sim.ggv.ggv_builder import GGVBuilder


_OUTPUT_DIR = "output"


def _ensure_output_dir() -> str:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR


def _safe_filename(event_name: str, suffix: str) -> str:
    safe = event_name.lower().replace(" ", "_")
    return os.path.join(_ensure_output_dir(), f"{safe}_{suffix}.png")


# ---------------------------------------------------------------------------
# 1. Track map with speed overlay
# ---------------------------------------------------------------------------

def plot_track_speed(result: EventResult, show: bool = False) -> plt.Figure:
    """Colour-coded track map: colour = speed."""
    states = result.states
    x = np.array([s.x for s in states])
    y = np.array([s.y for s in states])
    v = np.array([s.v for s in states]) * 3.6   # → km/h

    fig, ax = plt.subplots(figsize=(9, 7))
    points  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection
    norm = mcolors.Normalize(vmin=v.min(), vmax=v.max())
    lc   = LineCollection(segs, cmap="plasma", norm=norm, linewidth=2)
    lc.set_array(v[:-1])
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="Speed (km/h)")

    ax.set_xlim(x.min() - 5, x.max() + 5)
    ax.set_ylim(y.min() - 5, y.max() + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{result.event_name} — Track Speed Map")
    fig.tight_layout()
    fig.savefig(_safe_filename(result.event_name, "track_speed"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 2. g-g diagram
# ---------------------------------------------------------------------------

def plot_gg_diagram(
    result: EventResult,
    ggv: Optional[GGVBuilder] = None,
    show: bool = False,
) -> plt.Figure:
    """g-g scatter plot with optional GGV envelope overlay."""
    states = result.states
    ax_g   = np.array([s.ax for s in states]) / 9.81
    ay_g   = np.array([s.ay for s in states]) / 9.81

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ay_g, ax_g, s=2, alpha=0.5, c="steelblue", label="Sim data")

    if ggv is not None:
        # Draw GGV envelope at representative speeds
        v_samples = np.linspace(ggv.v_range[0], ggv.v_range[-1], 5)
        for v_s in v_samples:
            ay_env = ggv.ay_range
            axmax  = np.array([ggv.query_ax_max(v_s, a) for a in ay_env]) / 9.81
            axmin  = np.array([ggv.query_ax_min(v_s, a) for a in ay_env]) / 9.81
            ay_g_env = ay_env / 9.81
            ax.plot(ay_g_env, axmax, "r--", linewidth=0.8, alpha=0.5)
            ax.plot(ay_g_env, axmin, "b--", linewidth=0.8, alpha=0.5)

    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Lateral acceleration (g)")
    ax.set_ylabel("Longitudinal acceleration (g)")
    ax.set_title(f"{result.event_name} — g-g Diagram")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_safe_filename(result.event_name, "gg_diagram"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 3. Speed vs distance
# ---------------------------------------------------------------------------

def plot_speed_distance(result: EventResult, show: bool = False) -> plt.Figure:
    states = result.states
    s_km   = np.array([s.s for s in states]) / 1000.0
    v_kph  = np.array([s.v for s in states]) * 3.6

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s_km, v_kph, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"{result.event_name} — Speed vs Distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_safe_filename(result.event_name, "speed_distance"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 4. Battery: temperature and SOC vs distance
# ---------------------------------------------------------------------------

def plot_battery(result: EventResult, show: bool = False) -> plt.Figure:
    states = result.states
    s_km   = np.array([s.s for s in states]) / 1000.0
    temp   = np.array([s.battery_temp for s in states])
    soc    = np.array([s.SOC for s in states]) * 100.0   # %

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(s_km, temp, color="tomato", linewidth=1.2)
    ax1.set_ylabel("Battery Temperature (°C)")
    ax1.set_title(f"{result.event_name} — Battery State")
    ax1.grid(True, alpha=0.3)

    ax2.plot(s_km, soc, color="seagreen", linewidth=1.2)
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("State of Charge (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(_safe_filename(result.event_name, "battery"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 5. GGV surface visualisation
# ---------------------------------------------------------------------------

def plot_ggv_surface(ggv: GGVBuilder, show: bool = False) -> plt.Figure:
    """Plot the GGV envelope as a family of curves at different speeds."""
    fig, ax = plt.subplots(figsize=(8, 8))

    v_samples = np.linspace(ggv.v_range[0], ggv.v_range[-1], 8)
    cmap = cm.get_cmap("viridis", len(v_samples))

    for idx, v_s in enumerate(v_samples):
        ay_env = ggv.ay_range
        axmax  = np.array([ggv.query_ax_max(v_s, a) for a in ay_env]) / 9.81
        axmin  = np.array([ggv.query_ax_min(v_s, a) for a in ay_env]) / 9.81
        ay_g   = ay_env / 9.81
        colour = cmap(idx)
        label  = f"{v_s*3.6:.0f} km/h"
        ax.plot(ay_g, axmax, color=colour, linewidth=1.5, label=label)
        ax.plot(ay_g, axmin, color=colour, linewidth=1.5)

    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Lateral acceleration (g)")
    ax.set_ylabel("Longitudinal acceleration (g)")
    ax.set_title("GGV Performance Envelope")
    ax.legend(title="Speed", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(_ensure_output_dir(), "ggv_surface.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all plots for one event result
# ---------------------------------------------------------------------------

def generate_all_plots(
    result: EventResult,
    ggv: Optional[GGVBuilder] = None,
    show: bool = False,
) -> None:
    """Generate and save all standard plots for an event result."""
    plot_track_speed(result, show=show)
    plot_gg_diagram(result, ggv=ggv, show=show)
    plot_speed_distance(result, show=show)
    plot_battery(result, show=show)
    print(f"  Plots saved to '{_OUTPUT_DIR}/'")
