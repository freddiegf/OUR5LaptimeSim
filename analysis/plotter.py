"""
plotter.py
==========
All matplotlib figure generation for event results.

Plots per event:
  Acceleration — 5-panel telemetry vs time (distance, velocity, accel, power, limiting factor)
  Skidpad      — no plots (time only)
  Sprint       — 5-panel telemetry vs time + track speed map
  Endurance    — lap time bar chart + battery SOC & temperature vs laps
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

# Limiting factor category colours
_LF_COLORS = {
    "Power limit": "#e67e22",   # orange
    "Battery":     "#e74c3c",   # red
    "Tyre":        "#3498db",   # blue
    "Motor":       "#2ecc71",   # green
    "Braking":     "#95a5a6",   # grey
}


def _ensure_output_dir() -> str:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR


def _safe_filename(event_name: str, suffix: str) -> str:
    safe = event_name.lower().replace(" ", "_")
    return os.path.join(_ensure_output_dir(), f"{safe}_{suffix}.png")


# ---------------------------------------------------------------------------
# 1. Five-panel telemetry vs time (Acceleration & Sprint)
# ---------------------------------------------------------------------------

def plot_telemetry(result: EventResult, show: bool = False) -> plt.Figure:
    """5-panel time-domain plot: distance, velocity, accel, power, limiting factor."""
    states = result.states
    t   = np.array([s.t for s in states])
    d   = np.array([s.s for s in states])
    v   = np.array([s.v for s in states]) * 3.6       # km/h
    a   = np.array([s.ax for s in states]) / 9.81     # g
    pwr = np.array([s.power_demand for s in states]) / 1000.0  # kW
    lf  = [s.limiting_factor for s in states]

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.4]})

    # Distance
    axes[0].plot(t, d, color="seagreen", linewidth=1.2)
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title(f"{result.event_name} — Telemetry vs Time")
    axes[0].grid(True, alpha=0.3)

    # Velocity
    axes[1].plot(t, v, color="steelblue", linewidth=1.2)
    axes[1].set_ylabel("Speed (km/h)")
    axes[1].grid(True, alpha=0.3)

    # Acceleration
    axes[2].plot(t, a, color="tomato", linewidth=1.2)
    axes[2].axhline(0, color="k", linewidth=0.5)
    axes[2].set_ylabel("Acceleration (g)")
    axes[2].grid(True, alpha=0.3)

    # Power
    axes[3].plot(t, pwr, color="darkorange", linewidth=1.0)
    axes[3].set_ylabel("Terminal Power (kW)")
    axes[3].grid(True, alpha=0.3)

    # Limiting factor — colour strip
    ax_lf = axes[4]
    for cat, colour in _LF_COLORS.items():
        mask = np.array([f == cat for f in lf])
        if mask.any():
            ax_lf.fill_between(t, 0, 1, where=mask, color=colour,
                               alpha=0.85, label=cat, step="mid")
    ax_lf.set_ylim(0, 1)
    ax_lf.set_yticks([])
    ax_lf.set_ylabel("Limit")
    ax_lf.set_xlabel("Time (s)")
    ax_lf.legend(loc="upper center", fontsize=7, ncol=len(_LF_COLORS),
                 bbox_to_anchor=(0.5, -0.3))

    fig.tight_layout()
    fig.savefig(_safe_filename(result.event_name, "telemetry"), dpi=150,
                bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 2. Track map with speed overlay (Sprint)
# ---------------------------------------------------------------------------

def plot_track_speed(result: EventResult, show: bool = False) -> plt.Figure:
    """Colour-coded track map: colour = speed."""
    states = result.states
    x = np.array([s.x for s in states])
    y = np.array([s.y for s in states])
    v = np.array([s.v for s in states]) * 3.6   # km/h

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
# 3. Endurance — lap time bar chart
# ---------------------------------------------------------------------------

def plot_endurance_laptimes(result: EventResult, show: bool = False) -> plt.Figure:
    """Bar chart of lap times for the endurance event."""
    lap_times = result.lap_times
    if not lap_times:
        return None

    n_laps = len(lap_times)
    laps = np.arange(1, n_laps + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(laps, lap_times, color="steelblue", edgecolor="white",
                  linewidth=0.5)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(f"Endurance — Lap Times  (Total: {sum(lap_times):.1f} s)")
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight slowest / fastest
    if n_laps > 1:
        i_fast = int(np.argmin(lap_times))
        i_slow = int(np.argmax(lap_times))
        bars[i_fast].set_color("#2ecc71")
        bars[i_slow].set_color("#e74c3c")

    fig.tight_layout()
    fig.savefig(_safe_filename(result.event_name, "laptimes"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 4. Endurance — battery SOC & temperature vs laps
# ---------------------------------------------------------------------------

def plot_endurance_battery(result: EventResult, show: bool = False) -> plt.Figure:
    """Two-panel plot of battery SOC and temperature vs completed laps."""
    lap_socs  = result.lap_socs
    lap_temps = result.lap_temps
    if not lap_socs:
        return None

    n_laps = len(lap_socs)
    laps = np.arange(1, n_laps + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(laps, np.array(lap_socs) * 100.0, color="seagreen",
             marker="o", markersize=2, linewidth=1.2)
    ax1.set_ylabel("State of Charge (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Endurance — Battery State vs Laps")
    ax1.grid(True, alpha=0.3)

    ax2.plot(laps, lap_temps, color="tomato",
             marker="o", markersize=2, linewidth=1.2)
    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Battery Temperature (°C)")
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
    """Generate and save all plots for an event result (dispatches by event name)."""
    name = result.event_name.lower()

    if "accel" in name:
        plot_telemetry(result, show=show)

    elif "skidpad" in name:
        pass   # no plots — time only

    elif "sprint" in name:
        plot_telemetry(result, show=show)
        plot_track_speed(result, show=show)

    elif "endurance" in name:
        plot_endurance_laptimes(result, show=show)
        plot_endurance_battery(result, show=show)

    else:
        plot_telemetry(result, show=show)

    print(f"  Plots saved to '{_OUTPUT_DIR}/'")
