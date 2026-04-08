"""
comparison.py
=============
Multi-car comparison pipeline: load a comparison YAML, run all variants
across all events, and produce comparison tables and plots.

Usage (from main.py):
    python main.py --compare config/comparison_example.yaml
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import yaml

from sim.vehicle.car_params import CarParams, build_car_params
from sim.vehicle.tyre_model import TyreModel
from sim.vehicle.aero import AeroModel
from sim.vehicle.powertrain import Powertrain
from sim.vehicle.battery import BatteryModel
from sim.ggv.ggv_builder import GGVBuilder
from sim.solver.lap_solver import LapSolver
from sim.track.track_builder import build_track, load_track_from_yaml
from sim.events.acceleration_event import AccelerationEvent
from sim.events.skidpad_event import SkidpadEvent
from sim.events.sprint_event import SprintEvent
from sim.events.endurance_event import EnduranceEvent
from sim.events.base_event import EventResult


_OUTPUT_DIR = "output"

# GGV grid parameters (same as main.py)
_V_MIN    = 1.0
_V_MAX    = 42.0
_V_STEPS  = 200
_AY_MAX   = 32.0
_AY_STEPS = 400


@dataclass
class VariantResult:
    """Results for one car variant across all events."""
    label: str
    car: CarParams
    results: Dict[str, EventResult]
    battery_depleted_lap: Optional[int] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a deep copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _ensure_output_dir() -> str:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR


def _build_ggv(car, tyre_f, tyre_r, aero, pt) -> GGVBuilder:
    ggv = GGVBuilder(car, tyre_f, tyre_r, aero, pt)
    ggv.build(
        v_range=np.linspace(_V_MIN, _V_MAX, _V_STEPS),
        ay_range=np.linspace(-_AY_MAX, _AY_MAX, _AY_STEPS),
        n_iter=8,
    )
    return ggv


# ---------------------------------------------------------------------------
# Track helpers — map event name to track YAML
# ---------------------------------------------------------------------------

_EVENT_TRACKS = {
    "accel":     "config/track_acceleration.yaml",
    "skidpad":   "config/track_skidpad.yaml",
    "sprint":    "config/track_sprint.yaml",
    "endurance": "config/track_sprint.yaml",
}

_EVENT_CLASSES = {
    "accel":     AccelerationEvent,
    "skidpad":   SkidpadEvent,
    "sprint":    SprintEvent,
    "endurance": EnduranceEvent,
}


def _run_event(event_name: str, solver, track, battery) -> EventResult:
    """Run a single event and return its result."""
    cls = _EVENT_CLASSES[event_name]
    event = cls(solver, track, battery)
    return event.run()


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    variant_results: List[VariantResult],
    events: List[str],
) -> None:
    """Print a side-by-side comparison table to the console."""
    max_label = max(len(vr.label) for vr in variant_results)
    max_label = max(max_label, 7)  # minimum width for "Variant"

    print()
    print("=" * 78)
    print("  CAR COMPARISON — RESULTS SUMMARY")
    print("=" * 78)

    # --- Time comparison across events ---
    # Build header
    col_w = 12
    header = f"  {'Variant':<{max_label}}"
    for ev in events:
        header += f"  {ev.capitalize() + ' (s)':>{col_w}}"
    print(header)
    print("  " + "-" * (max_label + (col_w + 2) * len(events)))

    for vr in variant_results:
        row = f"  {vr.label:<{max_label}}"
        for ev in events:
            r = vr.results.get(ev)
            if r:
                row += f"  {r.total_time:>{col_w}.3f}"
            else:
                row += f"  {'—':>{col_w}}"
        # Mark battery depletion
        if vr.battery_depleted_lap is not None:
            row += f"  BATTERY DEPLETED lap {vr.battery_depleted_lap}"
        print(row)

    # --- Endurance detail (if endurance was run) ---
    if "endurance" in events:
        print()
        print("  Endurance Detail")
        header2 = (f"  {'Variant':<{max_label}}"
                   f"  {'Laps':>6}"
                   f"  {'SOC (%)':>8}"
                   f"  {'T_bat (C)':>10}")
        print(header2)
        print("  " + "-" * (max_label + 28))
        for vr in variant_results:
            r = vr.results.get("endurance")
            if not r:
                continue
            n_laps = len(r.lap_times)
            soc_pct = r.final_SOC * 100.0
            t_bat = r.final_battery_temp
            row = (f"  {vr.label:<{max_label}}"
                   f"  {n_laps:>6}"
                   f"  {soc_pct:>8.1f}"
                   f"  {t_bat:>10.1f}")
            if vr.battery_depleted_lap is not None:
                row += "   DEPLETED"
            print(row)

    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Comparison bar chart — event times
# ---------------------------------------------------------------------------

def plot_comparison_bars(
    variant_results: List[VariantResult],
    events: List[str],
    show: bool = False,
) -> plt.Figure:
    """One subplot per event, bars per variant."""
    n_events = len(events)
    n_variants = len(variant_results)
    colours = cm.get_cmap("Set2", max(n_variants, 3))

    fig, axes = plt.subplots(1, n_events, figsize=(4 * n_events, 5))
    if n_events == 1:
        axes = [axes]

    for ax, ev in zip(axes, events):
        times = []
        labels = []
        for vr in variant_results:
            r = vr.results.get(ev)
            times.append(r.total_time if r else 0.0)
            labels.append(vr.label)

        bars = ax.bar(range(n_variants), times,
                      color=[colours(i) for i in range(n_variants)],
                      edgecolor="white", linewidth=0.5)
        ax.set_title(ev.capitalize(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Time (s)")
        ax.set_xticks(range(n_variants))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate bar values
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{t:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Car Comparison — Event Times", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(_ensure_output_dir(), "comparison_event_times.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Comparison endurance battery overlay
# ---------------------------------------------------------------------------

def plot_comparison_endurance_battery(
    variant_results: List[VariantResult],
    show: bool = False,
) -> Optional[plt.Figure]:
    """Overlaid SOC and temperature vs laps for all variants."""
    # Filter to variants that have endurance results
    endurance_vrs = [vr for vr in variant_results
                     if "endurance" in vr.results and vr.results["endurance"].lap_socs]
    if not endurance_vrs:
        return None

    n = len(endurance_vrs)
    colours = cm.get_cmap("Set2", max(n, 3))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for i, vr in enumerate(endurance_vrs):
        r = vr.results["endurance"]
        laps = np.arange(1, len(r.lap_socs) + 1)
        c = colours(i)

        ax1.plot(laps, np.array(r.lap_socs) * 100.0, color=c,
                 label=vr.label, marker="o", markersize=2, linewidth=1.2)
        ax2.plot(laps, r.lap_temps, color=c,
                 label=vr.label, marker="o", markersize=2, linewidth=1.2)

        # Mark battery depletion
        if vr.battery_depleted_lap is not None:
            lap_d = vr.battery_depleted_lap
            ax1.axvline(lap_d, color=c, linestyle="--", alpha=0.7, linewidth=1)
            ax2.axvline(lap_d, color=c, linestyle="--", alpha=0.7, linewidth=1)
            ax1.annotate(f"depleted", xy=(lap_d, 5),
                         fontsize=6, color=c, ha="center",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax1.set_ylabel("State of Charge (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Endurance — Battery Comparison", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Battery Temperature (\u00b0C)")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(_ensure_output_dir(), "comparison_endurance_battery.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Main comparison entry point
# ---------------------------------------------------------------------------

def run_comparison(
    config_path: str,
    events: List[str],
    ds: float = 0.1,
    show_plots: bool = False,
) -> List[VariantResult]:
    """Run the full comparison pipeline."""
    # --- Load comparison config ---
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    base_yaml_path = config["base"]
    with open(base_yaml_path, "r") as fh:
        base_raw = yaml.safe_load(fh)

    variant_defs = config["variants"]
    if "events" in config:
        events = config["events"]

    print()
    print("=" * 60)
    print("  OUR FS Laptime Simulation — COMPARISON MODE")
    print("=" * 60)
    print(f"  Base config  : {base_yaml_path}")
    print(f"  Variants     : {len(variant_defs)}")
    print(f"  Events       : {', '.join(events)}")
    print()

    # --- Load tracks once (shared across all variants) ---
    tracks = {}
    for ev in events:
        track_yaml = _EVENT_TRACKS[ev]
        if track_yaml not in tracks:
            segs = load_track_from_yaml(track_yaml)
            tracks[track_yaml] = build_track(segs, ds=ds)

    # --- Run each variant ---
    variant_results: List[VariantResult] = []

    for vi, vdef in enumerate(variant_defs):
        label = vdef["label"]
        overrides = vdef.get("overrides", {})

        print(f"\n  [{vi+1}/{len(variant_defs)}] {label}")
        print(f"  " + "-" * 50)

        # Merge overrides into base config
        merged_raw = _deep_merge(base_raw, overrides)
        car = build_car_params(merged_raw)

        # Build subsystems
        tyre_f = TyreModel(car.front_tyre)
        tyre_r = TyreModel(car.rear_tyre)
        aero   = AeroModel(car.aero, car)
        pt     = Powertrain(car.powertrain)
        bat    = BatteryModel(car.battery)

        print(f"    Mass: {car.mass:.0f} kg | "
              f"Power: {car.powertrain.power_limit_kW:.0f} kW | "
              f"Battery: {car.battery.n_series}s{car.battery.n_parallel}p "
              f"({car.battery.pack_capacity_kWh:.1f} kWh)")

        # Build GGV
        print(f"    Building GGV...", end=" ", flush=True)
        t0 = time.perf_counter()
        ggv = _build_ggv(car, tyre_f, tyre_r, aero, pt)
        print(f"done ({time.perf_counter()-t0:.1f} s)")

        solver = LapSolver(ggv, car, aero, pt, bat)

        # Run events
        event_results: Dict[str, EventResult] = {}
        for ev in events:
            track_yaml = _EVENT_TRACKS[ev]
            track = tracks[track_yaml]
            print(f"    Running {ev}...", end=" ", flush=True)
            t0 = time.perf_counter()
            result = _run_event(ev, solver, track, bat)
            print(f"{result.total_time:.3f} s ({time.perf_counter()-t0:.1f} s)")
            event_results[ev] = result

        # Detect battery depletion in endurance
        battery_depleted_lap = None
        if "endurance" in event_results:
            endurance = event_results["endurance"]
            for i, soc in enumerate(endurance.lap_socs):
                if soc <= 0.01:
                    battery_depleted_lap = i + 1
                    break

        variant_results.append(VariantResult(
            label=label,
            car=car,
            results=event_results,
            battery_depleted_lap=battery_depleted_lap,
        ))

        # Free heavy telemetry data to save memory
        for ev_name, r in event_results.items():
            r.states = []

    # --- Output ---
    print_comparison_table(variant_results, events)
    plot_comparison_bars(variant_results, events, show=show_plots)
    if "endurance" in events:
        plot_comparison_endurance_battery(variant_results, show=show_plots)

    return variant_results
