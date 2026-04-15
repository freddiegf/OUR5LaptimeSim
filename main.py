"""
main.py
=======
CLI entry point for the OUR Formula Student Laptime Simulation.

Usage
-----
    python main.py [--car CONFIG] [--events EVENT [EVENT ...]] [--show-plots]

Events
------
    accel      75 m acceleration dash
    skidpad    Figure-8 skidpad (4 circles, 2 timed)
    sprint     One lap of the FS autocross track
    endurance  22 km of laps with continuous battery tracking

Example
-------
    python main.py --events accel skidpad sprint endurance
    python main.py --car config/car_default.yaml --events sprint --show-plots
    python3 main.py --compare config/comparison_sweep_example.yaml --events endurance
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import numpy as np

# --- Vehicle subsystems ---
from sim.vehicle.car_params import load_car_params
from sim.vehicle.tyre_model import TyreModel
from sim.vehicle.aero import AeroModel
from sim.vehicle.powertrain import Powertrain
from sim.vehicle.battery import BatteryModel

# --- Track ---
from sim.track.track_builder import (
    DEFAULT_SPRINT_LAYOUT,
    SPRINT_LAYOUTS,
    build_sprint_track,
    build_track,
    load_track_from_yaml,
)

# --- GGV & solver ---
from sim.ggv.ggv_builder import GGVBuilder
from sim.solver.lap_solver import LapSolver

# --- Events ---
from sim.events.acceleration_event import AccelerationEvent
from sim.events.skidpad_event import SkidpadEvent
from sim.events.sprint_event import SprintEvent
from sim.events.endurance_event import EnduranceEvent
from sim.events.base_event import EventResult

# --- Analysis ---
from analysis.plotter import generate_all_plots, plot_ggv_surface
from analysis.reporter import print_event_summary, print_all_results


# ---------------------------------------------------------------------------
# GGV grid parameters
# ---------------------------------------------------------------------------



_V_MIN   =  1.0    # m/s
_V_MAX   = 42.0    # m/s  (~150 km/h — generous upper bound for FS)
_V_STEPS = 200
_AY_MAX  = 32.0    # m/s² (just over 3 g)
_AY_STEPS = 400     # symmetric: -32 to +32


def build_ggv(car, tyre_f, tyre_r, aero, pt) -> GGVBuilder:
    print("  Building GGV surface…", end=" ", flush=True)
    t0  = time.perf_counter()
    ggv = GGVBuilder(car, tyre_f, tyre_r, aero, pt)
    ggv.build(
        v_range  = np.linspace(_V_MIN, _V_MAX, _V_STEPS),
        ay_range = np.linspace(-_AY_MAX, _AY_MAX, _AY_STEPS),
        n_iter   = 8,
    )
    print(f"done ({time.perf_counter()-t0:.1f} s)")
    return ggv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="OUR Formula Student Laptime Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--car", default="config/car_default.yaml",
        help="Path to car parameter YAML (default: config/car_default.yaml)",
    )
    parser.add_argument(
        "--events", nargs="+",
        choices=["accel", "skidpad", "sprint", "endurance"],
        default=["accel", "skidpad", "sprint", "endurance"],
        help="Events to simulate",
    )
    parser.add_argument(
        "--show-plots", action="store_true",
        help="Display plots interactively (in addition to saving them)",
    )
    parser.add_argument(
        "--ds", type=float, default=0.1,
        help="Track spatial resolution in metres (default: 0.1)",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Path to comparison YAML file (runs multi-car comparison mode)",
    )
    parser.add_argument(
        "--track-layout", choices=SPRINT_LAYOUTS, default=DEFAULT_SPRINT_LAYOUT,
        help=(
            "Sprint/endurance track layout. "
            "'fsuk2025' uses FSUK2025.svg scaled to 1 km/lap (default); "
            "'synthetic' uses config/track_sprint.yaml."
        ),
    )
    args = parser.parse_args(argv)

    # --- Comparison mode ---
    if args.compare:
        from analysis.comparison import run_comparison
        run_comparison(args.compare, args.events, ds=args.ds,
                       show_plots=args.show_plots,
                       track_layout=args.track_layout)
        return

    print()
    print("=" * 60)
    print("  OUR FS Laptime Simulation")
    print("=" * 60)
    print(f"  Car config   : {args.car}")
    print(f"  Events       : {', '.join(args.events)}")
    print(f"  Track ds     : {args.ds} m")
    print(f"  Track layout : {args.track_layout}  (sprint/endurance)")
    print()

    # --- Load car parameters ---
    print("  Loading car parameters…")
    car    = load_car_params(args.car)
    tyre_f = TyreModel(car.front_tyre)
    tyre_r = TyreModel(car.rear_tyre)
    aero   = AeroModel(car.aero, car)
    pt     = Powertrain(car.powertrain)
    bat    = BatteryModel(car.battery)

    print(f"    Mass       : {car.mass:.0f} kg")
    print(f"    Wheelbase  : {car.wheelbase:.3f} m")
    print(f"    CoG height : {car.cog_z:.3f} m")
    print(f"    Drivetrain : {car.powertrain.drivetrain_type}")
    print(f"    Battery    : {car.battery.pack_capacity_kWh:.1f} kWh @ "
          f"{car.battery.pack_V_nominal:.0f} V")
    print()

    # --- Build GGV surface (shared across all events) ---
    ggv    = build_ggv(car, tyre_f, tyre_r, aero, pt)
    solver = LapSolver(ggv, car, aero, pt, bat)

    if args.show_plots:
        plot_ggv_surface(ggv, show=args.show_plots)

    # --- Run events ---
    results: List[EventResult] = []

    if "accel" in args.events:
        print("\n  [Acceleration Event]")
        segs  = load_track_from_yaml("config/track_acceleration.yaml")
        track = build_track(segs, ds=args.ds)
        print(f"    Track length : {track.total_length:.1f} m")
        t0     = time.perf_counter()
        result = AccelerationEvent(solver, track, bat).run()
        print(f"    Solved in {time.perf_counter()-t0:.2f} s")
        print_event_summary(result)
        generate_all_plots(result, ggv=ggv, show=args.show_plots)
        results.append(result)

    if "skidpad" in args.events:
        print("\n  [Skidpad Event]")
        segs  = load_track_from_yaml("config/track_skidpad.yaml")
        track = build_track(segs, ds=args.ds)
        print(f"    Track length : {track.total_length:.1f} m")
        t0     = time.perf_counter()
        result = SkidpadEvent(solver, track, bat).run()
        print(f"    Solved in {time.perf_counter()-t0:.2f} s")
        print_event_summary(result)
        generate_all_plots(result, ggv=ggv, show=args.show_plots)
        results.append(result)

    if "sprint" in args.events:
        print("\n  [Sprint Event]")
        track = build_sprint_track(args.track_layout, ds=args.ds)
        print(f"    Track layout : {args.track_layout}")
        print(f"    Track length : {track.total_length:.1f} m")
        t0     = time.perf_counter()
        result = SprintEvent(solver, track, bat).run()
        print(f"    Solved in {time.perf_counter()-t0:.2f} s")
        print_event_summary(result)
        generate_all_plots(result, ggv=ggv, show=args.show_plots)
        results.append(result)

    if "endurance" in args.events:
        print("\n  [Endurance Event]  (this may take a moment…)")
        track = build_sprint_track(args.track_layout, ds=args.ds)
        print(f"    Track layout : {args.track_layout}")
        print(f"    Track length : {track.total_length:.1f} m/lap")
        t0     = time.perf_counter()
        result = EnduranceEvent(solver, track, bat).run()
        print(f"    Solved in {time.perf_counter()-t0:.2f} s")
        print_event_summary(result)
        generate_all_plots(result, ggv=ggv, show=args.show_plots)
        results.append(result)

    # --- Overall summary ---
    if len(results) > 1:
        print_all_results(results)


if __name__ == "__main__":
    main()
