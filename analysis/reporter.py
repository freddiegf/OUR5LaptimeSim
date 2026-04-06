"""
reporter.py
===========
Console summary tables for event results.
"""

from __future__ import annotations

import textwrap
from typing import List

from sim.events.base_event import EventResult


_SEP = "─" * 52


def print_event_summary(result: EventResult) -> None:
    """Print a formatted summary of a single event result."""
    print()
    print(_SEP)
    print(f"  {result.event_name.upper()}")
    print(_SEP)
    print(f"  Total time       : {result.total_time:.3f} s")
    print(f"  Total distance   : {result.total_distance_m/1000:.3f} km")
    print(f"  Peak speed       : {result.peak_speed_kph:.1f} km/h")
    print(f"  Peak long. accel : {result.peak_ax_g:+.2f} g")
    print(f"  Peak lat. accel  : {result.peak_ay_g:.2f} g")
    print(f"  Energy used      : {result.energy_used_kWh:.4f} kWh")
    print(f"  Final SOC        : {result.final_SOC*100:.1f} %")
    print(f"  Final batt. temp : {result.final_battery_temp:.1f} °C")
    print(_SEP)
    print()


def print_all_results(results: List[EventResult]) -> None:
    """Print a combined summary table for multiple events."""
    print()
    print("=" * 70)
    print("  FORMULA STUDENT SIMULATION — RESULTS SUMMARY")
    print("=" * 70)
    header = f"  {'Event':<16} {'Time (s)':>10} {'Dist (km)':>10} {'Vmax (kph)':>11} {'SOC (%)':>8} {'ΔT (°C)':>8}"
    print(header)
    print("-" * 70)
    for r in results:
        dT = r.final_battery_temp - r.states[0].battery_temp
        print(f"  {r.event_name:<16} {r.total_time:>10.3f} "
              f"{r.total_distance_m/1000:>10.3f} "
              f"{r.peak_speed_kph:>11.1f} "
              f"{r.final_SOC*100:>8.1f} "
              f"{dT:>+8.1f}")
    print("=" * 70)
    print()
