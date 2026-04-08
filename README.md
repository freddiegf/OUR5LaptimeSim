# OUR5 Laptime Simulation

Quasi-static laptime simulation for the OUR5 Formula Student car. Predicts lap times, battery state, and thermal behaviour across all four FS dynamic events.

## Quick Start

```bash
pip install -r requirements.txt

# Run all four events with the default car
python3 main.py

# Run specific events
python3 main.py --events accel skidpad sprint endurance

# Compare different car setups
python3 main.py --compare config/comparison_example.yaml
```

Plots and figures are saved to `output/`.

## Events

| Event | Description | Track |
|-------|-------------|-------|
| **Acceleration** | 75 m standing-start drag | Straight |
| **Skidpad** | Figure-8, two circles per direction | r = 9.125 m centreline, 3 m wide driving path |
| **Sprint** | Single lap of an autocross circuit | 891 m, 26 corners |
| **Endurance** | Laps until 22 km with continuous battery | Same as sprint, battery persists across laps |

Skidpad time is the **average** of one left and one right timed circle, per FS rules.

## Car Configuration

All car parameters are defined in a single YAML file (default: `config/car_default.yaml`):

| Category | Key Parameters |
|----------|---------------|
| **Vehicle** | `mass`, `wheelbase`, `cog_x`, `cog_z`, track widths |
| **Tyres** | `mu_x_nominal`, `mu_y_nominal`, load sensitivity, rolling radius |
| **Aero** | `Cl`, `Cd`, `area`, centre of pressure |
| **Powertrain** | `torque_curve`, `gear_ratio`, `drivetrain_efficiency`, `drivetrain_type`, `power_limit_kW` |
| **Battery** | `cell_capacity_Ah`, `cell_V_nominal`, `cell_R_int_Ohm`, `n_series`, `n_parallel` |

Pack-level quantities (voltage, resistance, capacity, thermal mass) are derived automatically from cell parameters and series/parallel config.

## Car Comparison Mode

Compare different car setups side-by-side across all events:

```bash
python3 main.py --compare config/comparison_example.yaml
```

Create a comparison YAML with a base config and named variants:

```yaml
base: config/car_default.yaml

variants:
  - label: "Baseline"
    overrides: {}

  - label: "Lightweight / Low Power"
    overrides:
      mass: 290
      powertrain:
        power_limit_kW: 30.0
      battery:
        n_parallel: 5

  - label: "High Aero"
    overrides:
      aero:
        Cl: 3.0
        Cd: 1.5
```

Any parameter from the car YAML can be overridden — only specify what differs from the base. The comparison produces:

- **Console table** — event times, SOC, battery temperature for each variant
- **Bar chart** — event times side-by-side (`output/comparison_event_times.png`)
- **Endurance battery overlay** — SOC and temperature vs laps (`output/comparison_endurance_battery.png`)

Battery depletion during endurance is clearly marked in both the table and plots.

## CLI Reference

```
python3 main.py [OPTIONS]

Options:
  --car PATH          Car parameter YAML (default: config/car_default.yaml)
  --events EVENT ...  Events to run: accel, skidpad, sprint, endurance
  --compare PATH      Comparison YAML file (multi-car comparison mode)
  --ds FLOAT          Track spatial resolution in metres (default: 0.1)
  --show-plots        Display plots interactively
```

## How It Works

```
Car YAML --> CarParams --> TyreModel, AeroModel, Powertrain, BatteryModel
                               |
Track YAML --> TrackProfile    |
                    |          v
                    +---> GGV Surface (v, ay) --> (ax_max, ax_min)
                    |          |
                    v          v
                    LapSolver.solve() --> [VehicleState]
                                               |
                                               v
                                          EventResult --> Plots + Tables
```

### GGV Surface

The GGV (g-g-v) envelope maps the car's performance: at any speed `v` and lateral acceleration `ay`, what are the maximum and minimum longitudinal accelerations? Built using:

- **4-wheel load transfer** — normal loads from mass distribution, aero downforce, and longitudinal/lateral acceleration
- **Traction ellipse** — couples longitudinal and lateral grip per wheel with load-dependent friction
- **Power limits** — motor torque curve and FS rules power ceiling

### Solver

Forward-backward quasi-static lap solver:

1. **Corner speed caps** — binary search at each track point to find the maximum speed where lateral grip is sufficient
2. **Forward pass** — accelerate from start, respecting GGV limits, power cap, and corner speed ceilings
3. **Backward pass** — brake backwards from finish, respecting GGV deceleration limits and corner caps
4. **Combine** — element-wise minimum of both passes gives the optimal speed profile
5. **Build states** — walk the speed profile computing forces, motor state, battery draw, and timing

### Battery Model

- Per-cell OCV lookup table (Molicel P42A 21700)
- Quadratic current solver: `R*I^2 - OCV*I + P = 0`
- Ohmic heating: `Q = I^2 * R_pack * dt` (no cooling model)
- SOC depletion tracked per step; battery state persists across endurance laps
- Maximum deliverable power `P_max = OCV^2 / (4*R_pack)` limits the car when SOC is low

## Project Structure

```
OUR5LaptimeSim/
  main.py                       # CLI entry point
  config/
    car_default.yaml            # Default car parameters
    comparison_example.yaml     # Example comparison config
    track_acceleration.yaml     # 75 m straight
    track_skidpad.yaml          # Figure-8 (r=9.125 m)
    track_sprint.yaml           # 891 m autocross circuit
  sim/
    vehicle/
      car_params.py             # Parameter dataclasses + YAML loader
      tyre_model.py             # Load-dependent friction, traction ellipse
      aero.py                   # Downforce and drag
      powertrain.py             # Motor torque curve + drivetrain
      battery.py                # Electrical + thermal battery model
    ggv/
      ggv_builder.py            # GGV performance envelope
    solver/
      lap_solver.py             # Forward-backward quasi-static solver
      vehicle_state.py          # Per-step state container
    track/
      track_builder.py          # Discretise track segments
      track_segment.py          # Segment dataclass
    events/
      base_event.py             # EventResult + abstract base class
      acceleration_event.py     # 75 m dash
      skidpad_event.py          # Figure-8 with timed circle extraction
      sprint_event.py           # Single lap
      endurance_event.py        # Multi-lap with persistent battery
  analysis/
    plotter.py                  # Per-event matplotlib figures
    comparison.py               # Multi-car comparison pipeline
    reporter.py                 # Console summary tables
  output/                       # Generated plots (git-ignored)
```

## Known Limitations

- No cooling model — battery temperature only rises
- No regenerative braking
- Lateral force distribution assumes equal slip angles (no front/rear balance)
- No tyre temperature model
- Steady-state (quasi-static) — no transient dynamics or weight transfer lag
