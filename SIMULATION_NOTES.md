# OUR5 Laptime Simulation — Notes & Results

## Bugs Fixed

### 1. Battery capacity unit error (`car_params.py`)

**Problem:** `pack_capacity_kWh` divided by 3600 instead of 1000 when converting Wh → kWh:
```python
self.pack_capacity_kWh = (pack_Ah * self.pack_V_nominal) / 3600.0  # WRONG
```
This made the capacity 3.6× too small (2.1 kWh instead of 7.6 kWh). Consequences:
- SOC depleted 3.6× too fast
- OCV dropped unrealistically → current spiked → I²R heating cascaded
- Battery thermal model appeared over-sensitive to parameters

**Fix:** `/ 3600.0` → `/ 1000.0`. Pack capacity is now correctly 7.62 kWh.

### 2. Corner speed caps not propagated through solver passes (`lap_solver.py`)

**Problem:** The binary-search corner speed cap was applied as post-processing in `_combine`, after the forward-backward passes had already completed. This created speed discontinuities at corner entry/exit that implied 20-150g of deceleration (GGV limit: ~1.5g). The root cause: both passes clip `ay` to 32 m/s² (grid max), but actual tyre limit is ~17 m/s², so neither pass correctly determined corner speeds on its own.

**Fix:** Corner caps are now pre-computed and enforced as hard ceilings inside both `_forward_pass` and `_backward_pass`. The backward pass correctly propagates braking zones from each corner, and the forward pass can't overshoot into corners.

### 3. Lateral load transfer normalization (`ggv_builder.py`)

**Problem:** Roll stiffness fractions used `m*g` (static weight) as denominator, but axle loads include aero downforce. Fractions summed to >1.0, overestimating lateral load transfer by ~12% at speed.

**Fix:** Normalized by `Fz_front + Fz_rear` (actual total vertical force).

### 4. Force reporting in `_build_states` (`lap_solver.py`)

**Problem:** `drive_force` and `brake_force` in VehicleState were set to maximum available capacity, not the actual demand matching `m*ax`. Post-processing force/energy analysis was inconsistent.

**Fix:** Forces computed from Newton's second law: `F = m*ax + drag`. Per-wheel Fx scaled proportionally to match actual demand.

### 5. Event boundary conditions

**Problem:** Sprint and endurance events forced the car to stop at the finish (v_final=0). In FS, the car crosses the finish line at speed.

**Fix:** Sprint uses v_final=1000 (unconstrained exit). Endurance uses v_final=1000 for all laps.

### 6. Skidpad track realism

**Problem:** Four circles back-to-back with no transition. No approach/exit straights.

**Fix:** Added approach straight (15m), crossing straight (3.5m), and exit straight (15m). Rolling start/finish at 10 m/s. Event code updated to extract timed portions from segment boundaries.

### 7. Sprint track realism

**Problem:** Track was a symmetric oval with only two corner radii (6m/8m chicanes + 10m slalom + 15m hairpins).

**Fix:** Redesigned as asymmetric ~637m layout with varied features: tight hairpin (r=5m), medium corners (r=7-10m), fast sweeper (r=20m), chicanes, slalom section, and two long straights.

### 8. Battery power not coupled to solver

**Problem:** The GGV pre-bakes the FS rules power limit (30 kW) but doesn't know about battery state. As SOC falls, the battery's maximum deliverable power `P_max = V_oc² / (4 × R_pack)` can drop below 30 kW, but the solver kept demanding full power regardless.

**Fix:** Added `BatteryModel.max_power()` method. All events now compute `min(rules_limit, battery_max_power)` and pass it to the solver via a new `power_limit_W` parameter. The forward pass caps acceleration based on the effective power limit. In endurance, the power limit is recomputed before each lap as SOC depletes.

### 9. Sprint/endurance track too many long straights

**Problem:** Previous 637m track had straights of 50-75m, totalling ~480m of straight (75% of track). Real FS autocross tracks have max straights ~55m and 20-30 corner segments.

**Fix:** Redesigned as ~891m layout with 26 corner segments: 2 hairpins (r=5m), 3 chicanes, 2 slaloms, 4 sweepers (r=20-35m), and 4 medium corners (r=8-15m). Max straight 55m. Dense direction changes throughout.

---

## Current Baseline Results (`car_default.yaml`)

| Event      | Time     | Peak Speed | Peak ax  | Peak ay  | Final SOC | ΔT Battery |
|------------|----------|------------|----------|----------|-----------|------------|
| Accel      | 5.29 s   | 77.0 km/h  | +1.14 g  | 0.00 g   | 99.7%     | +0.1°C     |
| Skidpad    | 8.83 s   | 52.2 km/h  | +0.56 g  | 1.58 g   | 99.8%     | +0.0°C     |
| Sprint     | 57.86 s  | 86.4 km/h  | +1.14 g  | 1.62 g   | 97.9%     | +0.4°C     |
| Endurance  | 1414 s   | 86.4 km/h  | +1.14 g  | 1.62 g   | 48.7%     | +12.4°C    |

---

## Skidpad Time Interpretation

The reported 8.63 s is the **sum of one left timed lap + one right timed lap**. Per circle: ~4.32 s.

Theoretical check: `v = sqrt(μy × g × r) = sqrt(1.60 × 9.81 × 7.625) = 10.94 m/s` → period = `2π × 7.625 / 10.94 = 4.38 s/circle`. Sim gives 4.32 s (slightly faster due to aero downforce). Physically correct.

---

### 10. Realistic OCV curve

**Problem:** Linear OCV model (`V_oc = V_nom × SOC`) vastly underestimates voltage at low SOC — at SOC=0.5, V_cell=1.8V vs real ~3.7V. This caused unrealistic current spikes and heating at moderate SOC.

**Fix:** Replaced with per-cell interpolation table from Molicel P42A datasheet: (0%, 2.5V), (20%, 3.2V), (40%, 3.6V), (60%, 3.8V), (80%, 3.95V), (100%, 4.2V). Pack voltage = cell_V × n_series.

### 11. Power limit as terminal power

**Problem:** The 20 kW FS rules power limit was treated as mechanical wheel power. The actual limit is measured at the battery terminals (V_terminal × I). Drivetrain efficiency η was not applied when converting to wheel force.

**Fix:** Power limit force at the wheel is now `F = P_terminal × η / v` in both the GGV builder and the forward pass. Battery P_demand is now terminal power = P_wheel / η. The `power_demand` field in VehicleState is now terminal electrical power, and `energy_used_kWh` is battery electrical energy.

### 12. Limiting factor tracking

**Problem:** No visibility into what constrains the car at each point on track.

**Fix:** Added `limiting_factor` field to VehicleState. At each step, the solver computes all four wheel-level force limits (tyre grip, motor torque, rules power, battery power) and reports which is tightest. Braking phases report "Braking". New telemetry plots include a colour-coded strip showing the active constraint.

---

## Known Limitations

- **Endurance battery temperature** now realistic at 32.4°C
  - Still no cooling model — temperature only rises
- No regenerative braking modelled
- Lateral force distribution assumes equal slip angles (no front/rear balance)
- No tyre temperature model

---

## Architecture

```
YAML → CarParams → {TyreModel, AeroModel, Powertrain, BatteryModel}
Track YAML → TrackProfile (s, κ, x, y)
CarParams + Tyres + Aero + Powertrain → GGVBuilder.build() → GGV surface
GGV + BatteryModel → LapSolver.solve(TrackProfile) → [VehicleState]
[VehicleState] → EventResult → Plotter + Reporter
```

Solver algorithm:
1. Pre-compute corner speed caps (binary search on GGV lateral limit)
2. Forward pass with caps (accelerate within GGV + cap ceiling)
3. Backward pass with caps (brake within GGV + cap ceiling)
4. Combine (element-wise minimum)
5. Build VehicleState list (forces, battery, timing)

GGV build: ~0.2 s. Full 4-event run: ~60 s (endurance: ~35 laps × solve).
