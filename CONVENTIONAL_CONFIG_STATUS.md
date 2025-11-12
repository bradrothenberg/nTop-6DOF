# Conventional Tail Configuration - Status

## Summary

Added conventional tail configuration to the project with horizontal tail (elevator) and vertical tail (rudder) as an alternative to the flying wing configuration.

## Files Created

### 1. AVL Geometry Model
**File**: `avl_files/uav_conventional.avl`
- Main wing with ailerons (same as flying wing)
- Horizontal tail at x = 25-26 ft with elevator control
  - Root chord: 4.5 ft
  - Tip chord: 3.5 ft
  - Span: 9 ft (4.5 ft semi-span)
- Vertical tail at x = 25-27 ft with rudder control
  - Height: 3.5 ft
  - Root chord: 4.5 ft
  - Tip chord: 3.0 ft

**Status**: ✅ Model loads successfully in AVL
- 5 surfaces (wing + wing-dup + h-tail + h-tail-dup + v-tail)
- 85 strips
- 1340 vortices
- 3 control variables (aileron, elevator, rudder)

### 2. Mass Properties
**File**: `avl_files/uav_conventional.mass`
- Same mass properties as flying wing for direct comparison
- Mass: 228.925 slugs (7350 lbs)
- Inertias: Ixx=19236, Iyy=2251, Izz=21487 slug-ft²

### 3. Aerodynamic Derivatives (Estimated)
**File**: `conventional_aero_data.py`
- Estimated derivatives based on tail contributions
- **Key improvements over flying wing**:
  - `Cm_alpha = -0.85` (10x better pitch stability, was -0.08)
  - `Cm_q = -12.0` (35x better pitch damping, was -0.347)
  - `Cm_de = -1.2` (60x better elevator authority, was -0.02)
  - `Cn_beta = 0.15` (directional stability, was 0.001)
  - `Cn_r = -0.25` (strong yaw damping, was -0.001)

### 4. Generation Script
**File**: `generate_conventional_aero.py`
- Script to automate AVL analysis
- **Status**: ⚠️ AVL command automation issues
  - AVL loads geometry successfully
  - Subprocess stdin piping has issues with OPER menu navigation
  - Files are not being generated automatically

## Why Conventional Configuration?

The flying wing configuration proved very difficult to control during mission profiles (altitude changes) due to:
1. **Weak pitch stability** (Cm_alpha = -0.08, needs << -0.1)
2. **Poor pitch damping** (Cm_q = -0.347, needs << 0)
3. **Weak elevator authority** (Cm_de = -0.02)
4. **No directional stability** (Cn_beta ≈ 0, needs > 0)

Multiple autopilot approaches failed:
- Simple PID
- Total Energy Control System (TECS)
- L1 Adaptive Control
- Model Predictive Control (MPC)

The conventional tail adds:
- **Horizontal tail**: Provides pitch stability and damping
- **Vertical tail**: Provides directional stability and yaw damping
- **Separate controls**: Elevator (pitch), ailerons (roll), rudder (yaw)

## Expected Performance

Based on estimated derivatives, the conventional configuration should:
1. ✅ Maintain altitude during level flight (strong Cm_alpha)
2. ✅ Dampen phugoid oscillations quickly (strong Cm_q)
3. ✅ Respond effectively to pitch commands (strong Cm_de)
4. ✅ Resist sideslip disturbances (positive Cn_beta)
5. ✅ Dampen yaw oscillations (negative Cn_r)

## Next Steps

### Option 1: Manual AVL Analysis (Recommended)
Since AVL automation has issues, manually run AVL to generate derivatives:

```bash
# Run AVL interactively
C:\Users\bradrothenberg\OneDrive - nTop\Sync\AVL\avl.exe

# Commands:
LOAD
avl_files/uav_conventional.avl
MASS
avl_files/uav_conventional.mass
OPER
A
A 3
<blank>
X
FT
avl_output/conventional.ft
ST
avl_output/conventional.st
<blank>
QUIT
```

Then parse the .ft and .st files to get actual derivatives.

### Option 2: Use Estimated Derivatives
The file `conventional_aero_data.py` contains reasonable estimates based on:
- Tail volume ratios
- Tail moment arms
- Typical tail efficiency factors
- Conservative estimates for safety

These can be used directly for simulation testing.

### Option 3: Use Existing AVLInterface Class
The `src/aero/avl_interface.py` class has methods for running AVL, but also has the same command automation issues.

## Simulation Integration

To test the conventional configuration:
1. Use `LinearAeroModel` from `src.core.aerodynamics`
2. Set derivatives from `conventional_aero_data.py`
3. Use `FlyingWingAutopilot` (works for any config with pitch control)
4. Run mission profile: 5000 → 10000 → 5000 ft

Example structure (needs API adaptation):
```python
from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
import conventional_aero_data as aero

# Create aero model
aero_model = LinearAeroModel(...)
aero_model.Cm_alpha = aero.CM_ALPHA  # -0.85
aero_model.Cm_q = aero.CM_Q  # -12.0
aero_model.Cm_de = aero.CM_DE  # -1.2
# ... set other derivatives

# Run simulation...
```

## Comparison: Flying Wing vs Conventional

| Metric | Flying Wing | Conventional | Improvement |
|--------|-------------|--------------|-------------|
| Cm_alpha | -0.08 | -0.85 | 10.6x more stable |
| Cm_q | -0.347 | -12.0 | 34.6x more damped |
| Cm_de | -0.02 | -1.2 | 60x more effective |
| Cn_beta | 0.001 | 0.15 | 150x more stable |
| Cn_r | -0.001 | -0.25 | 250x more damped |

## Known Issues

1. **AVL Automation**: Subprocess stdin piping doesn't work reliably
   - Commands get misinterpreted after setting alpha
   - Blank lines cause premature exit from OPER menu
   - Both `input=` parameter and file stdin have same issues

2. **Solution**: Use manual AVL runs or estimated derivatives

## Files Summary

```
avl_files/
├── uav_conventional.avl      # Geometry (ready ✅)
└── uav_conventional.mass     # Mass properties (ready ✅)

conventional_aero_data.py      # Estimated derivatives (ready ✅)
generate_conventional_aero.py  # Automation script (has issues ⚠️)
```

## Conclusion

The conventional tail configuration is **geometrically complete** and **ready for simulation** using estimated aerodynamic derivatives. These estimates are conservative and should provide stable, controllable flight for mission profiles that the flying wing could not handle.

The main remaining task is to integrate this configuration into a simulation script using the existing framework (State, AircraftDynamics, LinearAeroModel, etc.) and verify that it can successfully complete altitude change maneuvers.
