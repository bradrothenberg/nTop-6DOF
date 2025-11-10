# nTop 6-DOF Flight Dynamics Framework

A production-ready Python framework for 6-degree-of-freedom (6-DOF) flight dynamics simulation, integrated with AVL (Athena Vortex Lattice) and XFOIL for aerodynamic analysis of nTop-designed aircraft.

## Project Status

### Phase 1: AVL Geometry Generation & Validation ✅ (COMPLETED)

### Phase 2: Core 6-DOF Dynamics ✅ (COMPLETED)

**Completed Components:**

1. **Quaternion Mathematics** ([src/core/quaternion.py](src/core/quaternion.py))
   - ✅ Quaternion class with normalization
   - ✅ Euler angle conversions
   - ✅ Rotation matrix generation
   - ✅ Quaternion multiplication and kinematics

2. **State Vector** ([src/core/state.py](src/core/state.py))
   - ✅ Complete 13-state vector (position, velocity, quaternion, angular rates)
   - ✅ Array conversion for integration
   - ✅ Derived properties (altitude, airspeed, alpha, beta)
   - ✅ Euler angle utilities

3. **6-DOF Dynamics** ([src/core/dynamics.py](src/core/dynamics.py))
   - ✅ Rigid body equations of motion
   - ✅ Force and moment aggregation
   - ✅ Gravity model
   - ✅ Pluggable aerodynamics and propulsion

4. **Numerical Integrators** ([src/core/integrator.py](src/core/integrator.py))
   - ✅ RK4 (4th-order Runge-Kutta)
   - ✅ RK45 (adaptive step size)
   - ✅ Integration loop with time history

5. **Aerodynamic Models** ([src/core/aerodynamics.py](src/core/aerodynamics.py))
   - ✅ Constant coefficient model
   - ✅ Linear stability derivatives model
   - ✅ AVL table-based model with interpolation
   - ✅ Control surface effects

6. **Propulsion Models** ([src/core/propulsion.py](src/core/propulsion.py))
   - ✅ Constant thrust model
   - ✅ Propeller model
   - ✅ Thrust line offset moments
   - ✅ Combined force/moment model

### Phase 3: Supporting Systems ✅ (COMPLETED)

**Completed Components:**

1. **Standard Atmosphere** ([src/environment/atmosphere.py](src/environment/atmosphere.py))
   - ✅ US Standard Atmosphere 1976 model
   - ✅ Troposphere, stratosphere layers
   - ✅ Temperature, pressure, density computation
   - ✅ Speed of sound, viscosity
   - ✅ Dynamic pressure, Mach number, Reynolds number utilities

2. **Autopilot Controllers** ([src/control/autopilot.py](src/control/autopilot.py))
   - ✅ Generic PID controller with anti-windup
   - ✅ Altitude hold (cascaded pitch control)
   - ✅ Heading hold (cascaded roll control)
   - ✅ Airspeed hold (throttle control)

3. **Trim Solver** ([src/control/trim.py](src/control/trim.py))
   - ✅ Straight and level flight trim
   - ✅ Coordinated turn trim
   - ✅ Scipy-based optimization
   - ✅ Residual minimization

4. **AVL Aerodynamic Database** ([src/aero/avl_database.py](src/aero/avl_database.py))
   - ✅ Load AVL sweep data from CSV
   - ✅ Coefficient interpolation
   - ✅ Force and moment computation
   - ✅ Damping derivatives support

### Phase 4: Analysis Tools ✅ (COMPLETED)

**Completed Components:**

1. **Linearization** ([src/analysis/stability.py](src/analysis/stability.py))
   - ✅ Linearize dynamics about trim point
   - ✅ Extract A, B, C, D state-space matrices
   - ✅ Finite difference method for Jacobians
   - ✅ Full 13-state, 4-input linearized model

2. **Stability Analysis** ([src/analysis/stability.py](src/analysis/stability.py))
   - ✅ Eigenvalue and eigenvector computation
   - ✅ Dynamic mode identification
   - ✅ Mode classification (phugoid, short period, dutch roll, roll, spiral)
   - ✅ Damping ratio and natural frequency extraction
   - ✅ Stability assessment

3. **Frequency Response** ([src/analysis/frequency.py](src/analysis/frequency.py))
   - ✅ Bode plot computation (magnitude and phase)
   - ✅ Step response analysis
   - ✅ Impulse response analysis
   - ✅ Gain and phase margin calculation
   - ✅ Transfer function utilities

---

### Phase 1: AVL Geometry Generation & Validation ✅ (COMPLETED)

**Completed Components:**

1. **Geometry Analysis** ([src/io/geometry.py](src/io/geometry.py))
   - ✅ CSV point parser for LE/TE data
   - ✅ Wing geometry calculator (span, area, MAC, AR, sweep, taper, dihedral)
   - ✅ Tail surface estimator using volume coefficients
   - ✅ Unit conversion (inches → feet)

2. **Mass Properties** ([src/io/mass_properties.py](src/io/mass_properties.py))
   - ✅ Mass converter (lbm → slugs, kg)
   - ✅ Inertia converter (lbm·in² → slug·ft², kg·m²)
   - ✅ CG location converter (inches → feet, meters)
   - ✅ AVL .mass file generator

3. **AVL Geometry Generator** ([src/aero/avl_geometry.py](src/aero/avl_geometry.py))
   - ✅ Complete AVL .avl file writer
   - ✅ Wing surface from LE/TE points
   - ✅ Horizontal tail with elevator
   - ✅ Vertical tail with rudder
   - ✅ Flaperon control surfaces (80% chord, 75% span)
   - ✅ NACA 6-series airfoil integration

4. **Flight Conditions** ([src/aero/avl_run_cases.py](src/aero/avl_run_cases.py))
   - ✅ US Standard Atmosphere model
   - ✅ AVL .run file generator
   - ✅ Cruise condition (Mach 0.25 @ 20,000 ft)
   - ✅ Climb condition (Mach 0.20 @ 10,000 ft)
   - ✅ Landing condition (Mach 0.15 @ sea level)

5. **AVL Interface** ([src/aero/avl_interface.py](src/aero/avl_interface.py))
   - ✅ Subprocess interface to AVL executable
   - ✅ Command sequence generator
   - ✅ Output file parser (.ft, .st files)
   - ✅ Alpha sweep capability
   - ✅ Results plotting

6. **Project Structure**
   - ✅ Modular directory layout
   - ✅ requirements.txt with dependencies
   - ✅ Generated files in avl_files/

---

## Aircraft Configuration

### Wing Geometry (from nTop export)

| Parameter | Value | Units |
|-----------|-------|-------|
| **Span** | 19.89 | ft |
| **Area** | 199.94 | ft² |
| **Mean Aerodynamic Chord (MAC)** | 26.69 | ft |
| **Root Chord** | 22.40 | ft |
| **Tip Chord** | 0.20 | ft |
| **Taper Ratio** | 0.009 | - |
| **Aspect Ratio** | 1.98 | - |
| **LE Sweep** | 56.64 | deg |
| **c/4 Sweep** | 45.18 | deg |
| **Dihedral** | 6.00 | deg |

**Notes:**
- Very low aspect ratio (1.98) indicates delta wing or highly swept design
- Extreme taper ratio (0.009) means nearly pointed wing tips
- High sweep angles (56.6° LE) typical of high-speed UAV or flying wing

### Horizontal Tail (Estimated)

| Parameter | Value | Units |
|-----------|-------|-------|
| **Area** | 47.99 | ft² |
| **Span** | 8.72 | ft |
| **Chord** | 5.51 | ft |
| **Aspect Ratio** | 1.58 | - |
| **Volume Coefficient (V_H)** | 0.60 | - |
| **Moment Arm** | 66.72 | ft |
| **X Position** | 90.68 | ft |

### Vertical Tail (Estimated)

| Parameter | Value | Units |
|-----------|-------|-------|
| **Area** | 2.98 | ft² |
| **Height** | 2.11 | ft |
| **Chord** | 1.41 | ft |
| **Aspect Ratio** | 1.50 | - |
| **Volume Coefficient (V_V)** | 0.05 | - |
| **Moment Arm** | 66.72 | ft |
| **X Position** | 92.73 | ft |

### Mass Properties

| Parameter | Value | Units |
|-----------|-------|-------|
| **Mass** | 7,555.6 lbm / 234.8 slugs / 3,427.2 kg |
| **CG Location** | (12.92, 0.00, 0.05) | ft |
| **Ixx** | 14,908.4 | slug·ft² |
| **Iyy** | 2,318.4 | slug·ft² |
| **Izz** | 17,226.9 | slug·ft² |

### Control Surfaces

| Surface | Hinge Line | Span Extent | Type |
|---------|------------|-------------|------|
| **Flaperons** | 80% chord | 75% semispan | Antisymmetric (roll) |
| **Elevator** | 70% chord | Full span | Symmetric (pitch) |
| **Rudder** | 70% chord | Full height | Yaw |

### Airfoils

- **Wing**: NACA 64-212 (constant along span)
- **Horizontal Tail**: NACA 0012
- **Vertical Tail**: NACA 0012

---

## Flight Envelope

| Condition | Altitude | Mach | Velocity | Density | Purpose |
|-----------|----------|------|----------|---------|---------|
| **Cruise** | 20,000 ft | 0.25 | 259.2 ft/s | 0.001267 slug/ft³ | Loiter |
| **Climb** | 10,000 ft | 0.20 | 215.4 ft/s | 0.001756 slug/ft³ | Ascent |
| **Landing** | 0 ft | 0.15 | 167.4 ft/s | 0.002377 slug/ft³ | Approach |

---

## Generated Files

### AVL Files (in avl_files/)

1. **uav.avl** - Main geometry file
   - Wing with 7 spanwise sections
   - Horizontal tail (2 sections)
   - Vertical tail (2 sections)
   - Control surface definitions
   - Reference values (Sref, Cref, Bref, CG)

2. **uav.mass** - Mass properties file
   - Total mass in slugs
   - CG location in feet
   - Inertia tensor in slug·ft²

3. **uav.run** - Run cases file
   - 3 flight conditions defined
   - Atmospheric properties
   - Mass/inertia for each case

### Testing Scripts

1. **run_avl_simple.bat** - Windows batch file to run AVL manually
2. **test_avl.bat** - AVL geometry validation script

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- scipy
- matplotlib
- pandas
- pyyaml
- pytest

### 2. Generate AVL Geometry from nTop Data

```python
from src.aero.avl_geometry import generate_avl_geometry_from_csv

# Generate AVL files from CSV exports
wing, h_tail, v_tail, mass_props = generate_avl_geometry_from_csv(
    le_file="Data/LEpts.csv",
    te_file="Data/TEpts.csv",
    mass_file="Data/mass.csv",
    output_file="avl_files/uav.avl",
    aircraft_name="nTop_UAV"
)
```

### 3. Run AVL Analysis

#### Option A: Manual (Windows)
```bash
run_avl_simple.bat
```

#### Option B: Python Interface
```python
from src.aero.avl_interface import AVLInterface

avl = AVLInterface(r"C:\path\to\avl.exe")

# Single analysis point
result = avl.run_avl_case(
    avl_file="avl_files/uav.avl",
    mass_file="avl_files/uav.mass",
    alpha=2.0,
    beta=0.0,
    mach=0.25
)

print(f"CL = {result.CL}, CD = {result.CD}, L/D = {result.CL/result.CD}")

# Alpha sweep
results = avl.run_alpha_sweep(
    avl_file="avl_files/uav.avl",
    mass_file="avl_files/uav.mass",
    alpha_range=(-5, 15, 1),
    mach=0.25
)
```

### 4. Analyze Wing Geometry

```python
from src.io.geometry import read_csv_points, compute_wing_geometry

le_points = read_csv_points("Data/LEpts.csv", units='inches')
te_points = read_csv_points("Data/TEpts.csv", units='inches')

wing = compute_wing_geometry(le_points, te_points)

print(f"Wing Span: {wing.span:.2f} ft")
print(f"Wing Area: {wing.area:.2f} ft²")
print(f"Aspect Ratio: {wing.aspect_ratio:.2f}")
```

---

## Testing

### Test Coverage

All phases have comprehensive test suites:

- **Phase 1 Tests** ([tests/test_phase1.py](tests/test_phase1.py)): 11 tests for AVL geometry generation
- **Phase 2 Tests** ([tests/test_phase2.py](tests/test_phase2.py)): 23 tests for 6-DOF dynamics
- **Phase 3 Tests** ([tests/test_phase3.py](tests/test_phase3.py)): 22 tests for supporting systems
- **Phase 4 Tests** ([tests/test_phase4.py](tests/test_phase4.py)): 11 tests for analysis tools

Run all tests:
```bash
pytest tests/ -v
```

Total: **67 passing tests**

---

## Examples

### Phase 2: Basic 6-DOF Simulation
```bash
python examples/simple_6dof_sim.py
```
Demonstrates complete 6-DOF simulation with quaternion attitude, RK4 integration, and basic aerodynamics.

### Phase 3: Autopilot Demonstration
```bash
python examples/autopilot_demo.py
```
Demonstrates altitude hold, heading hold, and airspeed hold autopilots with standard atmosphere integration.

### Phase 4: Stability Analysis
```bash
python examples/stability_analysis_demo.py
```
Demonstrates linearization, eigenvalue analysis, mode identification, and frequency response (Bode plots, step responses).

---

## Directory Structure

```
nTop6DOF/
├── src/
│   ├── core/               # ✅ 6-DOF dynamics
│   │   ├── quaternion.py       # ✅ Quaternion mathematics
│   │   ├── state.py            # ✅ State vector
│   │   ├── dynamics.py         # ✅ Rigid body dynamics
│   │   ├── integrator.py       # ✅ RK4, RK45 integrators
│   │   ├── aerodynamics.py     # ✅ Aero models
│   │   └── propulsion.py       # ✅ Propulsion models
│   ├── environment/        # ✅ Atmosphere models
│   │   └── atmosphere.py       # ✅ US Standard Atmosphere 1976
│   ├── control/            # ✅ Autopilot and trim
│   │   ├── autopilot.py        # ✅ PID controllers
│   │   └── trim.py             # ✅ Trim solver
│   ├── analysis/           # ✅ Stability and frequency analysis
│   │   ├── stability.py        # ✅ Linearization and modes
│   │   └── frequency.py        # ✅ Bode plots, step response
│   ├── aero/
│   │   ├── avl_geometry.py     # ✅ AVL file generator
│   │   ├── avl_interface.py    # ✅ AVL Python interface
│   │   ├── avl_run_cases.py    # ✅ Run case generator
│   │   ├── avl_database.py     # ✅ AVL data interpolation
│   │   └── xfoil_interface.py  # TODO
│   ├── io/
│   │   ├── geometry.py         # ✅ LE/TE CSV parser
│   │   └── mass_properties.py  # ✅ Mass converter
│   ├── propulsion/             # TODO (advanced models)
│   ├── analysis/               # TODO
│   └── visualization/          # TODO
├── tests/
│   ├── test_phase1.py          # ✅ Phase 1 tests (11 tests)
│   ├── test_phase2.py          # ✅ Phase 2 tests (23 tests)
│   ├── test_phase3.py          # ✅ Phase 3 tests (22 tests)
│   └── test_phase4.py          # ✅ Phase 4 tests (11 tests)
├── examples/
│   ├── simple_6dof_sim.py      # ✅ Basic 6-DOF simulation
│   ├── autopilot_demo.py       # ✅ Autopilot demonstration
│   └── stability_analysis_demo.py  # ✅ Stability analysis
├── config/                     # TODO
├── Data/
│   ├── LEpts.csv              # ✅ Wing LE points (from nTop)
│   ├── TEpts.csv              # ✅ Wing TE points (from nTop)
│   └── mass.csv               # ✅ Mass properties (from nTop)
├── Docs/
│   ├── avl_doc.txt            # AVL documentation
│   └── AVL_User_Primer.pdf
├── Plan/
│   └── flight_6dof_project_plan.md
├── avl_files/
│   ├── uav.avl                        # ✅ Generated AVL geometry
│   ├── uav.mass                       # ✅ Generated mass file
│   ├── uav.run                        # ✅ Generated run cases
│   └── sample_aero_database.csv       # ✅ Sample aero database
├── requirements.txt           # ✅ Python dependencies
└── README.md                  # This file
```

---

## Next Steps (Phase 5 & Beyond)

### Phase 5: I/O and Configuration (Planned)
1. **Configuration Files**
   - YAML-based aircraft definitions
   - Easy aircraft setup from config files
   - Parameter sweeps and batch analysis
2. **File Parsers**
   - Enhanced AVL output parsing
   - XFOIL polar integration
   - Export/import utilities

### Medium Term:
- XFOIL integration for 2D airfoil polars
- Simple propulsion model
- Basic autopilot (PID altitude/heading hold)
- Linearization & stability analysis
- Eigenmode analysis (phugoid, short period, Dutch roll, etc.)

### Long Term:
- nTop workflow automation
- Parametric design sweeps
- Optimization interface
- Wind/turbulence models
- Advanced control laws
- Real-time visualization

---

## Important Notes

### Tail Geometry Assumptions
**⚠️ Current tail geometry is ESTIMATED using volume coefficients:**
- Horizontal tail: V_H = 0.6
- Vertical tail: V_V = 0.05
- Tail moment arms: 2.5 × wing MAC

**You should export tail surfaces from nTop using the same LE/TE point method:**
1. Export horizontal tail LEpts and TEpts to separate CSV files
2. Export vertical tail LEpts and TEpts to separate CSV files
3. Update avl_geometry.py to use actual geometry instead of estimates

### Airfoil Selection
Currently using **NACA 64-212** for the wing. For optimal performance at Mach 0.25 cruise:
- Consider NACA 64-series with design CL matching cruise condition
- Run XFOIL to generate polars if custom airfoil is needed
- Update airfoil parameter in avl_geometry.py

### Units Convention
- **Input**: US Customary (inches, lbm, lbm·in²)
- **AVL**: US Customary (feet, slugs, slug·ft²)
- **6-DOF (future)**: Can use either SI or US Customary

### AVL Path
Update AVL executable path in scripts:
```python
avl_exe = r"C:\Users\bradrothenberg\OneDrive - nTop\Sync\AVL\avl.exe"
```

---

## References

- **AVL Documentation**: `Docs/avl_doc.txt`, `Docs/AVL_User_Primer.pdf`
- **Project Plan**: `Plan/flight_6dof_project_plan.md`
- **AVL Website**: http://web.mit.edu/drela/Public/web/avl/
- **XFOIL Website**: https://web.mit.edu/drela/Public/web/xfoil/

---

## Contact & Support

For issues with:
- **nTop geometry export**: Check LE/TE point extraction from nTop
- **AVL errors**: Verify .avl file format, check AVL documentation
- **Python errors**: Ensure all dependencies are installed
- **Framework design**: Review `Plan/flight_6dof_project_plan.md`

---

## License

Internal nTop project - not for external distribution.

---

**Last Updated**: 2025-01-10
**Status**: Phases 1, 2, 3, 4 Complete - Core Framework with Analysis Tools
**Version**: 0.4.0-alpha
**Test Coverage**: 67 passing tests across 4 phases
