# nTop 6-DOF Flight Dynamics Framework

A production-ready Python framework for 6-degree-of-freedom (6-DOF) flight dynamics simulation, integrated with AVL (Athena Vortex Lattice) and XFOIL for aerodynamic analysis of nTop-designed aircraft.

## Project Status

✅ **ALL CORE PHASES COMPLETE** - Production-ready framework with 164 passing tests

For detailed phase-by-phase status, see [Plan/status.md](Plan/status.md)

**Key Achievements:**
- ✅ Complete 6-DOF dynamics with quaternion attitude
- ✅ Flying wing configuration with stable controlled flight
- ✅ FJ-44 turbofan integration with trim solver
- ✅ Triple-loop cascaded autopilot with stall protection
- ✅ Comprehensive testing (164 tests, all passing)
- ✅ Extensive documentation (1,100+ lines)

**Recent Updates:**
- Enhanced autopilot for flying wing stability
- Comprehensive user and tuning guides
- Full unit test coverage for autopilot controllers
- Performance benchmarks and validation

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Flying Wing Demo (Recommended)

```bash
python examples/flyingwing_stable_flight.py
```

Demonstrates the enhanced triple-loop cascaded autopilot achieving stable controlled flight.

### 3. Run Tests

```bash
pytest tests/ -v
```

All 164 tests should pass ✅

---

## Documentation

### User Guides
- **[FLYING_WING_AUTOPILOT_GUIDE.md](FLYING_WING_AUTOPILOT_GUIDE.md)** - Complete user guide (500+ lines)
- **[AUTOPILOT_TUNING_GUIDE.md](AUTOPILOT_TUNING_GUIDE.md)** - Practical tuning guide (600+ lines)

### Technical Documentation
- **[Plan/status.md](Plan/status.md)** - Detailed project status
- **[TRIM_STATUS.md](TRIM_STATUS.md)** - Trim analysis
- **[RK4_AUTOPILOT_RESULTS.md](RK4_AUTOPILOT_RESULTS.md)** - Integration testing

---

## Examples

| Example | Description |
|---------|-------------|
| **flyingwing_stable_flight.py** | Flying wing with enhanced autopilot (recommended) |
| simple_6dof_sim.py | Basic 6-DOF simulation |
| autopilot_demo.py | Altitude/heading/airspeed hold |
| stability_analysis_demo.py | Linearization and mode analysis |
| config_demo.py | YAML configuration system |
| visualization_demo.py | 3D plots and animations |

---

## Aircraft Configuration

### Wing Geometry (from nTop)

| Parameter | Value |
|-----------|-------|
| Span | 19.89 ft |
| Area | 199.94 ft² |
| MAC | 26.69 ft |
| Aspect Ratio | 1.98 |
| LE Sweep | 56.64° |
| Taper Ratio | 0.009 |

### Mass Properties

| Parameter | Value |
|-----------|-------|
| Mass | 234.8 slugs (7,555.6 lbm) |
| CG | (12.92, 0.00, 0.05) ft |
| Ixx, Iyy, Izz | 14,908 / 2,318 / 17,227 slug·ft² |

---

## Testing

**Total: 164 passing tests** ✅

- Phase 1 (AVL Geometry): 11 tests
- Phase 2 (6-DOF Dynamics): 23 tests
- Phase 3 (Supporting Systems): 22 tests
- Phase 4 (Analysis Tools): 11 tests
- Phase 5 (I/O & Configuration): 17 tests
- Phase 6 (Visualization): 19 tests
- Core Coverage: 29 tests
- Autopilot Controllers: 32 tests

---

## Directory Structure

```
nTop6DOF/
├── src/               # Core framework
│   ├── core/          # 6-DOF dynamics, quaternions, integrators
│   ├── environment/   # Standard atmosphere
│   ├── control/       # Autopilot and trim solvers
│   ├── analysis/      # Stability and frequency analysis
│   ├── aero/          # AVL interface and database
│   ├── io/            # Configuration and parsers
│   ├── visualization/ # Plotting and animation
│   └── simulation/    # Enhanced trim solvers
├── tests/             # Comprehensive test suite (164 tests)
├── examples/          # Demonstration scripts
├── config/            # YAML configuration files
├── Data/              # nTop exports (LE/TE points, mass)
├── Plan/              # Project planning and status
└── avl_files/         # Generated AVL files
```

---

## References

- **AVL Website**: http://web.mit.edu/drela/Public/web/avl/
- **Project Plan**: [Plan/flight_6dof_project_plan.md](Plan/flight_6dof_project_plan.md)
- **Project Status**: [Plan/status.md](Plan/status.md)

---

## License

MIT License - Copyright (c) 2025 Brad Rothenberg

---

**Last Updated**: 2025-11-10  
**Version**: 0.6.0-alpha  
**Test Coverage**: 164 passing tests ✅
