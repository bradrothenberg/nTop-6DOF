"""
XFOIL Integration Demo

Demonstrates using XFOIL polar data with the 6-DOF simulation for
high-fidelity airfoil performance with Reynolds number effects.

This example:
1. Creates synthetic XFOIL polar database for NACA 64-212 airfoil
2. Loads the polar database
3. Creates XFOIL-based aerodynamic model
4. Runs simulation with Reynolds-corrected aerodynamics
5. Compares with AVL-based model

Author: Claude Code
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aero.xfoil_database import create_example_database, XFOILDatabaseLoader
from src.core.aerodynamics import XFOILAeroModel, LinearAeroModel
from src.core.state import State
from src.core.dynamics import RigidBody6DOF
from src.core.propulsion import TurbofanModel
from src.core.integrator import integrate_rk4
from src.environment.atmosphere import StandardAtmosphere


def compare_aero_models():
    """
    Compare XFOIL-based model with AVL-based model across flight envelope.
    """
    print("="*60)
    print("XFOIL vs AVL Aerodynamic Model Comparison")
    print("="*60)

    # Create example XFOIL database
    print("\n1. Creating example XFOIL polar database...")
    create_example_database('xfoil_data')

    # Load database
    print("\n2. Loading XFOIL polar database...")
    loader = XFOILDatabaseLoader(database_dir='xfoil_data')
    polar_db = loader.auto_load_airfoil('NACA_64-212')

    if polar_db is None:
        print("ERROR: Could not load polar database!")
        return

    print(f"   Loaded database for {polar_db.airfoil_name}")
    print(f"   Reynolds numbers: {polar_db.reynolds_numbers / 1e6} million")

    # Aircraft parameters (flying wing from nTop)
    S_ref = 199.94      # ft²
    c_ref = 26.689      # ft (MAC)
    b_ref = 19.890      # ft
    AR = b_ref**2 / S_ref  # Aspect ratio ≈ 1.98

    # Create XFOIL-based model
    print("\n3. Creating XFOIL-based aerodynamic model...")
    xfoil_model = XFOILAeroModel(
        polar_database=polar_db,
        S_ref=S_ref,
        c_ref=c_ref,
        b_ref=b_ref,
        aspect_ratio=AR,
        oswald_efficiency=0.85
    )

    # Create AVL-based model for comparison
    print("4. Creating AVL-based aerodynamic model...")
    avl_model = LinearAeroModel(S_ref=S_ref, c_ref=c_ref, b_ref=b_ref)
    # Set typical values from flying wing AVL analysis
    avl_model.CL_0 = 0.2
    avl_model.CL_alpha = 0.11 * (180 / np.pi)  # Convert per degree to per radian
    avl_model.CD_0 = 0.006
    avl_model.CD_alpha2 = 0.05
    avl_model.Cm_0 = -0.05
    avl_model.Cm_alpha = -0.080

    # Test across flight envelope
    print("\n5. Testing across flight envelope...")
    print("   Altitude, Airspeed, Alpha -> CL, CD, L/D, Cm")
    print("   " + "-"*70)

    altitudes = [0, 5000, 10000, 20000]  # ft
    airspeeds = [200, 300, 400, 500, 600]  # ft/s
    alphas = np.radians([0, 2, 5, 8, 10])  # radians

    # Storage for comparison plots
    results_xfoil = {'alpha': [], 'CL': [], 'CD': [], 'LD': [], 'Cm': [], 'Re': []}
    results_avl = {'alpha': [], 'CL': [], 'CD': [], 'LD': [], 'Cm': []}

    # Test at typical cruise condition
    test_alt = 5000  # ft
    test_V = 548.5   # ft/s (Mach 0.5 at 5000 ft)

    atm = StandardAtmosphere()
    rho, _, _, _ = atm.get_conditions(test_alt)

    for alpha in alphas:
        # Create test state
        state = State()
        state.position = np.array([0, 0, -test_alt])
        state.velocity_body = np.array([test_V * np.cos(alpha), 0, test_V * np.sin(alpha)])
        state.set_euler_angles(0, alpha, 0)

        # XFOIL model
        forces_xfoil, moments_xfoil = xfoil_model.compute_forces_moments(state)
        q_bar = 0.5 * rho * test_V**2
        CL_xfoil = -forces_xfoil[2] / (q_bar * S_ref)
        CD_xfoil = -forces_xfoil[0] / (q_bar * S_ref)
        Cm_xfoil = moments_xfoil[1] / (q_bar * S_ref * c_ref)
        LD_xfoil = CL_xfoil / CD_xfoil if CD_xfoil > 0 else 0

        # Compute Reynolds number
        mu = 3.62e-7 * (1 + (test_alt / 50000))  # Simplified viscosity
        Re = rho * test_V * c_ref / mu

        results_xfoil['alpha'].append(np.degrees(alpha))
        results_xfoil['CL'].append(CL_xfoil)
        results_xfoil['CD'].append(CD_xfoil)
        results_xfoil['LD'].append(LD_xfoil)
        results_xfoil['Cm'].append(Cm_xfoil)
        results_xfoil['Re'].append(Re / 1e6)

        # AVL model
        forces_avl, moments_avl = avl_model.compute_forces_moments(state)
        CL_avl = -forces_avl[2] / (q_bar * S_ref)
        CD_avl = -forces_avl[0] / (q_bar * S_ref)
        Cm_avl = moments_avl[1] / (q_bar * S_ref * c_ref)
        LD_avl = CL_avl / CD_avl if CD_avl > 0 else 0

        results_avl['alpha'].append(np.degrees(alpha))
        results_avl['CL'].append(CL_avl)
        results_avl['CD'].append(CD_avl)
        results_avl['LD'].append(LD_avl)
        results_avl['Cm'].append(Cm_avl)

        print(f"   Alpha={np.degrees(alpha):4.1f}° | XFOIL: CL={CL_xfoil:5.3f} CD={CD_xfoil:6.4f} L/D={LD_xfoil:5.1f} Cm={Cm_xfoil:6.3f} Re={Re/1e6:4.1f}M")
        print(f"              | AVL:   CL={CL_avl:5.3f} CD={CD_avl:6.4f} L/D={LD_avl:5.1f} Cm={Cm_avl:6.3f}")
        print()

    # Plot comparison
    print("\n6. Plotting comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # CL vs alpha
    axes[0, 0].plot(results_xfoil['alpha'], results_xfoil['CL'], 'b-o', label='XFOIL', linewidth=2)
    axes[0, 0].plot(results_avl['alpha'], results_avl['CL'], 'r--s', label='AVL', linewidth=2)
    axes[0, 0].set_xlabel('Angle of Attack (deg)')
    axes[0, 0].set_ylabel('CL')
    axes[0, 0].set_title('Lift Coefficient vs Alpha')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Drag polar
    axes[0, 1].plot(results_xfoil['CD'], results_xfoil['CL'], 'b-o', label='XFOIL', linewidth=2)
    axes[0, 1].plot(results_avl['CD'], results_avl['CL'], 'r--s', label='AVL', linewidth=2)
    axes[0, 1].set_xlabel('CD')
    axes[0, 1].set_ylabel('CL')
    axes[0, 1].set_title('Drag Polar')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # L/D vs alpha
    axes[1, 0].plot(results_xfoil['alpha'], results_xfoil['LD'], 'b-o', label='XFOIL', linewidth=2)
    axes[1, 0].plot(results_avl['alpha'], results_avl['LD'], 'r--s', label='AVL', linewidth=2)
    axes[1, 0].set_xlabel('Angle of Attack (deg)')
    axes[1, 0].set_ylabel('L/D')
    axes[1, 0].set_title('Lift-to-Drag Ratio vs Alpha')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Cm vs alpha
    axes[1, 1].plot(results_xfoil['alpha'], results_xfoil['Cm'], 'b-o', label='XFOIL', linewidth=2)
    axes[1, 1].plot(results_avl['alpha'], results_avl['Cm'], 'r--s', label='AVL', linewidth=2)
    axes[1, 1].set_xlabel('Angle of Attack (deg)')
    axes[1, 1].set_ylabel('Cm')
    axes[1, 1].set_title('Pitch Moment Coefficient vs Alpha')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.savefig('output/xfoil_avl_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved comparison plot to output/xfoil_avl_comparison.png")

    plt.show()


def run_xfoil_simulation():
    """
    Run 6-DOF simulation using XFOIL-based aerodynamic model.
    """
    print("\n" + "="*60)
    print("6-DOF Simulation with XFOIL Aerodynamics")
    print("="*60)

    # Load XFOIL database
    print("\n1. Loading XFOIL polar database...")
    loader = XFOILDatabaseLoader(database_dir='xfoil_data')
    polar_db = loader.auto_load_airfoil('NACA_64-212')

    if polar_db is None:
        print("ERROR: Could not load polar database!")
        return

    # Aircraft parameters
    S_ref = 199.94
    c_ref = 26.689
    b_ref = 19.890
    AR = b_ref**2 / S_ref

    # Create XFOIL aerodynamic model
    print("\n2. Creating XFOIL aerodynamic model...")
    aero_model = XFOILAeroModel(
        polar_database=polar_db,
        S_ref=S_ref,
        c_ref=c_ref,
        b_ref=b_ref,
        aspect_ratio=AR,
        oswald_efficiency=0.85
    )

    # Mass properties
    mass = 234.8  # slugs
    Ixx = 14908  # slug-ft²
    Iyy = 2318
    Izz = 17227

    # Create propulsion model (FJ-44 turbofan)
    print("\n3. Creating propulsion model...")
    propulsion = TurbofanModel(
        max_thrust=1900.0,
        altitude_lapse_rate=0.7,
        thrust_line_offset=np.array([0.0, 0.0, 0.0])
    )

    # Create dynamics
    print("\n4. Creating 6-DOF dynamics...")
    dynamics = RigidBody6DOF(
        mass=mass,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        Ixz=0.0,
        aero_model=aero_model,
        propulsion_model=propulsion
    )

    # Initial state (level flight at 5000 ft, Mach 0.5)
    print("\n5. Setting initial conditions...")
    state0 = State()
    state0.position = np.array([0, 0, -5000])
    state0.velocity_body = np.array([548.5, 0, 0])  # ft/s
    state0.set_euler_angles(0, np.radians(1.75), 0)  # Trim alpha
    state0.angular_rates = np.array([0, 0, 0])

    # Control inputs (trimmed)
    controls = {
        'throttle': 0.093,
        'elevator': np.radians(-6.81)
    }

    print(f"\n   Initial altitude: {-state0.position[2]:.0f} ft")
    print(f"   Initial airspeed: {state0.airspeed:.1f} ft/s")
    print(f"   Initial alpha: {np.degrees(state0.alpha):.2f}°")

    # Integrate
    print("\n6. Running simulation (30 seconds)...")
    dt = 0.01
    t_final = 30.0

    time, states = integrate_rk4(dynamics, state0, controls, dt, t_final)

    print(f"   Simulation completed: {len(time)} time steps")

    # Extract results
    positions = np.array([s.position for s in states])
    velocities = np.array([s.velocity_body for s in states])
    euler_angles = np.array([s.get_euler_angles() for s in states])
    airspeeds = np.array([s.airspeed for s in states])
    alphas = np.array([np.degrees(s.alpha) for s in states])

    # Summary statistics
    print("\n7. Flight Statistics:")
    print(f"   Altitude change: {positions[0, 2] - positions[-1, 2]:.1f} ft")
    print(f"   Airspeed change: {airspeeds[-1] - airspeeds[0]:.1f} ft/s")
    print(f"   Average alpha: {np.mean(alphas):.2f}°")
    print(f"   Alpha range: [{np.min(alphas):.2f}, {np.max(alphas):.2f}]°")

    # Plot results
    print("\n8. Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Altitude
    axes[0, 0].plot(time, -positions[:, 2], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (ft)')
    axes[0, 0].set_title('Altitude vs Time')
    axes[0, 0].grid(True, alpha=0.3)

    # Airspeed
    axes[0, 1].plot(time, airspeeds, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Airspeed (ft/s)')
    axes[0, 1].set_title('Airspeed vs Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Angle of attack
    axes[1, 0].plot(time, alphas, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Alpha (deg)')
    axes[1, 0].set_title('Angle of Attack vs Time')
    axes[1, 0].grid(True, alpha=0.3)

    # 3D trajectory
    axes[1, 1].plot(positions[:, 0], positions[:, 1], 'k-', linewidth=2)
    axes[1, 1].scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
    axes[1, 1].scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End')
    axes[1, 1].set_xlabel('X (ft)')
    axes[1, 1].set_ylabel('Y (ft)')
    axes[1, 1].set_title('Ground Track')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')

    plt.tight_layout()
    plt.savefig('output/xfoil_simulation.png', dpi=150, bbox_inches='tight')
    print("   Saved simulation plot to output/xfoil_simulation.png")

    plt.show()

    print("\n✓ XFOIL simulation completed successfully!")


if __name__ == '__main__':
    # Create output directory
    Path('output').mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("XFOIL Integration Demo")
    print("="*60)
    print("\nThis demo shows how to use XFOIL airfoil polars with the")
    print("6-DOF simulation for high-fidelity Reynolds number effects.")
    print()

    # Part 1: Compare XFOIL vs AVL models
    compare_aero_models()

    # Part 2: Run simulation with XFOIL model
    run_xfoil_simulation()

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
