"""
Flying Wing Simulation with Hybrid XFOIL+AVL Aerodynamics

Combines the best of both worlds:
- XFOIL: 2D section data with Reynolds effects and viscous drag
- AVL: 3D stability derivatives and control effectiveness

This provides the most accurate aerodynamic modeling available in the framework.

Author: Claude Code
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aero.xfoil_database import create_example_database, XFOILDatabaseLoader
from src.core.hybrid_aero import HybridXFOILAVLModel
from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.propulsion import TurbofanModel
from src.control.autopilot import FlyingWingAutopilot


def main():
    """Run flying wing simulation with hybrid XFOIL+AVL aerodynamics."""

    print("="*70)
    print("Flying Wing Simulation with Hybrid XFOIL+AVL Aerodynamics")
    print("="*70)
    print()

    # === 1. Create/Load XFOIL Database ===
    print("1. Loading XFOIL polar database...")
    xfoil_dir = Path('xfoil_data')
    if not xfoil_dir.exists():
        print("   Creating example XFOIL database...")
        create_example_database('xfoil_data')

    loader = XFOILDatabaseLoader(database_dir='xfoil_data')
    polar_db = loader.auto_load_airfoil('NACA_64-212')

    if polar_db is None:
        print("ERROR: Could not load XFOIL database!")
        return

    print(f"   OK Loaded {polar_db.airfoil_name}")
    print(f"   OK Reynolds numbers: {polar_db.reynolds_numbers / 1e6} million")
    print()

    # === 2. Aircraft Configuration ===
    print("2. Configuring flying wing...")

    # Geometry (from nTop)
    S_ref = 199.94      # ft²
    c_ref = 26.689      # ft (MAC)
    b_ref = 19.890      # ft
    AR = b_ref**2 / S_ref  # Aspect ratio ≈ 1.98

    # Mass properties
    mass = 234.8        # slugs
    Ixx = 14908         # slug-ft²
    Iyy = 2318
    Izz = 17227

    print(f"   Wing area:      {S_ref:.2f} ft²")
    print(f"   Span:           {b_ref:.2f} ft")
    print(f"   MAC:            {c_ref:.2f} ft")
    print(f"   Aspect ratio:   {AR:.2f}")
    print(f"   Mass:           {mass:.1f} slugs")
    print()

    # === 3. Create Hybrid Aerodynamic Model ===
    print("3. Creating hybrid XFOIL+AVL aerodynamic model...")

    aero_model = HybridXFOILAVLModel(
        polar_database=polar_db,
        S_ref=S_ref,
        c_ref=c_ref,
        b_ref=b_ref,
        aspect_ratio=AR,
        oswald_efficiency=0.85
    )

    # Set AVL stability derivatives (from actual flying wing AVL analysis)
    avl_derivatives = {
        # Longitudinal (from AVL analysis)
        'CL_alpha': 0.11 * (180 / np.pi),  # per radian
        'Cm_0': 0.000061,
        'Cm_alpha': -0.079668,  # Static stability
        'Cm_q': -0.347,         # Pitch damping (strong for flying wing)

        # Lateral-directional
        'Cl_beta': -0.1,
        'Cl_p': -0.4,
        'Cl_r': 0.1,
        'Cn_beta': 0.1,
        'Cn_p': -0.05,
        'Cn_r': -0.001,  # Weak for flying wing
        'CY_beta': -0.2,

        # Control effectiveness (from AVL)
        'Cm_elevon': -0.02,      # Pitch control
        'Cl_elevon': -0.001536,  # Roll control (45x stronger than old flaperons)
    }

    aero_model.set_avl_derivatives(avl_derivatives)

    print("   OK XFOIL component: 2D polars with Reynolds effects")
    print("   OK AVL component:   3D stability derivatives")
    print(f"   OK Cm_alpha = {aero_model.Cm_alpha:.6f} (stable)")
    print(f"   OK Cm_q = {aero_model.Cm_q:.3f} (pitch damping)")
    print(f"   OK Cl_elevon = {aero_model.Cl_elevon:.6f} (roll control)")
    print()

    # === 4. Create Propulsion Model ===
    print("4. Creating FJ-44 turbofan model...")

    propulsion = TurbofanModel(
        thrust_max=1900.0,           # lbf
        altitude_lapse_rate=0.7,
        thrust_offset=np.array([0.0, 0.0, 0.0])
    )

    print("   OK Max thrust: 1900 lbf")
    print("   OK Altitude lapse: 0.7")
    print()

    # === 5. Create 6-DOF Dynamics ===
    print("5. Creating 6-DOF dynamics...")

    # Create inertia matrix
    inertia = np.array([[Ixx, 0.0, 0.0],
                        [0.0, Iyy, 0.0],
                        [0.0, 0.0, Izz]])

    dynamics = AircraftDynamics(
        mass=mass,
        inertia=inertia
    )

    # Set models
    dynamics.aero_model = aero_model  # Hybrid XFOIL+AVL
    dynamics.propulsion_model = propulsion

    print("   OK 6-DOF equations configured")
    print("   OK Using hybrid XFOIL+AVL aerodynamics")
    print()

    # === 6. Create Autopilot ===
    print("6. Configuring flying wing autopilot...")

    autopilot = FlyingWingAutopilot(
        # Inner loop (pitch rate damping)
        Kp_pitch_rate=0.15,
        Ki_pitch_rate=0.01,
        Kd_pitch_rate=0.0,

        # Middle loop (pitch attitude)
        Kp_pitch=0.8,
        Ki_pitch=0.05,
        Kd_pitch=0.15,

        # Outer loop (altitude hold)
        Kp_alt=0.003,
        Ki_alt=0.0002,
        Kd_alt=0.008,

        # Stall protection
        stall_speed=150.0,
        min_airspeed_margin=1.3,
        max_alpha=12.0  # degrees
    )

    # Set trim and targets
    autopilot.set_trim(np.radians(-6.81))
    autopilot.throttle_trim = 0.093
    autopilot.set_target_altitude(5000.0)

    print("   OK Triple-loop cascaded architecture")
    print("   OK Pitch rate damping (inner loop)")
    print("   OK Stall protection enabled")
    print(f"   OK Target altitude: 5000 ft")
    print()

    # === 7. Initial Conditions ===
    print("7. Setting initial conditions...")

    # Trimmed level flight at Mach 0.54, 5000 ft
    state0 = State()
    state0.position = np.array([0, 0, -5000])
    state0.velocity_body = np.array([600.0, 0, 0])  # ft/s (Mach 0.54)
    state0.set_euler_angles(0, np.radians(1.75), 0)  # Trim pitch
    state0.angular_rates = np.array([0, 0, 0])

    print(f"   Altitude:  {-state0.position[2]:.0f} ft")
    print(f"   Airspeed:  {state0.airspeed:.1f} ft/s (Mach {state0.airspeed/1116.45:.2f})")
    print(f"   Alpha:     {np.degrees(state0.alpha):.2f}°")
    print(f"   Pitch:     {np.degrees(state0.get_euler_angles()[1]):.2f}°")
    print()

    # === 8. Run Simulation ===
    print("8. Running simulation (10 seconds)...")

    dt = 0.01
    t_final = 10.0

    # Storage for history
    time = []
    states = []
    controls_history = {'elevon': [], 'throttle': []}

    # Initial state
    state = state0.copy()
    t = 0.0

    # Simulation loop with autopilot
    while t <= t_final:
        # Update autopilot
        elevon_cmd = autopilot.update(
            current_altitude=-state.position[2],
            current_pitch=state.get_euler_angles()[1],
            current_pitch_rate=state.angular_rates[1],
            current_airspeed=state.airspeed,
            current_alpha=state.alpha,
            dt=dt
        )

        # Control inputs
        controls = {
            'elevator': elevon_cmd,
            'throttle': autopilot.throttle_trim
        }

        # Store history
        time.append(t)
        states.append(state.copy())
        controls_history['elevon'].append(np.degrees(elevon_cmd))
        controls_history['throttle'].append(autopilot.throttle_trim)

        # Propagate dynamics (RK4 single step)
        state_dot = dynamics.compute_derivatives(state, controls)
        state_array = state.to_array()

        # RK4 integration
        k1 = state_dot
        state_temp = State()
        state_temp.from_array(state_array + 0.5 * dt * k1)
        k2 = dynamics.compute_derivatives(state_temp, controls)

        state_temp.from_array(state_array + 0.5 * dt * k2)
        k3 = dynamics.compute_derivatives(state_temp, controls)

        state_temp.from_array(state_array + dt * k3)
        k4 = dynamics.compute_derivatives(state_temp, controls)

        # Update state
        state_new_array = state_array + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state.from_array(state_new_array)
        state.quaternion.normalize()

        t += dt

    # Convert to arrays
    time = np.array(time)
    positions = np.array([s.position for s in states])
    velocities = np.array([s.velocity_body for s in states])
    euler_angles = np.array([s.get_euler_angles() for s in states])
    airspeeds = np.array([s.airspeed for s in states])
    alphas = np.array([np.degrees(s.alpha) for s in states])
    elevons = np.array(controls_history['elevon'])
    throttles = np.array(controls_history['throttle'])

    print(f"   OK Simulation complete: {len(time)} time steps")
    print()

    # === 9. Flight Statistics ===
    print("9. Flight Statistics:")
    print("   " + "-"*60)

    alt_change = positions[0, 2] - positions[-1, 2]
    speed_change = airspeeds[-1] - airspeeds[0]

    print(f"   Altitude change:     {alt_change:+.1f} ft")
    print(f"   Airspeed change:     {speed_change:+.1f} ft/s")
    print(f"   Altitude std dev:    {np.std(-positions[:, 2]):.1f} ft")
    print(f"   Airspeed std dev:    {np.std(airspeeds):.1f} ft/s")
    print(f"   Pitch std dev:       {np.degrees(np.std(euler_angles[:, 1])):.2f}°")
    print(f"   Roll std dev:        {np.degrees(np.std(euler_angles[:, 0])):.2f}°")
    print(f"   Alpha range:         [{np.min(alphas):.2f}, {np.max(alphas):.2f}]°")
    print(f"   Elevon range:        [{np.min(elevons):.2f}, {np.max(elevons):.2f}]°")
    print(f"   Stall protection:    {'ACTIVE' if autopilot.stall_protection_active else 'Not triggered'}")
    print()

    # === 10. Visualization ===
    print("10. Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Altitude
    axes[0, 0].plot(time, -positions[:, 2], 'b-', linewidth=2, label='Altitude')
    axes[0, 0].axhline(5000, color='k', linestyle='--', alpha=0.3, label='Target')
    axes[0, 0].set_xlabel('Time (s)', fontsize=11)
    axes[0, 0].set_ylabel('Altitude (ft)', fontsize=11)
    axes[0, 0].set_title('Altitude vs Time', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Airspeed
    axes[0, 1].plot(time, airspeeds, 'r-', linewidth=2, label='Airspeed')
    axes[0, 1].axhline(600, color='k', linestyle='--', alpha=0.3, label='Initial')
    axes[0, 1].set_xlabel('Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('Airspeed (ft/s)', fontsize=11)
    axes[0, 1].set_title('Airspeed vs Time', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Pitch and Alpha
    axes[1, 0].plot(time, np.degrees(euler_angles[:, 1]), 'g-', linewidth=2, label='Pitch angle')
    axes[1, 0].plot(time, alphas, 'm--', linewidth=2, label='Angle of attack')
    axes[1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[1, 0].set_ylabel('Angle (deg)', fontsize=11)
    axes[1, 0].set_title('Pitch & Alpha vs Time', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Control inputs
    axes[1, 1].plot(time, elevons, 'c-', linewidth=2, label='Elevon')
    axes[1, 1].plot(time, throttles * 100, 'orange', linewidth=2, label='Throttle (%)')
    axes[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes[1, 1].set_ylabel('Deflection (deg) / Throttle (%)', fontsize=11)
    axes[1, 1].set_title('Control Inputs vs Time', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Flying Wing - Hybrid XFOIL+AVL Aerodynamics',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save plot
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig('output/flyingwing_hybrid_aero.png', dpi=150, bbox_inches='tight')
    print("   OK Saved plot: output/flyingwing_hybrid_aero.png")

    # === 11. Create Animation ===
    print("\n11. Creating trajectory animation...")

    from src.visualization.animation import animate_trajectory

    # Use every 10th frame for speed
    frame_skip = 10

    anim = animate_trajectory(
        positions[::frame_skip],
        attitudes=euler_angles[::frame_skip],
        time=time[::frame_skip],
        save_path='output/flyingwing_hybrid_animation.gif',
        fps=30,
        interval=50
    )

    print("   OK Saved animation: output/flyingwing_hybrid_animation.gif")

    # Show plot (commented out for non-interactive runs)
    # plt.show()

    print()
    print("="*70)
    print("OK Simulation Complete!")
    print("="*70)
    print()
    print("The hybrid XFOIL+AVL model provides:")
    print("  • Accurate viscous drag from XFOIL 2D polars")
    print("  • Reynolds number effects on performance")
    print("  • Accurate stability derivatives from AVL")
    print("  • Proper control effectiveness from AVL")
    print()
    print("This is the most accurate aerodynamic modeling in the framework!")
    print()


if __name__ == '__main__':
    main()
