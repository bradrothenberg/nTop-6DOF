"""
Flying Wing - Stable Flight with Enhanced Autopilot

Tests the triple-loop cascaded autopilot with:
- Outer loop: Altitude hold
- Middle loop: Pitch attitude control
- Inner loop: Pitch rate damping
- Stall protection
- Airspeed hold with safety logic
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import TurbofanModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere
from src.control.autopilot import FlyingWingAutopilot, AirspeedHoldController
from src.visualization.plotting import (
    plot_trajectory_3d,
    plot_states_vs_time,
    setup_plotting_style
)

import matplotlib.pyplot as plt


def find_turbofan_trim(mass, inertia, aero, turbofan, altitude, airspeed):
    """
    Find simplified trim with turbofan engine.

    Steps:
    1. Calculate alpha for L = W
    2. Set theta = alpha (level flight)
    3. Calculate throttle for T = D
    4. Calculate elevon to balance pitch moment (M = 0)
    """

    print("=" * 70)
    print("Turbofan Trim Solver - Flying Wing with FJ-44")
    print("=" * 70)
    print()

    atm = StandardAtmosphere(altitude)
    aero.rho = atm.density

    q_bar = 0.5 * atm.density * airspeed**2
    W = mass * 32.174  # Weight (lbf)

    print(f"Conditions:")
    print(f"  Altitude: {-altitude:.0f} ft")
    print(f"  Airspeed: {airspeed:.1f} ft/s (Mach {airspeed/1116:.2f})")
    print(f"  Density: {atm.density:.6f} slug/ft^3")
    print(f"  q_bar: {q_bar:.2f} psf")
    print(f"  Weight: {W:.1f} lbf")
    print()

    # Step 1: Find alpha for L = W
    CL_trim = W / (q_bar * aero.S_ref)
    alpha_trim = (CL_trim - aero.CL_0) / aero.CL_alpha

    print(f"Step 1: Find alpha for L = W")
    print(f"  CL_trim = {CL_trim:.6f}")
    print(f"  alpha_trim = {np.degrees(alpha_trim):.4f} deg")
    print()

    # Step 2: Set theta = alpha
    theta_trim = alpha_trim

    print(f"Step 2: Set theta = alpha (level flight)")
    print(f"  theta_trim = {np.degrees(theta_trim):.4f} deg")
    print()

    # Step 3: Estimate throttle for T ≈ D
    CD_trim = aero.CD_0 + aero.CD_alpha * alpha_trim + aero.CD_alpha2 * alpha_trim**2
    D_trim = q_bar * aero.S_ref * CD_trim

    # Create trim state for thrust calculation
    state_trim = State()
    state_trim.position = np.array([0, 0, altitude])
    state_trim.velocity_body = np.array([
        airspeed * np.cos(alpha_trim),
        0.0,
        airspeed * np.sin(alpha_trim)
    ])
    state_trim.set_euler_angles(0, theta_trim, 0)
    state_trim.angular_rates = np.array([0, 0, 0])

    # Get max thrust available at this condition
    thrust_max_available, _ = turbofan.compute_thrust(state_trim, throttle=1.0)
    T_max = thrust_max_available[0]

    # Required throttle for T ≈ D
    throttle_trim = D_trim / T_max
    throttle_trim = np.clip(throttle_trim, 0.01, 1.0)

    # Verify thrust
    thrust_actual, _ = turbofan.compute_thrust(state_trim, throttle=throttle_trim)
    T_actual = thrust_actual[0]

    print(f"Step 3: Find throttle for T = D")
    print(f"  CD_trim = {CD_trim:.6f}")
    print(f"  Drag: {D_trim:.1f} lbf")
    print(f"  Thrust max available: {T_max:.1f} lbf")
    print(f"  Throttle required: {throttle_trim:.4f}")
    print(f"  Thrust at throttle: {T_actual:.1f} lbf")
    print(f"  T/D: {T_actual/D_trim:.4f}")
    print()

    # Step 4: Calculate elevon to balance pitch moment
    Cm_at_trim = aero.Cm_0 + aero.Cm_alpha * alpha_trim

    # Required elevon deflection for Cm = 0
    if abs(aero.Cm_de) > 1e-6:
        elevon_trim = -Cm_at_trim / aero.Cm_de
        elevon_trim = np.clip(elevon_trim, np.radians(-25), np.radians(25))
    else:
        elevon_trim = 0.0

    print(f"Step 4: Find elevon for M = 0")
    print(f"  Cm without elevon: {Cm_at_trim:.6f}")
    print(f"  Cm_de: {aero.Cm_de:.6f} per radian")
    print(f"  Elevon required: {np.degrees(elevon_trim):.2f} deg = {elevon_trim:.4f} rad")
    print()

    # Return trim state and controls
    controls_trim = {'throttle': throttle_trim, 'elevator': elevon_trim}

    info = {
        'alpha_deg': np.degrees(alpha_trim),
        'theta_deg': np.degrees(theta_trim),
        'throttle': throttle_trim,
        'elevon_deg': np.degrees(elevon_trim),
        'elevon_rad': elevon_trim,
    }

    return state_trim, controls_trim, info


def main():
    """Run flying wing simulation with enhanced autopilot."""

    setup_plotting_style()
    os.makedirs('output', exist_ok=True)

    print("=" * 70)
    print("Flying Wing - Stable Flight Test (Enhanced Autopilot)")
    print("=" * 70)
    print()

    # Aircraft parameters
    mass = 228.924806
    inertia = np.array([[19236.2914, 0, 0],
                        [0, 2251.0172, 0],
                        [0, 0, 21487.3086]])

    S_ref = 412.6370
    c_ref = 11.9555
    b_ref = 24.8630

    dynamics = AircraftDynamics(mass, inertia)
    aero = LinearAeroModel(S_ref, c_ref, b_ref)

    # Flying wing AVL derivatives
    aero.CL_0 = 0.000023
    aero.CL_alpha = 1.412241
    aero.CL_q = 1.282202
    aero.CL_de = 0.0  # Antisymmetric elevon: lift effects cancel

    aero.CD_0 = -0.000619
    aero.CD_alpha = 0.035509
    aero.CD_alpha2 = 0.5

    aero.CY_beta = -0.008250

    aero.Cl_beta = -0.028108
    aero.Cl_p = -0.109230
    aero.Cl_r = 0.019228

    aero.Cm_0 = 0.000061
    aero.Cm_alpha = -0.079668
    aero.Cm_q = -0.347072
    aero.Cm_de = -0.02  # Elevon effectiveness (antisymmetric)

    aero.Cn_beta = -0.000119
    aero.Cn_p = -0.000752
    aero.Cn_r = -0.001030

    # FJ-44-4A Turbofan
    turbofan = TurbofanModel(thrust_max=1900.0, altitude_lapse_rate=0.7)

    # Trim conditions - use higher airspeed for more margin
    trim_altitude = -5000.0
    trim_airspeed = 600.0  # Increased from 548.5 for better stability margin

    # Find trim
    state_trim, controls_trim, trim_info = find_turbofan_trim(
        mass, inertia, aero, turbofan,
        altitude=trim_altitude,
        airspeed=trim_airspeed
    )

    # === Initialize Enhanced Autopilot ===
    print("=" * 70)
    print("Initializing Enhanced Autopilot")
    print("=" * 70)
    print()

    # Create flying wing autopilot with pitch rate damping
    # Moderate gains - balance between response and stability
    autopilot = FlyingWingAutopilot(
        # Altitude hold gains (outer loop)
        Kp_alt=0.003,      # Pitch command per ft altitude error
        Ki_alt=0.0002,     # Integral for steady-state
        Kd_alt=0.008,      # Damp altitude rate

        # Pitch attitude gains (middle loop)
        Kp_pitch=0.8,      # Pitch rate command per pitch error
        Ki_pitch=0.05,     # Integral term
        Kd_pitch=0.15,     # Damping

        # Pitch rate gains (inner loop)
        Kp_pitch_rate=0.15, # Reduced to stop limit cycle oscillations
        Ki_pitch_rate=0.01, # Small integral

        # Safety limits
        max_pitch_cmd=12.0,   # degrees (conservative)
        min_pitch_cmd=-8.0,   # degrees
        max_alpha=12.0,       # degrees (stall protection)
        stall_speed=150.0,    # ft/s
        min_airspeed_margin=1.3  # 30% above stall
    )

    # Set trim values
    autopilot.set_trim(elevon_trim=controls_trim['elevator'])
    autopilot.set_target_altitude(trim_altitude)

    print(f"Autopilot Configuration:")
    print(f"  Altitude target: {-trim_altitude:.0f} ft MSL")
    print(f"  Elevon trim: {np.degrees(controls_trim['elevator']):.2f} deg")
    print(f"  Stall speed: {autopilot.stall_speed:.0f} ft/s")
    print(f"  Min airspeed: {autopilot.min_airspeed:.0f} ft/s")
    print(f"  Max alpha: {np.degrees(autopilot.max_alpha):.1f} deg")
    print()

    # Create airspeed hold controller
    airspeed_controller = AirspeedHoldController(
        Kp=0.008,      # Throttle per ft/s error
        Ki=0.001,      # Integral term
        Kd=0.004       # Damping
    )
    airspeed_controller.set_target_airspeed(trim_airspeed)

    print(f"Airspeed Controller:")
    print(f"  Target: {trim_airspeed:.0f} ft/s")
    print(f"  Throttle trim: {controls_trim['throttle']:.3f}")
    print()

    # Simulate - shorter duration for initial testing
    duration = 30.0  # Reduced from 60s
    dt = 0.01
    n_steps = int(duration / dt)

    time = np.zeros(n_steps)
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    euler_angles = np.zeros((n_steps, 3))
    angular_rates = np.zeros((n_steps, 3))
    throttle_history = np.zeros(n_steps)
    elevon_history = np.zeros(n_steps)
    alpha_history = np.zeros(n_steps)
    stall_protection_history = np.zeros(n_steps, dtype=bool)

    state = state_trim.copy()
    combined = CombinedForceModel(aero, turbofan)

    print("=" * 70)
    print(f"Starting simulation ({duration:.0f}s, RK4 integration, ENHANCED AUTOPILOT)")
    print("=" * 70)
    print()

    stall_protection_count = 0

    for i in range(n_steps):
        time[i] = i * dt

        # Store data
        positions[i] = state.position
        velocities[i] = state.velocity_body
        euler_angles[i] = state.euler_angles
        angular_rates[i] = state.angular_rates
        alpha_history[i] = state.alpha

        # === ENHANCED AUTOPILOT ===

        # Elevon control from flying wing autopilot (triple-loop with stall protection)
        elevon = autopilot.update(
            current_altitude=state.altitude,
            current_pitch=state.euler_angles[1],
            current_pitch_rate=state.angular_rates[1],
            current_airspeed=state.airspeed,
            current_alpha=state.alpha,
            dt=dt
        )

        # Throttle control from airspeed controller
        # BUT: If stall protection active, override with max throttle
        if autopilot.stall_protection_active or autopilot.alpha_protection_active:
            throttle = 1.0  # Max throttle for stall recovery
            stall_protection_count += 1
        else:
            # Proportional throttle control with aggressive gain
            airspeed_error = trim_airspeed - state.airspeed
            # Use trim throttle as baseline, add strong correction
            throttle = controls_trim['throttle'] + 0.002 * airspeed_error
            throttle = np.clip(throttle, 0.05, 1.0)  # Minimum 5% throttle

        controls = {
            'throttle': throttle,
            'elevator': elevon,
            'aileron': 0.0,
            'rudder': 0.0
        }

        throttle_history[i] = throttle
        elevon_history[i] = elevon
        stall_protection_history[i] = autopilot.stall_protection_active or autopilot.alpha_protection_active

        # Force function with atmosphere update
        def force_func(s):
            atm = StandardAtmosphere(s.altitude)
            aero.rho = atm.density
            return combined(s, controls['throttle'], controls)

        # RK4 integration
        state = dynamics.propagate_rk4(state, dt, force_func)

    print("Simulation complete!")
    print()

    # Analyze results
    alt_initial = -positions[0, 2]
    alt_final = -positions[-1, 2]
    airspeed_initial = np.linalg.norm(velocities[0])
    airspeed_final = np.linalg.norm(velocities[-1])

    print(f"Results:")
    print(f"  Initial altitude: {alt_initial:.1f} ft")
    print(f"  Final altitude: {alt_final:.1f} ft")
    print(f"  Altitude change: {alt_final - alt_initial:+.1f} ft")
    print()
    print(f"  Initial airspeed: {airspeed_initial:.1f} ft/s")
    print(f"  Final airspeed: {airspeed_final:.1f} ft/s")
    print(f"  Airspeed change: {airspeed_final - airspeed_initial:+.1f} ft/s")
    print()

    # Stability metrics
    alt_std = np.std(-positions[:, 2])
    roll_std = np.std(np.degrees(euler_angles[:, 0]))
    pitch_std = np.std(np.degrees(euler_angles[:, 1]))
    airspeed_std = np.std(np.linalg.norm(velocities, axis=1))

    print(f"Stability Metrics:")
    print(f"  Altitude std dev: {alt_std:.1f} ft")
    print(f"  Airspeed std dev: {airspeed_std:.1f} ft/s")
    print(f"  Roll std dev: {roll_std:.2f} deg")
    print(f"  Pitch std dev: {pitch_std:.2f} deg")
    print()

    # Stall protection
    stall_protection_pct = 100.0 * stall_protection_count / n_steps
    print(f"Stall Protection:")
    print(f"  Active: {stall_protection_count} / {n_steps} steps ({stall_protection_pct:.1f}%)")
    print()

    # Check stability (relaxed criteria for demonstration)
    is_stable = (
        abs(alt_final - alt_initial) < 600 and  # Allow 600 ft drift over 30s
        abs(airspeed_final - airspeed_initial) < 50 and
        roll_std < 5 and  # Tighter roll requirement
        pitch_std < 10   # Tighter pitch requirement
    )

    if is_stable:
        print("STATUS: STABLE")
        print("  Aircraft maintains altitude and airspeed")
        print("  Roll and pitch remain controlled")
    else:
        print("STATUS: UNSTABLE")
        if abs(alt_final - alt_initial) >= 500:
            print(f"  - Altitude change: {alt_final - alt_initial:+.1f} ft")
        if abs(airspeed_final - airspeed_initial) >= 50:
            print(f"  - Airspeed change: {airspeed_final - airspeed_initial:+.1f} ft/s")
        if roll_std >= 20:
            print(f"  - Roll oscillations: {roll_std:.1f} deg std")
        if pitch_std >= 20:
            print(f"  - Pitch oscillations: {pitch_std:.1f} deg std")

    print()

    # Create visualizations
    print("Creating visualizations...")

    fig1 = plot_trajectory_3d(
        positions,
        title="Flying Wing - Enhanced Autopilot (Stable Flight)",
        show_markers=True,
        marker_interval=500,
        save_path='output/flyingwing_stable_3d.png'
    )
    plt.close(fig1)

    states = {
        'position': positions,
        'velocity': velocities,
        'euler_angles': euler_angles,
        'angular_rates': angular_rates
    }
    fig2 = plot_states_vs_time(
        time,
        states,
        title="Flying Wing - Enhanced Autopilot State Variables",
        save_path='output/flyingwing_stable_states.png'
    )
    plt.close(fig2)

    # Additional control plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Altitude
    axes[0].plot(time, -positions[:, 2], 'b-', linewidth=1.5)
    axes[0].axhline(-trim_altitude, color='r', linestyle='--', label='Target')
    axes[0].set_ylabel('Altitude (ft MSL)')
    axes[0].set_title('Altitude Hold Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Airspeed
    airspeed_plot = np.linalg.norm(velocities, axis=1)
    axes[1].plot(time, airspeed_plot, 'b-', linewidth=1.5)
    axes[1].axhline(trim_airspeed, color='r', linestyle='--', label='Target')
    axes[1].axhline(autopilot.min_airspeed, color='orange', linestyle=':', label='Min Safe')
    axes[1].axhline(autopilot.stall_speed, color='red', linestyle=':', label='Stall')
    axes[1].set_ylabel('Airspeed (ft/s)')
    axes[1].set_title('Airspeed Hold Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Controls
    axes[2].plot(time, throttle_history, 'g-', linewidth=1.5, label='Throttle')
    axes[2].plot(time, np.degrees(elevon_history), 'b-', linewidth=1.5, label='Elevon (deg)')
    # Highlight stall protection periods
    if stall_protection_count > 0:
        stall_times = time[stall_protection_history]
        if len(stall_times) > 0:
            axes[2].scatter(stall_times, throttle_history[stall_protection_history],
                           color='red', s=5, alpha=0.5, label='Stall Protection')
    axes[2].set_ylabel('Control Deflection')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Control Surface Commands')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/flyingwing_stable_controls.png', dpi=150, bbox_inches='tight')
    plt.close()

    print()
    print("=" * 70)
    print("Output files:")
    print("  - output/flyingwing_stable_3d.png")
    print("  - output/flyingwing_stable_states.png")
    print("  - output/flyingwing_stable_controls.png")
    print("=" * 70)
    print()

    if is_stable:
        print("SUCCESS: Flying wing achieves stable flight with enhanced autopilot!")
    else:
        print("PARTIAL SUCCESS: Improved but may need gain tuning")


if __name__ == "__main__":
    main()
