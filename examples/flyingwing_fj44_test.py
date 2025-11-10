"""
Flying Wing Test with FJ-44 Turbofan Engine

Replaces the 50 HP propeller with an FJ-44-4A turbofan (1900 lbf thrust).
This should provide sufficient thrust for Mach 0.5 cruise.
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

    # Step 3: Calculate drag and required throttle
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

    # Calculate throttle for T = D
    # For turbofan, thrust is relatively constant with speed
    # T = thrust_max * throttle * altitude_factor * velocity_factor

    # Get max thrust available at this condition
    thrust_max_available, _ = turbofan.compute_thrust(state_trim, throttle=1.0)
    T_max = thrust_max_available[0]

    # Required throttle
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

    # Verify trim with full dynamics
    combined = CombinedForceModel(aero, turbofan)
    dynamics = AircraftDynamics(mass, inertia)

    def force_func(s):
        return combined(s, throttle_trim, {})

    state_dot = dynamics.state_derivative(state_trim, force_func)
    vel_dot = state_dot[3:6]
    omega_dot = state_dot[10:13]

    forces, moments = combined(state_trim, throttle_trim, {})
    Cm_trim = aero.Cm_0 + aero.Cm_alpha * alpha_trim

    print("Step 4: Verify trim")
    print(f"  Accelerations:")
    print(f"    ax: {vel_dot[0]:.4f} ft/s^2")
    print(f"    az: {vel_dot[2]:.4f} ft/s^2")
    print(f"    q_dot: {np.degrees(omega_dot[1]):.4f} deg/s^2")
    print()
    print(f"  Pitch moment: {moments[1]:.1f} ft-lbf")
    print(f"  Cm at trim: {Cm_trim:.6f}")
    print()

    is_acceptable = (
        abs(vel_dot[2]) < 1.0 and
        abs(vel_dot[0]) < 5.0 and
        T_actual / D_trim > 0.95  # At least 95% of required thrust
    )

    if is_acceptable:
        print("STATUS: ACCEPTABLE TRIM")
        print("  Vertical and forward accelerations near zero")
        print("  Thrust sufficient for cruise")
    else:
        print("STATUS: POOR TRIM")
        if abs(vel_dot[2]) >= 1.0:
            print(f"  - Large vertical accel: {vel_dot[2]:.2f} ft/s^2")
        if abs(vel_dot[0]) >= 5.0:
            print(f"  - Large forward accel: {vel_dot[0]:.2f} ft/s^2")
        if T_actual / D_trim < 0.95:
            print(f"  - Insufficient thrust: T/D = {T_actual/D_trim:.2f}")
    print()

    controls_trim = {'throttle': throttle_trim}

    info = {
        'alpha_deg': np.degrees(alpha_trim),
        'theta_deg': np.degrees(theta_trim),
        'throttle': throttle_trim,
        'CL_trim': CL_trim,
        'CD_trim': CD_trim,
        'Cm_trim': Cm_trim,
        'lift': q_bar * aero.S_ref * CL_trim,
        'drag': D_trim,
        'thrust': T_actual,
        'thrust_max': T_max,
        'pitch_moment': moments[1],
        'vel_dot': vel_dot,
        'omega_dot': omega_dot,
        'is_acceptable': is_acceptable
    }

    return state_trim, controls_trim, info


def main():
    """Run flying wing simulation with FJ-44 turbofan."""

    setup_plotting_style()
    os.makedirs('output', exist_ok=True)

    print("=" * 70)
    print("Flying Wing - FJ-44 Turbofan Test (Mach 0.5)")
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

    aero.Cn_beta = -0.000119
    aero.Cn_p = -0.000752
    aero.Cn_r = -0.001030

    # FJ-44-4A Turbofan (1900 lbf thrust)
    turbofan = TurbofanModel(thrust_max=1900.0, altitude_lapse_rate=0.7)

    # Find trim
    state_trim, controls_trim, trim_info = find_turbofan_trim(
        mass, inertia, aero, turbofan,
        altitude=-5000.0,
        airspeed=548.5
    )

    if not trim_info['is_acceptable']:
        print("WARNING: Trim solver did not find acceptable trim!")
        print("Proceeding anyway...")
        print()

    # Simulate
    duration = 60.0
    dt = 0.01
    n_steps = int(duration / dt)

    time = np.zeros(n_steps)
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    euler_angles = np.zeros((n_steps, 3))
    angular_rates = np.zeros((n_steps, 3))
    throttle_history = np.zeros(n_steps)

    state = state_trim.copy()
    combined = CombinedForceModel(aero, turbofan)

    print("Starting simulation (60s, NO AUTOPILOT)...")
    print()

    for i in range(n_steps):
        time[i] = i * dt

        # Update atmosphere
        atm = StandardAtmosphere(state.altitude)
        aero.rho = atm.density

        # Store data
        positions[i] = state.position
        velocities[i] = state.velocity_body
        euler_angles[i] = state.euler_angles
        angular_rates[i] = state.angular_rates
        throttle_history[i] = controls_trim['throttle']

        # Fixed controls (trim values)
        def force_func(s):
            return combined(s, controls_trim['throttle'], {})

        # State derivative
        state_dot = dynamics.state_derivative(state, force_func)

        # Euler integration
        x = state.to_array()
        x_new = x + state_dot * dt
        state.from_array(x_new)

        # Normalize quaternion
        state.q.normalize()

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

    print(f"Stability Metrics:")
    print(f"  Altitude std dev: {alt_std:.1f} ft")
    print(f"  Roll std dev: {roll_std:.2f} deg")
    print(f"  Pitch std dev: {pitch_std:.2f} deg")
    print()

    # Check stability
    is_stable = (
        abs(alt_final - alt_initial) < 1000 and
        abs(airspeed_final - airspeed_initial) < 100 and
        roll_std < 30 and
        pitch_std < 30
    )

    if is_stable:
        print("STATUS: STABLE")
        print("  Aircraft maintains altitude and airspeed")
    else:
        print("STATUS: UNSTABLE")
        if abs(alt_final - alt_initial) >= 1000:
            print(f"  - Large altitude change: {alt_final - alt_initial:+.1f} ft")
        if abs(airspeed_final - airspeed_initial) >= 100:
            print(f"  - Large airspeed change: {airspeed_final - airspeed_initial:+.1f} ft/s")
        if roll_std >= 30:
            print(f"  - Roll oscillations: {roll_std:.1f} deg std")
        if pitch_std >= 30:
            print(f"  - Pitch oscillations: {pitch_std:.1f} deg std")

    print()

    # Create visualizations
    print("Creating visualizations...")

    fig1 = plot_trajectory_3d(
        positions,
        title="Flying Wing - FJ-44 Turbofan (Mach 0.5)",
        show_markers=True,
        marker_interval=500,
        save_path='output/flyingwing_fj44_3d.png'
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
        title="Flying Wing - FJ-44 State Variables",
        save_path='output/flyingwing_fj44_states.png'
    )
    plt.close(fig2)

    print()
    print("=" * 70)
    print("Output files:")
    print("  - output/flyingwing_fj44_3d.png")
    print("  - output/flyingwing_fj44_states.png")
    print("=" * 70)
    print()

    print("=" * 70)
    print("TRIM SOLUTION SUMMARY:")
    print("=" * 70)
    print(f"  Engine: FJ-44-4A Turbofan")
    print(f"  Max Thrust (SL): 1900 lbf")
    print(f"  Thrust at trim: {trim_info['thrust']:.1f} lbf")
    print(f"  Thrust available: {trim_info['thrust_max']:.1f} lbf")
    print(f"  Throttle: {trim_info['throttle']:.1%}")
    print()
    print(f"  Alpha: {trim_info['alpha_deg']:.4f} deg")
    print(f"  Theta: {trim_info['theta_deg']:.4f} deg")
    print(f"  CL: {trim_info['CL_trim']:.6f}")
    print(f"  CD: {trim_info['CD_trim']:.6f}")
    print(f"  L/D: {trim_info['CL_trim']/trim_info['CD_trim']:.1f}")
    print()
    print(f"  Lift: {trim_info['lift']:.1f} lbf")
    print(f"  Drag: {trim_info['drag']:.1f} lbf")
    print(f"  T/D: {trim_info['thrust']/trim_info['drag']:.3f}")
    print("=" * 70)
    print()

    if is_stable:
        print("SUCCESS: Flying wing is stable with FJ-44 turbofan!")
    else:
        print("NEEDS WORK: Still some instability (likely pitch moment)")


if __name__ == "__main__":
    main()
