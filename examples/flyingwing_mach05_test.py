"""
Flying Wing Stability Test at Mach 0.5 with Manual Trim

Tests the flying wing starting from manually calculated trim conditions:
- Airspeed: 548.5 ft/s (Mach 0.5 at 5000 ft)
- Alpha: 1.75 deg (for L = W)
- Throttle: Adjusted to balance drag

This bypasses the trim solver and directly tests if the simulation is stable
when started from proper equilibrium conditions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import PropellerModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere
from src.visualization.plotting import (
    plot_trajectory_3d,
    plot_states_vs_time,
    setup_plotting_style
)

import matplotlib.pyplot as plt


def test_mach05_trim(duration=30.0, dt=0.01):
    """
    Test flying wing at Mach 0.5 with manual trim conditions.

    Trim conditions calculated:
    - V = 548.5 ft/s (Mach 0.5)
    - alpha = 1.75 deg (for CL = 0.0432, L = W)
    - theta = 1.75 deg (for level flight)
    - throttle = adjusted for thrust = drag
    """

    print("=" * 70)
    print("Flying Wing Mach 0.5 Test - Manual Trim")
    print("=" * 70)
    print()

    # Aircraft parameters
    mass = 228.924806  # slugs
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
    aero.Cm_alpha = -0.079668  # STABLE!
    aero.Cm_q = -0.347072

    aero.Cn_beta = -0.000119
    aero.Cn_p = -0.000752
    aero.Cn_r = -0.001030

    prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)
    combined_model = CombinedForceModel(aero, prop)

    # Trim conditions for Mach 0.5 at 5000 ft
    altitude = -5000.0
    airspeed = 548.5  # ft/s
    alpha_trim = np.radians(1.75)  # deg
    theta_trim = np.radians(1.75)  # deg (≈ alpha for level flight)

    # Estimate throttle needed for trim
    # At trim: Thrust = Drag
    atm = StandardAtmosphere(altitude)
    aero.rho = atm.density
    q_bar = 0.5 * atm.density * airspeed**2

    CL_trim = aero.CL_0 + aero.CL_alpha * alpha_trim
    CD_trim = aero.CD_0 + aero.CD_alpha * alpha_trim + aero.CD_alpha2 * alpha_trim**2

    L_trim = q_bar * S_ref * CL_trim
    D_trim = q_bar * S_ref * CD_trim

    # Estimate throttle (this is approximate, may need adjustment)
    throttle_trim = min(0.5, max(0.1, D_trim / 500.0))  # Rough estimate

    print(f"Trim Conditions (Manual):")
    print(f"  Altitude: {-altitude:.0f} ft")
    print(f"  Airspeed: {airspeed:.1f} ft/s (Mach 0.5)")
    print(f"  Alpha: {np.degrees(alpha_trim):.2f} deg")
    print(f"  Theta: {np.degrees(theta_trim):.2f} deg")
    print(f"  Throttle: {throttle_trim:.3f}")
    print()
    print(f"Expected Forces:")
    print(f"  Lift: {L_trim:.0f} lbf")
    print(f"  Weight: {mass * 32.174:.0f} lbf")
    print(f"  L/W: {L_trim / (mass * 32.174):.4f}")
    print(f"  Drag: {D_trim:.0f} lbf")
    print()

    # Initial state - MANUALLY TRIMMED
    state = State()
    state.position = np.array([0.0, 0.0, altitude])

    # Velocity in body frame at trim alpha
    state.velocity_body = np.array([
        airspeed * np.cos(alpha_trim),
        0.0,
        airspeed * np.sin(alpha_trim)
    ])

    # Attitude: theta = alpha for level flight
    state.set_euler_angles(0.0, theta_trim, 0.0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    # Fixed controls (trim values)
    throttle = throttle_trim
    elevon = 0.0

    print("Starting simulation...")
    print("  NO AUTOPILOT - just natural dynamics")
    print("  Testing if trim conditions remain stable")
    print()

    # Storage
    n_steps = int(duration / dt)
    time = np.zeros(n_steps)
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    euler_angles = np.zeros((n_steps, 3))
    angular_rates = np.zeros((n_steps, 3))

    # Simulate
    for i in range(n_steps):
        time[i] = i * dt

        altitude_current = state.altitude
        atm = StandardAtmosphere(altitude_current)
        aero.rho = atm.density

        # Store data
        positions[i] = state.position
        velocities[i] = state.velocity_body
        euler_angles[i] = state.euler_angles
        angular_rates[i] = state.angular_rates

        # Forces and moments with FIXED controls
        def force_func(s):
            forces, moments = combined_model(s, throttle)
            return forces, moments

        # State derivative
        state_dot_array = dynamics.state_derivative(state, force_func)

        # Euler integration
        x = state.to_array()
        x_new = x + state_dot_array * dt
        state.from_array(x_new)

    print("Simulation complete!")
    print()

    # Analyze results
    alt_final = -positions[-1, 2]
    airspeed_final = np.linalg.norm(velocities[-1])
    pitch_final = np.degrees(euler_angles[-1, 1])
    roll_final = np.degrees(euler_angles[-1, 0])

    alt_change = alt_final - (-altitude)
    airspeed_change = airspeed_final - airspeed

    print(f"Final State:")
    print(f"  Altitude: {alt_final:.1f} ft (change: {alt_change:+.1f} ft)")
    print(f"  Airspeed: {airspeed_final:.1f} ft/s (change: {airspeed_change:+.1f} ft/s)")
    print(f"  Pitch: {pitch_final:.2f} deg")
    print(f"  Roll: {roll_final:.2f} deg")
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

    # Check if stable
    is_stable = (
        abs(alt_change) < 500 and
        abs(airspeed_change) < 50 and
        roll_std < 10 and
        pitch_std < 5
    )

    if is_stable:
        print("STATUS: STABLE - Aircraft maintains trim!")
        print("  - Altitude deviation < 500 ft")
        print("  - Airspeed deviation < 50 ft/s")
        print("  - Attitude stable")
    else:
        print("STATUS: UNSTABLE - Trim not maintained")
        if abs(alt_change) >= 500:
            print(f"  - Large altitude change: {alt_change:+.1f} ft")
        if abs(airspeed_change) >= 50:
            print(f"  - Large airspeed change: {airspeed_change:+.1f} ft/s")
        if roll_std >= 10:
            print(f"  - Roll oscillations: {roll_std:.1f} deg std")
        if pitch_std >= 5:
            print(f"  - Pitch oscillations: {pitch_std:.1f} deg std")

    print()

    return {
        'time': time,
        'positions': positions,
        'velocities': velocities,
        'euler_angles': euler_angles,
        'angular_rates': angular_rates,
        'is_stable': is_stable
    }


def main():
    """Run Mach 0.5 trim test."""
    setup_plotting_style()
    os.makedirs('output', exist_ok=True)

    results = test_mach05_trim(duration=30.0, dt=0.01)

    print("Creating visualizations...")

    # 3D Trajectory
    fig1 = plot_trajectory_3d(
        results['positions'],
        title="Flying Wing Mach 0.5 Trim Test (Manual)",
        show_markers=True,
        marker_interval=250,
        save_path='output/flyingwing_mach05_trim_3d.png'
    )
    plt.close(fig1)

    # State Variables
    states = {
        'position': results['positions'],
        'velocity': results['velocities'],
        'euler_angles': results['euler_angles'],
        'angular_rates': results['angular_rates']
    }
    fig2 = plot_states_vs_time(
        results['time'],
        states,
        title="Flying Wing Mach 0.5 - State Variables",
        save_path='output/flyingwing_mach05_trim_states.png'
    )
    plt.close(fig2)

    print()
    print("=" * 70)
    print("Output files:")
    print("  - output/flyingwing_mach05_trim_3d.png")
    print("  - output/flyingwing_mach05_trim_states.png")
    print("=" * 70)
    print()

    if results['is_stable']:
        print("SUCCESS: Flying wing is stable at Mach 0.5!")
    else:
        print("NEEDS WORK: Check plots to diagnose remaining issues")


if __name__ == "__main__":
    main()
