"""
Flying Wing Test with Proper Trim and Sufficient Power

Uses simplified trim solver to find correct trim conditions,
and increased engine power (250 HP) to sustain Mach 0.5 flight.
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
from src.simulation.trim_simple import find_simple_trim_flying_wing
from src.visualization.plotting import (
    plot_trajectory_3d,
    plot_states_vs_time,
    setup_plotting_style
)

import matplotlib.pyplot as plt


def main():
    """Run flying wing simulation with proper trim."""

    setup_plotting_style()
    os.makedirs('output', exist_ok=True)

    print("=" * 70)
    print("Flying Wing - Proper Trim Test (Mach 0.5, 250 HP)")
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

    # INCREASED POWER: 250 HP (instead of 50 HP)
    prop = PropellerModel(power_max=250.0, prop_diameter=6.0, prop_efficiency=0.75)

    # Find trim
    state_trim, controls_trim, trim_info = find_simple_trim_flying_wing(
        mass, inertia, aero, prop,
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

    state = state_trim.copy()
    combined = CombinedForceModel(aero, prop)

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

        # Fixed controls (trim values)
        def force_func(s):
            return combined(s, controls_trim['throttle'], controls_trim)

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
        title="Flying Wing - Proper Trim (Mach 0.5, 250 HP)",
        show_markers=True,
        marker_interval=500,
        save_path='output/flyingwing_proper_trim_3d.png'
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
        title="Flying Wing - Proper Trim State Variables",
        save_path='output/flyingwing_proper_trim_states.png'
    )
    plt.close(fig2)

    print()
    print("=" * 70)
    print("Output files:")
    print("  - output/flyingwing_proper_trim_3d.png")
    print("  - output/flyingwing_proper_trim_states.png")
    print("=" * 70)
    print()

    if is_stable:
        print("SUCCESS: Flying wing is stable at Mach 0.5!")
    else:
        print("NEEDS WORK: Check plots for remaining issues")


if __name__ == "__main__":
    main()
