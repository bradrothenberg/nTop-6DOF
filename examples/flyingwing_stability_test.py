"""
Flying Wing Stability Test - NO AUTOPILOT

Test if the flying wing is naturally stable by simulating with NO control inputs.
Just let it fly straight and level with trim conditions.
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
from src.visualization.plotting import plot_trajectory_3d, plot_states_vs_time, setup_plotting_style

import matplotlib.pyplot as plt


def test_natural_stability(duration=30.0, dt=0.01):
    """Test natural stability with NO control inputs."""

    print("=" * 70)
    print("Flying Wing Natural Stability Test (NO AUTOPILOT)")
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
    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)

    # FLYING WING AVL DERIVATIVES
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
    aero.Cm_q = -0.347072      # Good damping

    aero.Cn_beta = -0.000119
    aero.Cn_p = -0.000752
    aero.Cn_r = -0.001030

    prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)
    combined_model = CombinedForceModel(aero, prop)

    # Initial state - trimmed flight
    # Trim alpha ≈ Cm_0 / Cm_alpha = 0.000061 / 0.079668 ≈ 0.04° ≈ 0°
    state = State()
    state.position = np.array([0.0, 0.0, -5000.0])
    state.velocity_body = np.array([200.0, 0.0, 0.0])
    state.set_euler_angles(0, 0.0, 0)  # Zero pitch for trim
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    # Fixed controls (trim values)
    throttle = 0.25  # Constant throttle
    elevon = 0.0     # NO control input

    print(f"Initial conditions:")
    print(f"  Altitude: 5000 ft")
    print(f"  Airspeed: 200 ft/s")
    print(f"  Pitch: 0 deg (TRIM)")
    print(f"  Throttle: {throttle}")
    print(f"  Elevon: {np.degrees(elevon)} deg")
    print()
    print("NO AUTOPILOT - just natural dynamics")
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

        altitude = state.altitude
        atm = StandardAtmosphere(altitude)
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
    print(f"Final State:")
    print(f"  Altitude: {-positions[-1, 2]:.1f} ft (started at 5000 ft)")
    print(f"  Airspeed: {np.linalg.norm(velocities[-1]):.1f} ft/s (started at 200 ft/s)")
    print(f"  Pitch: {np.degrees(euler_angles[-1, 1]):.1f} deg (started at 2 deg)")
    print(f"  Roll: {np.degrees(euler_angles[-1, 0]):.1f} deg (started at 0 deg)")
    print()
    print(f"Deviations:")
    print(f"  Altitude change: {-(positions[-1, 2] - positions[0, 2]):.1f} ft")
    print(f"  Max roll angle: {np.degrees(np.max(np.abs(euler_angles[:, 0]))):.1f} deg")
    print(f"  Max pitch angle: {np.degrees(np.max(euler_angles[:, 1])):.1f} deg")
    print(f"  Max roll rate: {np.degrees(np.max(np.abs(angular_rates[:, 0]))):.2f} deg/s")
    print(f"  Max pitch rate: {np.degrees(np.max(np.abs(angular_rates[:, 1]))):.2f} deg/s")
    print()

    # Check stability
    roll_std = np.std(np.degrees(euler_angles[:, 0]))
    pitch_std = np.std(np.degrees(euler_angles[:, 1]))

    print(f"Stability Metrics:")
    print(f"  Roll std dev: {roll_std:.2f} deg")
    print(f"  Pitch std dev: {pitch_std:.2f} deg")

    if roll_std < 5.0 and pitch_std < 5.0:
        print("  STATUS: STABLE ✓")
    elif roll_std < 30.0 and pitch_std < 20.0:
        print("  STATUS: Mildly unstable (may need control)")
    else:
        print("  STATUS: UNSTABLE")
    print()

    return {
        'time': time,
        'positions': positions,
        'velocities': velocities,
        'euler_angles': euler_angles,
        'angular_rates': angular_rates
    }


def main():
    """Run stability test."""
    setup_plotting_style()
    os.makedirs('output', exist_ok=True)

    results = test_natural_stability(duration=30.0, dt=0.01)

    print("Creating visualizations...")

    # 3D Trajectory
    fig1 = plot_trajectory_3d(
        results['positions'],
        title="Flying Wing Natural Stability (No Autopilot)",
        show_markers=True,
        marker_interval=250,
        save_path='output/flyingwing_stability_test_3d.png'
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
        title="Flying Wing Natural Stability - State Variables",
        save_path='output/flyingwing_stability_test_states.png'
    )
    plt.close(fig2)

    print()
    print("=" * 70)
    print("Output files:")
    print("  - output/flyingwing_stability_test_3d.png")
    print("  - output/flyingwing_stability_test_states.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
