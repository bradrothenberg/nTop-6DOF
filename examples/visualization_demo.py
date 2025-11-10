"""
Visualization Demonstration

Demonstrates the visualization capabilities of the nTop 6-DOF Flight Dynamics Framework.
Shows plotting and animation of flight data.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.quaternion import Quaternion
from src.core.dynamics import AircraftDynamics
from src.core.integrator import RK4Integrator
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import PropellerModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere
from src.control.autopilot import AltitudeHoldController, HeadingHoldController, AirspeedHoldController
from src.visualization.plotting import (
    plot_trajectory_3d,
    plot_states_vs_time,
    plot_controls_vs_time,
    plot_forces_moments,
    setup_plotting_style
)
from src.visualization.animation import animate_trajectory

import matplotlib.pyplot as plt


def simulate_maneuver(duration=60.0, dt=0.01):
    """
    Simulate a multi-axis maneuver with autopilot control.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds
    dt : float
        Time step in seconds

    Returns
    -------
    dict
        Dictionary containing simulation results
    """
    # Aircraft parameters (nTop UAV)
    mass = 234.8  # slugs
    inertia = np.array([
        [14908.4, 0, 0],
        [0, 2318.4, 0],
        [0, 0, 17226.9]
    ])

    # Reference geometry
    S_ref = 199.94  # ft²
    b_ref = 19.890  # ft
    c_ref = 26.689  # ft

    # Create models
    dynamics = AircraftDynamics(mass, inertia)

    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)
    aero.CL_0 = 0.2
    aero.CL_alpha = 4.5
    aero.CD_0 = 0.03
    aero.CD_alpha2 = 0.8
    aero.Cm_0 = 0.0
    aero.Cm_alpha = -0.6
    aero.Cm_q = -8.0
    aero.Cm_elevator = -1.2  # Elevator effectiveness
    aero.CY_beta = -0.3
    aero.Cl_beta = -0.1
    aero.Cl_p = -0.5
    aero.Cl_aileron = 0.15  # Aileron effectiveness
    aero.Cn_beta = 0.08
    aero.Cn_r = -0.2
    aero.Cn_aileron = -0.02

    prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)

    combined_model = CombinedForceModel(aero, prop)

    # Autopilots with more conservative gains
    alt_ctrl = AltitudeHoldController(Kp_alt=0.0015, Ki_alt=0.0002, Kd_alt=0.006,
                                      Kp_pitch=3.0, Ki_pitch=0.8, Kd_pitch=0.2)
    hdg_ctrl = HeadingHoldController(Kp_heading=0.8, Ki_heading=0.1, Kd_heading=0.15,
                                     Kp_roll=2.5, Ki_roll=0.5, Kd_roll=0.15)
    spd_ctrl = AirspeedHoldController(Kp=0.015, Ki=0.002, Kd=0.08)

    # Initial state
    state = State()
    state.position = np.array([0.0, 0.0, -5000.0])  # Start at 5000 ft
    state.velocity_body = np.array([200.0, 0.0, 0.0])  # 200 ft/s forward
    state.set_euler_angles(0, np.radians(2), 0)  # 2° pitch
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    # Use smaller time step for stability

    # Storage
    n_steps = int(duration / dt)
    time = np.zeros(n_steps)
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    euler_angles = np.zeros((n_steps, 3))
    angular_rates = np.zeros((n_steps, 3))
    controls_elevator = np.zeros(n_steps)
    controls_aileron = np.zeros(n_steps)
    controls_rudder = np.zeros(n_steps)
    controls_throttle = np.zeros(n_steps)
    forces_history = np.zeros((n_steps, 3))
    moments_history = np.zeros((n_steps, 3))

    # Autopilot targets
    alt_ctrl.set_target_altitude(6000.0)  # Climb to 6000 ft
    hdg_ctrl.set_target_heading(np.radians(90))  # Turn to 90° (East)
    spd_ctrl.set_target_airspeed(220.0)  # Increase speed to 220 ft/s

    print("Starting simulation...")
    print(f"Duration: {duration:.1f} s, dt: {dt} s")
    print(f"Initial altitude: 5000 ft")
    print(f"Target altitude: 6000 ft")
    print(f"Target heading: 90°")
    print(f"Target airspeed: 220 ft/s")
    print()

    # Simulate
    for i in range(n_steps):
        time[i] = i * dt

        # Atmosphere
        altitude = state.altitude
        atm = StandardAtmosphere(altitude)
        aero.rho = atm.density

        # Autopilot commands
        roll, pitch, yaw = state.euler_angles
        airspeed = state.airspeed

        elevator = alt_ctrl.update(altitude, pitch, dt)
        aileron = hdg_ctrl.update(yaw, roll, dt)
        rudder = 0.0  # Simple coordinated turn
        throttle = spd_ctrl.update(airspeed, dt)

        # Clamp controls
        elevator = np.clip(elevator, np.radians(-20), np.radians(20))
        aileron = np.clip(aileron, np.radians(-20), np.radians(20))
        throttle = np.clip(throttle, 0.0, 1.0)

        # Store data
        positions[i] = state.position
        velocities[i] = state.velocity_body
        euler_angles[i] = state.euler_angles
        angular_rates[i] = state.angular_rates
        controls_elevator[i] = elevator
        controls_aileron[i] = aileron
        controls_rudder[i] = rudder
        controls_throttle[i] = throttle

        # Compute forces and moments with control inputs
        def force_func_with_controls(s):
            forces, moments = combined_model(s, throttle)

            # Add control surface effects
            q_bar = atm.get_dynamic_pressure(s.airspeed)
            moments[1] += q_bar * S_ref * c_ref * aero.Cm_elevator * elevator
            moments[0] += q_bar * S_ref * b_ref * aero.Cl_aileron * aileron

            return forces, moments

        # Get forces and moments for storage
        F, M = force_func_with_controls(state)
        forces_history[i] = F
        moments_history[i] = M

        # State derivative
        state_dot_array = dynamics.state_derivative(state, force_func_with_controls)

        # Simple Euler integration
        x = state.to_array()
        x_new = x + state_dot_array * dt

        # Update state
        state.from_array(x_new)

    print("Simulation complete!")
    print(f"Final altitude: {-positions[-1, 2]:.1f} ft")
    print(f"Final heading: {np.degrees(euler_angles[-1, 2]):.1f}°")
    print(f"Final airspeed: {np.linalg.norm(velocities[-1]):.1f} ft/s")
    print()

    # Return results
    return {
        'time': time,
        'positions': positions,
        'velocities': velocities,
        'euler_angles': euler_angles,
        'angular_rates': angular_rates,
        'controls': {
            'elevator': controls_elevator,
            'aileron': controls_aileron,
            'rudder': controls_rudder,
            'throttle': controls_throttle
        },
        'forces': forces_history,
        'moments': moments_history
    }


def main():
    """Run visualization demonstration."""
    print("=" * 70)
    print("Phase 6: Visualization Demonstration")
    print("=" * 70)
    print()

    # Set up plotting style
    setup_plotting_style()

    # Run simulation (using default dt=0.01 for stability)
    results = simulate_maneuver(duration=60.0)

    print("Creating visualizations...")
    print()

    # 1. 3D Trajectory
    print("1. Plotting 3D trajectory...")
    fig1 = plot_trajectory_3d(
        results['positions'],
        title="Flight Path - Climb and Turn Maneuver",
        show_markers=True,
        marker_interval=50
    )

    # 2. State Variables
    print("2. Plotting state variables...")
    states_dict = {
        'position': results['positions'],
        'velocity': results['velocities'],
        'euler_angles': results['euler_angles'],
        'angular_rates': results['angular_rates']
    }
    fig2 = plot_states_vs_time(
        results['time'],
        states_dict,
        title="State Variables - Climb and Turn Maneuver"
    )

    # 3. Control Inputs
    print("3. Plotting control inputs...")
    fig3 = plot_controls_vs_time(
        results['time'],
        results['controls'],
        title="Control Inputs - Climb and Turn Maneuver"
    )

    # 4. Forces and Moments
    print("4. Plotting forces and moments...")
    fig4 = plot_forces_moments(
        results['time'],
        results['forces'],
        results['moments'],
        title="Forces and Moments - Climb and Turn Maneuver"
    )

    print()
    print("All plots created successfully!")
    print()

    # Save all figures
    print("Saving figures...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'trajectory_3d.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'state_variables.png'), dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'control_inputs.png'), dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'forces_moments.png'), dpi=150, bbox_inches='tight')

    print(f"Figures saved to: {output_dir}")
    print()

    # Create animation (optional - can be slow)
    create_anim = input("Create animation? (y/n): ").lower() == 'y'

    if create_anim:
        print()
        print("Creating animation (this may take a minute)...")

        # Subsample for faster animation
        subsample = 5
        positions_subsampled = results['positions'][::subsample]
        attitudes_subsampled = results['euler_angles'][::subsample]
        time_subsampled = results['time'][::subsample]

        anim_path = os.path.join(output_dir, 'trajectory_animation.gif')
        animate_trajectory(
            positions_subsampled,
            attitudes=attitudes_subsampled,
            time=time_subsampled,
            save_path=anim_path,
            fps=20,
            dpi=100
        )

        print(f"Animation saved to: {anim_path}")
        print()

    print("=" * 70)
    print("Visualization demonstration complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - 4 plots created")
    print(f"  - {len(results['time'])} simulation steps")
    print(f"  - {results['time'][-1]:.1f} seconds simulated")
    if create_anim:
        print(f"  - Animation created")
    print()


if __name__ == "__main__":
    main()
