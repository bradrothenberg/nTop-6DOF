"""
Flying Wing Demonstration

Demonstrates proper flying wing simulation with:
- Flaperon control model (roll + pitch + lift + drag coupling)
- Ixz product of inertia
- Flying wing coupling derivatives
- Improved stability
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import FlyingWingAeroModel
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


def simulate_flying_wing(duration=60.0, dt=0.01):
    """
    Simulate flying wing UAV with proper flaperon model and inertia.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds
    dt : float
        Time step (seconds)

    Returns
    -------
    dict
        Simulation results
    """
    print("Initializing flying wing simulation...")
    print()

    # Aircraft parameters
    mass = 234.8  # slugs (7,555 lbm)

    # Inertia tensor WITH Ixz product of inertia
    Ixx = 14908.4
    Iyy = 2318.4
    Izz = 17226.9
    Ixz = 0.10 * np.sqrt(Ixx * Izz)  # 10% coupling, ~1602.6 slug-ft^2

    inertia = np.array([[Ixx, 0, Ixz],
                        [0, Iyy, 0],
                        [Ixz, 0, Izz]])

    print(f"Aircraft Configuration:")
    print(f"  Mass: {mass:.1f} slugs ({mass * 32.174:.0f} lbm)")
    print(f"  Ixx: {Ixx:.1f} slug-ft^2")
    print(f"  Iyy: {Iyy:.1f} slug-ft^2")
    print(f"  Izz: {Izz:.1f} slug-ft^2")
    print(f"  Ixz: {Ixz:.1f} slug-ft^2 (NEW - pitch-roll coupling)")
    print()

    # Reference geometry (highly swept flying wing)
    S_ref = 199.94  # ft²
    b_ref = 19.890  # ft
    c_ref = 26.689  # ft
    AR = b_ref**2 / S_ref

    print(f"Wing Geometry:")
    print(f"  Span: {b_ref:.2f} ft")
    print(f"  Area: {S_ref:.2f} ft²")
    print(f"  Mean Chord: {c_ref:.2f} ft")
    print(f"  Aspect Ratio: {AR:.3f} (delta planform)")
    print()

    # Create models
    dynamics = AircraftDynamics(mass, inertia)

    # Use FlyingWingAeroModel with flaperon effects
    aero = FlyingWingAeroModel(S_ref, c_ref, b_ref, rho=0.002377, Ixz_factor=0.10)

    # Set force derivatives
    aero.CL_0 = 0.2
    aero.CL_alpha = 4.5
    aero.CD_0 = 0.03
    aero.CD_alpha2 = 0.8

    # Moment derivatives
    aero.Cm_0 = 0.0
    aero.Cm_alpha = -0.6
    aero.Cm_q = -8.0

    # Lateral derivatives
    aero.CY_beta = -0.3
    aero.Cl_beta = -0.1
    aero.Cl_p = -0.5
    aero.Cn_beta = 0.08
    aero.Cn_r = -0.2

    # Flaperon derivatives (set in FlyingWingAeroModel, but can override)
    aero.Cl_flaperon = 0.15     # Roll effectiveness
    aero.Cm_flaperon = -0.08    # Nose-down pitch coupling
    aero.CL_flaperon = 0.30     # Lift increase
    aero.CD_flaperon = 0.10     # Drag penalty

    # Flying wing coupling (already set in FlyingWingAeroModel)
    # aero.Cl_alpha = -0.08
    # aero.Cm_p = -0.10
    # aero.Cm_r = -0.04
    # aero.Cn_p = -0.02

    print("Aerodynamic Model: FlyingWingAeroModel")
    print("  Flaperon control: Roll + Pitch + Lift + Drag")
    print("  Coupling derivatives: Cl_alpha, Cm_p, Cm_r, Cn_p")
    print()

    # Propulsion
    prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)
    combined_model = CombinedForceModel(aero, prop)

    # Autopilots with reduced gains (flying wing is more sensitive)
    alt_ctrl = AltitudeHoldController(
        Kp_alt=0.0010,   # Reduced from 0.0015
        Ki_alt=0.0001,   # Reduced from 0.0002
        Kd_alt=0.004,    # Reduced from 0.006
        Kp_pitch=2.0,    # Reduced from 3.0
        Ki_pitch=0.5,    # Reduced from 0.8
        Kd_pitch=0.15    # Reduced from 0.2
    )

    hdg_ctrl = HeadingHoldController(
        Kp_heading=0.5,   # Reduced from 0.8
        Ki_heading=0.05,  # Reduced from 0.1
        Kd_heading=0.10,  # Reduced from 0.15
        Kp_roll=1.5,      # Reduced from 2.5
        Ki_roll=0.3,      # Reduced from 0.5
        Kd_roll=0.10      # Reduced from 0.15
    )

    spd_ctrl = AirspeedHoldController(
        Kp=0.010,   # Reduced from 0.015
        Ki=0.001,   # Reduced from 0.002
        Kd=0.05     # Reduced from 0.08
    )

    print("Autopilot Configuration:")
    print("  Reduced gains for flying wing sensitivity")
    print("  Altitude hold: 5000 -> 6000 ft")
    print("  Heading hold: 0 -> 90 deg")
    print("  Airspeed hold: 200 -> 220 ft/s")
    print()

    # Initial state
    state = State()
    state.position = np.array([0.0, 0.0, -5000.0])
    state.velocity_body = np.array([200.0, 0.0, 0.0])
    state.set_euler_angles(0, np.radians(2), 0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    # Storage
    n_steps = int(duration / dt)
    time = np.zeros(n_steps)
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    euler_angles = np.zeros((n_steps, 3))
    angular_rates = np.zeros((n_steps, 3))
    controls_flaperon = np.zeros(n_steps)
    controls_elevator = np.zeros(n_steps)
    controls_rudder = np.zeros(n_steps)
    controls_throttle = np.zeros(n_steps)
    forces_history = np.zeros((n_steps, 3))
    moments_history = np.zeros((n_steps, 3))

    # Autopilot targets
    alt_ctrl.set_target_altitude(6000.0)
    hdg_ctrl.set_target_heading(np.radians(90))
    spd_ctrl.set_target_airspeed(220.0)

    print("Starting simulation...")
    print(f"Duration: {duration:.1f} s, dt: {dt} s")
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

        # Altitude control (outputs elevator command)
        elevator = alt_ctrl.update(altitude, pitch, dt)

        # Heading control (outputs flaperon command for roll)
        flaperon = hdg_ctrl.update(yaw, roll, dt)

        # Airspeed control
        rudder = 0.0
        throttle = spd_ctrl.update(airspeed, dt)

        # Clamp controls
        elevator = np.clip(elevator, np.radians(-20), np.radians(20))
        flaperon = np.clip(flaperon, np.radians(-20), np.radians(20))
        throttle = np.clip(throttle, 0.0, 1.0)

        # Store data
        positions[i] = state.position
        velocities[i] = state.velocity_body
        euler_angles[i] = state.euler_angles
        angular_rates[i] = state.angular_rates
        controls_flaperon[i] = flaperon
        controls_elevator[i] = elevator
        controls_rudder[i] = rudder
        controls_throttle[i] = throttle

        # Compute forces and moments with control inputs
        def force_func_with_controls(s):
            # Pass controls to the aero model (it handles flaperon coupling)
            controls_dict = {
                'flaperon': flaperon,
                'elevator': elevator,
                'rudder': rudder
            }

            forces, moments = aero.compute_forces_moments(s, controls_dict)

            # Add propulsion
            F_prop, M_prop = prop.compute_thrust(s, throttle)
            forces += F_prop
            moments += M_prop

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
    print()

    # Analyze results
    print("Final State:")
    print(f"  Altitude: {-positions[-1, 2]:.1f} ft (target: 6000 ft)")
    print(f"  Heading: {np.degrees(euler_angles[-1, 2]):.1f}° (target: 90°)")
    print(f"  Airspeed: {np.linalg.norm(velocities[-1]):.1f} ft/s (target: 220 ft/s)")
    print()

    print("Flight Statistics:")
    print(f"  Distance traveled: {np.linalg.norm(positions[-1] - positions[0]):.1f} ft")
    print(f"  Max roll angle: {np.degrees(np.max(np.abs(euler_angles[:, 0]))):.1f}°")
    print(f"  Max pitch angle: {np.degrees(np.max(np.abs(euler_angles[:, 1]))):.1f}°")
    print(f"  Max roll rate: {np.degrees(np.max(np.abs(angular_rates[:, 0]))):.1f}°/s")
    print(f"  Max pitch rate: {np.degrees(np.max(np.abs(angular_rates[:, 1]))):.1f}°/s")
    print()

    # Return results
    return {
        'time': time,
        'positions': positions,
        'velocities': velocities,
        'euler_angles': euler_angles,
        'angular_rates': angular_rates,
        'controls_flaperon': controls_flaperon,
        'controls_elevator': controls_elevator,
        'controls_rudder': controls_rudder,
        'controls_throttle': controls_throttle,
        'forces': forces_history,
        'moments': moments_history
    }


def main():
    """Run flying wing demonstration."""
    print()
    print("=" * 70)
    print("Flying Wing UAV Demonstration")
    print("=" * 70)
    print()

    # Set up plotting style
    setup_plotting_style()

    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Run simulation
    results = simulate_flying_wing(duration=60.0, dt=0.01)

    print("Creating visualizations...")
    print()

    # 1. 3D Trajectory
    print("1. Plotting 3D trajectory...")
    fig1 = plot_trajectory_3d(
        results['positions'],
        title="Flying Wing Flight Trajectory (with Flaperon Model)",
        show_markers=True,
        marker_interval=500,
        save_path='output/flying_wing_trajectory_3d.png'
    )
    plt.close(fig1)

    # 2. State Variables
    print("2. Plotting state variables...")
    states = {
        'position': results['positions'],
        'velocity': results['velocities'],
        'euler_angles': results['euler_angles'],
        'angular_rates': results['angular_rates']
    }
    fig2 = plot_states_vs_time(
        results['time'],
        states,
        title="Flying Wing State Variables",
        save_path='output/flying_wing_state_variables.png'
    )
    plt.close(fig2)

    # 3. Control Inputs
    print("3. Plotting control inputs...")
    controls = {
        'flaperon': results['controls_flaperon'],
        'elevator': results['controls_elevator'],
        'rudder': results['controls_rudder'],
        'throttle': results['controls_throttle']
    }

    # Custom plot for controls with flaperon
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    axes[0].plot(results['time'], np.degrees(results['controls_flaperon']), 'b-', linewidth=1.5)
    axes[0].set_ylabel('Flaperon (deg)')
    axes[0].set_title('Flying Wing Control Inputs', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['time'], np.degrees(results['controls_elevator']), 'g-', linewidth=1.5)
    axes[1].set_ylabel('Elevator (deg)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(results['time'], np.degrees(results['controls_rudder']), 'r-', linewidth=1.5)
    axes[2].set_ylabel('Rudder (deg)')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(results['time'], results['controls_throttle'], 'm-', linewidth=1.5)
    axes[3].set_ylabel('Throttle')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('output/flying_wing_control_inputs.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. Forces and Moments
    print("4. Plotting forces and moments...")
    fig4 = plot_forces_moments(
        results['time'],
        results['forces'],
        results['moments'],
        title="Flying Wing Forces and Moments",
        save_path='output/flying_wing_forces_moments.png'
    )
    plt.close(fig4)

    print()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print()
    print("Output files:")
    print("  - output/flying_wing_trajectory_3d.png")
    print("  - output/flying_wing_state_variables.png")
    print("  - output/flying_wing_control_inputs.png")
    print("  - output/flying_wing_forces_moments.png")
    print()
    print("Flying wing simulation demonstrates:")
    print("  1. Flaperon control coupling (roll + pitch + lift + drag)")
    print("  2. Ixz product of inertia (~1602.6 slug-ft^2)")
    print("  3. Flying wing coupling derivatives (Cl_alpha, Cm_p, Cm_r, Cn_p)")
    print("  4. Improved stability compared to conventional aileron model")
    print()


if __name__ == "__main__":
    main()
