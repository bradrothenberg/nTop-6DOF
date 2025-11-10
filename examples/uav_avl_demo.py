"""
UAV Demonstration with ACTUAL AVL Data

Uses real stability derivatives extracted from AVL analysis
of the nTop UAV configuration.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
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

import matplotlib.pyplot as plt


def simulate_uav_with_avl_data(duration=60.0, dt=0.01):
    """
    Simulate nTop UAV with ACTUAL AVL stability derivatives.

    AVL Analysis Results (alpha = 2°):
    - CLa = 2.441430
    - Cma = 0.002710 (POSITIVE - statically UNSTABLE!)
    - Cmq = -0.232245
    - Clb = -0.041339
    - Clp = -0.173909
    - Cnb = 0.013061
    - Cnr = -0.017481
    """
    print("=" * 70)
    print("nTop UAV Simulation with AVL Data")
    print("=" * 70)
    print()

    # Aircraft parameters from uav.mass
    mass = 234.836794  # slugs
    inertia = np.array([[14908.4373, 0, 0],
                        [0, 2318.4433, 0],
                        [0, 0, 17226.8806]])

    # Reference geometry from uav.avl
    S_ref = 199.9374  # ft²
    c_ref = 26.6891   # ft
    b_ref = 19.8904   # ft

    print("Aircraft Configuration:")
    print(f"  Mass: {mass:.2f} slugs ({mass * 32.174:.0f} lbm)")
    print(f"  Sref: {S_ref:.2f} ft²")
    print(f"  cref: {c_ref:.2f} ft")
    print(f"  bref: {b_ref:.2f} ft")
    print(f"  CG: (12.9183, 0.0011, 0.0453) ft")
    print()

    # Create models
    dynamics = AircraftDynamics(mass, inertia)

    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)

    # ACTUAL AVL STABILITY DERIVATIVES (from uav_stability.txt)
    print("Setting AVL Stability Derivatives:")
    print()

    # Force derivatives
    aero.CL_0 = 0.08529 - 2.441430 * np.radians(2)  # CL at alpha=0 (back-calculate)
    aero.CL_alpha = 2.441430  # CLa (per radian)
    aero.CL_q = 1.319577      # CLq
    aero.CL_de = 0.006853     # CLd_elevator (VERY SMALL!)

    aero.CD_0 = 0.00122 - 0.069800 * np.radians(2)  # CD at alpha=0
    aero.CD_alpha = 0.069800  # CDa
    aero.CD_alpha2 = 0.5      # Induced drag (estimate)

    aero.CY_beta = -0.041976  # CYb
    aero.CY_dr = -0.000360    # CYd_rudder (VERY SMALL!)

    print(f"  CL_0 = {aero.CL_0:.6f}")
    print(f"  CL_alpha = {aero.CL_alpha:.6f} (per rad)")
    print(f"  CL_q = {aero.CL_q:.6f}")
    print(f"  CL_elevator = {aero.CL_de:.6f} (VERY WEAK!)")
    print()
    print(f"  CD_0 = {aero.CD_0:.6f}")
    print(f"  CD_alpha = {aero.CD_alpha:.6f}")
    print()

    # Moment derivatives
    aero.Cl_beta = -0.041339  # Clb (roll due to sideslip)
    aero.Cl_p = -0.173909     # Clp (roll damping)
    aero.Cl_r = 0.029142      # Clr
    aero.Cl_da = -0.000189    # Cld_flaperon (VERY SMALL! Almost zero)
    aero.Cl_dr = 0.000011     # Cld_rudder

    aero.Cm_0 = 0.00012 - 0.002710 * np.radians(2)  # Cm at alpha=0
    aero.Cm_alpha = 0.002710  # Cma (POSITIVE - UNSTABLE!)
    aero.Cm_q = -0.232245     # Cmq (pitch damping - WEAK)
    aero.Cm_de = -0.002755    # Cmd_elevator (VERY WEAK!)

    aero.Cn_beta = 0.013061   # Cnb (yaw stability - WEAK)
    aero.Cn_p = -0.003725     # Cnp
    aero.Cn_r = -0.017481     # Cnr (yaw damping - VERY WEAK)
    aero.Cn_da = 0.000014     # Cnd_flaperon
    aero.Cn_dr = 0.000187     # Cnd_rudder (VERY WEAK!)

    print(f"  Cm_0 = {aero.Cm_0:.6f}")
    print(f"  Cm_alpha = {aero.Cm_alpha:.6f} (POSITIVE - UNSTABLE!)")
    print(f"  Cm_q = {aero.Cm_q:.6f} (pitch damping - WEAK)")
    print(f"  Cm_elevator = {aero.Cm_de:.6f} (VERY WEAK!)")
    print()
    print(f"  Cl_beta = {aero.Cl_beta:.6f}")
    print(f"  Cl_p = {aero.Cl_p:.6f} (roll damping)")
    print(f"  Cl_flaperon = {aero.Cl_da:.6f} (ALMOST ZERO!)")
    print()
    print(f"  Cn_beta = {aero.Cn_beta:.6f} (yaw stability - WEAK)")
    print(f"  Cn_r = {aero.Cn_r:.6f} (yaw damping - WEAK)")
    print()

    print("WARNING: CRITICAL ISSUES DETECTED:")
    print("  1. Cm_alpha > 0 -> Aircraft is STATICALLY UNSTABLE in pitch")
    print("  2. Cm_q = -0.23 -> Very weak pitch damping")
    print("  3. Elevator effectiveness (Cm_de) is EXTREMELY weak (-0.00276)")
    print("  4. Flaperon effectiveness (Cl_da) is ALMOST ZERO (-0.000189)")
    print("  5. Yaw damping (Cn_r) is very weak (-0.0175)")
    print()
    print("This explains the instability! The aircraft has:")
    print("  - Positive pitch stiffness (nose-up divergence)")
    print("  - Weak control authority")
    print("  - Insufficient damping")
    print()

    # Propulsion
    prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)
    combined_model = CombinedForceModel(aero, prop)

    # Autopilots with VERY conservative gains
    alt_ctrl = AltitudeHoldController(
        Kp_alt=0.0005,   # Reduced further
        Ki_alt=0.00005,  # Reduced further
        Kd_alt=0.002,    # Reduced further
        Kp_pitch=1.0,    # Reduced from 2.0
        Ki_pitch=0.2,    # Reduced from 0.5
        Kd_pitch=0.05    # Reduced from 0.15
    )

    hdg_ctrl = HeadingHoldController(
        Kp_heading=0.2,   # Reduced from 0.5
        Ki_heading=0.02,  # Reduced from 0.05
        Kd_heading=0.05,  # Reduced from 0.10
        Kp_roll=0.5,      # Reduced from 1.5
        Ki_roll=0.1,      # Reduced from 0.3
        Kd_roll=0.05      # Reduced from 0.10
    )

    spd_ctrl = AirspeedHoldController(
        Kp=0.005,
        Ki=0.0005,
        Kd=0.02
    )

    # Initial state
    state = State()
    state.position = np.array([0.0, 0.0, -5000.0])
    state.velocity_body = np.array([200.0, 0.0, 0.0])
    state.set_euler_angles(0, np.radians(2), 0)  # Match AVL analysis alpha
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    # Storage
    n_steps = int(duration / dt)
    time = np.zeros(n_steps)
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    euler_angles = np.zeros((n_steps, 3))
    angular_rates = np.zeros((n_steps, 3))
    controls_elevator = np.zeros(n_steps)
    controls_flaperon = np.zeros(n_steps)
    controls_rudder = np.zeros(n_steps)
    controls_throttle = np.zeros(n_steps)
    forces_history = np.zeros((n_steps, 3))
    moments_history = np.zeros((n_steps, 3))

    # Autopilot targets (MODEST changes)
    alt_ctrl.set_target_altitude(5500.0)  # Only 500 ft climb
    hdg_ctrl.set_target_heading(np.radians(20))  # Only 20° turn
    spd_ctrl.set_target_airspeed(205.0)  # Only 5 ft/s increase

    print("Autopilot Targets (MODEST for unstable aircraft):")
    print("  Altitude: 5000 -> 5500 ft")
    print("  Heading: 0 -> 20 deg")
    print("  Airspeed: 200 -> 205 ft/s")
    print()

    print("Starting simulation...")
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
        flaperon = hdg_ctrl.update(yaw, roll, dt)  # Actually aileron
        rudder = 0.0
        throttle = spd_ctrl.update(airspeed, dt)

        # Clamp controls (tight limits due to weak effectiveness)
        elevator = np.clip(elevator, np.radians(-10), np.radians(10))
        flaperon = np.clip(flaperon, np.radians(-10), np.radians(10))
        throttle = np.clip(throttle, 0.0, 1.0)

        # Store data
        positions[i] = state.position
        velocities[i] = state.velocity_body
        euler_angles[i] = state.euler_angles
        angular_rates[i] = state.angular_rates
        controls_elevator[i] = elevator
        controls_flaperon[i] = flaperon
        controls_rudder[i] = rudder
        controls_throttle[i] = throttle

        # Compute forces and moments
        def force_func_with_controls(s):
            forces, moments = combined_model(s, throttle)

            # Add control surface effects
            q_bar = atm.get_dynamic_pressure(s.airspeed)
            moments[1] += q_bar * S_ref * c_ref * aero.Cm_de * elevator
            moments[0] += q_bar * S_ref * b_ref * aero.Cl_da * flaperon

            return forces, moments

        F, M = force_func_with_controls(state)
        forces_history[i] = F
        moments_history[i] = M

        # State derivative
        state_dot_array = dynamics.state_derivative(state, force_func_with_controls)

        # Euler integration
        x = state.to_array()
        x_new = x + state_dot_array * dt
        state.from_array(x_new)

    print("Simulation complete!")
    print()

    # Analyze results
    print("Final State:")
    print(f"  Altitude: {-positions[-1, 2]:.1f} ft (target: 5500 ft)")
    print(f"  Heading: {np.degrees(euler_angles[-1, 2]):.1f}° (target: 20°)")
    print(f"  Airspeed: {np.linalg.norm(velocities[-1]):.1f} ft/s (target: 205 ft/s)")
    print()

    print("Flight Statistics:")
    print(f"  Distance traveled: {np.linalg.norm(positions[-1] - positions[0]):.1f} ft")
    print(f"  Max roll angle: {np.degrees(np.max(np.abs(euler_angles[:, 0]))):.1f}°")
    print(f"  Max pitch angle: {np.degrees(np.max(np.abs(euler_angles[:, 1]))):.1f}°")
    print(f"  Max roll rate: {np.degrees(np.max(np.abs(angular_rates[:, 0]))):.1f}°/s")
    print(f"  Max pitch rate: {np.degrees(np.max(np.abs(angular_rates[:, 1]))):.1f}°/s")
    print()

    return {
        'time': time,
        'positions': positions,
        'velocities': velocities,
        'euler_angles': euler_angles,
        'angular_rates': angular_rates,
        'controls_elevator': controls_elevator,
        'controls_flaperon': controls_flaperon,
        'controls_rudder': controls_rudder,
        'controls_throttle': controls_throttle,
        'forces': forces_history,
        'moments': moments_history
    }


def main():
    """Run UAV demonstration with AVL data."""
    setup_plotting_style()
    os.makedirs('output', exist_ok=True)

    # Run simulation
    results = simulate_uav_with_avl_data(duration=60.0, dt=0.01)

    print("Creating visualizations...")
    print()

    # 1. 3D Trajectory
    fig1 = plot_trajectory_3d(
        results['positions'],
        title="nTop UAV Trajectory (with AVL Data)",
        show_markers=True,
        marker_interval=500,
        save_path='output/uav_avl_trajectory_3d.png'
    )
    plt.close(fig1)

    # 2. State Variables
    states = {
        'position': results['positions'],
        'velocity': results['velocities'],
        'euler_angles': results['euler_angles'],
        'angular_rates': results['angular_rates']
    }
    fig2 = plot_states_vs_time(
        results['time'],
        states,
        title="nTop UAV State Variables (AVL Data)",
        save_path='output/uav_avl_state_variables.png'
    )
    plt.close(fig2)

    # 3. Control Inputs
    controls = {
        'elevator': results['controls_elevator'],
        'flaperon': results['controls_flaperon'],
        'rudder': results['controls_rudder'],
        'throttle': results['controls_throttle']
    }
    fig3 = plot_controls_vs_time(
        results['time'],
        controls,
        title="nTop UAV Control Inputs (AVL Data)",
        save_path='output/uav_avl_control_inputs.png'
    )
    plt.close(fig3)

    # 4. Forces and Moments
    fig4 = plot_forces_moments(
        results['time'],
        results['forces'],
        results['moments'],
        title="nTop UAV Forces and Moments (AVL Data)",
        save_path='output/uav_avl_forces_moments.png'
    )
    plt.close(fig4)

    print()
    print("=" * 70)
    print("Output files:")
    print("  - output/uav_avl_trajectory_3d.png")
    print("  - output/uav_avl_state_variables.png")
    print("  - output/uav_avl_control_inputs.png")
    print("  - output/uav_avl_forces_moments.png")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
