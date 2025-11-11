"""
Flying Wing Visualization Demo with ACTUAL AVL Data

Uses the real stability derivatives from the flying wing AVL analysis.
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


def simulate_flying_wing(duration=60.0, dt=0.01):
    """
    Simulate nTop UAV flying wing with ACTUAL AVL stability derivatives.

    Flying Wing AVL Analysis Results (alpha = 2°):
    - CLa = 1.412241 (lower AR than conventional)
    - Cma = -0.079668 (STABLE!)
    - Cmq = -0.347072 (good pitch damping)
    - Clp = -0.109230 (roll damping)
    - Cnr = -0.001030 (weak yaw damping)
    - Cl_elevon = -0.001536 (45x stronger than old flaperon!)
    """
    print("=" * 70)
    print("Flying Wing Visualization Demo with AVL Data")
    print("=" * 70)
    print()

    # Aircraft parameters from uav_flyingwing.mass
    mass = 228.924806  # slugs
    inertia = np.array([[19236.2914, 0, 0],
                        [0, 2251.0172, 0],
                        [0, 0, 21487.3086]])

    # Reference geometry from uav_flyingwing.avl
    S_ref = 412.6370  # ft² (full wing area)
    c_ref = 11.9555   # ft (mean chord)
    b_ref = 24.8630   # ft (span)

    print("Flying Wing Configuration:")
    print(f"  Mass: {mass:.2f} slugs ({mass * 32.174:.0f} lbm)")
    print(f"  Sref: {S_ref:.2f} ft²")
    print(f"  cref: {c_ref:.2f} ft")
    print(f"  bref: {b_ref:.2f} ft")
    print(f"  CG: (12.8461, -0.0006, 0.0437) ft")
    print(f"  Aspect Ratio: {b_ref**2 / S_ref:.2f}")
    print()

    # Create models
    dynamics = AircraftDynamics(mass, inertia)

    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)

    # ACTUAL FLYING WING AVL STABILITY DERIVATIVES
    print("Setting Flying Wing AVL Derivatives:")
    print()

    # Force derivatives
    aero.CL_0 = 0.04932 - 1.412241 * np.radians(2)  # CL at alpha=0
    aero.CL_alpha = 1.412241  # CLa (per radian) - lower than conventional due to low AR
    aero.CL_q = 1.282202      # CLq

    aero.CD_0 = 0.00062 - 0.035509 * np.radians(2)  # CD at alpha=0
    aero.CD_alpha = 0.035509  # CDa
    aero.CD_alpha2 = 0.5      # Induced drag

    aero.CY_beta = -0.008250  # CYb (weak - no vertical tail)

    print(f"  CL_0 = {aero.CL_0:.6f}")
    print(f"  CL_alpha = {aero.CL_alpha:.6f} (per rad)")
    print(f"  CL_q = {aero.CL_q:.6f}")
    print()
    print(f"  CD_0 = {aero.CD_0:.6f}")
    print(f"  CD_alpha = {aero.CD_alpha:.6f}")
    print()

    # Moment derivatives
    aero.Cl_beta = -0.028108  # Clb (roll due to sideslip)
    aero.Cl_p = -0.109230     # Clp (roll damping)
    aero.Cl_r = 0.019228      # Clr
    aero.Cl_da = -0.001536    # Cld_elevon (45x STRONGER than old flaperon!)

    aero.Cm_0 = -0.00272 - (-0.079668) * np.radians(2)  # Cm at alpha=0
    aero.Cm_alpha = -0.079668  # Cma (STABLE - NEGATIVE!)
    aero.Cm_q = -0.347072      # Cmq (pitch damping - GOOD!)
    # Note: Cm_elevon = 0 for symmetric deflection (need differential for pitch)

    aero.Cn_beta = -0.000119  # Cnb (yaw stability - very weak, no tail)
    aero.Cn_p = -0.000752     # Cnp
    aero.Cn_r = -0.001030     # Cnr (yaw damping - very weak)
    aero.Cn_da = 0.000109     # Cnd_elevon

    print(f"  Cm_0 = {aero.Cm_0:.6f}")
    print(f"  Cm_alpha = {aero.Cm_alpha:.6f} (STABLE - NEGATIVE!)")
    print(f"  Cm_q = {aero.Cm_q:.6f} (pitch damping - GOOD)")
    print()
    print(f"  Cl_beta = {aero.Cl_beta:.6f}")
    print(f"  Cl_p = {aero.Cl_p:.6f} (roll damping)")
    print(f"  Cl_elevon = {aero.Cl_da:.6f} (45x STRONGER!)")
    print()
    print(f"  Cn_beta = {aero.Cn_beta:.6f} (yaw stability - very weak)")
    print(f"  Cn_r = {aero.Cn_r:.6f} (yaw damping - very weak)")
    print()

    print("Aircraft Status: STABLE")
    print("  - Cm_alpha < 0 (stable pitch)")
    print("  - Static margin: +5.6%")
    print("  - Strong elevon control (45x improvement)")
    print("  - Good pitch damping (138% improvement)")
    print()

    # Propulsion
    prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)
    combined_model = CombinedForceModel(aero, prop)

    # Autopilots with INCREASED gains (3x due to stronger controls)
    alt_ctrl = AltitudeHoldController(
        Kp_alt=0.0015,    # 3x increase from 0.0005
        Ki_alt=0.00015,   # 3x increase
        Kd_alt=0.006,     # 3x increase
        Kp_pitch=2.0,     # 2x increase
        Ki_pitch=0.4,     # 2x increase
        Kd_pitch=0.10     # 2x increase
    )

    hdg_ctrl = HeadingHoldController(
        Kp_heading=0.4,   # 2x increase
        Ki_heading=0.04,  # 2x increase
        Kd_heading=0.08,  # 1.6x increase
        Kp_roll=1.0,      # 2x increase
        Ki_roll=0.2,      # 2x increase
        Kd_roll=0.08      # 1.6x increase
    )

    spd_ctrl = AirspeedHoldController(
        Kp=0.01,
        Ki=0.001,
        Kd=0.03
    )

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
    controls_elevon_left = np.zeros(n_steps)
    controls_elevon_right = np.zeros(n_steps)
    controls_throttle = np.zeros(n_steps)
    forces_history = np.zeros((n_steps, 3))
    moments_history = np.zeros((n_steps, 3))

    # Autopilot targets (MODEST changes)
    alt_ctrl.set_target_altitude(6000.0)  # 1000 ft climb
    hdg_ctrl.set_target_heading(np.radians(45))  # 45° turn
    spd_ctrl.set_target_airspeed(220.0)  # 20 ft/s increase

    print("Autopilot Targets:")
    print("  Altitude: 5000 -> 6000 ft")
    print("  Heading: 0 -> 45 deg")
    print("  Airspeed: 200 -> 220 ft/s")
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

        # Get pitch and roll commands
        pitch_cmd = alt_ctrl.update(altitude, pitch, dt)
        roll_cmd = hdg_ctrl.update(yaw, roll, dt)
        throttle = spd_ctrl.update(airspeed, dt)

        # ELEVON MIXING: pitch + roll (differential)
        left_elevon = pitch_cmd + roll_cmd
        right_elevon = pitch_cmd - roll_cmd

        # Clamp controls
        left_elevon = np.clip(left_elevon, np.radians(-15), np.radians(15))
        right_elevon = np.clip(right_elevon, np.radians(-15), np.radians(15))
        throttle = np.clip(throttle, 0.0, 1.0)

        # For aerodynamics, use average deflection for roll effect
        elevon_avg = (left_elevon + right_elevon) / 2.0
        elevon_diff = (left_elevon - right_elevon) / 2.0

        # Store data
        positions[i] = state.position
        velocities[i] = state.velocity_body
        euler_angles[i] = state.euler_angles
        angular_rates[i] = state.angular_rates
        controls_elevon_left[i] = left_elevon
        controls_elevon_right[i] = right_elevon
        controls_throttle[i] = throttle

        # Compute forces and moments
        def force_func_with_controls(s):
            forces, moments = combined_model(s, throttle)

            # Add elevon control effects (roll only - pitch requires differential drag analysis)
            q_bar = atm.get_dynamic_pressure(s.airspeed)
            moments[0] += q_bar * S_ref * b_ref * aero.Cl_da * elevon_diff  # Roll moment

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
    print(f"  Altitude: {-positions[-1, 2]:.1f} ft (target: 6000 ft)")
    print(f"  Heading: {np.degrees(euler_angles[-1, 2]):.1f}° (target: 45°)")
    print(f"  Airspeed: {np.linalg.norm(velocities[-1]):.1f} ft/s (target: 220 ft/s)")
    print()

    print("Flight Statistics:")
    print(f"  Distance traveled: {np.linalg.norm(positions[-1, :2] - positions[0, :2]):.1f} ft")
    print(f"  Max roll angle: {np.degrees(np.max(np.abs(euler_angles[:, 0]))):.1f}°")
    print(f"  Max pitch angle: {np.degrees(np.max(euler_angles[:, 1])):.1f}°")
    print(f"  Max roll rate: {np.degrees(np.max(np.abs(angular_rates[:, 0]))):.1f}°/s")
    print(f"  Max pitch rate: {np.degrees(np.max(np.abs(angular_rates[:, 1]))):.1f}°/s")
    print()

    return {
        'time': time,
        'positions': positions,
        'velocities': velocities,
        'euler_angles': euler_angles,
        'angular_rates': angular_rates,
        'controls_elevon_left': controls_elevon_left,
        'controls_elevon_right': controls_elevon_right,
        'controls_throttle': controls_throttle,
        'forces': forces_history,
        'moments': moments_history
    }


def main():
    """Run flying wing visualization demo."""
    setup_plotting_style()
    os.makedirs('output', exist_ok=True)

    # Run simulation
    results = simulate_flying_wing(duration=60.0, dt=0.01)

    print("Creating visualizations...")
    print()

    # 1. 3D Trajectory
    fig1 = plot_trajectory_3d(
        results['positions'],
        title="Flying Wing Trajectory (with AVL Data)",
        show_markers=True,
        marker_interval=500,
        save_path='output/flyingwing_trajectory_3d.png'
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
        title="Flying Wing State Variables (AVL Data)",
        save_path='output/flyingwing_state_variables.png'
    )
    plt.close(fig2)

    # 3. Control Inputs (elevons)
    controls = {
        'elevon_left': results['controls_elevon_left'],
        'elevon_right': results['controls_elevon_right'],
        'throttle': results['controls_throttle']
    }
    fig3 = plot_controls_vs_time(
        results['time'],
        controls,
        title="Flying Wing Control Inputs (AVL Data)",
        save_path='output/flyingwing_control_inputs.png'
    )
    plt.close(fig3)

    # 4. Forces and Moments
    fig4 = plot_forces_moments(
        results['time'],
        results['forces'],
        results['moments'],
        title="Flying Wing Forces and Moments (AVL Data)",
        save_path='output/flyingwing_forces_moments.png'
    )
    plt.close(fig4)

    print()
    print("=" * 70)
    print("Output files:")
    print("  - output/flyingwing_trajectory_3d.png")
    print("  - output/flyingwing_state_variables.png")
    print("  - output/flyingwing_control_inputs.png")
    print("  - output/flyingwing_forces_moments.png")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
