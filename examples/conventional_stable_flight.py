"""
Conventional Tail Configuration - Stable Flight with AVL Aerodynamics

Tests the conventional tail UAV with:
- AVL-generated stability derivatives
- Horizontal tail with elevator for pitch control
- Vertical tail with rudder for yaw control
- Wing ailerons for roll control
- Triple-loop autopilot for altitude hold

Expected performance (based on AVL data):
- Excellent pitch stability (Cm_alpha = -1.643, 20x better than flying wing)
- Strong pitch damping (Cm_q = -6.197, 18x better than flying wing)
- Good elevator authority (Cm_de = -0.4116)
- Excellent directional stability (Cn_beta = 0.053)
- Strong yaw damping (Cn_r = -0.126)
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
import conventional_aero_data_avl as aero_data


def find_turbofan_trim(mass, inertia, aero, turbofan, altitude, airspeed):
    """
    Find simplified trim with turbofan engine for conventional tail configuration.

    Steps:
    1. Calculate alpha for L = W
    2. Set theta = alpha (level flight)
    3. Calculate throttle for T = D
    4. Calculate elevator to balance pitch moment (M = 0)
    """

    print("=" * 70)
    print("Turbofan Trim Solver - Conventional Tail Configuration")
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

    # Step 2: Set theta = alpha (level flight)
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

    print(f"Step 3: Throttle for T = D")
    print(f"  CD_trim = {CD_trim:.6f}")
    print(f"  D_trim = {D_trim:.1f} lbf")
    print(f"  T_max_available = {T_max:.1f} lbf")
    print(f"  throttle_trim = {throttle_trim:.4f} ({100*throttle_trim:.1f}%)")
    print()

    # Step 4: Calculate elevator for M = 0
    # Pitching moment: Cm = Cm_0 + Cm_alpha * alpha + Cm_de * elevator
    # For trim: Cm = 0, so elevator = -(Cm_0 + Cm_alpha * alpha) / Cm_de
    Cm_trim = aero.Cm_0 + aero.Cm_alpha * alpha_trim
    elevator_trim = -Cm_trim / aero.Cm_de

    print(f"Step 4: Elevator for M = 0")
    print(f"  Cm_0 = {aero.Cm_0:.6f}")
    print(f"  Cm_alpha * alpha = {aero.Cm_alpha * alpha_trim:.6f}")
    print(f"  Cm_trim = {Cm_trim:.6f}")
    print(f"  Cm_de = {aero.Cm_de:.6f}")
    print(f"  elevator_trim = {np.degrees(elevator_trim):.4f} deg")
    print()

    print("=" * 70)
    print("TRIM SOLUTION")
    print("=" * 70)
    print(f"alpha     = {np.degrees(alpha_trim):7.3f} deg")
    print(f"theta     = {np.degrees(theta_trim):7.3f} deg")
    print(f"throttle  = {throttle_trim:7.4f}")
    print(f"elevator  = {np.degrees(elevator_trim):7.3f} deg")
    print("=" * 70)
    print()

    return {
        'alpha': alpha_trim,
        'theta': theta_trim,
        'throttle': throttle_trim,
        'elevator': elevator_trim,
        'aileron': 0.0,
        'rudder': 0.0,
        'CL': CL_trim,
        'CD': CD_trim,
    }


def main():
    """Run conventional tail stable flight simulation."""

    print("\n" + "=" * 70)
    print("Conventional Tail Configuration - Stable Flight Simulation")
    print("Using AVL-Generated Aerodynamic Derivatives")
    print("=" * 70 + "\n")

    # Aircraft mass properties (same as flying wing for comparison)
    mass = 228.925  # slugs
    inertia = np.array([
        [19236.29, 0.0, 0.0],
        [0.0, 2251.02, 0.0],
        [0.0, 0.0, 21487.31]
    ])  # slug-ft^2

    # Create aerodynamic model with AVL derivatives
    aero = LinearAeroModel(
        S_ref=aero_data.SREF,
        c_ref=aero_data.CREF,
        b_ref=aero_data.BREF,
        rho=0.002377  # sea level (will be updated)
    )

    # Set longitudinal derivatives
    aero.CL_0 = aero_data.TRIM_CL - aero_data.CL_ALPHA * np.radians(aero_data.TRIM_ALPHA)
    aero.CL_alpha = aero_data.CL_ALPHA
    aero.CL_q = aero_data.CL_Q
    aero.CL_de = aero_data.CL_DE_RAD  # Elevator control (converted to per radian)

    aero.CD_0 = 0.006  # Use XFOIL drag estimate (more realistic than AVL CD_trim)
    aero.CD_alpha = aero_data.CD_ALPHA
    aero.CD_alpha2 = 0.05  # Generic induced drag term
    aero.CD_q = aero_data.CD_Q

    aero.Cm_0 = aero_data.TRIM_CM - aero_data.CM_ALPHA * np.radians(aero_data.TRIM_ALPHA)
    aero.Cm_alpha = aero_data.CM_ALPHA  # EXCELLENT: -1.643
    aero.Cm_q = aero_data.CM_Q  # EXCELLENT: -6.197
    aero.Cm_de = aero_data.CM_DE_RAD  # Elevator control: -0.4116/rad

    # Set lateral-directional derivatives
    aero.CY_beta = aero_data.CY_BETA
    aero.CY_p = aero_data.CY_P
    aero.CY_r = aero_data.CY_R

    aero.Cl_beta = aero_data.CL_BETA  # Dihedral effect
    aero.Cl_p = aero_data.CL_P  # Roll damping
    aero.Cl_r = aero_data.CL_R
    aero.Cl_da = aero_data.CL_DA_RAD  # Aileron control

    aero.Cn_beta = aero_data.CN_BETA  # EXCELLENT: 0.053
    aero.Cn_p = aero_data.CN_P
    aero.Cn_r = aero_data.CN_R  # EXCELLENT: -0.126
    aero.Cn_dr = aero_data.CN_DR_RAD  # Rudder control (appears to be ~0)

    print("Aerodynamic Model Configuration:")
    print(f"  S_ref = {aero.S_ref:.2f} ft^2")
    print(f"  c_ref = {aero.c_ref:.2f} ft")
    print(f"  b_ref = {aero.b_ref:.2f} ft")
    print(f"  CL_alpha = {aero.CL_alpha:.4f} /rad")
    print(f"  Cm_alpha = {aero.Cm_alpha:.4f} /rad (EXCELLENT!)")
    print(f"  Cm_q = {aero.Cm_q:.4f} /rad (EXCELLENT!)")
    print(f"  Cm_de = {aero.Cm_de:.4f} /rad")
    print(f"  Cn_beta = {aero.Cn_beta:.4f} /rad (EXCELLENT!)")
    print(f"  Cn_r = {aero.Cn_r:.4f} /rad (EXCELLENT!)")
    print()

    # Create turbofan engine
    turbofan = TurbofanModel(thrust_max=1900.0, altitude_lapse_rate=0.7)

    # Create dynamics
    dynamics = AircraftDynamics(mass, inertia)

    # Flight condition
    altitude = -5000.0  # ft (negative in NED frame)
    airspeed = 600.0  # ft/s (Mach 0.54 at 5000 ft)

    # Find trim
    trim = find_turbofan_trim(mass, inertia, aero, turbofan, altitude, airspeed)

    # Initial state at trim
    state = State()
    state.position = np.array([0.0, 0.0, altitude])
    state.velocity_body = np.array([
        airspeed * np.cos(trim['alpha']),
        0.0,
        airspeed * np.sin(trim['alpha'])
    ])
    state.set_euler_angles(0.0, trim['theta'], 0.0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    # Create autopilot (can reuse FlyingWingAutopilot - it just controls pitch)
    # Tuned gains for conventional configuration
    autopilot = FlyingWingAutopilot(
        target_altitude=-altitude,
        trim_throttle=trim['throttle'],
        trim_elevon=trim['elevator'],  # Use as elevator trim
        Kp_alt=0.005,  # Altitude control
        Ki_alt=0.0005,
        Kd_alt=0.012,
        Kp_pitch=0.8,  # Pitch attitude control
        Ki_pitch=0.05,
        Kd_pitch=0.15,
        Kp_pitch_rate=0.15,  # Pitch rate damping
        Ki_pitch_rate=0.01,
        target_airspeed=airspeed,
        stall_airspeed=150.0
    )

    airspeed_controller = AirspeedHoldController(
        target_airspeed=airspeed,
        trim_throttle=trim['throttle'],
        Kp=0.015  # Increased for higher drag
    )

    # Simulation parameters
    dt = 0.01  # 10ms time step
    duration = 30.0  # 30 seconds
    num_steps = int(duration / dt)

    # Data storage
    time_history = []
    states_history = []
    controls_history = []

    # Simulation loop
    print("Running simulation...")
    print(f"  Duration: {duration} seconds")
    print(f"  Time step: {dt} seconds")
    print(f"  Total steps: {num_steps}")
    print()

    for step in range(num_steps):
        t = step * dt

        # Compute autopilot commands
        elevon_cmd = autopilot.update(state, dt)  # Use as elevator command
        throttle_cmd = airspeed_controller.update(state, dt)

        # Control inputs for conventional configuration
        # elevon_cmd becomes elevator, set aileron=0, rudder=0
        controls = {
            'throttle': throttle_cmd,
            'elevator': elevon_cmd,  # Elevator control
            'aileron': 0.0,  # No lateral control for now
            'rudder': 0.0   # No yaw control for now
        }

        # Compute derivatives
        state_dot = dynamics.compute_derivatives(state, controls)

        # Integrate (Euler method)
        state.position += state_dot.velocity_ned * dt
        state.velocity_body += state_dot.acceleration_body * dt
        state.angular_rates += state_dot.angular_acceleration * dt

        # Update quaternion
        q = state.quaternion
        q_dot = 0.5 * np.array([
            -q[1]*state.angular_rates[0] - q[2]*state.angular_rates[1] - q[3]*state.angular_rates[2],
            q[0]*state.angular_rates[0] + q[2]*state.angular_rates[2] - q[3]*state.angular_rates[1],
            q[0]*state.angular_rates[1] - q[1]*state.angular_rates[2] + q[3]*state.angular_rates[0],
            q[0]*state.angular_rates[2] + q[1]*state.angular_rates[1] - q[2]*state.angular_rates[0]
        ])
        state.quaternion += q_dot * dt
        state.quaternion /= np.linalg.norm(state.quaternion)  # Normalize

        # Store data
        time_history.append(t)
        states_history.append(state.copy())
        controls_history.append(controls.copy())

        # Progress indicator
        if step % 1000 == 0:
            alt_ft = -state.position[2]
            V_body = np.linalg.norm(state.velocity_body)
            print(f"  t = {t:.1f}s: alt = {alt_ft:.0f} ft, V = {V_body:.1f} ft/s, "
                  f"elev = {np.degrees(elevon_cmd):.2f}°, throttle = {throttle_cmd:.3f}")

    print("\nSimulation complete!")
    print()

    # Analysis
    print("=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    altitudes = np.array([-s.position[2] for s in states_history])
    airspeeds = np.array([np.linalg.norm(s.velocity_body) for s in states_history])
    euler = np.array([s.euler_angles() for s in states_history])
    rolls = np.degrees(euler[:, 0])
    pitches = np.degrees(euler[:, 1])

    print(f"Altitude:")
    print(f"  Initial: {altitudes[0]:.1f} ft")
    print(f"  Final: {altitudes[-1]:.1f} ft")
    print(f"  Change: {altitudes[-1] - altitudes[0]:+.1f} ft")
    print(f"  Std dev: {np.std(altitudes):.1f} ft")
    print()

    print(f"Airspeed:")
    print(f"  Initial: {airspeeds[0]:.1f} ft/s")
    print(f"  Final: {airspeeds[-1]:.1f} ft/s")
    print(f"  Change: {airspeeds[-1] - airspeeds[0]:+.1f} ft/s")
    print(f"  Std dev: {np.std(airspeeds):.1f} ft/s")
    print()

    print(f"Roll angle:")
    print(f"  Std dev: {np.std(rolls):.2f}°")
    print()

    print(f"Pitch angle:")
    print(f"  Std dev: {np.std(pitches):.2f}°")
    print()

    print("=" * 70)

    # Visualization
    setup_plotting_style()

    # 3D trajectory
    fig = plot_trajectory_3d(states_history, title="Conventional Tail - Stable Flight (AVL Aerodynamics)")
    fig.savefig("output/conventional_stable_3d.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # State history
    fig = plot_states_vs_time(time_history, states_history,
                              title="Conventional Tail - State Variables (AVL Aerodynamics)")
    fig.savefig("output/conventional_stable_states.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Control history
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    elevators = np.degrees([c['elevator'] for c in controls_history])
    ailerons = np.degrees([c['aileron'] for c in controls_history])
    rudders = np.degrees([c['rudder'] for c in controls_history])
    throttles = [c['throttle'] for c in controls_history]

    axes[0].plot(time_history, altitudes, 'b-', linewidth=1.5)
    axes[0].axhline(5000, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0].set_ylabel('Altitude (ft)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_history, airspeeds, 'g-', linewidth=1.5)
    axes[1].axhline(600, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[1].set_ylabel('Airspeed (ft/s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time_history, elevators, 'b-', linewidth=1.5, label='Elevator')
    axes[2].plot(time_history, ailerons, 'g-', linewidth=1.5, label='Aileron')
    axes[2].plot(time_history, rudders, 'r-', linewidth=1.5, label='Rudder')
    axes[2].set_ylabel('Control Surfaces (°)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(time_history, throttles, 'orange', linewidth=1.5)
    axes[3].set_ylabel('Throttle')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylim([0, 1])
    axes[3].grid(True, alpha=0.3)

    fig.suptitle('Conventional Tail - Control Performance (AVL Aerodynamics)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig("output/conventional_stable_controls.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nPlots saved to output/ directory")
    print("  - conventional_stable_3d.png")
    print("  - conventional_stable_states.png")
    print("  - conventional_stable_controls.png")
    print()


if __name__ == "__main__":
    main()
