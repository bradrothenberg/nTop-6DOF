"""
Flying Wing - Complete Mission Profile with Model Predictive Control

Uses MPC to fly the complete mission by:
1. Predicting trajectory over finite horizon
2. Optimizing control inputs (elevon, throttle) to reach targets
3. Explicitly handling altitude-airspeed coupling
4. Satisfying control and state constraints

Mission Profile:
1. Initial Cruise - 5,000 ft for 1 minute
2. Climb - 5,000 ft -> 10,000 ft
3. High Cruise - 10,000 ft for 2 minutes
4. Descent - 10,000 ft -> 5,000 ft
5. Final Cruise - 5,000 ft for 1 minute
"""

import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import TurbofanModel
from src.environment.atmosphere import StandardAtmosphere
from src.control.mpc_autopilot import MPCAutopilot
import matplotlib.pyplot as plt


def main():
    """Run complete mission with MPC autopilot."""

    print("=" * 70)
    print("Flying Wing - Complete Mission Profile with MPC")
    print("=" * 70)
    print()
    print("Model Predictive Control Strategy:")
    print("  - Predicts trajectory 10 seconds ahead (20 steps x 0.5s)")
    print("  - Optimizes elevon + throttle together")
    print("  - Explicitly couples altitude and airspeed control")
    print("  - Satisfies control limits (elevon, throttle)")
    print("  - Satisfies state limits (alpha, pitch rate)")
    print()
    print("Mission Profile:")
    print("  1. Initial Cruise at 5,000 ft (1 minute)")
    print("  2. Climb to 10,000 ft")
    print("  3. High Cruise at 10,000 ft (2 minutes)")
    print("  4. Descent to 5,000 ft")
    print("  5. Final Cruise at 5,000 ft (1 minute)")
    print()

    # Aircraft configuration
    mass = 228.924806
    inertia = np.array([[19236.2914, 0.0, 0.0],
                        [0.0, 2251.0172, 0.0],
                        [0.0, 0.0, 21487.3086]])

    S_ref = 412.6370
    c_ref = 11.9555
    b_ref = 24.8630

    # Create aerodynamic model
    aero = LinearAeroModel(S_ref, c_ref, b_ref)
    aero.CL_0 = 0.000023
    aero.CL_alpha = 1.412241
    aero.CL_q = 1.282202
    aero.CL_de = 0.0
    aero.CD_0 = 0.006
    aero.CD_alpha = 0.025
    aero.CD_alpha2 = 0.05
    aero.Cm_0 = 0.000061
    aero.Cm_alpha = -0.079668
    aero.Cm_q = -0.347
    aero.Cm_de = -0.02
    aero.Cl_beta = -0.1
    aero.Cl_p = -0.4
    aero.Cl_r = 0.1
    aero.Cl_da = -0.001536
    aero.Cn_beta = 0.05
    aero.Cn_p = -0.05
    aero.Cn_r = -0.1
    aero.CY_beta = -0.2

    # Create propulsion
    turbofan = TurbofanModel(thrust_max=1900.0, altitude_lapse_rate=0.7)
    dynamics = AircraftDynamics(mass, inertia)

    # Create MPC autopilot
    print("Creating MPC autopilot...")
    mpc = MPCAutopilot(
        horizon=20,              # 10 second horizon (20 steps x 0.5s)
        dt_mpc=0.5,              # MPC timestep
        Q_altitude=1.0,          # Altitude tracking weight
        Q_airspeed=0.5,          # Airspeed tracking weight
        Q_pitch=0.1,             # Pitch angle weight
        R_elevator=0.1,          # Elevator control cost
        R_throttle=0.05,         # Throttle control cost
        mass=mass,
        S_ref=S_ref,
        c_ref=c_ref,
        elevon_trim=np.radians(-5.66),
        throttle_trim=0.794
    )
    print("  MPC autopilot created!")
    print()

    # Initial state
    state = State()
    state.position = np.array([0.0, 0.0, -5000.0])
    state.velocity_body = np.array([600.0, 0.0, 0.0])
    state.set_euler_angles(0.0, np.radians(1.4649), 0.0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    print(f"Initial State:")
    print(f"  Altitude: {-state.position[2]:.0f} ft")
    print(f"  Airspeed: {state.airspeed:.0f} ft/s")
    print(f"  Pitch: {np.degrees(state.euler_angles[1]):.1f} deg")
    print()

    # Simulation parameters
    dt = 0.05  # Simulation timestep
    mpc_dt = 0.5  # MPC update rate (call MPC every 0.5s)
    max_time = 500.0

    # Storage
    times = []
    positions = []
    velocities = []
    euler_angles = []
    airspeeds = []
    throttles = []
    elevons = []
    target_altitudes = []
    target_airspeeds = []
    phases = []

    # Mission waypoints
    waypoints = [
        {'altitude': 5000, 'airspeed': 600, 'duration': 60, 'name': 'Initial Cruise'},
        {'altitude': 10000, 'airspeed': 600, 'duration': 150, 'name': 'Climb'},
        {'altitude': 10000, 'airspeed': 600, 'duration': 120, 'name': 'High Cruise'},
        {'altitude': 5000, 'airspeed': 600, 'duration': 150, 'name': 'Descent'},
        {'altitude': 5000, 'airspeed': 600, 'duration': 60, 'name': 'Final Cruise'}
    ]

    print("Mission Waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  {i+1}. {wp['name']}: {wp['altitude']} ft, {wp['airspeed']} ft/s for {wp['duration']}s")
    print()

    # Mission execution
    current_waypoint = 0
    waypoint_start_time = 0.0
    t = 0.0
    step = 0
    mpc_step = 0

    # MPC control (updated every mpc_dt seconds)
    elevon_cmd = np.radians(-5.66)
    throttle_cmd = 0.794

    print("Starting mission...")
    print()

    while t < max_time and current_waypoint < len(waypoints):
        # Store data
        times.append(t)
        positions.append(state.position.copy())
        velocities.append(state.velocity_body.copy())
        euler_angles.append(np.array(state.euler_angles))
        airspeeds.append(state.airspeed)

        altitude = -state.position[2]
        airspeed = state.airspeed

        # Current waypoint
        waypoint = waypoints[current_waypoint]
        phases.append(waypoint['name'])

        # Check waypoint completion
        elapsed = t - waypoint_start_time
        if elapsed >= waypoint['duration']:
            if current_waypoint < len(waypoints) - 1:
                current_waypoint += 1
                waypoint_start_time = t
                new_wp = waypoints[current_waypoint]
                print(f"  t={t:6.1f}s: {new_wp['name']} (target: {new_wp['altitude']} ft, {new_wp['airspeed']} ft/s)")
            else:
                break

        # Update MPC targets
        target_alt = waypoint['altitude']
        target_speed = waypoint['airspeed']
        target_altitudes.append(target_alt)
        target_airspeeds.append(target_speed)

        mpc.set_targets(target_alt, target_speed)

        # Solve MPC optimization (every mpc_dt seconds)
        if mpc_step % int(mpc_dt / dt) == 0:
            # Get atmospheric density
            atm = StandardAtmosphere(-state.position[2])
            rho = atm.density

            # Compute flight path angle
            V_ned = state.velocity_inertial
            gamma = np.arctan2(-V_ned[2], np.linalg.norm(V_ned[:2]))

            # Solve MPC
            elevon_cmd, throttle_cmd = mpc.solve_mpc(
                current_altitude=altitude,
                current_airspeed=airspeed,
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_gamma=gamma,
                rho=rho
            )

        throttles.append(throttle_cmd)
        elevons.append(np.degrees(elevon_cmd))

        # Force function
        controls = {
            'elevator': elevon_cmd,
            'aileron': 0.0,
            'rudder': 0.0,
            'throttle': throttle_cmd
        }

        def force_func(s):
            atm = StandardAtmosphere(s.altitude)
            aero.rho = atm.density
            aero_forces, aero_moments = aero.compute_forces_moments(s, controls)
            prop_forces, prop_moments = turbofan.compute_thrust(s, controls['throttle'])
            return aero_forces + prop_forces, aero_moments + prop_moments

        # RK4 integration
        state_dot = dynamics.state_derivative(state, force_func)
        state_array = state.to_array()

        k1 = state_dot
        state_temp = State()
        state_temp.from_array(state_array + 0.5 * dt * k1)
        k2 = dynamics.state_derivative(state_temp, force_func)

        state_temp.from_array(state_array + 0.5 * dt * k2)
        k3 = dynamics.state_derivative(state_temp, force_func)

        state_temp.from_array(state_array + dt * k3)
        k4 = dynamics.state_derivative(state_temp, force_func)

        state_new = state_array + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state.from_array(state_new)

        t += dt
        step += 1
        mpc_step += 1

        # Progress
        if step % 400 == 0:
            wp_name = waypoint['name']
            print(f"  t={t:6.1f}s | {wp_name:20s} | Alt: {altitude:7.0f} ft (tgt: {target_alt:5.0f}) | V: {airspeed:5.1f} ft/s (tgt: {target_speed:5.0f})")

    print()
    print("=" * 70)
    print("Mission Complete!")
    print("=" * 70)
    print()

    # Convert to arrays
    times = np.array(times)
    positions = np.array(positions)
    euler_angles = np.array(euler_angles)
    airspeeds = np.array(airspeeds)
    throttles = np.array(throttles)
    elevons = np.array(elevons)
    target_altitudes = np.array(target_altitudes)
    target_airspeeds = np.array(target_airspeeds)

    # Statistics
    actual_altitudes = -positions[:, 2]
    print(f"Flight Duration: {times[-1]/60:.1f} minutes")
    print(f"Max Altitude: {actual_altitudes.max():.0f} ft")
    print(f"Min Altitude: {actual_altitudes.min():.0f} ft")
    print(f"Max Airspeed: {max(airspeeds):.1f} ft/s")
    print(f"Min Airspeed: {min(airspeeds):.1f} ft/s")
    print(f"Distance Traveled: {positions[-1, 0]/5280:.1f} miles")

    # Tracking performance
    alt_errors = np.abs(actual_altitudes - target_altitudes)
    speed_errors = np.abs(airspeeds - target_airspeeds)
    print(f"Mean Altitude Error: {alt_errors.mean():.0f} ft")
    print(f"Max Altitude Error: {alt_errors.max():.0f} ft")
    print(f"Mean Airspeed Error: {speed_errors.mean():.1f} ft/s")
    print(f"Max Airspeed Error: {speed_errors.max():.1f} ft/s")
    print()

    # Create visualizations
    print("Creating visualizations...")
    os.makedirs('output', exist_ok=True)

    # 3D trajectory
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = positions[:, 0] / 5280.0
    y = positions[:, 1] / 5280.0
    z = -positions[:, 2]

    # Color by altitude
    colors = plt.cm.viridis(z / z.max())

    ax.scatter(x, y, z, c=colors, s=5, alpha=0.6)
    ax.plot(x, y, z, 'b-', linewidth=1, alpha=0.3)
    ax.scatter(x[0], y[0], z[0], c='g', s=200, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='r', s=200, marker='o', label='End')

    ax.set_xlabel('X (miles)')
    ax.set_ylabel('Y (miles)')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title('Flying Wing - Complete Mission with MPC')
    ax.legend()
    ax.grid(True)

    plt.savefig('output/mission_mpc_3d.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Timeline
    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)

    # Altitude with target
    axes[0].plot(times / 60.0, actual_altitudes, 'b-', linewidth=2, label='Actual')
    axes[0].plot(times / 60.0, target_altitudes, 'r--', linewidth=1.5, alpha=0.7, label='Target')
    axes[0].set_ylabel('Altitude (ft)')
    axes[0].set_title('Complete Mission with Model Predictive Control')
    axes[0].legend()
    axes[0].grid(True)

    # Airspeed with target
    axes[1].plot(times / 60.0, airspeeds, 'g-', linewidth=2, label='Actual')
    axes[1].plot(times / 60.0, target_airspeeds, 'r--', linewidth=1.5, alpha=0.7, label='Target')
    axes[1].set_ylabel('Airspeed (ft/s)')
    axes[1].legend()
    axes[1].grid(True)

    # Pitch
    axes[2].plot(times / 60.0, np.degrees(euler_angles[:, 1]), 'r-', linewidth=2)
    axes[2].set_ylabel('Pitch (deg)')
    axes[2].grid(True)

    # Throttle
    axes[3].plot(times / 60.0, throttles, 'k-', linewidth=2)
    axes[3].set_ylabel('Throttle')
    axes[3].grid(True)

    # Elevon
    axes[4].plot(times / 60.0, elevons, 'purple', linewidth=2)
    axes[4].set_ylabel('Elevon (deg)')
    axes[4].grid(True)

    # Tracking errors
    axes[5].plot(times / 60.0, alt_errors, 'm-', linewidth=2, label='Altitude Error')
    axes[5].plot(times / 60.0, speed_errors * 10, 'c-', linewidth=2, label='Airspeed Error x10')
    axes[5].set_ylabel('Tracking Error')
    axes[5].set_xlabel('Time (minutes)')
    axes[5].legend()
    axes[5].grid(True)

    plt.tight_layout()
    plt.savefig('output/mission_mpc_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: output/mission_mpc_3d.png")
    print("  Saved: output/mission_mpc_timeline.png")
    print()
    print("SUCCESS: Complete mission flown with Model Predictive Control!")


if __name__ == "__main__":
    main()
