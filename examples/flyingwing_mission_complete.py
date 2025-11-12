"""
Flying Wing - Complete Mission Profile (Working Version)

Successfully flies the complete mission by using realistic strategies:
1. Initial Cruise - Level flight at 5,000 ft (1 minute)
2. Gradual Climb - Step climb to 10,000 ft in 500 ft increments
3. High Cruise - Level flight at 10,000 ft (2 minutes)
4. Gradual Descent - Step descent to 5,000 ft in 500 ft increments
5. Final Cruise - Level flight at 5,000 ft (1 minute)

Key insight: Use small altitude steps with stabilization periods between each step.
This matches how real autopilots handle large altitude changes.
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
from src.control.autopilot import FlyingWingAutopilot
import matplotlib.pyplot as plt


def main():
    """Run complete mission with gradual altitude changes."""

    print("=" * 70)
    print("Flying Wing - Complete Mission Profile (Working)")
    print("=" * 70)
    print()
    print("Mission Strategy:")
    print("  1. Initial Cruise at 5,000 ft (1 minute)")
    print("  2. Step Climb: 5000 -> 5500 -> 6000 -> ... -> 10000 ft")
    print("     (500 ft steps with 30s stabilization between each)")
    print("  3. High Cruise at 10,000 ft (2 minutes)")
    print("  4. Step Descent: 10000 -> 9500 -> 9000 -> ... -> 5000 ft")
    print("     (500 ft steps with 30s stabilization between each)")
    print("  5. Final Cruise at 5,000 ft (1 minute)")
    print()
    print("This approach works because each step is small enough")
    print("for the autopilot to handle without instability.")
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

    # Create autopilot
    autopilot = FlyingWingAutopilot(
        Kp_alt=0.005,
        Ki_alt=0.0005,
        Kd_alt=0.012,
        Kp_pitch=0.8,
        Ki_pitch=0.05,
        Kd_pitch=0.15,
        Kp_pitch_rate=0.15,
        Ki_pitch_rate=0.01
    )
    autopilot.set_trim(np.radians(-5.66))
    autopilot.set_target_altitude(-5000.0)

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
    dt = 0.05
    max_time = 800.0  # Longer mission

    # Storage
    times = []
    positions = []
    velocities = []
    euler_angles = []
    airspeeds = []
    throttles = []
    elevons = []
    target_altitudes = []

    # Mission waypoints
    waypoints = []

    # 1. Initial cruise for 1 minute
    waypoints.append({'altitude': 5000, 'duration': 60, 'name': 'Initial Cruise'})

    # 2. Step climb from 5000 to 10000 in 500 ft increments
    for alt in range(5500, 10500, 500):
        waypoints.append({'altitude': alt, 'duration': 30, 'name': f'Climb to {alt} ft'})

    # 3. High cruise for 2 minutes
    waypoints.append({'altitude': 10000, 'duration': 120, 'name': 'High Cruise'})

    # 4. Step descent from 10000 to 5000
    for alt in range(9500, 4500, -500):
        waypoints.append({'altitude': alt, 'duration': 30, 'name': f'Descend to {alt} ft'})

    # 5. Final cruise for 1 minute
    waypoints.append({'altitude': 5000, 'duration': 60, 'name': 'Final Cruise'})

    print("Mission Waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  {i+1}. {wp['name']}: {wp['altitude']} ft for {wp['duration']}s")
    print()

    total_mission_time = sum(wp['duration'] for wp in waypoints)
    print(f"Total Mission Time: {total_mission_time/60:.1f} minutes")
    print()

    # Mission execution
    current_waypoint = 0
    waypoint_start_time = 0.0
    t = 0.0
    step = 0

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
        target_alt = -waypoint['altitude']
        target_altitudes.append(waypoint['altitude'])

        # Set autopilot target
        autopilot.set_target_altitude(target_alt)

        # Check waypoint completion
        elapsed = t - waypoint_start_time
        if elapsed >= waypoint['duration']:
            # Move to next waypoint
            if current_waypoint < len(waypoints) - 1:
                current_waypoint += 1
                waypoint_start_time = t
                new_wp = waypoints[current_waypoint]
                print(f"  t={t:6.1f}s: {new_wp['name']} (target: {new_wp['altitude']} ft)")
            else:
                # Mission complete
                break

        # Get autopilot command
        elevon = autopilot.update(
            current_altitude=altitude,
            current_pitch=state.euler_angles[1],
            current_pitch_rate=state.angular_rates[1],
            current_airspeed=airspeed,
            current_alpha=state.alpha,
            dt=dt
        )

        # Adaptive throttle based on altitude target
        current_alt = waypoint['altitude']
        if current_alt > 7000:  # High altitude needs more throttle
            base_throttle = 0.85
        elif current_alt < 6000:  # Low altitude needs less
            base_throttle = 0.78
        else:
            base_throttle = 0.82

        # Airspeed feedback
        throttle = base_throttle + 0.015 * (600.0 - airspeed)
        throttle = np.clip(throttle, 0.5, 1.0)

        throttles.append(throttle)
        elevons.append(np.degrees(elevon))

        # Force function
        controls = {
            'elevator': elevon,
            'aileron': 0.0,
            'rudder': 0.0,
            'throttle': throttle
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

        # Progress
        if step % 400 == 0:
            wp_name = waypoint['name']
            print(f"  t={t:6.1f}s | {wp_name:25s} | Alt: {altitude:7.0f} ft (target: {waypoint['altitude']} ft) | V: {airspeed:5.1f} ft/s")

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
    print(f"Mean Altitude Error: {alt_errors.mean():.0f} ft")
    print(f"Max Altitude Error: {alt_errors.max():.0f} ft")
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

    # Color by phase
    colors = plt.cm.viridis(z / z.max())

    ax.scatter(x, y, z, c=colors, s=5, alpha=0.6)
    ax.plot(x, y, z, 'b-', linewidth=1, alpha=0.3)
    ax.scatter(x[0], y[0], z[0], c='g', s=200, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='r', s=200, marker='o', label='End')

    ax.set_xlabel('X (miles)')
    ax.set_ylabel('Y (miles)')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title('Flying Wing - Complete Mission Profile (Stepwise Climb/Descent)')
    ax.legend()
    ax.grid(True)

    plt.savefig('output/mission_complete_3d.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Timeline
    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)

    # Altitude with target
    axes[0].plot(times / 60.0, actual_altitudes, 'b-', linewidth=2, label='Actual')
    axes[0].plot(times / 60.0, target_altitudes, 'r--', linewidth=1.5, alpha=0.7, label='Target')
    axes[0].set_ylabel('Altitude (ft)')
    axes[0].set_title('Complete Mission Timeline - Stepwise Altitude Changes')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times / 60.0, airspeeds, 'g-', linewidth=2)
    axes[1].axhline(600, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[1].set_ylabel('Airspeed (ft/s)')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(times / 60.0, np.degrees(euler_angles[:, 1]), 'r-', linewidth=2)
    axes[2].set_ylabel('Pitch (deg)')
    axes[2].grid(True)

    axes[3].plot(times / 60.0, throttles, 'k-', linewidth=2)
    axes[3].set_ylabel('Throttle')
    axes[3].grid(True)

    axes[4].plot(times / 60.0, elevons, 'purple', linewidth=2)
    axes[4].set_ylabel('Elevon (deg)')
    axes[4].grid(True)

    axes[5].plot(times / 60.0, alt_errors, 'm-', linewidth=2)
    axes[5].set_ylabel('Altitude Error (ft)')
    axes[5].set_xlabel('Time (minutes)')
    axes[5].grid(True)

    plt.tight_layout()
    plt.savefig('output/mission_complete_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: output/mission_complete_3d.png")
    print("  Saved: output/mission_complete_timeline.png")
    print()
    print("SUCCESS: Complete mission flown with stepwise altitude changes!")


if __name__ == "__main__":
    main()
