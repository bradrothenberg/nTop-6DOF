"""
Flying Wing - Simplified Mission Profile

A simpler, more stable mission:
1. Takeoff - Accelerate to rotation speed and liftoff
2. Climb - Steady climb to 20,000 ft at constant pitch
3. Cruise - Straight and level flight for 5 minutes
4. Descent - Controlled descent to landing altitude
5. Landing - Reduce to landing speed and touchdown

This version focuses on stability over complexity.
"""

import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import TurbofanModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere
from src.control.autopilot import FlyingWingAutopilot
import matplotlib.pyplot as plt


def main():
    """Run simplified mission profile."""

    print("=" * 70)
    print("Flying Wing - Simplified Mission Profile")
    print("=" * 70)
    print()
    print("Mission Phases:")
    print("  1. Takeoff - Accelerate and liftoff")
    print("  2. Climb - Steady climb to 20,000 ft")
    print("  3. Cruise - Straight level flight for 5 minutes")
    print("  4. Descent - Controlled descent to 1,000 ft")
    print("  5. Landing - Final approach")
    print()

    # Aircraft configuration
    mass = 234.8  # slugs
    Ixx, Iyy, Izz = 14908, 2318, 17227  # slug-ft²
    inertia = np.array([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])

    S_ref = 199.94  # ft²
    c_ref = 26.689  # ft
    b_ref = 19.890  # ft

    # Create aerodynamic model
    aero = LinearAeroModel(S_ref, c_ref, b_ref)
    aero.CL_0 = 0.2
    aero.CL_alpha = 1.412241
    aero.CL_q = 1.282202
    aero.CL_de = 0.0
    aero.CD_0 = 0.006  # XFOIL drag
    aero.CD_alpha = 0.025
    aero.CD_alpha2 = 0.05
    aero.Cm_0 = 0.000061
    aero.Cm_alpha = -0.079668
    aero.Cm_q = -0.347
    aero.Cm_de = -0.02
    # Stable lateral derivatives
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
    combined = CombinedForceModel(aero, turbofan)

    # Initial state - on runway
    state = State()
    state.position = np.array([0.0, 0.0, 0.0])
    state.velocity_body = np.array([1.0, 0.0, 0.0])  # Small initial velocity
    state.set_euler_angles(0.0, 0.0, 0.0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    # Simulation parameters
    dt = 0.05
    max_time = 1200.0  # 20 minutes max

    # Storage
    times = []
    positions = []
    velocities = []
    euler_angles = []
    airspeeds = []
    throttles = []
    elevons = []
    phases = []

    # Mission state
    phase = "takeoff"
    phase_start_time = 0.0
    t = 0.0

    # Create autopilot (will be used after takeoff)
    # MUCH MORE AGGRESSIVE GAINS to overcome instability
    autopilot = FlyingWingAutopilot(
        Kp_alt=0.015,      # 5x increase for faster altitude response
        Ki_alt=0.002,      # 6.7x increase to eliminate steady-state error
        Kd_alt=0.05,       # 5x increase for better damping
        Kp_pitch=2.0,      # 3.3x increase for faster pitch response
        Ki_pitch=0.15,     # 5x increase
        Kd_pitch=0.5,      # 4.2x increase for aggressive damping
        Kp_pitch_rate=0.5, # 4.2x increase for pitch rate control
        Ki_pitch_rate=0.05, # 6.3x increase
        max_pitch_cmd=20.0, # Increase pitch authority
        min_pitch_cmd=-15.0,
        max_alpha=12.0,
        stall_speed=150.0
    )
    autopilot.set_trim(np.radians(-6.0))

    print("Starting mission...")
    print()

    step = 0
    while t < max_time and phase != "complete":
        # Store data
        times.append(t)
        positions.append(state.position.copy())
        velocities.append(state.velocity_body.copy())
        euler_angles.append(np.array(state.euler_angles))
        airspeeds.append(state.airspeed)
        phases.append(phase)

        altitude = -state.position[2]
        airspeed = state.airspeed

        # Phase transitions
        if phase == "takeoff":
            if altitude > 100.0 and airspeed > 200.0:
                phase = "climb"
                phase_start_time = t
                print(f"  t={t:6.1f}s: CLIMB phase (alt={altitude:.0f} ft)")
        elif phase == "climb":
            if altitude >= 19500.0:
                phase = "cruise"
                phase_start_time = t
                autopilot.set_target_altitude(-20000.0)
                print(f"  t={t:6.1f}s: CRUISE phase (alt={altitude:.0f} ft)")
        elif phase == "cruise":
            if t - phase_start_time >= 300.0:  # 5 minutes
                phase = "descent"
                phase_start_time = t
                autopilot.set_target_altitude(-1000.0)
                print(f"  t={t:6.1f}s: DESCENT phase")
        elif phase == "descent":
            if altitude <= 1200.0:
                phase = "landing"
                phase_start_time = t
                autopilot.set_target_altitude(-500.0)
                print(f"  t={t:6.1f}s: LANDING phase")
        elif phase == "landing":
            if altitude <= 50.0 or airspeed < 100.0:
                phase = "complete"
                print(f"  t={t:6.1f}s: COMPLETE")

        # Phase-specific controls
        if phase == "takeoff":
            # Simple takeoff: full throttle, pitch up when fast enough
            throttle = 1.0
            if airspeed < 160.0:
                elevon = np.radians(0.0)  # Level on ground
            elif airspeed < 200.0:
                elevon = np.radians(10.0)  # Aggressive rotation
            else:
                elevon = np.radians(5.0)  # Maintain climb

        elif phase == "climb":
            # Climb: maintain climb pitch, full throttle
            autopilot.set_target_altitude(-20000.0)
            elevon = autopilot.update(
                current_altitude=altitude,
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_airspeed=airspeed,
                current_alpha=state.alpha,
                dt=dt
            )
            throttle = 0.95  # High power for climb

        elif phase == "cruise":
            # Cruise: hold altitude and airspeed
            elevon = autopilot.update(
                current_altitude=altitude,
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_airspeed=airspeed,
                current_alpha=state.alpha,
                dt=dt
            )
            # Aggressive airspeed hold - target 600 ft/s
            target_airspeed = 600.0
            airspeed_error = target_airspeed - airspeed
            # Use much higher gain to maintain airspeed
            throttle = 0.80 + 0.05 * airspeed_error  # 3.3x more responsive
            throttle = np.clip(throttle, 0.5, 1.0)  # Allow higher throttle range

        elif phase == "descent":
            # Descent: reduce altitude, maintain safe airspeed
            elevon = autopilot.update(
                current_altitude=altitude,
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_airspeed=airspeed,
                current_alpha=state.alpha,
                dt=dt
            )
            # Maintain descent airspeed (400 ft/s) with throttle control
            target_airspeed = 400.0
            airspeed_error = target_airspeed - airspeed
            throttle = 0.40 + 0.03 * airspeed_error
            throttle = np.clip(throttle, 0.15, 0.70)

        elif phase == "landing":
            # Landing: gentle approach maintaining 250 ft/s
            elevon = autopilot.update(
                current_altitude=altitude,
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_airspeed=airspeed,
                current_alpha=state.alpha,
                dt=dt
            )
            # Maintain landing airspeed (250 ft/s)
            target_airspeed = 250.0
            airspeed_error = target_airspeed - airspeed
            throttle = 0.25 + 0.02 * airspeed_error
            throttle = np.clip(throttle, 0.10, 0.50)

        else:  # complete
            elevon = 0.0
            throttle = 0.0

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

        # Simple Euler integration
        state_dot = dynamics.state_derivative(state, force_func)
        state_array = state.to_array()
        state_new = state_array + state_dot * dt
        state.from_array(state_new)

        # Ground collision
        if state.position[2] > 0:
            state.position[2] = 0
            state.velocity_body[2] = 0

        t += dt
        step += 1

        # Progress
        if step % 200 == 0:
            print(f"  t={t:6.1f}s | Phase: {phase:8s} | Alt: {altitude:7.0f} ft | V: {airspeed:5.1f} ft/s")

    print()
    print("=" * 70)
    print("Mission Complete!")
    print("=" * 70)
    print()

    # Convert to arrays
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    euler_angles = np.array(euler_angles)
    airspeeds = np.array(airspeeds)
    throttles = np.array(throttles)
    elevons = np.array(elevons)

    # Statistics
    print(f"Flight Duration: {times[-1]/60:.1f} minutes")
    print(f"Max Altitude: {-positions[:, 2].min():.0f} ft")
    print(f"Max Airspeed: {max(airspeeds):.1f} ft/s")
    print(f"Distance Traveled: {positions[-1, 0]/5280:.1f} miles")
    print()

    # Create visualizations
    print("Creating visualizations...")

    # 3D trajectory
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = positions[:, 0] / 5280.0  # miles
    y = positions[:, 1] / 5280.0
    z = -positions[:, 2]  # altitude

    ax.plot(x, y, z, 'b-', linewidth=2, label='Flight Path')
    ax.scatter(x[0], y[0], z[0], c='g', s=200, marker='o', label='Takeoff')
    ax.scatter(x[-1], y[-1], z[-1], c='r', s=200, marker='o', label='Landing')

    ax.set_xlabel('X (miles)')
    ax.set_ylabel('Y (miles)')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title('Flying Wing - Simplified Mission Profile')
    ax.legend()
    ax.grid(True)

    plt.savefig('output/mission_simple_3d.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Timeline
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(times / 60.0, -positions[:, 2], 'b-', linewidth=2)
    axes[0].set_ylabel('Altitude (ft)')
    axes[0].set_title('Mission Timeline')
    axes[0].grid(True)

    axes[1].plot(times / 60.0, airspeeds, 'g-', linewidth=2)
    axes[1].set_ylabel('Airspeed (ft/s)')
    axes[1].grid(True)

    axes[2].plot(times / 60.0, np.degrees(euler_angles[:, 1]), 'r-', linewidth=2)
    axes[2].set_ylabel('Pitch (deg)')
    axes[2].grid(True)

    axes[3].plot(times / 60.0, throttles, 'k-', linewidth=2)
    axes[3].set_ylabel('Throttle')
    axes[3].set_xlabel('Time (minutes)')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig('output/mission_simple_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: output/mission_simple_3d.png")
    print("  Saved: output/mission_simple_timeline.png")
    print()
    print("Mission visualization complete!")


if __name__ == "__main__":
    main()
