"""
Flying Wing - Mission Profile with Gain Scheduling

Demonstrates a complete mission using gain-scheduled autopilot:
1. Initial Cruise - Level flight at 5,000 ft (1 minute)
2. Climb - Ascend to 10,000 ft
3. High Cruise - Level flight at 10,000 ft (2 minutes)
4. Descent - Descend to 5,000 ft
5. Final Cruise - Return to 5,000 ft level flight (1 minute)

Uses gain scheduling to handle large altitude changes stably.
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
from src.control.gain_scheduled_autopilot import GainScheduledAutopilot
import matplotlib.pyplot as plt


def main():
    """Run mission profile with gain scheduling."""

    print("=" * 70)
    print("Flying Wing - Mission Profile (Gain Scheduled)")
    print("=" * 70)
    print()
    print("Mission Phases:")
    print("  1. Initial Cruise - Level flight at 5,000 ft (1 minute)")
    print("  2. Climb - Ascend to 10,000 ft")
    print("  3. High Cruise - Level flight at 10,000 ft (2 minutes)")
    print("  4. Descent - Descend to 5,000 ft")
    print("  5. Final Cruise - Return to 5,000 ft level flight (1 minute)")
    print()
    print("Using gain scheduling for stable altitude changes")
    print()

    # Aircraft configuration - EXACT match to working flyingwing_hybrid_aero.py
    mass = 228.924806  # slugs
    inertia = np.array([[19236.2914, 0.0, 0.0],
                        [0.0, 2251.0172, 0.0],
                        [0.0, 0.0, 21487.3086]])

    S_ref = 412.6370  # ft²
    c_ref = 11.9555  # ft
    b_ref = 24.8630  # ft

    # Create aerodynamic model - EXACT coefficients from working example
    aero = LinearAeroModel(S_ref, c_ref, b_ref)
    aero.CL_0 = 0.000023
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
    # Generic lateral derivatives
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

    # Create GAIN-SCHEDULED autopilot
    autopilot = GainScheduledAutopilot()
    autopilot.set_trim(np.radians(-5.66))  # CORRECT trim from working example

    # Initial state - CORRECT trim conditions from working example
    state = State()
    state.position = np.array([0.0, 0.0, -5000.0])  # 5000 ft altitude (NED)
    state.velocity_body = np.array([600.0, 0.0, 0.0])  # 600 ft/s
    state.set_euler_angles(0.0, np.radians(1.4649), 0.0)  # CORRECT trim pitch
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    print(f"Initial State (at trim):")
    print(f"  Altitude: {-state.position[2]:.0f} ft")
    print(f"  Airspeed: {state.airspeed:.0f} ft/s")
    print(f"  Pitch: {np.degrees(state.euler_angles[1]):.1f} deg")
    print(f"  Autopilot mode: {autopilot.get_current_mode()}")
    print()

    # Simulation parameters
    dt = 0.05
    max_time = 400.0  # 6.7 minutes

    # Storage
    times = []
    positions = []
    velocities = []
    euler_angles = []
    airspeeds = []
    throttles = []
    elevons = []
    phases = []
    modes = []

    # Mission state
    phase = "initial_cruise"
    phase_start_time = 0.0
    t = 0.0

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
        modes.append(autopilot.get_current_mode())

        altitude = -state.position[2]
        airspeed = state.airspeed

        # Phase transitions with gain mode changes
        if phase == "initial_cruise":
            if t - phase_start_time >= 60.0:  # 1 minute
                phase = "climb"
                phase_start_time = t
                autopilot.set_target_altitude(-10000.0)
                autopilot.set_mode('climb', dt)  # Switch to climb gains
                print(f"  t={t:6.1f}s: CLIMB phase (switching to climb gains)")

        elif phase == "climb":
            if altitude >= 9500.0:
                phase = "high_cruise"
                phase_start_time = t
                autopilot.set_target_altitude(-10000.0)
                autopilot.set_mode('cruise', dt)  # Switch to cruise gains
                print(f"  t={t:6.1f}s: HIGH CRUISE phase at {altitude:.0f} ft (switching to cruise gains)")

        elif phase == "high_cruise":
            if t - phase_start_time >= 120.0:  # 2 minutes
                phase = "descent"
                phase_start_time = t
                autopilot.set_target_altitude(-5000.0)
                autopilot.set_mode('descent', dt)  # Switch to descent gains
                print(f"  t={t:6.1f}s: DESCENT phase (switching to descent gains)")

        elif phase == "descent":
            if altitude <= 5500.0:
                phase = "final_cruise"
                phase_start_time = t
                autopilot.set_target_altitude(-5000.0)
                autopilot.set_mode('cruise', dt)  # Switch to cruise gains
                print(f"  t={t:6.1f}s: FINAL CRUISE phase (switching to cruise gains)")

        elif phase == "final_cruise":
            if t - phase_start_time >= 60.0:  # 1 minute
                phase = "complete"
                print(f"  t={t:6.1f}s: COMPLETE")

        # Get elevon command from autopilot
        elevon = autopilot.update(
            current_altitude=altitude,
            current_pitch=state.euler_angles[1],
            current_pitch_rate=state.angular_rates[1],
            current_airspeed=airspeed,
            current_alpha=state.alpha,
            dt=dt
        )

        # Phase-specific throttle control
        target_airspeed = 600.0  # Maintain trim airspeed throughout
        airspeed_error = target_airspeed - airspeed

        if phase == "initial_cruise" or phase == "final_cruise":
            throttle = 0.80 + 0.015 * airspeed_error
            throttle = np.clip(throttle, 0.5, 1.0)
        elif phase == "climb":
            throttle = 0.90 + 0.015 * airspeed_error
            throttle = np.clip(throttle, 0.7, 1.0)
        elif phase == "high_cruise":
            throttle = 0.85 + 0.015 * airspeed_error
            throttle = np.clip(throttle, 0.6, 1.0)
        elif phase == "descent":
            throttle = 0.70 + 0.015 * airspeed_error
            throttle = np.clip(throttle, 0.4, 0.9)
        else:  # complete
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
        if step % 200 == 0:
            print(f"  t={t:6.1f}s | Phase: {phase:15s} | Mode: {autopilot.get_current_mode():7s} | Alt: {altitude:7.0f} ft | V: {airspeed:5.1f} ft/s")

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
    print(f"Min Altitude: {-positions[:, 2].max():.0f} ft")
    print(f"Max Airspeed: {max(airspeeds):.1f} ft/s")
    print(f"Min Airspeed: {min(airspeeds):.1f} ft/s")
    print(f"Distance Traveled: {positions[-1, 0]/5280:.1f} miles")
    print()

    # Create visualizations
    print("Creating visualizations...")
    os.makedirs('output', exist_ok=True)

    # 3D trajectory
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = positions[:, 0] / 5280.0  # miles
    y = positions[:, 1] / 5280.0
    z = -positions[:, 2]  # altitude

    ax.plot(x, y, z, 'b-', linewidth=2, label='Flight Path')
    ax.scatter(x[0], y[0], z[0], c='g', s=200, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='r', s=200, marker='o', label='End')

    ax.set_xlabel('X (miles)')
    ax.set_ylabel('Y (miles)')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title('Flying Wing - Gain-Scheduled Mission Profile')
    ax.legend()
    ax.grid(True)

    plt.savefig('output/mission_scheduled_3d.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Timeline
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(times / 60.0, -positions[:, 2], 'b-', linewidth=2)
    axes[0].set_ylabel('Altitude (ft)')
    axes[0].set_title('Mission Timeline - Gain Scheduled Autopilot')
    axes[0].grid(True)

    axes[1].plot(times / 60.0, airspeeds, 'g-', linewidth=2)
    axes[1].set_ylabel('Airspeed (ft/s)')
    axes[1].grid(True)

    axes[2].plot(times / 60.0, np.degrees(euler_angles[:, 1]), 'r-', linewidth=2)
    axes[2].set_ylabel('Pitch (deg)')
    axes[2].grid(True)

    axes[3].plot(times / 60.0, throttles, 'k-', linewidth=2)
    axes[3].set_ylabel('Throttle')
    axes[3].grid(True)

    axes[4].plot(times / 60.0, elevons, 'm-', linewidth=2)
    axes[4].set_ylabel('Elevon (deg)')
    axes[4].set_xlabel('Time (minutes)')
    axes[4].grid(True)

    plt.tight_layout()
    plt.savefig('output/mission_scheduled_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: output/mission_scheduled_3d.png")
    print("  Saved: output/mission_scheduled_timeline.png")
    print()
    print("Mission visualization complete!")


if __name__ == "__main__":
    main()
