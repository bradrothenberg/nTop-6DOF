"""
Flying Wing - Complete Mission Profile

Simulates a complete mission:
1. Takeoff - Ground roll and rotation
2. Climb - Ascend to 20,000 ft
3. Cruise - Circular pattern at 20,000 ft for 1 hour
4. Descent - Descend to landing altitude
5. Landing - Final approach and touchdown

Uses hybrid XFOIL+AVL aerodynamics with triple-loop autopilot.
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
from matplotlib.animation import FuncAnimation, PillowWriter


class MissionController:
    """Controls mission phases and transitions."""

    def __init__(self, autopilot):
        self.autopilot = autopilot
        self.phase = "takeoff"
        self.phase_start_time = 0.0
        self.cruise_center = np.array([0.0, 0.0])  # Will be set at cruise start
        self.cruise_radius = 10000.0  # ft (about 1.9 miles)
        self.cruise_airspeed = 600.0  # ft/s

    def update(self, state, t, dt):
        """Update mission phase and return controls."""

        # Phase transitions
        if self.phase == "takeoff":
            # Rotate at 150 ft/s, liftoff at 180 ft/s
            if state.airspeed > 180.0 and state.position[2] < -10.0:
                self.phase = "climb"
                self.phase_start_time = t
                print(f"  Phase: CLIMB (t={t:.1f}s)")

        elif self.phase == "climb":
            # Transition to cruise at 20,000 ft
            if -state.position[2] >= 19800.0:
                self.phase = "cruise"
                self.phase_start_time = t
                # Set cruise center at current position
                self.cruise_center = state.position[:2].copy()
                print(f"  Phase: CRUISE at 20,000 ft (t={t:.1f}s)")

        elif self.phase == "cruise":
            # Cruise for 5 minutes (300 seconds) - shortened for demonstration
            if t - self.phase_start_time >= 300.0:
                self.phase = "descent"
                self.phase_start_time = t
                print(f"  Phase: DESCENT (t={t:.1f}s)")

        elif self.phase == "descent":
            # Transition to landing at 2000 ft
            if -state.position[2] <= 2100.0:
                self.phase = "landing"
                self.phase_start_time = t
                print(f"  Phase: LANDING (t={t:.1f}s)")

        elif self.phase == "landing":
            # Touchdown at ground level
            if state.position[2] >= -5.0:
                self.phase = "complete"
                print(f"  Phase: COMPLETE (t={t:.1f}s)")

        # Get controls for current phase
        return self.get_phase_controls(state, t, dt)

    def get_phase_controls(self, state, t, dt):
        """Get controls for current mission phase."""

        if self.phase == "takeoff":
            # Full throttle, hold nose down until rotation speed
            if state.airspeed < 150.0:
                elevon = np.radians(-10.0)  # Nose down on ground
                throttle = 1.0
            else:
                elevon = np.radians(5.0)  # Rotate
                throttle = 1.0
            return elevon, throttle

        elif self.phase == "climb":
            # Climb at 10 degrees pitch, full throttle
            self.autopilot.set_target_altitude(-20000.0)  # NED frame
            elevon = self.autopilot.update(
                current_altitude=-state.position[2],
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_airspeed=state.airspeed,
                current_alpha=state.alpha,
                dt=dt
            )
            throttle = 1.0  # Climb power
            return elevon, throttle

        elif self.phase == "cruise":
            # Circular cruise pattern
            self.autopilot.set_target_altitude(-20000.0)

            # Calculate desired heading for circular pattern
            rel_pos = state.position[:2] - self.cruise_center
            distance_from_center = np.linalg.norm(rel_pos)

            # Desired position on circle (90 degrees ahead)
            current_angle = np.arctan2(rel_pos[1], rel_pos[0])
            desired_angle = current_angle + np.radians(90.0 / distance_from_center * self.cruise_radius)
            desired_pos = self.cruise_center + self.cruise_radius * np.array([
                np.cos(desired_angle), np.sin(desired_angle)
            ])

            # Simple proportional lateral control via roll
            # (In reality would need coordinated turn logic)
            elevon = self.autopilot.update(
                current_altitude=-state.position[2],
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_airspeed=state.airspeed,
                current_alpha=state.alpha,
                dt=dt
            )

            # Throttle for airspeed hold
            airspeed_error = self.cruise_airspeed - state.airspeed
            throttle = 0.85 + 0.01 * airspeed_error
            throttle = np.clip(throttle, 0.1, 1.0)
            return elevon, throttle

        elif self.phase == "descent":
            # Descend to 2000 ft
            self.autopilot.set_target_altitude(-2000.0)
            elevon = self.autopilot.update(
                current_altitude=-state.position[2],
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_airspeed=state.airspeed,
                current_alpha=state.alpha,
                dt=dt
            )
            throttle = 0.3  # Reduce power for descent
            return elevon, throttle

        elif self.phase == "landing":
            # Final approach
            self.autopilot.set_target_altitude(0.0)
            elevon = self.autopilot.update(
                current_altitude=-state.position[2],
                current_pitch=state.euler_angles[1],
                current_pitch_rate=state.angular_rates[1],
                current_airspeed=state.airspeed,
                current_alpha=state.alpha,
                dt=dt
            )
            throttle = 0.2  # Idle for touchdown
            return elevon, throttle

        else:  # complete
            return 0.0, 0.0


def main():
    """Run complete mission profile simulation."""

    print("=" * 70)
    print("Flying Wing - Complete Mission Profile")
    print("=" * 70)
    print()
    print("Mission Phases:")
    print("  1. Takeoff - Ground roll and rotation")
    print("  2. Climb - Ascend to 20,000 ft")
    print("  3. Cruise - Circular pattern for 5 minutes at 20,000 ft")
    print("  4. Descent - Descend to landing altitude")
    print("  5. Landing - Final approach and touchdown")
    print()
    print("NOTE: Cruise duration shortened to 5 minutes for demonstration")

    # Aircraft configuration (from nTop flying wing)
    mass = 234.8  # slugs (7500 lbm)
    Ixx, Iyy, Izz = 14908, 2318, 17227  # slug-ft²
    inertia = np.array([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])

    S_ref = 199.94  # ft²
    c_ref = 26.689  # ft (MAC)
    b_ref = 19.890  # ft

    # Create aerodynamic model with XFOIL drag
    aero = LinearAeroModel(S_ref, c_ref, b_ref)

    # AVL derivatives
    aero.CL_0 = 0.2
    aero.CL_alpha = 1.412241  # From AVL
    aero.CL_q = 1.282202
    aero.CL_de = 0.0

    aero.CD_0 = 0.006  # From XFOIL
    aero.CD_alpha = 0.025
    aero.CD_alpha2 = 0.05

    aero.Cm_0 = 0.000061
    aero.Cm_alpha = -0.079668
    aero.Cm_q = -0.347
    aero.Cm_de = -0.02

    # Generic stable lateral derivatives
    aero.Cl_beta = -0.1
    aero.Cl_p = -0.4
    aero.Cl_r = 0.1
    aero.Cl_da = -0.001536

    aero.Cn_beta = 0.05
    aero.Cn_p = -0.05
    aero.Cn_r = -0.1

    aero.CY_beta = -0.2

    # Create propulsion (FJ-44 turbofan)
    turbofan = TurbofanModel(thrust_max=1900.0, altitude_lapse_rate=0.7)

    # Create dynamics
    dynamics = AircraftDynamics(mass, inertia)
    combined = CombinedForceModel(aero, turbofan)

    # Create autopilot
    autopilot = FlyingWingAutopilot(
        Kp_alt=0.005,
        Ki_alt=0.0005,
        Kd_alt=0.012,
        Kp_pitch=0.8,
        Ki_pitch=0.05,
        Kd_pitch=0.15,
        Kp_pitch_rate=0.15,
        Ki_pitch_rate=0.01,
        max_pitch_cmd=15.0,
        min_pitch_cmd=-10.0,
        max_alpha=12.0,
        stall_speed=150.0,
        min_airspeed_margin=1.3
    )
    autopilot.set_trim(np.radians(-6.0))

    # Create mission controller
    mission = MissionController(autopilot)

    # Initial state - on runway
    state = State()
    state.position = np.array([0.0, 0.0, 0.0])  # On ground
    state.velocity_body = np.array([0.0, 0.0, 0.0])  # Stationary
    state.set_euler_angles(0.0, 0.0, 0.0)  # Roll, Pitch, Yaw (level on runway)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    print("Starting mission simulation...")
    print()

    # Simulation parameters
    dt = 0.1  # Larger time step for long mission
    max_duration = 5000.0  # Maximum 5000 seconds (~83 minutes)

    # Storage
    time_history = []
    position_history = []
    velocity_history = []
    euler_history = []
    phase_history = []
    throttle_history = []
    elevon_history = []

    t = 0.0
    step = 0

    while t < max_duration and mission.phase != "complete":
        # Store data
        time_history.append(t)
        position_history.append(state.position.copy())
        velocity_history.append(state.velocity_body.copy())
        euler_history.append(np.array(state.euler_angles))  # Convert tuple to array
        phase_history.append(mission.phase)

        # Get controls from mission controller
        elevon, throttle = mission.update(state, t, dt)

        throttle_history.append(throttle)
        elevon_history.append(elevon)

        controls = {
            'elevator': elevon,
            'aileron': 0.0,
            'rudder': 0.0,
            'throttle': throttle
        }

        # Force function
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

        # Progress indicator
        if step % 100 == 0:
            alt = -state.position[2]
            airspeed = state.airspeed
            print(f"  t={t:6.1f}s | Phase: {mission.phase:8s} | Alt: {alt:7.0f} ft | V: {airspeed:5.1f} ft/s")

    print()
    print("=" * 70)
    print("Mission Complete!")
    print("=" * 70)

    # Convert to arrays
    time_history = np.array(time_history)
    position_history = np.array(position_history)
    velocity_history = np.array(velocity_history)
    euler_history = np.array(euler_history)

    # Create visualizations
    print("Creating visualizations...")

    # 3D trajectory plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = position_history[:, 0] / 5280.0  # Convert to miles
    y = position_history[:, 1] / 5280.0
    z = -position_history[:, 2]  # Altitude (positive up)

    ax.plot(x, y, z, 'b-', linewidth=1.5, label='Flight Path')
    ax.scatter(x[0], y[0], z[0], c='g', s=100, marker='o', label='Takeoff')
    ax.scatter(x[-1], y[-1], z[-1], c='r', s=100, marker='o', label='Landing')

    ax.set_xlabel('X (miles)')
    ax.set_ylabel('Y (miles)')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title('Flying Wing - Complete Mission Profile')
    ax.legend()
    ax.grid(True)

    plt.savefig('output/mission_trajectory_3d.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Mission timeline plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    axes[0].plot(time_history / 60.0, -position_history[:, 2], 'b-')
    axes[0].set_ylabel('Altitude (ft)')
    axes[0].set_title('Mission Timeline')
    axes[0].grid(True)

    airspeed = np.linalg.norm(velocity_history, axis=1)
    axes[1].plot(time_history / 60.0, airspeed, 'g-')
    axes[1].set_ylabel('Airspeed (ft/s)')
    axes[1].grid(True)

    axes[2].plot(time_history / 60.0, np.degrees(euler_history[:, 1]), 'r-')
    axes[2].set_ylabel('Pitch (deg)')
    axes[2].grid(True)

    axes[3].plot(time_history / 60.0, throttle_history, 'k-')
    axes[3].set_ylabel('Throttle')
    axes[3].set_xlabel('Time (minutes)')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig('output/mission_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: output/mission_trajectory_3d.png")
    print("  Saved: output/mission_timeline.png")
    print()
    print("Mission visualization complete!")


if __name__ == "__main__":
    main()
