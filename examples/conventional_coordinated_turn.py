"""
Conventional Tail Configuration - Coordinated Turn Maneuver

Demonstrates a coordinated turn using the table-based aerodynamic model.

Turn sequence:
1. Straight and level flight (0-10s)
2. Roll into 30° bank angle (10-15s)
3. Maintain coordinated turn (15-40s)
4. Roll back to wings level (40-45s)
5. Straight and level flight (45-60s)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.table_aero import TableAeroModel
from src.core.propulsion import TurbofanModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere
from src.visualization.plotting import (
    plot_trajectory_3d,
    plot_states_vs_time,
    setup_plotting_style
)

import matplotlib.pyplot as plt
from pathlib import Path


class CoordinatedTurnAutopilot:
    """
    Autopilot for coordinated turns.

    Controls:
    - Roll: Commands bank angle
    - Pitch: Maintains altitude
    - Yaw: Coordinates turn (sideslip ~ 0)
    """

    def __init__(self,
                 # Roll control gains
                 Kp_roll=0.8, Ki_roll=0.1, Kd_roll=0.3,
                 # Pitch control gains (altitude hold)
                 Kp_alt=0.0005, Ki_alt=0.00002, Kd_alt=0.001,
                 Kp_pitch=0.1, Ki_pitch=0.01, Kd_pitch=0.02,
                 Kp_pitch_rate=0.02, Ki_pitch_rate=0.002,
                 # Yaw control gains (coordination)
                 Kp_yaw=0.5, Kd_yaw=0.1):

        # Roll control
        self.Kp_roll = Kp_roll
        self.Ki_roll = Ki_roll
        self.Kd_roll = Kd_roll
        self.roll_error_integral = 0
        self.prev_roll_error = 0

        # Pitch control (altitude hold)
        self.Kp_alt = Kp_alt
        self.Ki_alt = Ki_alt
        self.Kd_alt = Kd_alt
        self.Kp_pitch = Kp_pitch
        self.Ki_pitch = Ki_pitch
        self.Kd_pitch = Kd_pitch
        self.Kp_pitch_rate = Kp_pitch_rate
        self.Ki_pitch_rate = Ki_pitch_rate

        self.alt_error_integral = 0
        self.pitch_error_integral = 0
        self.pitch_rate_error_integral = 0
        self.prev_alt_error = 0

        # Yaw control (sideslip coordination)
        self.Kp_yaw = Kp_yaw
        self.Kd_yaw = Kd_yaw

        # Target values
        self.target_roll = 0
        self.target_altitude = 5000

    def set_target_roll(self, roll_deg):
        """Set target bank angle (degrees)."""
        self.target_roll = np.radians(roll_deg)

    def set_target_altitude(self, altitude_ft):
        """Set target altitude (feet)."""
        self.target_altitude = altitude_ft

    def update(self, state, dt):
        """
        Update autopilot and return control commands.

        Returns:
        --------
        controls : dict
            'aileron': Aileron deflection (radians)
            'elevator': Elevator deflection (radians)
            'rudder': Rudder deflection (radians)
        """

        # === ROLL CONTROL (Bank Angle) ===
        roll, pitch, yaw = state.euler_angles
        p, q, r = state.angular_rates

        roll_error = self.target_roll - roll
        self.roll_error_integral += roll_error * dt
        roll_error_derivative = (roll_error - self.prev_roll_error) / dt

        # PID roll control -> aileron
        aileron = (self.Kp_roll * roll_error +
                   self.Ki_roll * self.roll_error_integral +
                   self.Kd_roll * roll_error_derivative)

        aileron = np.clip(aileron, np.radians(-25), np.radians(25))
        self.prev_roll_error = roll_error

        # === PITCH CONTROL (Altitude Hold) ===
        altitude = state.altitude
        alt_error = self.target_altitude - altitude
        self.alt_error_integral += alt_error * dt
        alt_error_derivative = (alt_error - self.prev_alt_error) / dt

        # Outer loop: altitude -> pitch command
        pitch_cmd = (self.Kp_alt * alt_error +
                     self.Ki_alt * self.alt_error_integral +
                     self.Kd_alt * alt_error_derivative)

        pitch_cmd = np.clip(pitch_cmd, np.radians(-10), np.radians(10))
        self.prev_alt_error = alt_error

        # Middle loop: pitch -> pitch rate command
        pitch_error = pitch_cmd - pitch
        self.pitch_error_integral += pitch_error * dt

        pitch_rate_cmd = (self.Kp_pitch * pitch_error +
                          self.Ki_pitch * self.pitch_error_integral +
                          self.Kd_pitch * (pitch_error - getattr(self, 'prev_pitch_error', 0)) / dt)

        self.prev_pitch_error = pitch_error

        # Inner loop: pitch rate -> elevator
        pitch_rate_error = pitch_rate_cmd - q
        self.pitch_rate_error_integral += pitch_rate_error * dt

        elevator = (self.Kp_pitch_rate * pitch_rate_error +
                    self.Ki_pitch_rate * self.pitch_rate_error_integral)

        elevator = np.clip(elevator, np.radians(-25), np.radians(25))

        # === YAW CONTROL (Sideslip Coordination) ===
        # Estimate sideslip from lateral velocity
        V_body = state.velocity_body
        V = np.linalg.norm(V_body)
        if V > 1.0:
            beta = np.arcsin(np.clip(V_body[1] / V, -1, 1))
        else:
            beta = 0

        # Coordinate turn: use rudder to zero sideslip
        rudder = -(self.Kp_yaw * beta + self.Kd_yaw * r)
        rudder = np.clip(rudder, np.radians(-25), np.radians(25))

        return {
            'aileron': aileron,
            'elevator': elevator,
            'rudder': rudder
        }


def main():
    """Run coordinated turn simulation."""

    print("\n" + "=" * 70)
    print("Conventional Tail - Coordinated Turn Simulation")
    print("Using Table-Based Aerodynamic Model from AVL Data")
    print("=" * 70 + "\n")

    # Aircraft mass properties
    mass = 228.925  # slugs
    inertia = np.array([
        [19236.29, 0.0, 0.0],
        [0.0, 2251.02, 0.0],
        [0.0, 0.0, 21487.31]
    ])  # slug-ft^2

    # Create table-based aerodynamic model
    table_file = Path("aero_tables/conventional_aero_table.csv")
    if not table_file.exists():
        print(f"ERROR: Aero table not found: {table_file}")
        print("Run generate_aero_table_from_runfile.py first!")
        exit(1)

    aero = TableAeroModel(
        table_file=str(table_file),
        S_ref=199.94,
        c_ref=26.689,
        b_ref=19.890,
        rho=0.002377  # will be updated
    )

    # Create turbofan engine
    turbofan = TurbofanModel(thrust_max=1900.0, altitude_lapse_rate=0.7)

    # Create dynamics
    dynamics = AircraftDynamics(mass, inertia)

    # Flight condition
    altitude = -5000.0  # ft (negative in NED frame)
    airspeed = 450.0  # ft/s

    # Initial state - trimmed straight and level flight
    state = State()
    state.position = np.array([0.0, 0.0, altitude])

    # Start with small angle of attack for lift
    alpha_trim = np.radians(2.5)
    state.velocity_body = np.array([
        airspeed * np.cos(alpha_trim),
        0.0,
        airspeed * np.sin(alpha_trim)
    ])
    state.set_euler_angles(0.0, alpha_trim, 0.0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    print(f"Initial conditions:")
    print(f"  Altitude: {-altitude:.0f} ft")
    print(f"  Airspeed: {airspeed:.0f} ft/s")
    print(f"  Alpha: {np.degrees(alpha_trim):.2f}°")
    print()

    # Create autopilot
    autopilot = CoordinatedTurnAutopilot()
    autopilot.set_target_altitude(-altitude)

    # Simulation parameters
    dt = 0.01  # 10ms time step
    duration = 60.0  # 60 seconds
    num_steps = int(duration / dt)

    # Data storage
    time_history = []
    states_history = []
    controls_history = []

    # Throttle setting (constant for simplicity)
    throttle = 0.60  # 60% throttle

    print("Turn sequence:")
    print("  0-10s:  Straight and level")
    print("  10-15s: Roll into 30° bank")
    print("  15-40s: Maintain coordinated turn")
    print("  40-45s: Roll back to wings level")
    print("  45-60s: Straight and level")
    print()
    print("Running simulation...")
    print()

    # Simulation loop
    for step in range(num_steps):
        t = step * dt

        # Turn profile
        if t < 10.0:
            # Straight and level
            target_roll_deg = 0
        elif t < 15.0:
            # Roll into turn (30° bank)
            target_roll_deg = 30.0 * (t - 10.0) / 5.0
        elif t < 40.0:
            # Maintain turn
            target_roll_deg = 30.0
        elif t < 45.0:
            # Roll out of turn
            target_roll_deg = 30.0 * (1.0 - (t - 40.0) / 5.0)
        else:
            # Straight and level
            target_roll_deg = 0

        autopilot.set_target_roll(target_roll_deg)

        # Compute autopilot commands
        controls = autopilot.update(state, dt)
        controls['throttle'] = throttle

        # Store data
        time_history.append(t)
        states_history.append(state.copy())
        controls_history.append(controls.copy())

        # Force function with atmosphere update
        def force_func(s):
            atm = StandardAtmosphere(s.altitude)
            aero.rho = atm.density

            # Compute aero forces
            aero_forces, aero_moments = aero.compute_forces_moments(s, controls)

            # Compute thrust forces
            thrust_forces, _ = turbofan.compute_thrust(s, controls['throttle'])

            # Total forces and moments
            total_forces = aero_forces + thrust_forces
            total_moments = aero_moments

            return total_forces, total_moments

        # RK4 integration
        state = dynamics.propagate_rk4(state, dt, force_func)

        # Progress indicator
        if step % 1000 == 0:
            roll, pitch, yaw = state.euler_angles
            alt_ft = -state.position[2]
            V = np.linalg.norm(state.velocity_body)
            print(f"  t = {t:.1f}s: alt = {alt_ft:.0f} ft, V = {V:.1f} ft/s, "
                  f"roll = {np.degrees(roll):.1f}°, heading = {np.degrees(yaw):.1f}°")

    print("\nSimulation complete!")
    print()

    # Analysis
    print("=" * 70)
    print("TURN PERFORMANCE")
    print("=" * 70)

    positions = np.array([s.position for s in states_history])
    velocities = np.array([s.velocity_body for s in states_history])
    euler = np.array([s.euler_angles for s in states_history])

    altitudes = -positions[:, 2]
    rolls = np.degrees(euler[:, 0])
    pitches = np.degrees(euler[:, 1])
    headings = np.degrees(euler[:, 2])

    # Turn analysis (15-40s)
    turn_start_idx = int(15.0 / dt)
    turn_end_idx = int(40.0 / dt)

    heading_change = headings[turn_end_idx] - headings[turn_start_idx]
    if heading_change < -180:
        heading_change += 360

    turn_radius = np.mean(np.sqrt(
        (positions[turn_start_idx:turn_end_idx, 0] - np.mean(positions[turn_start_idx:turn_end_idx, 0]))**2 +
        (positions[turn_start_idx:turn_end_idx, 1] - np.mean(positions[turn_start_idx:turn_end_idx, 1]))**2
    ))

    print(f"Turn metrics (15-40s):")
    print(f"  Bank angle: {np.mean(rolls[turn_start_idx:turn_end_idx]):.1f}° (target: 30°)")
    print(f"  Heading change: {heading_change:.1f}°")
    print(f"  Turn radius: {turn_radius:.0f} ft ({turn_radius/6076:.2f} nm)")
    print(f"  Altitude change: {altitudes[turn_end_idx] - altitudes[turn_start_idx]:+.0f} ft")
    print()

    print(f"Overall:")
    print(f"  Altitude std dev: {np.std(altitudes):.1f} ft")
    print(f"  Final position: ({positions[-1, 0]:.0f}, {positions[-1, 1]:.0f}) ft")
    print()

    print("=" * 70)

    # Visualization
    setup_plotting_style()

    # 3D trajectory with turn highlighted
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Convert to conventional axes (North, East, Down -> X, Y, -Z)
    x = positions[:, 1] / 6076  # East in nm
    y = positions[:, 0] / 6076  # North in nm
    z = -positions[:, 2]        # Up in ft

    # Plot full trajectory
    ax.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')

    # Highlight turn segment
    x_turn = x[turn_start_idx:turn_end_idx]
    y_turn = y[turn_start_idx:turn_end_idx]
    z_turn = z[turn_start_idx:turn_end_idx]
    ax.plot(x_turn, y_turn, z_turn, 'r-', linewidth=3, label='Coordinated Turn')

    # Mark start and end
    ax.scatter([x[0]], [y[0]], [z[0]], c='g', s=100, marker='o', label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, marker='s', label='End')

    ax.set_xlabel('East (nm)')
    ax.set_ylabel('North (nm)')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title('Coordinated Turn - 3D Trajectory', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("output/conventional_turn_3d.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Time history plots
    fig, axes = plt.subplots(5, 1, figsize=(12, 14))

    # Altitude
    axes[0].plot(time_history, altitudes, 'b-', linewidth=1.5)
    axes[0].axhline(5000, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0].axvspan(15, 40, color='yellow', alpha=0.2, label='Turn')
    axes[0].set_ylabel('Altitude (ft)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Roll angle
    axes[1].plot(time_history, rolls, 'g-', linewidth=1.5, label='Actual')
    axes[1].plot([10, 15], [0, 30], 'r--', linewidth=2, label='Target')
    axes[1].plot([15, 40], [30, 30], 'r--', linewidth=2)
    axes[1].plot([40, 45], [30, 0], 'r--', linewidth=2)
    axes[1].axvspan(15, 40, color='yellow', alpha=0.2)
    axes[1].set_ylabel('Roll Angle (°)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Heading
    axes[2].plot(time_history, headings, 'purple', linewidth=1.5)
    axes[2].axvspan(15, 40, color='yellow', alpha=0.2)
    axes[2].set_ylabel('Heading (°)')
    axes[2].grid(True, alpha=0.3)

    # Control surfaces
    ailerons = np.degrees([c['aileron'] for c in controls_history])
    elevators = np.degrees([c['elevator'] for c in controls_history])
    rudders = np.degrees([c['rudder'] for c in controls_history])

    axes[3].plot(time_history, ailerons, 'b-', linewidth=1.5, label='Aileron')
    axes[3].plot(time_history, elevators, 'g-', linewidth=1.5, label='Elevator')
    axes[3].plot(time_history, rudders, 'r-', linewidth=1.5, label='Rudder')
    axes[3].axvspan(15, 40, color='yellow', alpha=0.2)
    axes[3].set_ylabel('Control Surfaces (°)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # Ground track
    axes[4].plot(positions[:, 1]/6076, positions[:, 0]/6076, 'b-', linewidth=2)
    axes[4].plot(positions[turn_start_idx:turn_end_idx, 1]/6076,
                 positions[turn_start_idx:turn_end_idx, 0]/6076,
                 'r-', linewidth=3, label='Turn')
    axes[4].scatter([positions[0, 1]/6076], [positions[0, 0]/6076], c='g', s=100, marker='o', label='Start')
    axes[4].scatter([positions[-1, 1]/6076], [positions[-1, 0]/6076], c='r', s=100, marker='s', label='End')
    axes[4].set_xlabel('East (nm)')
    axes[4].set_ylabel('North (nm)')
    axes[4].set_title('Ground Track')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    axes[4].set_aspect('equal')

    fig.suptitle('Coordinated Turn Performance - Table-Based Aero Model', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig("output/conventional_turn_performance.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nPlots saved to output/ directory")
    print("  - conventional_turn_3d.png")
    print("  - conventional_turn_performance.png")
    print()


if __name__ == "__main__":
    main()
