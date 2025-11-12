"""
Conventional Tail Configuration - Coordinated Turn with Trim Solver

Demonstrates a properly trimmed coordinated turn using table-based aerodynamics.

Features:
- Trim solver to find level flight condition
- Airspeed hold controller
- Coordinated turn autopilot (roll, pitch, yaw)

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
from src.core.propulsion import TurbofanModel
from src.environment.atmosphere import StandardAtmosphere
from src.visualization.plotting import setup_plotting_style

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize_scalar


def find_trim(mass, aero, turbofan, altitude, airspeed, max_iterations=20):
    """
    Find trim condition for level flight using table-based aero.

    Returns:
    --------
    trim : dict
        'alpha': Angle of attack (rad)
        'theta': Pitch angle (rad)
        'elevator': Elevator deflection (rad)
        'throttle': Throttle setting
    """

    print("=" * 70)
    print("Trim Solver - Table-Based Aerodynamics")
    print("=" * 70)
    print()

    atm = StandardAtmosphere(altitude)
    aero.rho = atm.density

    q_bar = 0.5 * atm.density * airspeed**2
    W = mass * 32.174  # Weight (lbf)

    print(f"Conditions:")
    print(f"  Altitude: {-altitude:.0f} ft")
    print(f"  Airspeed: {airspeed:.0f} ft/s (Mach {airspeed/atm.speed_of_sound:.2f})")
    print(f"  q_bar: {q_bar:.2f} psf")
    print(f"  Weight: {W:.1f} lbf")
    print()

    # Iterative trim solver
    # Start with initial guess
    alpha_deg = 3.0
    elevator_deg = -5.0

    for iteration in range(max_iterations):
        # Get coefficients from table
        CL, CD, Cm = aero.get_coefficients(alpha_deg, elevator_deg)

        # Lift and drag
        L = q_bar * aero.S_ref * CL
        D = q_bar * aero.S_ref * CD
        M = q_bar * aero.S_ref * aero.c_ref * Cm

        # Errors
        lift_error = L - W
        moment_error = M

        if iteration == 0:
            print(f"Initial guess:")
            print(f"  alpha = {alpha_deg:.3f}°, elevator = {elevator_deg:.3f}°")
            print(f"  CL = {CL:.4f}, CD = {CD:.5f}, Cm = {Cm:.4f}")
            print(f"  L = {L:.1f} lbf (need {W:.1f}), error = {lift_error:.1f} lbf")
            print(f"  M = {M:.1f} ft-lbf (need 0), error = {moment_error:.1f} ft-lbf")
            print()

        # Check convergence
        if abs(lift_error) < 10.0 and abs(moment_error) < 100.0:
            print(f"Converged in {iteration + 1} iterations")
            break

        # Update alpha to match lift
        # CL_target = W / (q_bar * S_ref)
        CL_target = W / (q_bar * aero.S_ref)

        # Estimate new alpha (rough linear approximation)
        dCL_dalpha = 0.05  # ~3 /rad for typical airfoil
        alpha_correction = (CL_target - CL) / dCL_dalpha
        alpha_deg += alpha_correction * 0.5  # Damping factor

        # Update elevator to zero moment
        # Estimate dCm/de from table
        # From AVL data: dCm/de = -0.00712 per degree = -0.408 per radian
        Cm_sensitivity = -0.00712  # Per degree (negative means up-elev gives negative Cm)
        # To drive Cm toward zero, we need: delta_e = -Cm / (dCm/de)
        # If Cm is negative, we need positive (up) elevator correction
        elevator_correction = -Cm / Cm_sensitivity
        elevator_deg += elevator_correction * 0.3  # Damping factor (reduced for stability)

        # Clamp to reasonable ranges
        alpha_deg = np.clip(alpha_deg, -2, 10)
        elevator_deg = np.clip(elevator_deg, -15, 5)

    # Final coefficients
    CL, CD, Cm = aero.get_coefficients(alpha_deg, elevator_deg)
    L = q_bar * aero.S_ref * CL
    D = q_bar * aero.S_ref * CD

    # Find throttle for T = D
    alpha_rad = np.radians(alpha_deg)
    theta_rad = alpha_rad  # Level flight

    state_trim = State()
    state_trim.position = np.array([0, 0, altitude])
    state_trim.velocity_body = np.array([
        airspeed * np.cos(alpha_rad),
        0.0,
        airspeed * np.sin(alpha_rad)
    ])
    state_trim.set_euler_angles(0, theta_rad, 0)
    state_trim.angular_rates = np.array([0, 0, 0])

    thrust_max, _ = turbofan.compute_thrust(state_trim, throttle=1.0)
    T_max = thrust_max[0]

    throttle_trim = D / T_max
    throttle_trim = np.clip(throttle_trim, 0.01, 1.0)

    print()
    print("=" * 70)
    print("TRIM SOLUTION")
    print("=" * 70)
    print(f"alpha     = {alpha_deg:7.3f} deg")
    print(f"theta     = {np.degrees(theta_rad):7.3f} deg")
    print(f"elevator  = {elevator_deg:7.3f} deg")
    print(f"throttle  = {throttle_trim:7.4f} ({100*throttle_trim:.1f}%)")
    print()
    print(f"CL        = {CL:7.4f}")
    print(f"CD        = {CD:7.5f}")
    print(f"Cm        = {Cm:7.4f}")
    print()
    print(f"Lift      = {L:7.1f} lbf (Weight = {W:.1f} lbf)")
    print(f"Drag      = {D:7.1f} lbf (Thrust = {D:.1f} lbf @ {100*throttle_trim:.1f}%)")
    print(f"Moment    = {q_bar * aero.S_ref * aero.c_ref * Cm:7.1f} ft-lbf")
    print("=" * 70)
    print()

    return {
        'alpha': alpha_rad,
        'theta': theta_rad,
        'elevator': np.radians(elevator_deg),
        'throttle': throttle_trim,
        'CL': CL,
        'CD': CD,
        'Cm': Cm
    }


class CoordinatedTurnAutopilot:
    """Autopilot for coordinated turns with airspeed hold."""

    def __init__(self,
                 # Roll control
                 Kp_roll=0.5, Ki_roll=0.05, Kd_roll=0.2,
                 # Pitch control (altitude hold)
                 Kp_alt=0.0003, Ki_alt=0.00001, Kd_alt=0.0006,
                 Kp_pitch=0.08, Ki_pitch=0.008, Kd_pitch=0.015,
                 Kp_pitch_rate=0.015, Ki_pitch_rate=0.001,
                 # Yaw control (coordination)
                 Kp_yaw=0.3, Kd_yaw=0.08,
                 # Airspeed control
                 Kp_airspeed=0.01, Ki_airspeed=0.002):

        # Roll control
        self.Kp_roll = Kp_roll
        self.Ki_roll = Ki_roll
        self.Kd_roll = Kd_roll
        self.roll_integral = 0
        self.prev_roll = 0

        # Pitch control
        self.Kp_alt = Kp_alt
        self.Ki_alt = Ki_alt
        self.Kd_alt = Kd_alt
        self.Kp_pitch = Kp_pitch
        self.Ki_pitch = Ki_pitch
        self.Kd_pitch = Kd_pitch
        self.Kp_pitch_rate = Kp_pitch_rate
        self.Ki_pitch_rate = Ki_pitch_rate

        self.alt_integral = 0
        self.pitch_integral = 0
        self.pitch_rate_integral = 0
        self.prev_alt = 0
        self.prev_pitch = 0

        # Yaw control
        self.Kp_yaw = Kp_yaw
        self.Kd_yaw = Kd_yaw

        # Airspeed control
        self.Kp_airspeed = Kp_airspeed
        self.Ki_airspeed = Ki_airspeed
        self.airspeed_integral = 0

        # Targets
        self.target_roll = 0
        self.target_altitude = 5000
        self.target_airspeed = 450
        self.trim_throttle = 0.6
        self.trim_elevator = 0

    def set_trim(self, trim):
        """Set trim values."""
        self.trim_throttle = trim['throttle']
        self.trim_elevator = trim['elevator']

    def set_target_roll(self, roll_deg):
        """Set target bank angle (degrees)."""
        self.target_roll = np.radians(roll_deg)

    def set_target_altitude(self, altitude_ft):
        """Set target altitude (feet)."""
        self.target_altitude = altitude_ft

    def set_target_airspeed(self, airspeed_fps):
        """Set target airspeed (ft/s)."""
        self.target_airspeed = airspeed_fps

    def update(self, state, dt):
        """Update autopilot and return control commands."""

        roll, pitch, yaw = state.euler_angles
        p, q, r = state.angular_rates
        altitude = state.altitude
        V = np.linalg.norm(state.velocity_body)

        # === ROLL CONTROL ===
        roll_error = self.target_roll - roll
        self.roll_integral += roll_error * dt
        self.roll_integral = np.clip(self.roll_integral, -0.5, 0.5)  # Anti-windup
        roll_rate = (roll - self.prev_roll) / dt if dt > 0 else 0

        aileron = (self.Kp_roll * roll_error +
                  self.Ki_roll * self.roll_integral -
                  self.Kd_roll * roll_rate)
        aileron = np.clip(aileron, np.radians(-20), np.radians(20))
        self.prev_roll = roll

        # === PITCH CONTROL (Triple loop: altitude -> pitch -> pitch rate) ===
        # Outer loop: altitude -> pitch command
        alt_error = self.target_altitude - altitude
        self.alt_integral += alt_error * dt
        self.alt_integral = np.clip(self.alt_integral, -5000, 5000)
        alt_rate = (altitude - self.prev_alt) / dt if dt > 0 else 0

        pitch_cmd = (self.Kp_alt * alt_error +
                    self.Ki_alt * self.alt_integral -
                    self.Kd_alt * alt_rate)
        pitch_cmd = np.clip(pitch_cmd, np.radians(-8), np.radians(8))
        self.prev_alt = altitude

        # Middle loop: pitch -> pitch rate command
        pitch_error = pitch_cmd - pitch
        self.pitch_integral += pitch_error * dt
        self.pitch_integral = np.clip(self.pitch_integral, -0.5, 0.5)
        pitch_rate = (pitch - self.prev_pitch) / dt if dt > 0 else 0

        pitch_rate_cmd = (self.Kp_pitch * pitch_error +
                         self.Ki_pitch * self.pitch_integral -
                         self.Kd_pitch * pitch_rate)
        self.prev_pitch = pitch

        # Inner loop: pitch rate -> elevator
        pitch_rate_error = pitch_rate_cmd - q
        self.pitch_rate_integral += pitch_rate_error * dt
        self.pitch_rate_integral = np.clip(self.pitch_rate_integral, -0.3, 0.3)

        elevator = self.trim_elevator + (self.Kp_pitch_rate * pitch_rate_error +
                                         self.Ki_pitch_rate * self.pitch_rate_integral)
        elevator = np.clip(elevator, np.radians(-20), np.radians(10))

        # === YAW CONTROL (Sideslip coordination) ===
        if V > 10.0:
            beta = np.arcsin(np.clip(state.velocity_body[1] / V, -1, 1))
        else:
            beta = 0

        rudder = -(self.Kp_yaw * beta + self.Kd_yaw * r)
        rudder = np.clip(rudder, np.radians(-20), np.radians(20))

        # === AIRSPEED CONTROL ===
        airspeed_error = self.target_airspeed - V
        self.airspeed_integral += airspeed_error * dt
        self.airspeed_integral = np.clip(self.airspeed_integral, -100, 100)

        throttle = self.trim_throttle + (self.Kp_airspeed * airspeed_error +
                                         self.Ki_airspeed * self.airspeed_integral)
        throttle = np.clip(throttle, 0.05, 1.0)

        return {
            'aileron': aileron,
            'elevator': elevator,
            'rudder': rudder,
            'throttle': throttle
        }


def main():
    """Run trimmed coordinated turn simulation."""

    print("\n" + "=" * 70)
    print("Conventional Tail - Trimmed Coordinated Turn")
    print("Table-Based Aerodynamics with Trim Solver")
    print("=" * 70 + "\n")

    # Aircraft properties
    mass = 228.925  # slugs
    inertia = np.array([
        [19236.29, 0.0, 0.0],
        [0.0, 2251.02, 0.0],
        [0.0, 0.0, 21487.31]
    ])

    # Create aero model
    table_file = Path("aero_tables/conventional_aero_table.csv")
    aero = TableAeroModel(
        table_file=str(table_file),
        S_ref=199.94,
        c_ref=26.689,
        b_ref=19.890,
        rho=0.002377
    )

    # Create engine
    turbofan = TurbofanModel(thrust_max=1900.0, altitude_lapse_rate=0.7)

    # Create dynamics
    dynamics = AircraftDynamics(mass, inertia)

    # Flight condition
    altitude = -5000.0  # ft
    airspeed = 350.0   # ft/s (reduced from 450 to achieve trim within aero table bounds)

    # Find trim
    trim = find_trim(mass, aero, turbofan, altitude, airspeed)

    # Initialize state at trim
    state = State()
    state.position = np.array([0.0, 0.0, altitude])
    state.velocity_body = np.array([
        airspeed * np.cos(trim['alpha']),
        0.0,
        airspeed * np.sin(trim['alpha'])
    ])
    state.set_euler_angles(0.0, trim['theta'], 0.0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    # Create autopilot
    autopilot = CoordinatedTurnAutopilot()
    autopilot.set_trim(trim)
    autopilot.set_target_altitude(-altitude)
    autopilot.set_target_airspeed(airspeed)

    # Simulation parameters
    dt = 0.01
    duration = 60.0
    num_steps = int(duration / dt)

    # Data storage
    time_history = []
    states_history = []
    controls_history = []

    print("Turn sequence:")
    print("  0-10s:  Straight and level")
    print("  10-15s: Roll into 30° bank")
    print("  15-40s: Coordinated turn")
    print("  40-45s: Roll out")
    print("  45-60s: Straight and level")
    print()
    print("Running simulation...")
    print()

    # Simulation loop
    for step in range(num_steps):
        t = step * dt

        # Turn profile
        if t < 10.0:
            target_roll_deg = 0
        elif t < 15.0:
            target_roll_deg = 30.0 * (t - 10.0) / 5.0
        elif t < 40.0:
            target_roll_deg = 30.0
        elif t < 45.0:
            target_roll_deg = 30.0 * (1.0 - (t - 40.0) / 5.0)
        else:
            target_roll_deg = 0

        autopilot.set_target_roll(target_roll_deg)

        # Update autopilot
        controls = autopilot.update(state, dt)

        # Store data
        time_history.append(t)
        states_history.append(state.copy())
        controls_history.append(controls.copy())

        # Forces with atmosphere update
        def force_func(s):
            atm = StandardAtmosphere(s.altitude)
            aero.rho = atm.density

            aero_forces, aero_moments = aero.compute_forces_moments(s, controls)
            thrust_forces, _ = turbofan.compute_thrust(s, controls['throttle'])

            return aero_forces + thrust_forces, aero_moments

        # Integrate
        state = dynamics.propagate_rk4(state, dt, force_func)

        # Progress
        if step % 1000 == 0:
            roll, pitch, yaw = state.euler_angles
            alt_ft = -state.position[2]
            V = np.linalg.norm(state.velocity_body)
            print(f"  t = {t:.1f}s: alt = {alt_ft:.0f} ft, V = {V:.1f} ft/s, "
                  f"roll = {np.degrees(roll):.1f}°, hdg = {np.degrees(yaw):.1f}°")

    print("\nSimulation complete!")
    print()

    # Analysis
    positions = np.array([s.position for s in states_history])
    velocities = np.array([s.velocity_body for s in states_history])
    euler = np.array([s.euler_angles for s in states_history])

    altitudes = -positions[:, 2]
    airspeeds = np.array([np.linalg.norm(v) for v in velocities])
    rolls = np.degrees(euler[:, 0])
    headings = np.degrees(euler[:, 2])

    # Turn metrics
    turn_start = int(15.0 / dt)
    turn_end = int(40.0 / dt)

    print("=" * 70)
    print("TURN PERFORMANCE")
    print("=" * 70)
    print(f"During turn (15-40s):")
    print(f"  Bank angle: {np.mean(rolls[turn_start:turn_end]):.1f}° (target: 30°)")
    print(f"  Heading change: {headings[turn_end] - headings[turn_start]:.1f}°")
    print(f"  Altitude change: {altitudes[turn_end] - altitudes[turn_start]:+.0f} ft")
    print(f"  Airspeed change: {airspeeds[turn_end] - airspeeds[turn_start]:+.1f} ft/s")
    print()
    print(f"Overall statistics:")
    print(f"  Altitude std dev: {np.std(altitudes):.1f} ft")
    print(f"  Airspeed std dev: {np.std(airspeeds):.1f} ft/s")
    print("=" * 70)
    print()

    # Plots
    setup_plotting_style()

    fig, axes = plt.subplots(6, 1, figsize=(12, 16))

    # Altitude
    axes[0].plot(time_history, altitudes, 'b-', linewidth=1.5)
    axes[0].axhline(5000, color='r', linestyle='--', alpha=0.5)
    axes[0].axvspan(15, 40, color='yellow', alpha=0.2, label='Turn')
    axes[0].set_ylabel('Altitude (ft)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Airspeed
    axes[1].plot(time_history, airspeeds, 'g-', linewidth=1.5)
    axes[1].axhline(450, color='r', linestyle='--', alpha=0.5)
    axes[1].axvspan(15, 40, color='yellow', alpha=0.2)
    axes[1].set_ylabel('Airspeed (ft/s)')
    axes[1].grid(True, alpha=0.3)

    # Roll
    axes[2].plot(time_history, rolls, 'purple', linewidth=1.5)
    axes[2].plot([10, 15, 40, 45], [0, 30, 30, 0], 'r--', linewidth=2, label='Target')
    axes[2].axvspan(15, 40, color='yellow', alpha=0.2)
    axes[2].set_ylabel('Roll (°)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Heading
    axes[3].plot(time_history, headings, 'orange', linewidth=1.5)
    axes[3].axvspan(15, 40, color='yellow', alpha=0.2)
    axes[3].set_ylabel('Heading (°)')
    axes[3].grid(True, alpha=0.3)

    # Controls
    ailerons = np.degrees([c['aileron'] for c in controls_history])
    elevators = np.degrees([c['elevator'] for c in controls_history])
    rudders = np.degrees([c['rudder'] for c in controls_history])

    axes[4].plot(time_history, ailerons, 'b-', linewidth=1.5, label='Aileron')
    axes[4].plot(time_history, elevators, 'g-', linewidth=1.5, label='Elevator')
    axes[4].plot(time_history, rudders, 'r-', linewidth=1.5, label='Rudder')
    axes[4].axvspan(15, 40, color='yellow', alpha=0.2)
    axes[4].set_ylabel('Controls (°)')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    # Ground track
    axes[5].plot(positions[:, 1]/6076, positions[:, 0]/6076, 'b-', linewidth=2)
    axes[5].plot(positions[turn_start:turn_end, 1]/6076,
                positions[turn_start:turn_end, 0]/6076,
                'r-', linewidth=3, label='Turn')
    axes[5].scatter([0], [0], c='g', s=100, marker='o', label='Start')
    axes[5].set_xlabel('East (nm)')
    axes[5].set_ylabel('North (nm)')
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()
    axes[5].set_aspect('equal')

    fig.suptitle('Coordinated Turn - Trimmed Flight', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig("output/conventional_turn_trimmed.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("Plot saved: output/conventional_turn_trimmed.png")
    print()

    # Save flight history for animation
    import pickle
    history = {
        'time': np.array(time_history),
        'position': positions,
        'attitude': euler,
        'velocity': velocities
    }
    history_file = Path("output/flight_history.pkl")
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
    print(f"Flight history saved: {history_file}")
    print()


if __name__ == "__main__":
    main()
