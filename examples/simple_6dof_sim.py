"""
Simple 6-DOF Flight Simulation Example

Demonstrates complete 6-DOF simulation using:
- Quaternion attitude representation
- RK4 integration
- Simple aerodynamic model
- Propulsion model
- nTop UAV parameters
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.integrator import RK4Integrator
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import PropellerModel, CombinedForceModel


def main():
    print("=" * 60)
    print("6-DOF Flight Simulation - nTop UAV")
    print("=" * 60)
    print()

    # ========================================
    # Aircraft Parameters (from nTop UAV)
    # ========================================
    mass = 234.8  # slugs
    inertia = np.array([
        [14908.4, 0, 0],
        [0, 2318.4, 0],
        [0, 0, 17226.9]
    ])  # slug·ft²

    S_ref = 199.94  # ft²
    c_ref = 26.689  # ft
    b_ref = 19.890  # ft

    print("Aircraft Parameters:")
    print(f"  Mass: {mass:.1f} slugs ({mass * 32.174:.0f} lbm)")
    print(f"  Wing area: {S_ref:.2f} ft²")
    print(f"  Wing span: {b_ref:.2f} ft")
    print(f"  Mean chord: {c_ref:.2f} ft")
    print()

    # ========================================
    # Create Models
    # ========================================

    # Dynamics
    dynamics = AircraftDynamics(mass, inertia)

    # Aerodynamics - linear model with reasonable derivatives
    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)
    # Tuned for stable flight
    aero.CL_0 = 0.2
    aero.CL_alpha = 4.5
    aero.CD_0 = 0.03
    aero.CD_alpha2 = 0.8
    aero.Cm_0 = 0.0
    aero.Cm_alpha = -0.6  # Stable
    aero.Cm_q = -8.0      # Pitch damping

    # Propulsion - 50 HP propeller
    propulsion = PropellerModel(power_max=50.0, prop_diameter=6.0,
                                 prop_efficiency=0.75)

    # Combined force model
    force_model = CombinedForceModel(aero, propulsion)

    print("Models created:")
    print("  - Linear aerodynamic model (stability derivatives)")
    print("  - Propeller model (50 HP)")
    print()

    # ========================================
    # Initial Conditions
    # ========================================

    state0 = State()
    state0.position = np.array([0.0, 0.0, -5000.0])  # 5000 ft altitude, NED
    state0.velocity_body = np.array([200.0, 0.0, 0.0])  # 200 ft/s forward
    state0.set_euler_angles(0, np.radians(3), 0)  # 3° pitch for climb
    state0.angular_rates = np.array([0.0, 0.0, 0.0])

    print("Initial Conditions:")
    print(f"  Position: {state0.position} ft (NED)")
    print(f"  Altitude: {state0.altitude:.0f} ft")
    print(f"  Airspeed: {state0.airspeed:.1f} ft/s")
    print(f"  Pitch: {np.degrees(state0.euler_angles[1]):.1f}°")
    print()

    # ========================================
    # Simulation Setup
    # ========================================

    dt = 0.01  # 10 ms time step
    t_final = 30.0  # 30 seconds

    integrator = RK4Integrator(dt=dt)

    # Throttle schedule
    throttle = 0.8  # 80% throttle

    # Create derivative function with throttle
    def derivative_func(s):
        return dynamics.state_derivative(s, lambda st: force_model(st, throttle))

    print(f"Simulation Parameters:")
    print(f"  Time step: {dt*1000:.1f} ms")
    print(f"  Duration: {t_final:.1f} seconds")
    print(f"  Throttle: {throttle*100:.0f}%")
    print()

    # ========================================
    # Run Simulation
    # ========================================

    print("Running simulation...")
    t_hist, x_hist = integrator.integrate(state0, (0, t_final), derivative_func)
    print(f"Completed {len(t_hist)} time steps")
    print()

    # ========================================
    # Extract Results
    # ========================================

    n_steps = len(t_hist)
    altitude = np.zeros(n_steps)
    airspeed = np.zeros(n_steps)
    pitch = np.zeros(n_steps)
    roll = np.zeros(n_steps)
    yaw = np.zeros(n_steps)
    alpha = np.zeros(n_steps)
    x_pos = np.zeros(n_steps)
    y_pos = np.zeros(n_steps)
    pitch_rate = np.zeros(n_steps)

    for i in range(n_steps):
        state_i = State()
        state_i.from_array(x_hist[i, :])

        altitude[i] = state_i.altitude
        airspeed[i] = state_i.airspeed
        euler = state_i.euler_angles
        roll[i] = np.degrees(euler[0])
        pitch[i] = np.degrees(euler[1])
        yaw[i] = np.degrees(euler[2])
        alpha[i] = np.degrees(state_i.alpha)
        x_pos[i] = state_i.position[0]
        y_pos[i] = state_i.position[1]
        pitch_rate[i] = np.degrees(state_i.angular_rates[1])

    # ========================================
    # Display Final State
    # ========================================

    state_final = State()
    state_final.from_array(x_hist[-1, :])

    print("Final State:")
    print(f"  Position: [{x_pos[-1]:.1f}, {y_pos[-1]:.1f}, {state_final.position[2]:.1f}] ft (NED)")
    print(f"  Altitude: {altitude[-1]:.0f} ft (change: {altitude[-1] - altitude[0]:+.0f} ft)")
    print(f"  Airspeed: {airspeed[-1]:.1f} ft/s (change: {airspeed[-1] - airspeed[0]:+.1f} ft/s)")
    print(f"  Pitch: {pitch[-1]:.1f}° (change: {pitch[-1] - pitch[0]:+.1f}°)")
    print(f"  Alpha: {alpha[-1]:.1f}°")
    print()

    # ========================================
    # Plot Results
    # ========================================

    print("Generating plots...")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('6-DOF Flight Simulation - nTop UAV', fontsize=14, fontweight='bold')

    # Altitude
    axes[0, 0].plot(t_hist, altitude, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (ft)')
    axes[0, 0].set_title('Altitude History')
    axes[0, 0].grid(True, alpha=0.3)

    # Airspeed
    axes[0, 1].plot(t_hist, airspeed, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Airspeed (ft/s)')
    axes[0, 1].set_title('Airspeed History')
    axes[0, 1].grid(True, alpha=0.3)

    # Euler angles
    axes[1, 0].plot(t_hist, roll, 'b-', label='Roll', linewidth=1.5)
    axes[1, 0].plot(t_hist, pitch, 'r-', label='Pitch', linewidth=1.5)
    axes[1, 0].plot(t_hist, yaw, 'g-', label='Yaw', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angle (deg)')
    axes[1, 0].set_title('Euler Angles')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Alpha and pitch rate
    axes[1, 1].plot(t_hist, alpha, 'b-', label='Alpha', linewidth=1.5)
    ax2 = axes[1, 1].twinx()
    ax2.plot(t_hist, pitch_rate, 'r--', label='Pitch rate', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Alpha (deg)', color='b')
    ax2.set_ylabel('Pitch rate (deg/s)', color='r')
    axes[1, 1].set_title('Angle of Attack & Pitch Rate')
    axes[1, 1].grid(True, alpha=0.3)

    # Ground track
    axes[2, 0].plot(x_pos / 5280, y_pos / 5280, 'b-', linewidth=2)
    axes[2, 0].plot(x_pos[0] / 5280, y_pos[0] / 5280, 'go', markersize=10, label='Start')
    axes[2, 0].plot(x_pos[-1] / 5280, y_pos[-1] / 5280, 'ro', markersize=10, label='End')
    axes[2, 0].set_xlabel('X Position (miles)')
    axes[2, 0].set_ylabel('Y Position (miles)')
    axes[2, 0].set_title('Ground Track')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axis('equal')

    # 3D trajectory
    ax3d = fig.add_subplot(3, 2, 6, projection='3d')
    ax3d.plot(x_pos / 5280, y_pos / 5280, altitude, 'b-', linewidth=2)
    ax3d.plot([x_pos[0] / 5280], [y_pos[0] / 5280], [altitude[0]], 'go', markersize=10)
    ax3d.plot([x_pos[-1] / 5280], [y_pos[-1] / 5280], [altitude[-1]], 'ro', markersize=10)
    ax3d.set_xlabel('X (miles)')
    ax3d.set_ylabel('Y (miles)')
    ax3d.set_zlabel('Altitude (ft)')
    ax3d.set_title('3D Trajectory')

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(os.path.dirname(__file__), 'simulation_results.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    print()

    plt.close('all')  # Close figures instead of showing

    print("=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
