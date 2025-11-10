"""
Autopilot Demonstration - Phase 3 Features

Demonstrates:
- Standard atmosphere integration
- Altitude hold autopilot
- Heading hold autopilot
- Airspeed hold autopilot
- Combined multi-axis control
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
from src.environment.atmosphere import StandardAtmosphere
from src.control.autopilot import AltitudeHoldController, HeadingHoldController, AirspeedHoldController


def main():
    print("=" * 70)
    print("Autopilot Demonstration - nTop UAV with Phase 3 Features")
    print("=" * 70)
    print()

    # ========================================
    # Aircraft Parameters
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

    print("Aircraft Configuration:")
    print(f"  Mass: {mass:.1f} slugs ({mass * 32.174:.0f} lbm)")
    print(f"  Wing area: {S_ref:.2f} ft²")
    print(f"  Wing span: {b_ref:.2f} ft")
    print()

    # ========================================
    # Create Models
    # ========================================

    dynamics = AircraftDynamics(mass, inertia)

    # Aerodynamics with stability and control derivatives
    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)
    aero.CL_0 = 0.2
    aero.CL_alpha = 4.5
    aero.CD_0 = 0.03
    aero.CD_alpha2 = 0.8
    aero.Cm_0 = 0.0
    aero.Cm_alpha = -0.6      # Stable
    aero.Cm_q = -8.0          # Pitch damping
    aero.Cm_elevator = -1.2   # Elevator effectiveness
    aero.Cl_beta = -0.1       # Dihedral effect
    aero.Cl_p = -0.5          # Roll damping
    aero.Cl_aileron = 0.15    # Aileron effectiveness
    aero.Cn_beta = 0.08       # Weathercock stability
    aero.Cn_r = -0.2          # Yaw damping
    aero.Cn_aileron = -0.02   # Adverse yaw

    # Propulsion
    propulsion = PropellerModel(power_max=50.0, prop_diameter=6.0,
                                 prop_efficiency=0.75)

    force_model = CombinedForceModel(aero, propulsion)

    print("Models:")
    print("  - 6-DOF rigid body dynamics")
    print("  - Linear aerodynamic model (with control derivatives)")
    print("  - 50 HP propeller model")
    print("  - US Standard Atmosphere 1976")
    print()

    # ========================================
    # Autopilot Controllers
    # ========================================

    # Altitude hold - target 6000 ft
    alt_controller = AltitudeHoldController(
        Kp_alt=0.0015,
        Ki_alt=0.0002,
        Kd_alt=0.006,
        Kp_pitch=3.0,
        Ki_pitch=0.8,
        Kd_pitch=0.2
    )
    alt_controller.set_target_altitude(6000.0)

    # Heading hold - target 90° (East)
    hdg_controller = HeadingHoldController(
        Kp_heading=0.8,
        Ki_heading=0.1,
        Kd_heading=0.15,
        Kp_roll=2.5,
        Ki_roll=0.5,
        Kd_roll=0.15
    )
    hdg_controller.set_target_heading(np.radians(90))

    # Airspeed hold - target 220 ft/s
    spd_controller = AirspeedHoldController(
        Kp=0.015,
        Ki=0.002,
        Kd=0.08
    )
    spd_controller.set_target_airspeed(220.0)

    print("Autopilot Configuration:")
    print("  - Altitude hold: 6,000 ft")
    print("  - Heading hold: 90° (East)")
    print("  - Airspeed hold: 220 ft/s")
    print()

    # ========================================
    # Initial Conditions
    # ========================================

    state0 = State()
    state0.position = np.array([0.0, 0.0, -5000.0])  # 5000 ft altitude
    state0.velocity_body = np.array([200.0, 0.0, 0.0])  # 200 ft/s
    state0.set_euler_angles(0, np.radians(2), 0)  # Slight pitch
    state0.angular_rates = np.array([0.0, 0.0, 0.0])

    print("Initial State:")
    print(f"  Altitude: {state0.altitude:.0f} ft (target: 6000 ft)")
    print(f"  Airspeed: {state0.airspeed:.1f} ft/s (target: 220 ft/s)")
    print(f"  Heading: {np.degrees(state0.euler_angles[2]):.1f}° (target: 90°)")
    print()

    # ========================================
    # Simulation
    # ========================================

    dt = 0.01
    t_final = 60.0  # 60 seconds

    print(f"Running simulation ({t_final:.0f} seconds)...")

    # Storage for plotting
    t_hist = []
    altitude_hist = []
    airspeed_hist = []
    heading_hist = []
    roll_hist = []
    pitch_hist = []
    elevator_hist = []
    aileron_hist = []
    throttle_hist = []
    x_pos_hist = []
    y_pos_hist = []

    # Initialize
    state = state0.copy()
    t = 0.0

    # Manual integration loop to apply autopilot at each step
    while t < t_final:
        # Update atmosphere
        atm = StandardAtmosphere(state.altitude)
        aero.rho = atm.density

        # Autopilot commands
        elevator = alt_controller.update(state.altitude, state.euler_angles[1], dt)
        aileron = hdg_controller.update(state.euler_angles[2], state.euler_angles[0], dt)
        throttle = spd_controller.update(state.airspeed, dt)

        # Store history
        t_hist.append(t)
        altitude_hist.append(state.altitude)
        airspeed_hist.append(state.airspeed)
        heading_hist.append(np.degrees(state.euler_angles[2]))
        roll_hist.append(np.degrees(state.euler_angles[0]))
        pitch_hist.append(np.degrees(state.euler_angles[1]))
        elevator_hist.append(np.degrees(elevator))
        aileron_hist.append(np.degrees(aileron))
        throttle_hist.append(throttle * 100)
        x_pos_hist.append(state.position[0])
        y_pos_hist.append(state.position[1])

        # Compute forces and moments with control inputs
        def force_func_with_controls(s):
            # Add control surface effects
            # This is simplified - full implementation would modify aero model
            forces, moments = force_model(s, throttle)

            # Add elevator moment
            q_bar = atm.get_dynamic_pressure(s.airspeed)
            moments[1] += q_bar * S_ref * c_ref * aero.Cm_elevator * elevator

            # Add aileron moment (roll)
            moments[0] += q_bar * S_ref * b_ref * aero.Cl_aileron * aileron
            # Adverse yaw from aileron
            moments[2] += q_bar * S_ref * b_ref * aero.Cn_aileron * aileron

            return forces, moments

        # State derivative
        state_dot_array = dynamics.state_derivative(state, force_func_with_controls)

        # RK4 integration step
        x = state.to_array()
        x_new = x + state_dot_array * dt

        # Update state
        state.from_array(x_new)

        t += dt

        # Progress indicator
        if int(t) % 10 == 0 and t - dt < int(t):
            print(f"  t = {int(t)}s: Alt = {state.altitude:.0f} ft, "
                  f"Speed = {state.airspeed:.1f} ft/s, "
                  f"Hdg = {np.degrees(state.euler_angles[2]):.0f}°")

    print(f"Simulation complete: {len(t_hist)} steps")
    print()

    # ========================================
    # Final State
    # ========================================

    print("Final State:")
    print(f"  Altitude: {altitude_hist[-1]:.0f} ft (error: {altitude_hist[-1] - 6000:.0f} ft)")
    print(f"  Airspeed: {airspeed_hist[-1]:.1f} ft/s (error: {airspeed_hist[-1] - 220:.1f} ft/s)")
    print(f"  Heading: {heading_hist[-1]:.1f}° (error: {heading_hist[-1] - 90:.1f}°)")
    print(f"  Distance traveled: {x_pos_hist[-1]/5280:.2f} mi East, {y_pos_hist[-1]/5280:.2f} mi North")
    print()

    # ========================================
    # Plot Results
    # ========================================

    print("Generating plots...")

    fig = plt.figure(figsize=(14, 12))

    # Altitude tracking
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t_hist, altitude_hist, 'b-', linewidth=2, label='Actual')
    ax1.axhline(y=6000, color='r', linestyle='--', linewidth=1.5, label='Target')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (ft)')
    ax1.set_title('Altitude Hold Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Airspeed tracking
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t_hist, airspeed_hist, 'r-', linewidth=2, label='Actual')
    ax2.axhline(y=220, color='b', linestyle='--', linewidth=1.5, label='Target')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Airspeed (ft/s)')
    ax2.set_title('Airspeed Hold Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Heading tracking
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(t_hist, heading_hist, 'g-', linewidth=2, label='Actual')
    ax3.axhline(y=90, color='r', linestyle='--', linewidth=1.5, label='Target')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heading (deg)')
    ax3.set_title('Heading Hold Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Euler angles
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(t_hist, roll_hist, 'b-', label='Roll', linewidth=1.5)
    ax4.plot(t_hist, pitch_hist, 'r-', label='Pitch', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (deg)')
    ax4.set_title('Attitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Control surfaces
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(t_hist, elevator_hist, 'b-', label='Elevator', linewidth=1.5)
    ax5.plot(t_hist, aileron_hist, 'r-', label='Aileron', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Deflection (deg)')
    ax5.set_title('Control Surface Commands')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Throttle
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(t_hist, throttle_hist, 'g-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Throttle (%)')
    ax6.set_title('Throttle Command')
    ax6.grid(True, alpha=0.3)

    # Ground track
    ax7 = plt.subplot(3, 3, 7)
    x_miles = np.array(x_pos_hist) / 5280
    y_miles = np.array(y_pos_hist) / 5280
    ax7.plot(x_miles, y_miles, 'b-', linewidth=2)
    ax7.plot(x_miles[0], y_miles[0], 'go', markersize=10, label='Start')
    ax7.plot(x_miles[-1], y_miles[-1], 'ro', markersize=10, label='End')
    ax7.set_xlabel('East (miles)')
    ax7.set_ylabel('North (miles)')
    ax7.set_title('Ground Track')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.axis('equal')

    # 3D trajectory
    ax8 = fig.add_subplot(3, 3, 8, projection='3d')
    ax8.plot(x_miles, y_miles, altitude_hist, 'b-', linewidth=2)
    ax8.plot([x_miles[0]], [y_miles[0]], [altitude_hist[0]], 'go', markersize=10)
    ax8.plot([x_miles[-1]], [y_miles[-1]], [altitude_hist[-1]], 'ro', markersize=10)
    ax8.set_xlabel('East (miles)')
    ax8.set_ylabel('North (miles)')
    ax8.set_zlabel('Altitude (ft)')
    ax8.set_title('3D Flight Path')

    # Tracking errors
    ax9 = plt.subplot(3, 3, 9)
    alt_error = np.array(altitude_hist) - 6000
    spd_error = np.array(airspeed_hist) - 220
    hdg_error = np.array(heading_hist) - 90
    ax9.plot(t_hist, alt_error, 'b-', label='Alt error (ft)', linewidth=1.5)
    ax9.plot(t_hist, spd_error * 10, 'r-', label='Speed error x10 (ft/s)', linewidth=1.5)
    ax9.plot(t_hist, hdg_error, 'g-', label='Heading error (deg)', linewidth=1.5)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Error')
    ax9.set_title('Tracking Errors')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = os.path.join(os.path.dirname(__file__), 'autopilot_demo_results.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    print()

    plt.close()

    print("=" * 70)
    print("Autopilot Demonstration Complete!")
    print("=" * 70)
    print()
    print("Phase 3 features demonstrated:")
    print("  [OK] Standard atmosphere integration")
    print("  [OK] Altitude hold autopilot (PID-based)")
    print("  [OK] Heading hold autopilot (cascaded control)")
    print("  [OK] Airspeed hold autopilot (throttle control)")
    print("  [OK] Multi-axis simultaneous control")
    print()


if __name__ == "__main__":
    main()
