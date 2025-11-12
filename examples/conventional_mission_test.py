"""
Test mission profile with conventional tail configuration.

Uses estimated aerodynamic derivatives for conventional configuration
with separate elevator, ailerons, and rudder.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dynamics.state import State6DOF
from src.dynamics.dynamics import Dynamics6DOF
from src.core.aerodynamics import AerodynamicModel
from src.control.autopilot import SimpleAutopilot
from src.atmosphere.standard_atmosphere import StandardAtmosphere
from src.utils.constants import FT2M, M2FT
import conventional_aero_data as aero

def main():
    print("=" * 70)
    print("Conventional Tail Configuration - Mission Profile Test")
    print("=" * 70)
    print()

    # Mission parameters
    initial_altitude = 5000  # ft
    cruise_altitude = 10000  # ft
    target_airspeed = 600  # ft/s
    duration = 120  # seconds

    # Mass properties (same as flying wing for comparison)
    mass = 228.924806  # slugs
    Ixx = 19236.2914  # slug-ft^2
    Iyy = 2251.0172
    Izz = 21487.3086

    # Aerodynamic reference
    S_ref = 412.637  # ft^2
    c_ref = 11.9555  # ft
    b_ref = 24.863  # ft

    # Initial conditions at 5000 ft
    atm = StandardAtmosphere()
    rho, _, _,  _ = atm.get_properties(initial_altitude)

    # Trim state (estimated for conventional config)
    trim_alpha = np.deg2rad(aero.TRIM_ALPHA)
    V = target_airspeed
    q_inf = 0.5 * rho * V**2

    # Initial state
    state = State6DOF()
    state.position = np.array([0.0, 0.0, -initial_altitude])  # NED frame
    state.velocity = np.array([V * np.cos(trim_alpha), 0.0, V * np.sin(trim_alpha)])
    state.attitude = np.array([0.0, trim_alpha, 0.0])  # roll, pitch, yaw
    state.angular_velocity = np.array([0.0, 0.0, 0.0])

    # Aerodynamic model with conventional configuration derivatives
    aero_model = AerodynamicModel(
        S_ref=S_ref,
        c_ref=c_ref,
        b_ref=b_ref,
        mass=mass,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz
    )

    # Set stability derivatives from estimated conventional config
    aero_model.CL_0 = aero.TRIM_CL
    aero_model.CL_alpha = aero.CL_ALPHA
    aero_model.CL_q = aero.CL_Q
    aero_model.CL_de = aero.CL_DE

    aero_model.CD_0 = aero.TRIM_CD

    aero_model.Cm_0 = 0.0
    aero_model.Cm_alpha = aero.CM_ALPHA
    aero_model.Cm_q = aero.CM_Q
    aero_model.Cm_de = aero.CM_DE

    aero_model.CY_beta = aero.CY_BETA
    aero_model.Cl_beta = aero.CL_BETA
    aero_model.Cl_p = aero.CL_P
    aero_model.Cl_r = aero.CL_R
    aero_model.Cl_da = aero.CL_DA

    aero_model.Cn_beta = aero.CN_BETA
    aero_model.Cn_p = aero.CN_P
    aero_model.Cn_r = aero.CN_R
    aero_model.Cn_dr = aero.CN_DR

    print("Aerodynamic Model:")
    print(f"  CL_alpha = {aero_model.CL_alpha:.4f} /rad")
    print(f"  Cm_alpha = {aero_model.Cm_alpha:.4f} /rad (should be << -0.1 for stability)")
    print(f"  Cm_q     = {aero_model.Cm_q:.4f} /rad (should be << 0 for damping)")
    print(f"  Cn_beta  = {aero_model.Cn_beta:.4f} /rad (should be > 0 for directional stability)")
    print()

    # 6-DOF dynamics
    dynamics = Dynamics6DOF(aero_model)

    # Simple autopilot
    autopilot = SimpleAutopilot(
        target_altitude=initial_altitude,
        target_airspeed=target_airspeed,
        Kp_alt=0.005,
        Ki_alt=0.0005,
        Kd_alt=0.012,
        Kp_pitch=0.8,
        Ki_pitch=0.05,
        Kd_pitch=0.15,
        Kp_pitch_rate=0.15,
        Ki_pitch_rate=0.01,
        throttle_gain=0.015
    )

    # Time settings
    dt = 0.01
    t = 0.0
    steps = int(duration / dt)

    # Storage
    time_hist = []
    altitude_hist = []
    airspeed_hist = []
    pitch_hist = []
    roll_hist = []
    elevator_hist = []
    throttle_hist = []
    alpha_hist = []

    # Mission phases
    phase_times = [30, 60, 90, 120]  # seconds
    phase_altitudes = [5000, 10000, 10000, 5000]  # ft
    current_phase = 0

    print("Starting simulation...")
    print(f"  Duration: {duration} seconds")
    print(f"  Time step: {dt} seconds")
    print()

    for step in range(steps):
        # Update mission phase
        if current_phase < len(phase_times) - 1 and t >= phase_times[current_phase]:
            current_phase += 1
            autopilot.target_altitude = phase_altitudes[current_phase]
            print(f"Phase {current_phase + 1}: Target altitude = {phase_altitudes[current_phase]} ft")

        # Get atmospheric properties
        altitude_ft = -state.position[2]
        rho, _, _, _ = atm.get_properties(altitude_ft)

        # Compute airspeed
        V_body = state.velocity
        airspeed = np.linalg.norm(V_body)

        # Compute angle of attack
        if V_body[0] > 0.1:
            alpha = np.arctan2(V_body[2], V_body[0])
        else:
            alpha = 0.0

        # Autopilot commands (elevator only, no flaperon mixing for conventional config)
        elevator, throttle = autopilot.update(state, dt)

        # Control limits
        elevator = np.clip(elevator, -25, 25)  # degrees
        throttle = np.clip(throttle, 0, 1)

        # Compute forces and moments
        control_surfaces = {
            'elevator': elevator,
            'aileron': 0.0,  # No aileron input for now
            'rudder': 0.0  # No rudder input for now
        }

        forces, moments = dynamics.compute_forces_moments(
            state, rho, control_surfaces
        )

        # Apply turbofan thrust (FJ-44 model)
        thrust = 3600 * throttle  # lbf
        forces[0] += thrust

        # Update dynamics
        state_dot = dynamics.compute_state_derivative(state, forces, moments)
        state.integrate(state_dot, dt)

        # Store history
        time_hist.append(t)
        altitude_hist.append(altitude_ft)
        airspeed_hist.append(airspeed)
        pitch_hist.append(np.rad2deg(state.attitude[1]))
        roll_hist.append(np.rad2deg(state.attitude[0]))
        elevator_hist.append(elevator)
        throttle_hist.append(throttle * 100)
        alpha_hist.append(np.rad2deg(alpha))

        t += dt

        # Progress update
        if step % 1000 == 0:
            print(f"  t={t:6.1f}s  Alt={altitude_ft:7.1f} ft  V={airspeed:6.1f} ft/s  "
                  f"pitch={np.rad2deg(state.attitude[1]):5.1f}°  elev={elevator:5.1f}°")

    print()
    print("Simulation complete!")
    print()

    # Statistics
    alt_mean = np.mean(altitude_hist)
    alt_std = np.std(altitude_hist)
    alt_change = altitude_hist[-1] - altitude_hist[0]

    V_mean = np.mean(airspeed_hist)
    V_std = np.std(airspeed_hist)
    V_change = airspeed_hist[-1] - airspeed_hist[0]

    pitch_std = np.std(pitch_hist)
    roll_std = np.std(roll_hist)

    print("Statistics:")
    print(f"  Altitude:")
    print(f"    Mean:   {alt_mean:7.1f} ft")
    print(f"    Std:    {alt_std:7.1f} ft")
    print(f"    Change: {alt_change:+7.1f} ft")
    print()
    print(f"  Airspeed:")
    print(f"    Mean:   {V_mean:7.1f} ft/s")
    print(f"    Std:    {V_std:7.1f} ft/s")
    print(f"    Change: {V_change:+7.1f} ft/s")
    print()
    print(f"  Attitude:")
    print(f"    Pitch std: {pitch_std:5.2f}°")
    print(f"    Roll std:  {roll_std:5.2f}°")
    print()

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Conventional Tail Configuration - Mission Profile", fontsize=14, fontweight='bold')

    # Altitude
    axes[0, 0].plot(time_hist, altitude_hist, 'b-', linewidth=1.5)
    axes[0, 0].axhline(y=initial_altitude, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(y=cruise_altitude, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (ft)')
    axes[0, 0].set_title('Altitude vs Time')
    axes[0, 0].grid(True, alpha=0.3)

    # Airspeed
    axes[0, 1].plot(time_hist, airspeed_hist, 'r-', linewidth=1.5)
    axes[0, 1].axhline(y=target_airspeed, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Airspeed (ft/s)')
    axes[0, 1].set_title('Airspeed vs Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Pitch attitude
    axes[1, 0].plot(time_hist, pitch_hist, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Pitch (deg)')
    axes[1, 0].set_title('Pitch Attitude')
    axes[1, 0].grid(True, alpha=0.3)

    # Roll attitude
    axes[1, 1].plot(time_hist, roll_hist, 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Roll (deg)')
    axes[1, 1].set_title('Roll Attitude')
    axes[1, 1].grid(True, alpha=0.3)

    # Elevator
    axes[2, 0].plot(time_hist, elevator_hist, 'c-', linewidth=1.5)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Elevator (deg)')
    axes[2, 0].set_title('Elevator Deflection')
    axes[2, 0].grid(True, alpha=0.3)

    # Throttle
    axes[2, 1].plot(time_hist, throttle_hist, 'orange', linewidth=1.5)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Throttle (%)')
    axes[2, 1].set_title('Throttle Setting')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "conventional_mission_test.png"
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
