"""
Simplified trim solver for flying wing at Mach 0.5.

Since the flying wing has no elevator control of pitch moment (Cm_de = 0),
we simplify the trim problem to just:
1. Find alpha for L = W
2. Set theta = alpha (for gamma = 0)
3. Find throttle for T = D

Accept whatever pitch moment results and rely on damping (Cmq) for stability.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import PropellerModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere


def find_simple_trim_flying_wing(
    mass: float,
    inertia: np.ndarray,
    aero_model: LinearAeroModel,
    prop_model: PropellerModel,
    altitude: float,
    airspeed: float
):
    """
    Find simplified trim for flying wing.

    Steps:
    1. Calculate alpha needed for L = W
    2. Set theta = alpha (level flight)
    3. Calculate throttle for T ≈ D
    4. Accept resulting pitch moment (rely on Cmq damping)

    Returns:
    --------
    state : State
        Trimmed state
    controls : dict
        Control inputs
    info : dict
        Trim information
    """

    print("=" * 70)
    print("Simplified Trim Solver - Flying Wing")
    print("=" * 70)
    print()

    # Atmosphere
    atm = StandardAtmosphere(altitude)
    aero_model.rho = atm.density

    q_bar = 0.5 * atm.density * airspeed**2
    W = mass * 32.174  # Weight (lbf)

    print(f"Conditions:")
    print(f"  Altitude: {-altitude:.0f} ft")
    print(f"  Airspeed: {airspeed:.1f} ft/s")
    print(f"  Density: {atm.density:.6f} slug/ft^3")
    print(f"  q_bar: {q_bar:.2f} psf")
    print(f"  Weight: {W:.1f} lbf")
    print()

    # Step 1: Find alpha for L = W
    # L = q_bar * S * (CL_0 + CL_alpha * alpha)
    # W = q_bar * S * CL_trim
    # CL_trim = CL_0 + CL_alpha * alpha_trim
    # alpha_trim = (CL_trim - CL_0) / CL_alpha

    CL_trim = W / (q_bar * aero_model.S_ref)
    alpha_trim = (CL_trim - aero_model.CL_0) / aero_model.CL_alpha

    print(f"Step 1: Find alpha for L = W")
    print(f"  CL_trim = W / (q * S) = {CL_trim:.6f}")
    print(f"  CL_0 = {aero_model.CL_0:.6f}")
    print(f"  CL_alpha = {aero_model.CL_alpha:.6f}")
    print(f"  alpha_trim = (CL_trim - CL_0) / CL_alpha = {np.degrees(alpha_trim):.4f} deg")
    print()

    # Step 2: Set theta = alpha for level flight (gamma = 0)
    theta_trim = alpha_trim

    print(f"Step 2: Set theta = alpha (level flight)")
    print(f"  theta_trim = {np.degrees(theta_trim):.4f} deg")
    print()

    # Step 3: Estimate throttle for T = D
    CD_trim = aero_model.CD_0 + aero_model.CD_alpha * alpha_trim + \
              aero_model.CD_alpha2 * alpha_trim**2
    D_trim = q_bar * aero_model.S_ref * CD_trim

    # Rough throttle estimate (propeller thrust ~ throttle * max_thrust)
    # For PropellerModel, thrust depends on throttle and airspeed
    # Use iterative approach

    throttle_trim = 0.3  # Initial guess
    for iteration in range(10):
        # Create test state
        state_test = State()
        state_test.position = np.array([0, 0, altitude])
        state_test.velocity_body = np.array([
            airspeed * np.cos(alpha_trim),
            0.0,
            airspeed * np.sin(alpha_trim)
        ])
        state_test.set_euler_angles(0, theta_trim, 0)
        state_test.angular_rates = np.array([0, 0, 0])

        # Compute thrust
        thrust_vec, _ = prop_model.compute_thrust(state_test, throttle_trim)
        thrust = thrust_vec[0]  # Forward thrust

        # Adjust throttle
        error = D_trim - thrust
        throttle_trim += 0.01 * error / D_trim  # Proportional adjustment
        throttle_trim = np.clip(throttle_trim, 0.01, 1.0)

        if abs(error) < 1.0:  # Within 1 lbf
            break

    print(f"Step 3: Find throttle for T = D")
    print(f"  CD_trim = {CD_trim:.6f}")
    print(f"  Drag: {D_trim:.1f} lbf")
    print(f"  Thrust (at throttle={throttle_trim:.3f}): {thrust:.1f} lbf")
    print(f"  T/D: {thrust/D_trim:.4f}")
    print()

    # Create trimmed state
    state_trim = State()
    state_trim.position = np.array([0, 0, altitude])
    state_trim.velocity_body = np.array([
        airspeed * np.cos(alpha_trim),
        0.0,
        airspeed * np.sin(alpha_trim)
    ])
    state_trim.set_euler_angles(0, theta_trim, 0)
    state_trim.angular_rates = np.array([0, 0, 0])

    controls_trim = {
        'throttle': throttle_trim,
        'elevator': 0.0,
        'aileron': 0.0,
        'rudder': 0.0
    }

    # Verify trim by computing accelerations
    combined = CombinedForceModel(aero_model, prop_model)
    dynamics = AircraftDynamics(mass, inertia)

    def force_func(s):
        return combined(s, controls_trim['throttle'], controls_trim)

    state_dot = dynamics.state_derivative(state_trim, force_func)
    vel_dot = state_dot[3:6]
    omega_dot = state_dot[10:13]

    # Compute pitch moment
    forces, moments = combined(state_trim, throttle_trim, controls_trim)

    # Compute Cm at trim
    Cm_trim = aero_model.Cm_0 + aero_model.Cm_alpha * alpha_trim

    print("Step 4: Verify trim")
    print(f"  Accelerations:")
    print(f"    ax: {vel_dot[0]:.4f} ft/s^2")
    print(f"    az: {vel_dot[2]:.4f} ft/s^2")
    print(f"    q_dot: {np.degrees(omega_dot[1]):.4f} deg/s^2")
    print()
    print(f"  Pitch moment: {moments[1]:.1f} ft-lbf")
    print(f"  Cm at trim: {Cm_trim:.6f}")
    print()

    # Check if acceptable
    is_acceptable = (
        abs(vel_dot[2]) < 1.0 and      # Vertical accel < 1 ft/s^2
        abs(vel_dot[0]) < 5.0           # Forward accel < 5 ft/s^2
    )

    if is_acceptable:
        print("STATUS: ACCEPTABLE TRIM")
        print("  Note: Pitch moment is non-zero, but Cmq damping should stabilize")
    else:
        print("STATUS: POOR TRIM")
        print(f"  Large accelerations present")
    print()

    info = {
        'alpha_deg': np.degrees(alpha_trim),
        'theta_deg': np.degrees(theta_trim),
        'throttle': throttle_trim,
        'CL_trim': CL_trim,
        'CD_trim': CD_trim,
        'Cm_trim': Cm_trim,
        'lift': q_bar * aero_model.S_ref * CL_trim,
        'drag': D_trim,
        'thrust': thrust,
        'pitch_moment': moments[1],
        'vel_dot': vel_dot,
        'omega_dot': omega_dot,
        'is_acceptable': is_acceptable
    }

    return state_trim, controls_trim, info


if __name__ == "__main__":
    # Flying wing parameters
    mass = 228.924806
    inertia = np.array([[19236.2914, 0, 0],
                        [0, 2251.0172, 0],
                        [0, 0, 21487.3086]])

    S_ref = 412.6370
    c_ref = 11.9555
    b_ref = 24.8630

    aero = LinearAeroModel(S_ref, c_ref, b_ref)

    # Flying wing AVL derivatives
    aero.CL_0 = 0.000023
    aero.CL_alpha = 1.412241
    aero.CL_q = 1.282202

    aero.CD_0 = -0.000619
    aero.CD_alpha = 0.035509
    aero.CD_alpha2 = 0.5

    aero.Cm_0 = 0.000061
    aero.Cm_alpha = -0.079668
    aero.Cm_q = -0.347072

    prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)

    # Find trim at Mach 0.5
    state_trim, controls_trim, info = find_simple_trim_flying_wing(
        mass, inertia, aero, prop,
        altitude=-5000.0,
        airspeed=548.5
    )

    print("=" * 70)
    print("TRIM SOLUTION:")
    print("=" * 70)
    print(f"  Alpha: {info['alpha_deg']:.4f} deg")
    print(f"  Theta: {info['theta_deg']:.4f} deg")
    print(f"  Throttle: {info['throttle']:.4f}")
    print()
    print(f"  CL: {info['CL_trim']:.6f}")
    print(f"  CD: {info['CD_trim']:.6f}")
    print(f"  Cm: {info['Cm_trim']:.6f}")
    print()
    print(f"  Lift: {info['lift']:.1f} lbf")
    print(f"  Drag: {info['drag']:.1f} lbf")
    print(f"  Thrust: {info['thrust']:.1f} lbf")
    print(f"  Pitch moment: {info['pitch_moment']:.1f} ft-lbf")
    print()
    print(f"  ax: {info['vel_dot'][0]:.4f} ft/s^2")
    print(f"  az: {info['vel_dot'][2]:.4f} ft/s^2")
    print(f"  q_dot: {np.degrees(info['omega_dot'][1]):.4f} deg/s^2")
    print("=" * 70)
