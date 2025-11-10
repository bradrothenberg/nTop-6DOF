"""
Debug aerodynamic forces and moments to understand instability.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.aerodynamics import LinearAeroModel
from src.environment.atmosphere import StandardAtmosphere


def test_aerodynamic_signs():
    """Test if aerodynamic model has correct signs."""

    print("=" * 70)
    print("Aerodynamic Force and Moment Sign Test")
    print("=" * 70)
    print()

    # Setup
    S_ref = 412.6370
    c_ref = 11.9555
    b_ref = 24.8630

    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)

    # Flying wing derivatives
    aero.CL_0 = 0.000023
    aero.CL_alpha = 1.412241
    aero.Cm_0 = 0.000061
    aero.Cm_alpha = -0.079668  # STABLE (negative)
    aero.Cm_q = -0.347072

    # Create a state at alpha = 2 deg
    state = State()
    state.velocity_body = np.array([200.0, 0.0, 0.0])
    state.set_euler_angles(0, np.radians(2), 0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    atm = StandardAtmosphere(5000)
    aero.rho = atm.density

    # Compute forces and moments
    forces, moments = aero(state)

    print(f"Test Case: alpha = 2 deg, V = 200 ft/s, altitude = 5000 ft")
    print()
    print(f"State:")
    print(f"  alpha = {np.degrees(state.alpha):.2f} deg")
    print(f"  beta = {np.degrees(state.beta):.2f} deg")
    print(f"  q_bar = {atm.get_dynamic_pressure(200):.4f} psf")
    print()

    # Expected coefficients
    q_bar = atm.get_dynamic_pressure(200)
    alpha_rad = state.alpha

    CL_expected = aero.CL_0 + aero.CL_alpha * alpha_rad
    Cm_expected = aero.Cm_0 + aero.Cm_alpha * alpha_rad

    print(f"Expected coefficients:")
    print(f"  CL = {CL_expected:.6f}")
    print(f"  Cm = {Cm_expected:.6f}")
    print()

    # Actual forces/moments
    L_body = -forces[2]  # Lift is -Z in body frame
    M_pitch = moments[1]

    CL_actual = L_body / (q_bar * S_ref)
    Cm_actual = M_pitch / (q_bar * S_ref * c_ref)

    print(f"Actual from aero model:")
    print(f"  CL = {CL_actual:.6f}")
    print(f"  Cm = {Cm_actual:.6f}")
    print()

    print(f"Forces (body frame):")
    print(f"  Fx = {forces[0]:.2f} lbf")
    print(f"  Fy = {forces[1]:.2f} lbf")
    print(f"  Fz = {forces[2]:.2f} lbf")
    print(f"  Lift (-Fz) = {-forces[2]:.2f} lbf")
    print()

    print(f"Moments (body frame):")
    print(f"  Mx (roll) = {moments[0]:.2f} lb-ft")
    print(f"  My (pitch) = {moments[1]:.2f} lb-ft")
    print(f"  Mz (yaw) = {moments[2]:.2f} lb-ft")
    print()

    # Test stability: increase alpha, check if pitch moment is restoring
    print("Stability Test:")
    print("-" * 70)

    for alpha_deg in [0, 2, 4, 6, 8, 10]:
        state2 = State()
        state2.velocity_body = np.array([200.0, 0.0, 0.0])
        state2.set_euler_angles(0, np.radians(alpha_deg), 0)
        state2.angular_rates = np.array([0.0, 0.0, 0.0])

        _, moments2 = aero(state2)
        M_pitch2 = moments2[1]

        print(f"  alpha = {alpha_deg:2d} deg  ->  M_pitch = {M_pitch2:8.1f} lb-ft")

    print()
    print("For STABLE aircraft:")
    print("  - As alpha INCREASES (nose up)")
    print("  - Pitch moment should be NEGATIVE (nose down)")
    print("  - This restores the aircraft to trim")
    print()

    # Check Cm_alpha sign
    state1 = State()
    state1.velocity_body = np.array([200.0, 0.0, 0.0])
    state1.set_euler_angles(0, np.radians(2), 0)
    state1.angular_rates = np.array([0.0, 0.0, 0.0])

    state2 = State()
    state2.velocity_body = np.array([200.0, 0.0, 0.0])
    state2.set_euler_angles(0, np.radians(4), 0)
    state2.angular_rates = np.array([0.0, 0.0, 0.0])

    _, M1 = aero(state1)
    _, M2 = aero(state2)

    dM = M2[1] - M1[1]
    dalpha = np.radians(2)

    Cm_alpha_actual = dM / (q_bar * S_ref * c_ref * dalpha)

    print(f"Measured Cm_alpha:")
    print(f"  dM/dalpha = {dM / dalpha:.2f} lb-ft/rad")
    print(f"  Cm_alpha = {Cm_alpha_actual:.6f}")
    print(f"  Expected: {aero.Cm_alpha:.6f}")
    print()

    if Cm_alpha_actual < 0:
        print("✓ Cm_alpha is NEGATIVE -> STABLE")
    else:
        print("✗ Cm_alpha is POSITIVE -> UNSTABLE!")


if __name__ == "__main__":
    test_aerodynamic_signs()
