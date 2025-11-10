"""
Flying Wing Diagnostics

Analyzes inertia tensor, control coupling, and stability characteristics
for flying wing UAV configuration.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.environment.atmosphere import StandardAtmosphere

import matplotlib.pyplot as plt


def analyze_inertia_tensor():
    """
    Analyze inertia tensor for flying wing geometry.

    For a swept flying wing, the product of inertia Ixz is typically
    non-zero and can be significant.
    """
    print("=" * 70)
    print("INERTIA TENSOR ANALYSIS")
    print("=" * 70)
    print()

    # Current inertia tensor
    mass = 234.8  # slugs
    inertia = np.array([[14908.4, 0, 0],
                        [0, 2318.4, 0],
                        [0, 0, 17226.9]])

    print("Current Inertia Tensor (slug-ft²):")
    print(f"  Ixx = {inertia[0,0]:.1f}")
    print(f"  Iyy = {inertia[1,1]:.1f}")
    print(f"  Izz = {inertia[2,2]:.1f}")
    print(f"  Ixy = {inertia[0,1]:.1f}")
    print(f"  Iyz = {inertia[1,2]:.1f}")
    print(f"  Ixz = {inertia[0,2]:.1f}")
    print()

    # Check principal axis ratios
    print("Inertia Ratios:")
    print(f"  Ixx/Iyy = {inertia[0,0]/inertia[1,1]:.3f}")
    print(f"  Izz/Iyy = {inertia[2,2]/inertia[1,1]:.3f}")
    print(f"  Ixx/Izz = {inertia[0,0]/inertia[2,2]:.3f}")
    print()

    # For a flying wing, Ixz is typically 5-15% of Ixx
    print("Expected Ixz for Swept Flying Wing:")
    print(f"  5% of Ixx:  {0.05 * inertia[0,0]:.1f}")
    print(f"  10% of Ixx: {0.10 * inertia[0,0]:.1f}")
    print(f"  15% of Ixx: {0.15 * inertia[0,0]:.1f}")
    print()

    # Wing geometry
    b_ref = 19.890  # ft (span)
    S_ref = 199.94  # ft² (area)
    c_ref = 26.689  # ft (mean chord)

    # Estimate aspect ratio and sweep
    AR = b_ref**2 / S_ref
    print(f"Wing Geometry:")
    print(f"  Span: {b_ref:.2f} ft")
    print(f"  Area: {S_ref:.2f} ft²")
    print(f"  Mean Chord: {c_ref:.2f} ft")
    print(f"  Aspect Ratio: {AR:.3f}")
    print()

    # Very low AR indicates highly swept delta
    if AR < 2.5:
        print("WARNING: LOW ASPECT RATIO DETECTED")
        print("   This indicates a highly swept flying wing (delta planform)")
        print("   Expected characteristics:")
        print("   - Significant Ixz product of inertia (pitch-roll coupling)")
        print("   - Strong vortex lift at high alpha")
        print("   - Nonlinear pitching moment")
        print()

    # Estimate Ixz based on typical flying wing geometry
    # For a swept wing, Ixz ≈ (sweep angle factor) * sqrt(Ixx * Izz)
    estimated_Ixz = 0.10 * np.sqrt(inertia[0,0] * inertia[2,2])
    print(f"Estimated Ixz (10% coupling): {estimated_Ixz:.1f} slug-ft²")
    print()

    return inertia, estimated_Ixz


def analyze_control_coupling():
    """
    Analyze how flaperon deflections affect multiple axes.
    """
    print("=" * 70)
    print("FLAPERON CONTROL COUPLING ANALYSIS")
    print("=" * 70)
    print()

    print("Flaperons vs. Conventional Ailerons:")
    print()
    print("CONVENTIONAL AILERON (current model):")
    print("  Left up, Right down -> Roll moment only")
    print("  dCl = Cl_aileron * delta_a")
    print("  dCm = 0")
    print("  dCL = 0")
    print()

    print("FLAPERON (actual flying wing):")
    print("  Differential deflection (roll control):")
    print("    Left up, Right down -> Roll + Pitch + Drag")
    print("    dCl = Cl_flaperon * delta_f")
    print("    dCm = Cm_flaperon * delta_f  (typically nose-down)")
    print("    dCD = CD_flaperon * |delta_f|")
    print()
    print("  Symmetric deflection (pitch/flap control):")
    print("    Both down -> Pitch down + Lift up")
    print("    dCm = Cm_flap * delta_f")
    print("    dCL = CL_flap * delta_f")
    print()

    # Typical derivative values for flaperons
    print("Typical Flaperon Derivatives (per radian):")
    print("  Cl_flaperon:  0.10 to 0.20  (roll effectiveness)")
    print("  Cm_flaperon: -0.05 to -0.15 (nose-down pitching)")
    print("  CL_flaperon:  0.20 to 0.40  (lift increase)")
    print("  CD_flaperon:  0.05 to 0.15  (drag penalty)")
    print()

    return {
        'Cl_flaperon': 0.15,
        'Cm_flaperon': -0.08,
        'CL_flaperon': 0.30,
        'CD_flaperon': 0.10
    }


def analyze_flying_wing_coupling():
    """
    Analyze flying wing specific coupling derivatives.
    """
    print("=" * 70)
    print("FLYING WING COUPLING DERIVATIVES")
    print("=" * 70)
    print()

    print("Missing Coupling Terms in Current Model:")
    print()

    print("1. Cl_alpha (Roll due to angle of attack):")
    print("   For swept wings, alpha induces roll moment")
    print("   Typical value: -0.05 to -0.15 per radian")
    print("   Effect: Aircraft rolls when pitching")
    print()

    print("2. Cm_p (Pitch due to roll rate):")
    print("   Roll rate causes asymmetric lift distribution")
    print("   Typical value: -0.05 to -0.15")
    print("   Effect: Pitch-roll coupling in dynamics")
    print()

    print("3. Cm_r (Pitch due to yaw rate):")
    print("   Yaw rate affects wing flow")
    print("   Typical value: -0.02 to -0.08")
    print("   Effect: Adverse pitch in turns")
    print()

    print("4. Cn_p (Yaw due to roll rate):")
    print("   Roll induces sideslip and yaw")
    print("   Typical value: -0.01 to -0.05")
    print("   Effect: Proverse yaw in roll")
    print()

    coupling_derivatives = {
        'Cl_alpha': -0.08,
        'Cm_p': -0.10,
        'Cm_r': -0.04,
        'Cn_p': -0.02
    }

    return coupling_derivatives


def test_stability_with_coupling():
    """
    Test trajectory stability with and without coupling terms.
    """
    print("=" * 70)
    print("STABILITY TEST: WITH/WITHOUT COUPLING")
    print("=" * 70)
    print()

    # Aircraft parameters
    mass = 234.8
    inertia = np.array([[14908.4, 0, 0],
                        [0, 2318.4, 0],
                        [0, 0, 17226.9]])

    # Add estimated Ixz
    inertia_coupled = inertia.copy()
    Ixz = 0.10 * np.sqrt(inertia[0,0] * inertia[2,2])
    inertia_coupled[0, 2] = Ixz
    inertia_coupled[2, 0] = Ixz

    S_ref = 199.94
    b_ref = 19.890
    c_ref = 26.689

    print("Testing two configurations:")
    print("  1. Current: Diagonal inertia, no coupling derivatives")
    print("  2. Improved: Ixz product, coupling derivatives")
    print()

    # Create dynamics
    dynamics1 = AircraftDynamics(mass, inertia)
    dynamics2 = AircraftDynamics(mass, inertia_coupled)

    # Create aero models
    atm = StandardAtmosphere(5000)

    aero1 = LinearAeroModel(S_ref, c_ref, b_ref, rho=atm.density)
    aero1.CL_0 = 0.2
    aero1.CL_alpha = 4.5
    aero1.CD_0 = 0.03
    aero1.CD_alpha2 = 0.8
    aero1.Cm_0 = 0.0
    aero1.Cm_alpha = -0.6
    aero1.Cm_q = -8.0
    aero1.Cl_beta = -0.1
    aero1.Cl_p = -0.5
    aero1.Cn_beta = 0.08
    aero1.Cn_r = -0.2

    aero2 = LinearAeroModel(S_ref, c_ref, b_ref, rho=atm.density)
    aero2.CL_0 = 0.2
    aero2.CL_alpha = 4.5
    aero2.CD_0 = 0.03
    aero2.CD_alpha2 = 0.8
    aero2.Cm_0 = 0.0
    aero2.Cm_alpha = -0.6
    aero2.Cm_q = -8.0
    aero2.Cl_beta = -0.1
    aero2.Cl_p = -0.5
    aero2.Cl_alpha = -0.08  # NEW: Roll due to alpha
    aero2.Cn_beta = 0.08
    aero2.Cn_r = -0.2
    aero2.Cn_p = -0.02  # NEW: Yaw due to roll rate
    aero2.Cm_p = -0.10  # NEW: Pitch due to roll rate
    aero2.Cm_r = -0.04  # NEW: Pitch due to yaw rate

    # Initial state (trim condition)
    state1 = State()
    state1.position = np.array([0, 0, -5000])
    state1.velocity_body = np.array([200, 0, 0])
    state1.set_euler_angles(0, np.radians(2), 0)
    state1.angular_rates = np.array([0, 0, 0])

    state2 = State()
    state2.position = np.array([0, 0, -5000])
    state2.velocity_body = np.array([200, 0, 0])
    state2.set_euler_angles(0, np.radians(2), 0)
    state2.angular_rates = np.array([0, 0, 0])

    # Simulate short period with small perturbation
    dt = 0.01
    n_steps = 500  # 5 seconds

    def force_func1(s):
        return aero1.compute_forces_moments(s)

    def force_func2(s):
        return aero2.compute_forces_moments(s)

    # Apply small pitch perturbation
    state1.angular_rates[1] = np.radians(5)  # 5°/s pitch rate
    state2.angular_rates[1] = np.radians(5)

    roll_history1 = []
    pitch_history1 = []
    roll_history2 = []
    pitch_history2 = []

    print("Simulating 5-second response to 5°/s pitch rate perturbation...")
    print()

    for i in range(n_steps):
        # Config 1
        state_dot1 = dynamics1.state_derivative(state1, force_func1)
        x1 = state1.to_array()
        state1.from_array(x1 + state_dot1 * dt)
        roll_history1.append(np.degrees(state1.euler_angles[0]))
        pitch_history1.append(np.degrees(state1.euler_angles[1]))

        # Config 2
        state_dot2 = dynamics2.state_derivative(state2, force_func2)
        x2 = state2.to_array()
        state2.from_array(x2 + state_dot2 * dt)
        roll_history2.append(np.degrees(state2.euler_angles[0]))
        pitch_history2.append(np.degrees(state2.euler_angles[1]))

    # Analyze results
    roll_std1 = np.std(roll_history1)
    pitch_std1 = np.std(pitch_history1)
    roll_std2 = np.std(roll_history2)
    pitch_std2 = np.std(pitch_history2)

    print("Configuration 1 (Diagonal Inertia, No Coupling):")
    print(f"  Roll std: {roll_std1:.2f}°")
    print(f"  Pitch std: {pitch_std1:.2f}°")
    print()

    print("Configuration 2 (Ixz + Coupling Derivatives):")
    print(f"  Roll std: {roll_std2:.2f}°")
    print(f"  Pitch std: {pitch_std2:.2f}°")
    print()

    # Plot comparison
    time = np.arange(n_steps) * dt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(time, roll_history1, 'b-', label='Config 1 (No Coupling)', linewidth=2)
    axes[0].plot(time, roll_history2, 'r-', label='Config 2 (With Coupling)', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Roll Angle (deg)')
    axes[0].set_title('Roll Response to Pitch Perturbation')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, pitch_history1, 'b-', label='Config 1 (No Coupling)', linewidth=2)
    axes[1].plot(time, pitch_history2, 'r-', label='Config 2 (With Coupling)', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Pitch Angle (deg)')
    axes[1].set_title('Pitch Response to Pitch Perturbation')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('output/coupling_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: output/coupling_comparison.png")
    print()

    return {
        'config1': {'roll_std': roll_std1, 'pitch_std': pitch_std1},
        'config2': {'roll_std': roll_std2, 'pitch_std': pitch_std2}
    }


def main():
    """Run all diagnostics."""
    print()
    print("=" * 70)
    print(" " * 15 + "FLYING WING UAV DIAGNOSTICS")
    print("=" * 70)
    print()

    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Run diagnostics
    inertia, estimated_Ixz = analyze_inertia_tensor()
    flaperon_derivs = analyze_control_coupling()
    coupling_derivs = analyze_flying_wing_coupling()
    stability_results = test_stability_with_coupling()

    # Summary
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print()

    print("KEY FINDINGS:")
    print()
    print("1. INERTIA TENSOR:")
    print(f"   - Current Ixz = 0 slug-ft^2")
    print(f"   - Estimated Ixz ~= {estimated_Ixz:.1f} slug-ft^2 (10% coupling)")
    print(f"   - Recommendation: Add Ixz to inertia matrix")
    print()

    print("2. CONTROL MODEL:")
    print("   - Current: Ailerons (roll only)")
    print("   - Actual: Flaperons (roll + pitch + lift + drag)")
    print("   - Recommendation: Implement flaperon model")
    print()

    print("3. COUPLING DERIVATIVES:")
    print("   - Missing: Cl_alpha, Cm_p, Cm_r, Cn_p")
    print("   - Effect: Unrealistic pitch-roll coupling")
    print("   - Recommendation: Add flying wing coupling terms")
    print()

    print("4. STABILITY TEST:")
    c1 = stability_results['config1']
    c2 = stability_results['config2']
    print(f"   - Without coupling: Roll std = {c1['roll_std']:.2f}°, Pitch std = {c1['pitch_std']:.2f}°")
    print(f"   - With coupling: Roll std = {c2['roll_std']:.2f}°, Pitch std = {c2['pitch_std']:.2f}°")
    print()

    print("NEXT STEPS:")
    print("  1. Update inertia tensor with estimated Ixz")
    print("  2. Implement flaperon aerodynamic model")
    print("  3. Add flying wing coupling derivatives")
    print("  4. Retune autopilot gains for coupled dynamics")
    print()


if __name__ == "__main__":
    main()
