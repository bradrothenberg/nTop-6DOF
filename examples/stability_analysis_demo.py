"""
Stability Analysis Demonstration - Phase 4 Features

Demonstrates:
- Linearization about trim point
- Eigenvalue and mode analysis
- Stability derivatives
- Frequency response (Bode plots, step response)
- Mode identification (phugoid, short period, etc.)
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
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import ConstantThrustModel, CombinedForceModel
from src.control.trim import TrimSolver
from src.analysis.stability import StabilityAnalyzer
from src.analysis.frequency import FrequencyAnalyzer
from src.environment.atmosphere import StandardAtmosphere


def main():
    print("=" * 70)
    print("Stability Analysis Demonstration - nTop UAV")
    print("=" * 70)
    print()

    # ========================================
    # Aircraft Setup
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
    print()

    # ========================================
    # Dynamics and Aerodynamics
    # ========================================

    dynamics = AircraftDynamics(mass, inertia)

    # Realistic aerodynamic model
    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)
    aero.CL_0 = 0.2
    aero.CL_alpha = 4.5
    aero.CL_q = 3.5
    aero.CL_elevator = 0.4

    aero.CD_0 = 0.03
    aero.CD_alpha2 = 0.8

    aero.Cm_0 = 0.0
    aero.Cm_alpha = -0.6      # Stable (negative)
    aero.Cm_q = -8.0          # Pitch damping
    aero.Cm_elevator = -1.2   # Elevator effectiveness

    aero.CY_beta = -0.3
    aero.Cl_beta = -0.1       # Dihedral effect
    aero.Cl_p = -0.5          # Roll damping
    aero.Cl_r = 0.1
    aero.Cl_aileron = 0.15

    aero.Cn_beta = 0.08       # Weathercock stability
    aero.Cn_p = -0.05
    aero.Cn_r = -0.2          # Yaw damping
    aero.Cn_aileron = -0.02

    propulsion = ConstantThrustModel(thrust=600.0)
    force_model = CombinedForceModel(aero, propulsion)

    print("Aerodynamic Model:")
    print(f"  CL_alpha: {aero.CL_alpha:.2f}")
    print(f"  Cm_alpha: {aero.Cm_alpha:.2f} (stability)")
    print(f"  Cm_q: {aero.Cm_q:.2f} (pitch damping)")
    print()

    # ========================================
    # Find Trim Condition
    # ========================================

    print("Finding trim for straight and level flight...")

    def dynamics_func(state, controls):
        # Update atmosphere
        atm = StandardAtmosphere(state.altitude)
        aero.rho = atm.density

        throttle = controls.get('throttle', 0.5)
        elevator = controls.get('elevator', 0.0)

        # Get forces
        forces, moments = force_model(state, throttle)

        # Add elevator moment
        q_bar = atm.get_dynamic_pressure(state.airspeed)
        moments[1] += q_bar * S_ref * c_ref * aero.Cm_elevator * elevator

        return dynamics.state_derivative(state, lambda s: (forces, moments))

    # Trim solver
    trim_solver = TrimSolver(dynamics_func)

    trim_state, trim_controls, trim_info = trim_solver.trim_straight_level(
        altitude=5000.0,
        airspeed=200.0,
        verbose=False
    )

    print(f"Trim found:")
    print(f"  Alpha: {trim_info['alpha_deg']:.2f} deg")
    print(f"  Theta: {trim_info['theta_deg']:.2f} deg")
    print(f"  Elevator: {trim_info['elevator_deg']:.2f} deg")
    print(f"  Throttle: {trim_info['throttle_pct']:.1f}%")
    print(f"  Residual norm: {trim_info['residual_norm']:.2e}")
    print()

    # ========================================
    # Linearization
    # ========================================

    print("Linearizing dynamics about trim...")

    analyzer = StabilityAnalyzer(dynamics_func)
    linear_model = analyzer.linearize(trim_state, trim_controls, eps=1e-6)

    print(f"Linearized model: {linear_model.n_states} states, {linear_model.n_inputs} inputs")
    print()

    # ========================================
    # Stability Analysis
    # ========================================

    print("Performing stability analysis...")
    print()

    # Print detailed report
    analyzer.print_stability_report(linear_model)

    # ========================================
    # Frequency Response Analysis
    # ========================================

    print("=" * 70)
    print("FREQUENCY RESPONSE ANALYSIS")
    print("=" * 70)
    print()

    freq_analyzer = FrequencyAnalyzer(linear_model)

    # Control inputs: 0=elevator, 1=aileron, 2=rudder, 3=throttle
    # State outputs: 3=u, 4=v, 5=w, 10=p, 11=q, 12=r

    print("Generating frequency response plots...")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Elevator to pitch rate (q)
    omega1, mag1, phase1 = freq_analyzer.bode(input_idx=0, output_idx=11)

    ax1 = plt.subplot(3, 3, 1)
    ax1.semilogx(omega1, mag1, 'b-', linewidth=2)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Elevator to Pitch Rate')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    ax2 = plt.subplot(3, 3, 4)
    ax2.semilogx(omega1, phase1, 'r-', linewidth=2)
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.axhline(y=-180, color='k', linestyle='--', alpha=0.5)

    # 2. Elevator to u velocity
    omega2, mag2, phase2 = freq_analyzer.bode(input_idx=0, output_idx=3)

    ax3 = plt.subplot(3, 3, 2)
    ax3.semilogx(omega2, mag2, 'b-', linewidth=2)
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_title('Elevator to Axial Velocity')
    ax3.grid(True, which='both', alpha=0.3)

    ax4 = plt.subplot(3, 3, 5)
    ax4.semilogx(omega2, phase2, 'r-', linewidth=2)
    ax4.set_xlabel('Frequency (rad/s)')
    ax4.set_ylabel('Phase (deg)')
    ax4.grid(True, which='both', alpha=0.3)

    # 3. Aileron to roll rate (p)
    omega3, mag3, phase3 = freq_analyzer.bode(input_idx=1, output_idx=10)

    ax5 = plt.subplot(3, 3, 3)
    ax5.semilogx(omega3, mag3, 'b-', linewidth=2)
    ax5.set_ylabel('Magnitude (dB)')
    ax5.set_title('Aileron to Roll Rate')
    ax5.grid(True, which='both', alpha=0.3)

    ax6 = plt.subplot(3, 3, 6)
    ax6.semilogx(omega3, phase3, 'r-', linewidth=2)
    ax6.set_xlabel('Frequency (rad/s)')
    ax6.set_ylabel('Phase (deg)')
    ax6.grid(True, which='both', alpha=0.3)

    # 4. Step responses
    t1, y1 = freq_analyzer.step_response(input_idx=0, output_idx=11, t_final=10.0)
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(t1, y1, 'b-', linewidth=2)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Pitch Rate (rad/s)')
    ax7.set_title('Elevator Step Response')
    ax7.grid(True, alpha=0.3)

    t2, y2 = freq_analyzer.step_response(input_idx=0, output_idx=3, t_final=20.0)
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(t2, y2, 'r-', linewidth=2)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('u Velocity (ft/s)')
    ax8.set_title('Elevator to Velocity')
    ax8.grid(True, alpha=0.3)

    t3, y3 = freq_analyzer.step_response(input_idx=1, output_idx=10, t_final=5.0)
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(t3, y3, 'g-', linewidth=2)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Roll Rate (rad/s)')
    ax9.set_title('Aileron Step Response')
    ax9.grid(True, alpha=0.3)

    plt.suptitle('nTop UAV - Frequency Response Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(os.path.dirname(__file__), 'stability_analysis_results.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    print()

    plt.close()

    # ========================================
    # Summary
    # ========================================

    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()
    print("Phase 4 features demonstrated:")
    print("  [OK] Linearization about trim point")
    print("  [OK] Eigenvalue and mode analysis")
    print("  [OK] Stability assessment")
    print("  [OK] Dynamic mode identification")
    print("  [OK] Bode plots (frequency response)")
    print("  [OK] Step response analysis")
    print()
    print("The aircraft is", "STABLE" if linear_model.is_stable() else "UNSTABLE")
    print()


if __name__ == "__main__":
    main()
