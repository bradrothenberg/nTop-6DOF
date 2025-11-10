"""
Stability Analysis Tools

Provides linearization and stability analysis for 6-DOF flight dynamics:
- Linearize dynamics about trim point
- Extract A, B, C, D state-space matrices
- Compute eigenvalues and eigenvectors
- Identify and analyze dynamic modes
"""

import numpy as np
from typing import Callable, Dict, Tuple, List, Optional
from dataclasses import dataclass
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.state import State


@dataclass
class DynamicMode:
    """
    Represents a dynamic mode of the aircraft.

    Attributes
    ----------
    name : str
        Mode name (e.g., 'Phugoid', 'Short Period', 'Dutch Roll')
    eigenvalue : complex
        Complex eigenvalue
    damping_ratio : float
        Damping ratio ζ
    natural_frequency : float
        Natural frequency ωn (rad/s)
    period : float
        Period of oscillation (seconds)
    time_to_half : float
        Time to half amplitude (seconds)
    eigenvector : ndarray
        Normalized eigenvector
    """
    name: str
    eigenvalue: complex
    damping_ratio: float
    natural_frequency: float
    period: float
    time_to_half: float
    eigenvector: np.ndarray


class LinearizedModel:
    """
    Linearized state-space representation of aircraft dynamics.

    State-space form: dx/dt = A*x + B*u
                      y = C*x + D*u

    Where:
    - x: state perturbation vector (13x1)
    - u: control input vector (4x1: elevator, aileron, rudder, throttle)
    - y: output vector (same as state for full-state feedback)

    Attributes
    ----------
    A : ndarray
        System matrix (13x13)
    B : ndarray
        Input matrix (13x4)
    C : ndarray
        Output matrix (13x13, identity for full state)
    D : ndarray
        Feedthrough matrix (13x4, zeros)
    trim_state : State
        Trim state about which linearization is performed
    trim_controls : dict
        Trim control inputs
    """

    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 trim_state: State,
                 trim_controls: Dict[str, float]):
        """Initialize linearized model."""
        self.A = A
        self.B = B
        self.C = np.eye(A.shape[0])  # Full state output
        self.D = np.zeros((A.shape[0], B.shape[1]))
        self.trim_state = trim_state
        self.trim_controls = trim_controls

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        """Number of control inputs."""
        return self.B.shape[1]

    def eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of system matrix A.

        Returns
        -------
        ndarray
            Complex eigenvalues
        """
        return np.linalg.eigvals(self.A)

    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of system matrix A.

        Returns
        -------
        eigenvalues : ndarray
            Complex eigenvalues
        eigenvectors : ndarray
            Eigenvectors (columns)
        """
        return np.linalg.eig(self.A)

    def is_stable(self) -> bool:
        """
        Check if system is stable (all eigenvalues have negative real parts).

        Returns
        -------
        bool
            True if stable, False otherwise
        """
        eigs = self.eigenvalues()
        return np.all(np.real(eigs) < 0)


class StabilityAnalyzer:
    """
    Stability analysis for 6-DOF flight dynamics.

    Performs linearization about trim, eigenvalue analysis,
    and mode identification.
    """

    def __init__(self, dynamics_function: Callable):
        """
        Initialize stability analyzer.

        Parameters
        ----------
        dynamics_function : callable
            Function that computes state derivatives: f(state, controls) -> state_dot
        """
        self.dynamics_function = dynamics_function

    def linearize(self,
                  trim_state: State,
                  trim_controls: Dict[str, float],
                  eps: float = 1e-6) -> LinearizedModel:
        """
        Linearize dynamics about trim point using finite differences.

        Parameters
        ----------
        trim_state : State
            Trim state
        trim_controls : dict
            Trim control inputs
        eps : float, optional
            Perturbation size for finite differences

        Returns
        -------
        LinearizedModel
            Linearized state-space model
        """
        # State vector dimension
        n = 13

        # Control vector dimension (elevator, aileron, rudder, throttle)
        m = 4
        control_names = ['elevator', 'aileron', 'rudder', 'throttle']

        # Initialize matrices
        A = np.zeros((n, n))
        B = np.zeros((n, m))

        # Nominal state derivative
        x0 = trim_state.to_array()
        f0 = self.dynamics_function(trim_state, trim_controls)

        # Compute A matrix (df/dx) using forward differences
        for i in range(n):
            # Perturb state
            x_pert = x0.copy()
            x_pert[i] += eps

            state_pert = State()
            state_pert.from_array(x_pert)

            # Compute perturbed derivative
            f_pert = self.dynamics_function(state_pert, trim_controls)

            # Finite difference
            A[:, i] = (f_pert - f0) / eps

        # Compute B matrix (df/du) using forward differences
        for i, control_name in enumerate(control_names):
            # Perturb control
            controls_pert = trim_controls.copy()
            controls_pert[control_name] += eps

            # Compute perturbed derivative
            f_pert = self.dynamics_function(trim_state, controls_pert)

            # Finite difference
            B[:, i] = (f_pert - f0) / eps

        return LinearizedModel(A, B, trim_state, trim_controls)

    def identify_modes(self, linear_model: LinearizedModel) -> List[DynamicMode]:
        """
        Identify and characterize dynamic modes.

        Parameters
        ----------
        linear_model : LinearizedModel
            Linearized model

        Returns
        -------
        list of DynamicMode
            Identified dynamic modes sorted by natural frequency
        """
        eigenvalues, eigenvectors = linear_model.eigenvectors()

        modes = []

        # Process each eigenvalue
        processed = set()
        for i, lam in enumerate(eigenvalues):
            if i in processed:
                continue

            # For complex conjugate pairs, process together
            if np.imag(lam) != 0:
                # Find conjugate
                conj_idx = None
                for j in range(i + 1, len(eigenvalues)):
                    if np.isclose(eigenvalues[j], np.conj(lam)):
                        conj_idx = j
                        break

                if conj_idx is not None:
                    processed.add(conj_idx)

                # Complex eigenvalue: λ = σ ± jω
                sigma = np.real(lam)
                omega = np.abs(np.imag(lam))

                # Natural frequency and damping
                omega_n = np.sqrt(sigma**2 + omega**2)
                zeta = -sigma / omega_n if omega_n > 0 else 0

                # Period and time to half
                period = 2 * np.pi / omega if omega > 0 else np.inf
                time_to_half = np.log(2) / (-sigma) if sigma < 0 else np.inf

                # Identify mode type based on state participation
                mode_name = self._identify_mode_type(eigenvectors[:, i], linear_model.trim_state)

                mode = DynamicMode(
                    name=mode_name,
                    eigenvalue=lam,
                    damping_ratio=zeta,
                    natural_frequency=omega_n,
                    period=period,
                    time_to_half=time_to_half,
                    eigenvector=eigenvectors[:, i]
                )

                modes.append(mode)

            else:
                # Real eigenvalue
                sigma = np.real(lam)
                time_const = 1 / (-sigma) if sigma != 0 else np.inf

                mode_name = self._identify_mode_type(eigenvectors[:, i], linear_model.trim_state)

                mode = DynamicMode(
                    name=mode_name,
                    eigenvalue=lam,
                    damping_ratio=1.0,  # Overdamped
                    natural_frequency=np.abs(sigma),
                    period=np.inf,
                    time_to_half=np.log(2) * time_const if sigma < 0 else np.inf,
                    eigenvector=eigenvectors[:, i]
                )

                modes.append(mode)

            processed.add(i)

        # Sort by natural frequency (descending)
        modes.sort(key=lambda m: m.natural_frequency, reverse=True)

        return modes

    def _identify_mode_type(self, eigenvector: np.ndarray, trim_state: State) -> str:
        """
        Identify mode type based on eigenvector participation.

        Parameters
        ----------
        eigenvector : ndarray
            Mode eigenvector
        trim_state : State
            Trim state

        Returns
        -------
        str
            Mode name
        """
        # State indices: [x, y, z, u, v, w, q0, q1, q2, q3, p, q, r]
        # 0-2: position, 3-5: velocity, 6-9: quaternion, 10-12: rates

        # Compute relative participation (magnitude)
        participation = np.abs(eigenvector)
        total = np.sum(participation)
        if total > 0:
            participation /= total

        # Analyze dominant states
        pos_part = np.sum(participation[0:3])      # Position
        vel_part = np.sum(participation[3:6])      # Velocity
        att_part = np.sum(participation[6:10])     # Attitude (quaternion)
        rate_part = np.sum(participation[10:13])   # Angular rates

        u_part = participation[3]  # Axial velocity
        w_part = participation[5]  # Vertical velocity
        q_part = participation[11]  # Pitch rate

        v_part = participation[4]  # Side velocity
        p_part = participation[10]  # Roll rate
        r_part = participation[12]  # Yaw rate

        # Mode identification heuristics
        # Short period: high pitch rate and angle of attack (w)
        if q_part > 0.3 and w_part > 0.1:
            return "Short Period"

        # Phugoid: high u and position participation, low pitch rate
        if u_part > 0.2 and pos_part > 0.2 and q_part < 0.1:
            return "Phugoid"

        # Dutch roll: high yaw rate and sideslip (v)
        if r_part > 0.3 and v_part > 0.1:
            return "Dutch Roll"

        # Roll mode: high roll rate
        if p_part > 0.4:
            return "Roll"

        # Spiral mode: slow convergence with position and heading
        if pos_part > 0.3 and rate_part < 0.2:
            return "Spiral"

        # Default
        return "Unidentified"

    def print_stability_report(self, linear_model: LinearizedModel):
        """
        Print comprehensive stability analysis report.

        Parameters
        ----------
        linear_model : LinearizedModel
            Linearized model
        """
        print("=" * 70)
        print("STABILITY ANALYSIS REPORT")
        print("=" * 70)
        print()

        # Trim conditions
        print("Trim Conditions:")
        print(f"  Altitude: {linear_model.trim_state.altitude:.1f} ft")
        print(f"  Airspeed: {linear_model.trim_state.airspeed:.1f} ft/s")
        alpha_deg = np.degrees(linear_model.trim_state.alpha)
        beta_deg = np.degrees(linear_model.trim_state.beta)
        print(f"  Alpha: {alpha_deg:.2f} deg")
        print(f"  Beta: {beta_deg:.2f} deg")
        print()

        print("Trim Controls:")
        for name, value in linear_model.trim_controls.items():
            if 'throttle' in name:
                print(f"  {name.capitalize()}: {value*100:.1f}%")
            else:
                print(f"  {name.capitalize()}: {np.degrees(value):.2f} deg")
        print()

        # Stability
        is_stable = linear_model.is_stable()
        print(f"System Stability: {'STABLE' if is_stable else 'UNSTABLE'}")
        print()

        # Modes
        modes = self.identify_modes(linear_model)

        print("Dynamic Modes:")
        print("-" * 70)
        print(f"{'Mode':<15} {'Freq (rad/s)':<12} {'Damp Ratio':<12} {'Period (s)':<12} {'T_half (s)':<12}")
        print("-" * 70)

        for mode in modes:
            period_str = f"{mode.period:.2f}" if mode.period < 1000 else "N/A"
            t_half_str = f"{mode.time_to_half:.2f}" if mode.time_to_half < 1000 else "N/A"

            print(f"{mode.name:<15} {mode.natural_frequency:<12.4f} {mode.damping_ratio:<12.4f} "
                  f"{period_str:<12} {t_half_str:<12}")

        print()


def test_stability():
    """Test stability analysis with simple example."""
    print("=" * 60)
    print("Stability Analysis Test")
    print("=" * 60)
    print()
    print("Note: This is a basic test. Full testing requires complete")
    print("      dynamics model (see Phase 4 integration tests)")
    print()


if __name__ == "__main__":
    test_stability()
