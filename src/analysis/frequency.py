"""
Frequency Response Analysis

Provides frequency-domain analysis tools for control system design:
- Bode plots (magnitude and phase)
- Step response
- Impulse response
- Transfer functions
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, List
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.analysis.stability import LinearizedModel


class FrequencyAnalyzer:
    """
    Frequency-domain analysis for linearized aircraft dynamics.

    Computes transfer functions, frequency responses, and time responses
    for control system analysis and design.
    """

    def __init__(self, linear_model: LinearizedModel):
        """
        Initialize frequency analyzer.

        Parameters
        ----------
        linear_model : LinearizedModel
            Linearized state-space model
        """
        self.linear_model = linear_model
        self.sys = signal.StateSpace(
            linear_model.A,
            linear_model.B,
            linear_model.C,
            linear_model.D
        )

    def bode(self,
             input_idx: int = 0,
             output_idx: int = 3,
             omega: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Bode plot data (frequency response).

        Parameters
        ----------
        input_idx : int, optional
            Control input index (0=elevator, 1=aileron, 2=rudder, 3=throttle)
        output_idx : int, optional
            Output state index (default: 3 = u velocity)
        omega : ndarray, optional
            Frequency vector (rad/s). If None, automatically generated.

        Returns
        -------
        omega : ndarray
            Frequency vector (rad/s)
        magnitude : ndarray
            Magnitude response (dB)
        phase : ndarray
            Phase response (degrees)
        """
        # Create SISO system for specific input-output pair
        A = self.linear_model.A
        B = self.linear_model.B[:, input_idx:input_idx+1]
        C = self.linear_model.C[output_idx:output_idx+1, :]
        D = self.linear_model.D[output_idx:output_idx+1, input_idx:input_idx+1]

        sys_siso = signal.StateSpace(A, B, C, D)

        # Compute frequency response
        if omega is None:
            omega = np.logspace(-2, 2, 1000)  # 0.01 to 100 rad/s

        omega, h = signal.freqs_zpk(*signal.ss2zpk(A, B, C, D), worN=omega)

        # Convert to magnitude (dB) and phase (deg)
        magnitude = 20 * np.log10(np.abs(h))
        phase = np.angle(h, deg=True)

        return omega, magnitude, phase

    def step_response(self,
                      input_idx: int = 0,
                      output_idx: int = 3,
                      t_final: float = 20.0,
                      n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute step response.

        Parameters
        ----------
        input_idx : int, optional
            Control input index
        output_idx : int, optional
            Output state index
        t_final : float, optional
            Final time (seconds)
        n_points : int, optional
            Number of time points

        Returns
        -------
        t : ndarray
            Time vector (seconds)
        y : ndarray
            Step response
        """
        # Create SISO system
        A = self.linear_model.A
        B = self.linear_model.B[:, input_idx:input_idx+1]
        C = self.linear_model.C[output_idx:output_idx+1, :]
        D = self.linear_model.D[output_idx:output_idx+1, input_idx:input_idx+1]

        sys_siso = signal.StateSpace(A, B, C, D)

        # Time vector
        t = np.linspace(0, t_final, n_points)

        # Compute step response
        t_out, y_out = signal.step(sys_siso, T=t)

        return t_out, y_out.flatten()

    def impulse_response(self,
                         input_idx: int = 0,
                         output_idx: int = 3,
                         t_final: float = 20.0,
                         n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute impulse response.

        Parameters
        ----------
        input_idx : int, optional
            Control input index
        output_idx : int, optional
            Output state index
        t_final : float, optional
            Final time (seconds)
        n_points : int, optional
            Number of time points

        Returns
        -------
        t : ndarray
            Time vector (seconds)
        y : ndarray
            Impulse response
        """
        # Create SISO system
        A = self.linear_model.A
        B = self.linear_model.B[:, input_idx:input_idx+1]
        C = self.linear_model.C[output_idx:output_idx+1, :]
        D = self.linear_model.D[output_idx:output_idx+1, input_idx:input_idx+1]

        sys_siso = signal.StateSpace(A, B, C, D)

        # Time vector
        t = np.linspace(0, t_final, n_points)

        # Compute impulse response
        t_out, y_out = signal.impulse(sys_siso, T=t)

        return t_out, y_out.flatten()

    def gain_margin_phase_margin(self,
                                   input_idx: int = 0,
                                   output_idx: int = 3) -> Tuple[float, float, float, float]:
        """
        Compute gain and phase margins.

        Parameters
        ----------
        input_idx : int, optional
            Control input index
        output_idx : int, optional
            Output state index

        Returns
        -------
        gm : float
            Gain margin (dB)
        pm : float
            Phase margin (degrees)
        wgc : float
            Gain crossover frequency (rad/s)
        wpc : float
            Phase crossover frequency (rad/s)
        """
        # Create SISO system
        A = self.linear_model.A
        B = self.linear_model.B[:, input_idx:input_idx+1]
        C = self.linear_model.C[output_idx:output_idx+1, :]
        D = self.linear_model.D[output_idx:output_idx+1, input_idx:input_idx+1]

        sys_siso = signal.StateSpace(A, B, C, D)

        # Convert to transfer function
        num, den = signal.ss2tf(A, B.flatten(), C.flatten(), D.flatten())
        sys_tf = signal.TransferFunction(num, den)

        # Compute margins
        gm, pm, wpc, wgc = signal.margin(sys_tf)

        # Convert gain margin to dB
        gm_db = 20 * np.log10(gm) if gm > 0 else np.inf

        return gm_db, pm, wgc, wpc

    def plot_bode(self,
                  input_idx: int = 0,
                  output_idx: int = 3,
                  input_name: str = "Control",
                  output_name: str = "State",
                  save_path: Optional[str] = None):
        """
        Plot Bode diagram.

        Parameters
        ----------
        input_idx : int, optional
            Control input index
        output_idx : int, optional
            Output state index
        input_name : str, optional
            Input label for plot
        output_name : str, optional
            Output label for plot
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt

        omega, magnitude, phase = self.bode(input_idx, output_idx)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Magnitude plot
        ax1.semilogx(omega, magnitude, 'b-', linewidth=2)
        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.set_title(f'Bode Plot: {input_name} to {output_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Phase plot
        ax2.semilogx(omega, phase, 'r-', linewidth=2)
        ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
        ax2.set_ylabel('Phase (deg)', fontsize=12)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.axhline(y=-180, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Bode plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_step_response(self,
                           input_idx: int = 0,
                           output_idx: int = 3,
                           input_name: str = "Control",
                           output_name: str = "State",
                           save_path: Optional[str] = None):
        """
        Plot step response.

        Parameters
        ----------
        input_idx : int, optional
            Control input index
        output_idx : int, optional
            Output state index
        input_name : str, optional
            Input label for plot
        output_name : str, optional
            Output label for plot
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt

        t, y = self.step_response(input_idx, output_idx)

        plt.figure(figsize=(10, 6))
        plt.plot(t, y, 'b-', linewidth=2)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(f'{output_name} Response', fontsize=12)
        plt.title(f'Step Response: {input_name} to {output_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Add rise time and settling time annotations if applicable
        if np.max(np.abs(y)) > 0:
            final_value = y[-1]
            if final_value > 0:
                # Rise time (10% to 90%)
                idx_10 = np.where(y >= 0.1 * final_value)[0]
                idx_90 = np.where(y >= 0.9 * final_value)[0]
                if len(idx_10) > 0 and len(idx_90) > 0:
                    rise_time = t[idx_90[0]] - t[idx_10[0]]
                    plt.text(0.02, 0.98, f'Rise Time: {rise_time:.2f} s',
                             transform=plt.gca().transAxes,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Step response plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def test_frequency():
    """Test frequency analysis."""
    print("=" * 60)
    print("Frequency Response Analysis Test")
    print("=" * 60)
    print()
    print("Note: This is a basic test. Full testing requires complete")
    print("      linearized model (see Phase 4 integration tests)")
    print()


if __name__ == "__main__":
    test_frequency()
