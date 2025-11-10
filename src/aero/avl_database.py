"""
AVL Aerodynamic Database

Loads AVL sweep data and provides interpolation for simulation.
Can be used to build aerodynamic models from AVL analysis results.
"""

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
import pandas as pd
from typing import Dict, Optional, Tuple
import os


class AVLDatabase:
    """
    Aerodynamic database from AVL sweep data.

    Stores force and moment coefficients as functions of:
    - Angle of attack (alpha)
    - Sideslip angle (beta) - optional
    - Control deflections - optional

    Provides interpolation for simulation use.

    Parameters
    ----------
    alpha_table : array_like
        Angle of attack values (radians)
    data_table : dict
        Dictionary of coefficient arrays: {'CL': [...], 'CD': [...], ...}
    S_ref : float
        Reference area (ft²)
    c_ref : float
        Reference chord (ft)
    b_ref : float
        Reference span (ft)
    """

    def __init__(self,
                 alpha_table: np.ndarray,
                 data_table: Dict[str, np.ndarray],
                 S_ref: float,
                 c_ref: float,
                 b_ref: float):
        """Initialize AVL database."""
        self.alpha_table = np.asarray(alpha_table)
        self.data_table = {key: np.asarray(val) for key, val in data_table.items()}
        self.S_ref = S_ref
        self.c_ref = c_ref
        self.b_ref = b_ref

        # Create interpolators
        self._build_interpolators()

    def _build_interpolators(self):
        """Build interpolation functions for all coefficients."""
        self.interpolators = {}

        for coeff_name, coeff_data in self.data_table.items():
            # Use linear interpolation with extrapolation
            self.interpolators[coeff_name] = interp1d(
                self.alpha_table,
                coeff_data,
                kind='linear',
                fill_value='extrapolate'
            )

    def get_coefficients(self, alpha: float) -> Dict[str, float]:
        """
        Get aerodynamic coefficients at specified angle of attack.

        Parameters
        ----------
        alpha : float
            Angle of attack (radians)

        Returns
        -------
        dict
            Dictionary of coefficients: {'CL': ..., 'CD': ..., ...}
        """
        coeffs = {}
        for coeff_name, interpolator in self.interpolators.items():
            coeffs[coeff_name] = float(interpolator(alpha))

        return coeffs

    def get_forces_moments(self,
                            alpha: float,
                            q_bar: float,
                            angular_rates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute aerodynamic forces and moments.

        Parameters
        ----------
        alpha : float
            Angle of attack (radians)
        q_bar : float
            Dynamic pressure (lbf/ft²)
        angular_rates : array_like, optional
            Body angular rates [p, q, r] (rad/s) for damping derivatives

        Returns
        -------
        forces_body : ndarray
            Aerodynamic forces in body frame [Fx, Fy, Fz] (lbf)
        moments_body : ndarray
            Aerodynamic moments in body frame [L, M, N] (ft·lbf)
        """
        # Get coefficients
        coeffs = self.get_coefficients(alpha)

        CL = coeffs.get('CL', 0.0)
        CD = coeffs.get('CD', 0.0)
        CY = coeffs.get('CY', 0.0)
        Cl = coeffs.get('Cl', 0.0)
        Cm = coeffs.get('Cm', 0.0)
        Cn = coeffs.get('Cn', 0.0)

        # Add damping derivatives if angular rates provided
        if angular_rates is not None:
            p, q, r = angular_rates

            # Normalize angular rates
            p_hat = p * self.b_ref / (2.0 * np.sqrt(q_bar * 2 / 0.002377))  # Rough airspeed estimate
            q_hat = q * self.c_ref / (2.0 * np.sqrt(q_bar * 2 / 0.002377))
            r_hat = r * self.b_ref / (2.0 * np.sqrt(q_bar * 2 / 0.002377))

            # Damping derivatives (approximate values, should come from AVL)
            Cl += coeffs.get('Cl_p', -0.5) * p_hat + coeffs.get('Cl_r', 0.1) * r_hat
            Cm += coeffs.get('Cm_q', -8.0) * q_hat
            Cn += coeffs.get('Cn_p', -0.05) * p_hat + coeffs.get('Cn_r', -0.2) * r_hat

        # Convert wind axis coefficients to body axis forces
        # Wind axis: X = -D, Z = -L (drag opposes velocity, lift perpendicular)
        # For small beta, body ≈ wind axis
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        # Force coefficients in body frame
        # CX = -CD*cos(alpha) + CL*sin(alpha)
        # CZ = -CD*sin(alpha) - CL*cos(alpha)
        CX = -CD * cos_alpha + CL * sin_alpha
        CZ = -CD * sin_alpha - CL * cos_alpha

        # Forces
        Fx = q_bar * self.S_ref * CX
        Fy = q_bar * self.S_ref * CY
        Fz = q_bar * self.S_ref * CZ

        forces_body = np.array([Fx, Fy, Fz])

        # Moments
        L_aero = q_bar * self.S_ref * self.b_ref * Cl
        M_aero = q_bar * self.S_ref * self.c_ref * Cm
        N_aero = q_bar * self.S_ref * self.b_ref * Cn

        moments_body = np.array([L_aero, M_aero, N_aero])

        return forces_body, moments_body

    @staticmethod
    def from_avl_sweep(filepath: str, S_ref: float, c_ref: float, b_ref: float) -> 'AVLDatabase':
        """
        Load database from AVL alpha sweep results.

        Parameters
        ----------
        filepath : str
            Path to CSV file with AVL sweep data
            Expected columns: alpha (deg), CL, CD, Cm, etc.
        S_ref : float
            Reference area (ft²)
        c_ref : float
            Reference chord (ft)
        b_ref : float
            Reference span (ft)

        Returns
        -------
        AVLDatabase
            Loaded database
        """
        # Read CSV
        df = pd.read_csv(filepath)

        # Convert alpha to radians if in degrees
        if 'alpha' in df.columns:
            alpha_table = np.radians(df['alpha'].values)
        elif 'Alpha' in df.columns:
            alpha_table = np.radians(df['Alpha'].values)
        else:
            raise ValueError("CSV must contain 'alpha' or 'Alpha' column")

        # Extract coefficient data
        data_table = {}
        for col in df.columns:
            if col.lower() != 'alpha':
                data_table[col] = df[col].values

        return AVLDatabase(alpha_table, data_table, S_ref, c_ref, b_ref)

    def save(self, filepath: str):
        """
        Save database to CSV file.

        Parameters
        ----------
        filepath : str
            Output filepath
        """
        # Build dataframe
        data = {'alpha_deg': np.degrees(self.alpha_table)}
        data.update(self.data_table)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        print(f"Saved AVL database to: {filepath}")

    def plot_polars(self, save_path: Optional[str] = None):
        """
        Plot aerodynamic polars.

        Parameters
        ----------
        save_path : str, optional
            Path to save figure (if None, displays plot)
        """
        import matplotlib.pyplot as plt

        alpha_deg = np.degrees(self.alpha_table)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('AVL Aerodynamic Database', fontsize=14, fontweight='bold')

        # CL vs alpha
        if 'CL' in self.data_table:
            axes[0, 0].plot(alpha_deg, self.data_table['CL'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Angle of Attack (deg)')
            axes[0, 0].set_ylabel('CL')
            axes[0, 0].set_title('Lift Coefficient')
            axes[0, 0].grid(True, alpha=0.3)

        # CD vs alpha
        if 'CD' in self.data_table:
            axes[0, 1].plot(alpha_deg, self.data_table['CD'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Angle of Attack (deg)')
            axes[0, 1].set_ylabel('CD')
            axes[0, 1].set_title('Drag Coefficient')
            axes[0, 1].grid(True, alpha=0.3)

        # Cm vs alpha
        if 'Cm' in self.data_table:
            axes[1, 0].plot(alpha_deg, self.data_table['Cm'], 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Angle of Attack (deg)')
            axes[1, 0].set_ylabel('Cm')
            axes[1, 0].set_title('Pitching Moment Coefficient')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # L/D vs alpha
        if 'CL' in self.data_table and 'CD' in self.data_table:
            L_D = self.data_table['CL'] / (self.data_table['CD'] + 1e-10)
            axes[1, 1].plot(alpha_deg, L_D, 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Angle of Attack (deg)')
            axes[1, 1].set_ylabel('L/D')
            axes[1, 1].set_title('Lift-to-Drag Ratio')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        else:
            plt.show()

        plt.close()


def create_sample_database():
    """Create sample AVL database for testing."""
    print("=" * 60)
    print("AVL Database - Sample Creation")
    print("=" * 60)
    print()

    # Sample data (typical subsonic wing)
    alpha_table = np.radians(np.linspace(-5, 15, 21))

    # Simple aerodynamic model
    CL_table = 0.2 + 5.0 * alpha_table
    CD_table = 0.02 + 0.05 * alpha_table**2
    Cm_table = 0.05 - 0.5 * alpha_table

    data_table = {
        'CL': CL_table,
        'CD': CD_table,
        'Cm': Cm_table,
        'CY': np.zeros_like(alpha_table),
        'Cl': np.zeros_like(alpha_table),
        'Cn': np.zeros_like(alpha_table)
    }

    # Create database
    db = AVLDatabase(
        alpha_table=alpha_table,
        data_table=data_table,
        S_ref=199.94,
        c_ref=26.689,
        b_ref=19.890
    )

    print("Sample database created:")
    print(f"  Alpha range: {np.degrees(alpha_table[0]):.1f}° to {np.degrees(alpha_table[-1]):.1f}°")
    print(f"  Number of points: {len(alpha_table)}")
    print(f"  S_ref: {db.S_ref:.2f} ft²")
    print()

    # Test interpolation
    alpha_test = np.radians(5)
    coeffs = db.get_coefficients(alpha_test)
    print(f"Coefficients at alpha = {np.degrees(alpha_test):.1f}°:")
    print(f"  CL: {coeffs['CL']:.4f}")
    print(f"  CD: {coeffs['CD']:.4f}")
    print(f"  Cm: {coeffs['Cm']:.4f}")
    print()

    return db


if __name__ == "__main__":
    db = create_sample_database()

    # Save database
    output_dir = os.path.join(os.path.dirname(__file__), '../../avl_files')
    os.makedirs(output_dir, exist_ok=True)

    db.save(os.path.join(output_dir, 'sample_aero_database.csv'))
    print()
