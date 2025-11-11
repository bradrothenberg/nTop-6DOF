"""
XFOIL Interface

Provides Python interface to XFOIL for 2D airfoil analysis:
- Airfoil polar generation (CL, CD, CM vs alpha)
- Reynolds number effects
- Boundary layer and viscous flow analysis
- Integration with 6-DOF aerodynamic models
"""

import subprocess
import os
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import re


@dataclass
class AirfoilPolar:
    """
    Container for XFOIL airfoil polar data.

    Attributes
    ----------
    alpha : np.ndarray
        Angle of attack (degrees)
    CL : np.ndarray
        Lift coefficient
    CD : np.ndarray
        Drag coefficient
    CDp : np.ndarray
        Pressure drag coefficient
    CM : np.ndarray
        Moment coefficient about quarter chord
    Top_Xtr : np.ndarray
        Top transition location (x/c)
    Bot_Xtr : np.ndarray
        Bottom transition location (x/c)
    reynolds : float
        Reynolds number
    mach : float
        Mach number
    ncrit : float
        Ncrit value used
    airfoil_name : str
        Airfoil identifier
    """
    alpha: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CDp: np.ndarray
    CM: np.ndarray
    Top_Xtr: np.ndarray
    Bot_Xtr: np.ndarray
    reynolds: float
    mach: float
    ncrit: float
    airfoil_name: str

    def __post_init__(self):
        """Validate data arrays."""
        n = len(self.alpha)
        arrays = [self.CL, self.CD, self.CDp, self.CM, self.Top_Xtr, self.Bot_Xtr]
        for arr in arrays:
            if len(arr) != n:
                raise ValueError("All polar data arrays must have same length")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert polar to pandas DataFrame."""
        return pd.DataFrame({
            'alpha': self.alpha,
            'CL': self.CL,
            'CD': self.CD,
            'CDp': self.CDp,
            'CM': self.CM,
            'Top_Xtr': self.Top_Xtr,
            'Bot_Xtr': self.Bot_Xtr
        })

    def save_csv(self, filename: str):
        """Save polar data to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

        # Add header with metadata
        with open(filename, 'r') as f:
            content = f.read()

        header = f"""# XFOIL Polar Data
# Airfoil: {self.airfoil_name}
# Reynolds: {self.reynolds:.0f}
# Mach: {self.mach:.4f}
# Ncrit: {self.ncrit}
"""
        with open(filename, 'w') as f:
            f.write(header + content)


class XFOILInterface:
    """
    Python interface to XFOIL airfoil analysis program.

    Provides methods to:
    - Run XFOIL polar analysis
    - Generate airfoil coordinates
    - Compute aerodynamic coefficients
    - Export results for integration with 6-DOF simulation

    Parameters
    ----------
    xfoil_exe : str
        Path to XFOIL executable
    work_dir : Optional[str]
        Working directory for temporary files (default: temp directory)

    Examples
    --------
    >>> xfoil = XFOILInterface("C:/path/to/xfoil.exe")
    >>> polar = xfoil.run_polar("NACA 64-212", Re=3e6, alpha_range=(-5, 15, 0.5))
    >>> print(f"Max L/D: {max(polar.CL / polar.CD):.2f}")
    """

    def __init__(self, xfoil_exe: str, work_dir: Optional[str] = None):
        """Initialize XFOIL interface."""
        self.xfoil_exe = xfoil_exe

        if not os.path.exists(xfoil_exe):
            raise FileNotFoundError(f"XFOIL executable not found: {xfoil_exe}")

        if work_dir is None:
            self.work_dir = tempfile.mkdtemp(prefix='xfoil_')
        else:
            self.work_dir = work_dir
            os.makedirs(work_dir, exist_ok=True)

    def generate_naca_airfoil(self, naca_code: str, n_points: int = 160) -> np.ndarray:
        """
        Generate NACA airfoil coordinates using XFOIL.

        Parameters
        ----------
        naca_code : str
            NACA 4-digit or 5-digit code (e.g., "2412", "64-212")
        n_points : int
            Number of points on airfoil surface

        Returns
        -------
        coords : np.ndarray
            Airfoil coordinates, shape (n_points, 2) [x, y]
        """
        coord_file = os.path.join(self.work_dir, f"{naca_code}.dat")

        commands = [
            f"NACA {naca_code}",
            f"PPAR",
            f"N {n_points}",
            "",
            "",
            f"SAVE {coord_file}",
            "",
            "QUIT"
        ]

        self._run_xfoil(commands)

        # Read coordinates
        coords = self._read_airfoil_file(coord_file)
        return coords

    def load_airfoil(self, filename: str) -> np.ndarray:
        """
        Load airfoil coordinates from file.

        Parameters
        ----------
        filename : str
            Path to airfoil coordinate file

        Returns
        -------
        coords : np.ndarray
            Airfoil coordinates, shape (n, 2)
        """
        return self._read_airfoil_file(filename)

    def run_polar(
        self,
        airfoil: str,
        reynolds: float,
        mach: float = 0.0,
        alpha_range: Tuple[float, float, float] = (-5, 15, 0.5),
        ncrit: float = 9.0,
        max_iter: int = 200,
        airfoil_file: Optional[str] = None
    ) -> AirfoilPolar:
        """
        Run XFOIL polar analysis.

        Parameters
        ----------
        airfoil : str
            NACA code (e.g., "2412") or airfoil name
        reynolds : float
            Reynolds number
        mach : float, optional
            Mach number (default: 0.0)
        alpha_range : tuple of float, optional
            (alpha_start, alpha_end, alpha_step) in degrees
        ncrit : float, optional
            Ncrit value for transition (default: 9.0)
            Lower = free transition, Higher = forced transition
        max_iter : int, optional
            Maximum viscous iterations
        airfoil_file : Optional[str], optional
            Path to airfoil file (if not NACA)

        Returns
        -------
        polar : AirfoilPolar
            Polar data including CL, CD, CM vs alpha
        """
        polar_file = os.path.join(self.work_dir, "polar.txt")

        # Remove old polar file
        if os.path.exists(polar_file):
            os.remove(polar_file)

        # Build command sequence
        commands = []

        # Load airfoil
        if airfoil_file is not None:
            commands.append(f"LOAD {airfoil_file}")
        elif airfoil.upper().startswith("NACA"):
            commands.append(f"NACA {airfoil.replace('NACA', '').strip()}")
        else:
            raise ValueError("Must provide either NACA code or airfoil_file")

        # Smooth and panel
        commands.extend([
            "PPAR",
            "N 200",  # 200 panels
            "",
            "",
            "OPER"
        ])

        # Set operating conditions
        commands.extend([
            f"VISC {reynolds:.0f}",
            f"MACH {mach:.4f}",
            f"VPAR",
            f"N {ncrit}",
            "",
            f"ITER {max_iter}"
        ])

        # Polar accumulation
        commands.extend([
            "PACC",
            polar_file,  # Save file
            "",          # No dump file
        ])

        # Alpha sweep
        alpha_start, alpha_end, alpha_step = alpha_range
        commands.append(f"ASEQ {alpha_start} {alpha_end} {alpha_step}")

        # Close polar and quit
        commands.extend([
            "PACC",  # Close polar accumulation
            "",
            "QUIT"
        ])

        # Run XFOIL
        self._run_xfoil(commands)

        # Parse polar file
        polar = self._parse_polar_file(polar_file, reynolds, mach, ncrit, airfoil)

        return polar

    def run_multi_reynolds(
        self,
        airfoil: str,
        reynolds_list: List[float],
        mach: float = 0.0,
        alpha_range: Tuple[float, float, float] = (-5, 15, 0.5),
        ncrit: float = 9.0,
        airfoil_file: Optional[str] = None
    ) -> List[AirfoilPolar]:
        """
        Run XFOIL analysis at multiple Reynolds numbers.

        Parameters
        ----------
        airfoil : str
            NACA code or airfoil name
        reynolds_list : List[float]
            List of Reynolds numbers to analyze
        mach : float, optional
            Mach number
        alpha_range : tuple, optional
            Alpha sweep range
        ncrit : float, optional
            Ncrit value
        airfoil_file : Optional[str], optional
            Path to airfoil file

        Returns
        -------
        polars : List[AirfoilPolar]
            List of polar data for each Reynolds number
        """
        polars = []
        for Re in reynolds_list:
            print(f"Running XFOIL analysis at Re = {Re:.2e}...")
            polar = self.run_polar(
                airfoil, Re, mach, alpha_range, ncrit, airfoil_file=airfoil_file
            )
            polars.append(polar)
        return polars

    def _run_xfoil(self, commands: List[str]):
        """
        Run XFOIL with given command sequence.

        Parameters
        ----------
        commands : List[str]
            List of XFOIL commands
        """
        # Write command file
        cmd_file = os.path.join(self.work_dir, "xfoil_commands.txt")
        with open(cmd_file, 'w') as f:
            f.write('\n'.join(commands))

        # Run XFOIL
        with open(cmd_file, 'r') as f_in:
            result = subprocess.run(
                [self.xfoil_exe],
                stdin=f_in,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.work_dir,
                text=True,
                timeout=60
            )

        if result.returncode != 0:
            print("XFOIL stderr:", result.stderr)

    def _read_airfoil_file(self, filename: str) -> np.ndarray:
        """Read airfoil coordinates from file."""
        coords = []
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Skip header
        start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line[0].isalpha():
                start_idx = i
                break

        # Read coordinates
        for line in lines[start_idx:]:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        coords.append([x, y])
                    except ValueError:
                        continue

        return np.array(coords)

    def _parse_polar_file(
        self,
        filename: str,
        reynolds: float,
        mach: float,
        ncrit: float,
        airfoil_name: str
    ) -> AirfoilPolar:
        """Parse XFOIL polar output file."""
        data = {
            'alpha': [],
            'CL': [],
            'CD': [],
            'CDp': [],
            'CM': [],
            'Top_Xtr': [],
            'Bot_Xtr': []
        }

        with open(filename, 'r') as f:
            lines = f.readlines()

        # Find data start
        data_start = 0
        for i, line in enumerate(lines):
            if 'alpha' in line.lower() and 'CL' in line:
                data_start = i + 2
                break

        # Parse data lines
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 7:
                try:
                    data['alpha'].append(float(parts[0]))
                    data['CL'].append(float(parts[1]))
                    data['CD'].append(float(parts[2]))
                    data['CDp'].append(float(parts[3]))
                    data['CM'].append(float(parts[4]))
                    data['Top_Xtr'].append(float(parts[5]))
                    data['Bot_Xtr'].append(float(parts[6]))
                except (ValueError, IndexError):
                    continue

        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])

        return AirfoilPolar(
            alpha=data['alpha'],
            CL=data['CL'],
            CD=data['CD'],
            CDp=data['CDp'],
            CM=data['CM'],
            Top_Xtr=data['Top_Xtr'],
            Bot_Xtr=data['Bot_Xtr'],
            reynolds=reynolds,
            mach=mach,
            ncrit=ncrit,
            airfoil_name=airfoil_name
        )


def plot_polar(polar: AirfoilPolar, save_path: Optional[str] = None):
    """
    Plot airfoil polar data.

    Parameters
    ----------
    polar : AirfoilPolar
        Polar data to plot
    save_path : Optional[str]
        Path to save plot (if None, display instead)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # CL vs alpha
    axes[0, 0].plot(polar.alpha, polar.CL, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Alpha (deg)')
    axes[0, 0].set_ylabel('CL')
    axes[0, 0].set_title(f'{polar.airfoil_name} - Lift Coefficient')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].axvline(0, color='k', linestyle='--', linewidth=0.5)

    # CD vs alpha
    axes[0, 1].plot(polar.alpha, polar.CD, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Alpha (deg)')
    axes[0, 1].set_ylabel('CD')
    axes[0, 1].set_title('Drag Coefficient')
    axes[0, 1].grid(True, alpha=0.3)

    # CL vs CD (drag polar)
    axes[1, 0].plot(polar.CD, polar.CL, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('CD')
    axes[1, 0].set_ylabel('CL')
    axes[1, 0].set_title('Drag Polar')
    axes[1, 0].grid(True, alpha=0.3)

    # L/D vs alpha
    LD = polar.CL / np.where(polar.CD > 1e-6, polar.CD, 1e-6)
    axes[1, 1].plot(polar.alpha, LD, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Alpha (deg)')
    axes[1, 1].set_ylabel('L/D')
    axes[1, 1].set_title(f'L/D Ratio (max: {np.max(LD):.1f})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linestyle='--', linewidth=0.5)

    # Add info text
    info_text = f"Re = {polar.reynolds:.2e}\nMach = {polar.mach:.3f}\nNcrit = {polar.ncrit}"
    fig.text(0.99, 0.01, info_text, ha='right', va='bottom',
             fontsize=9, family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("XFOIL Interface Example")
    print("=" * 60)

    # Note: Update this path to your XFOIL executable
    xfoil_exe = "xfoil.exe"  # Or "C:/path/to/xfoil.exe"

    if os.path.exists(xfoil_exe):
        xfoil = XFOILInterface(xfoil_exe)

        # Run polar analysis for NACA 64-212
        print("Running polar analysis for NACA 64-212...")
        polar = xfoil.run_polar(
            airfoil="NACA 64-212",
            reynolds=3e6,
            mach=0.25,
            alpha_range=(-5, 15, 0.5)
        )

        print(f"\nResults:")
        print(f"  Alpha range: {polar.alpha[0]:.1f} to {polar.alpha[-1]:.1f} deg")
        print(f"  CL range: {np.min(polar.CL):.3f} to {np.max(polar.CL):.3f}")
        print(f"  Max L/D: {np.max(polar.CL / polar.CD):.1f}")

        # Plot results
        plot_polar(polar, save_path="xfoil_polar.png")
        print(f"\nPlot saved to: xfoil_polar.png")
    else:
        print(f"XFOIL executable not found: {xfoil_exe}")
        print("Please update the path in this script.")
