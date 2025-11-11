"""
XFOIL Polar Database Loader

This module provides functionality to load and manage multiple XFOIL polar files,
interpolate between Reynolds numbers, and integrate with the aerodynamic models.

Author: Claude Code
Date: 2025-11-10
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import csv
from dataclasses import dataclass
from scipy.interpolate import interp1d, RegularGridInterpolator


@dataclass
class PolarDatabase:
    """
    Container for multiple airfoil polars at different Reynolds numbers.

    Attributes:
        airfoil_name: Name of the airfoil
        reynolds_numbers: List of Reynolds numbers with polar data
        polars: Dictionary mapping Reynolds numbers to polar data
        mach: Mach number for the polars (typically constant)
        ncrit: N-critical value used in XFOIL
    """
    airfoil_name: str
    reynolds_numbers: np.ndarray
    polars: Dict[float, Dict[str, np.ndarray]]
    mach: float
    ncrit: float

    def interpolate_coefficients(
        self,
        reynolds: float,
        alpha: float
    ) -> Tuple[float, float, float]:
        """
        Interpolate aerodynamic coefficients at a given Reynolds number and angle of attack.

        Args:
            reynolds: Reynolds number
            alpha: Angle of attack (degrees)

        Returns:
            Tuple of (CL, CD, CM) at the specified Re and alpha
        """
        # Clamp Reynolds number to available range
        re_min = self.reynolds_numbers[0]
        re_max = self.reynolds_numbers[-1]
        reynolds = np.clip(reynolds, re_min, re_max)

        # Get polars for bounding Reynolds numbers
        re_low_idx = np.searchsorted(self.reynolds_numbers, reynolds) - 1
        re_low_idx = max(0, re_low_idx)
        re_high_idx = min(re_low_idx + 1, len(self.reynolds_numbers) - 1)

        re_low = self.reynolds_numbers[re_low_idx]
        re_high = self.reynolds_numbers[re_high_idx]

        # Interpolate at each Reynolds number
        if re_low == re_high:
            # Exact match or at boundary
            polar = self.polars[re_low]
            CL = np.interp(alpha, polar['alpha'], polar['CL'])
            CD = np.interp(alpha, polar['alpha'], polar['CD'])
            CM = np.interp(alpha, polar['alpha'], polar['CM'])
        else:
            # Interpolate between two Reynolds numbers
            polar_low = self.polars[re_low]
            polar_high = self.polars[re_high]

            CL_low = np.interp(alpha, polar_low['alpha'], polar_low['CL'])
            CL_high = np.interp(alpha, polar_high['alpha'], polar_high['CL'])

            CD_low = np.interp(alpha, polar_low['alpha'], polar_low['CD'])
            CD_high = np.interp(alpha, polar_high['alpha'], polar_high['CD'])

            CM_low = np.interp(alpha, polar_low['alpha'], polar_low['CM'])
            CM_high = np.interp(alpha, polar_high['alpha'], polar_high['CM'])

            # Linear interpolation in Reynolds number
            t = (reynolds - re_low) / (re_high - re_low)
            CL = CL_low + t * (CL_high - CL_low)
            CD = CD_low + t * (CD_high - CD_low)
            CM = CM_low + t * (CM_high - CM_low)

        return CL, CD, CM

    def get_CLmax(self, reynolds: float) -> float:
        """
        Get maximum lift coefficient at a given Reynolds number.

        Args:
            reynolds: Reynolds number

        Returns:
            Maximum CL
        """
        # Find bounding Reynolds numbers
        re_low_idx = np.searchsorted(self.reynolds_numbers, reynolds) - 1
        re_low_idx = max(0, re_low_idx)
        re_high_idx = min(re_low_idx + 1, len(self.reynolds_numbers) - 1)

        re_low = self.reynolds_numbers[re_low_idx]
        re_high = self.reynolds_numbers[re_high_idx]

        # Get CLmax at each Reynolds number
        CLmax_low = np.max(self.polars[re_low]['CL'])
        CLmax_high = np.max(self.polars[re_high]['CL'])

        # Interpolate
        if re_low == re_high:
            return CLmax_low
        else:
            t = (reynolds - re_low) / (re_high - re_low)
            return CLmax_low + t * (CLmax_high - CLmax_low)

    def get_alpha_zero_lift(self, reynolds: float) -> float:
        """
        Get angle of attack for zero lift at a given Reynolds number.

        Args:
            reynolds: Reynolds number

        Returns:
            Alpha at CL=0 (degrees)
        """
        # Find bounding Reynolds numbers
        re_low_idx = np.searchsorted(self.reynolds_numbers, reynolds) - 1
        re_low_idx = max(0, re_low_idx)
        re_high_idx = min(re_low_idx + 1, len(self.reynolds_numbers) - 1)

        re_low = self.reynolds_numbers[re_low_idx]
        re_high = self.reynolds_numbers[re_high_idx]

        # Find alpha at CL=0 for each Reynolds number
        polar_low = self.polars[re_low]
        alpha0_low = np.interp(0.0, polar_low['CL'], polar_low['alpha'])

        polar_high = self.polars[re_high]
        alpha0_high = np.interp(0.0, polar_high['CL'], polar_high['alpha'])

        # Interpolate
        if re_low == re_high:
            return alpha0_low
        else:
            t = (reynolds - re_low) / (re_high - re_low)
            return alpha0_low + t * (alpha0_high - alpha0_low)


class XFOILDatabaseLoader:
    """
    Load and manage XFOIL polar databases from CSV files.
    """

    def __init__(self, database_dir: Optional[str] = None):
        """
        Initialize the database loader.

        Args:
            database_dir: Directory containing XFOIL polar CSV files.
                         If None, uses 'xfoil_data/' in current directory.
        """
        if database_dir is None:
            self.database_dir = Path.cwd() / 'xfoil_data'
        else:
            self.database_dir = Path(database_dir)

        self.databases: Dict[str, PolarDatabase] = {}

    def load_polar_csv(self, csv_path: str) -> Dict[str, np.ndarray]:
        """
        Load a single XFOIL polar from CSV file.

        CSV format should have header with metadata and data columns:
        # Airfoil: NACA 64-212
        # Reynolds: 3000000
        # Mach: 0.25
        # Ncrit: 9.0
        alpha,CL,CD,CDp,CM,Top_Xtr,Bot_Xtr
        -5.0,0.1234,0.0089,0.0045,-0.0123,0.9876,0.9543
        ...

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary with polar data and metadata
        """
        csv_path = Path(csv_path)

        metadata = {}
        data = {}

        with open(csv_path, 'r') as f:
            # Read metadata from header comments
            lines = f.readlines()

        # Parse metadata
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
            else:
                # Found first non-comment line (should be header)
                data_start_idx = i
                break

        # Parse header
        header_line = lines[data_start_idx].strip()
        columns = [col.strip() for col in header_line.split(',')]

        # Initialize data arrays
        for col in columns:
            data[col] = []

        # Parse data rows
        for line in lines[data_start_idx + 1:]:
            line = line.strip()
            if line and not line.startswith('#'):
                values = line.split(',')
                if len(values) == len(columns):
                    for i, col in enumerate(columns):
                        try:
                            data[col].append(float(values[i]))
                        except (ValueError, IndexError):
                            pass

        # Convert to numpy arrays
        for key in data:
            if len(data[key]) > 0:
                data[key] = np.array(data[key])

        # Add metadata
        data['metadata'] = metadata

        return data

    def load_airfoil_database(
        self,
        airfoil_name: str,
        csv_files: List[str]
    ) -> PolarDatabase:
        """
        Load multiple polars for an airfoil at different Reynolds numbers.

        Args:
            airfoil_name: Name of the airfoil
            csv_files: List of CSV file paths, each containing a polar at different Re

        Returns:
            PolarDatabase object
        """
        polars = {}
        reynolds_numbers = []
        mach = None
        ncrit = None

        for csv_file in csv_files:
            data = self.load_polar_csv(csv_file)

            # Extract metadata
            metadata = data.pop('metadata', {})
            reynolds = float(metadata.get('Reynolds', 0))

            if mach is None:
                mach = float(metadata.get('Mach', 0.0))
            if ncrit is None:
                ncrit = float(metadata.get('Ncrit', 9.0))

            reynolds_numbers.append(reynolds)
            polars[reynolds] = data

        # Sort by Reynolds number
        reynolds_numbers = np.array(sorted(reynolds_numbers))

        return PolarDatabase(
            airfoil_name=airfoil_name,
            reynolds_numbers=reynolds_numbers,
            polars=polars,
            mach=mach or 0.0,
            ncrit=ncrit or 9.0
        )

    def auto_load_airfoil(self, airfoil_name: str) -> Optional[PolarDatabase]:
        """
        Automatically load all polars for an airfoil from the database directory.

        Looks for files matching pattern: {airfoil_name}_Re*.csv

        Args:
            airfoil_name: Name of the airfoil (e.g., "NACA_64-212")

        Returns:
            PolarDatabase if files found, None otherwise
        """
        # Find all CSV files for this airfoil
        pattern = f"{airfoil_name}_Re*.csv"
        csv_files = list(self.database_dir.glob(pattern))

        if not csv_files:
            print(f"No polar files found for {airfoil_name} in {self.database_dir}")
            return None

        csv_files = [str(f) for f in sorted(csv_files)]

        print(f"Found {len(csv_files)} polar files for {airfoil_name}")

        database = self.load_airfoil_database(airfoil_name, csv_files)
        self.databases[airfoil_name] = database

        return database

    def get_database(self, airfoil_name: str) -> Optional[PolarDatabase]:
        """
        Get a loaded polar database by airfoil name.

        Args:
            airfoil_name: Name of the airfoil

        Returns:
            PolarDatabase if loaded, None otherwise
        """
        return self.databases.get(airfoil_name)


def create_example_database(output_dir: str = 'xfoil_data'):
    """
    Create example XFOIL polar CSV files for demonstration.

    This generates synthetic polar data for a NACA 64-212 airfoil
    at multiple Reynolds numbers for testing purposes.

    Args:
        output_dir: Directory to save CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    reynolds_list = [1e6, 3e6, 6e6, 9e6]
    airfoil_name = "NACA_64-212"

    for re in reynolds_list:
        # Generate synthetic polar data (example only)
        alpha = np.linspace(-5, 15, 41)

        # Lift coefficient (typical NACA 6-series)
        CLa = 0.11  # per degree
        CL0 = 0.2
        CL = CL0 + CLa * alpha
        CL = np.clip(CL, -0.5, 1.6)  # Stall limits

        # Drag coefficient (parabolic polar approximation)
        CD0 = 0.006 + 0.001 * (1 - re / 1e7)  # Reynolds effect
        K = 0.05
        CD = CD0 + K * CL**2

        # Pressure drag (portion of total drag)
        CDp = CD * 0.6

        # Moment coefficient (cambered airfoil)
        CM0 = -0.05
        CMa = -0.003  # per degree
        CM = CM0 + CMa * alpha

        # Transition locations (move aft with increasing Re)
        Top_Xtr = 0.90 + 0.05 * (re / 1e7)
        Bot_Xtr = 0.85 + 0.05 * (re / 1e7)
        Top_Xtr = np.full_like(alpha, min(Top_Xtr, 1.0))
        Bot_Xtr = np.full_like(alpha, min(Bot_Xtr, 1.0))

        # Save to CSV
        filename = output_path / f"{airfoil_name}_Re{int(re/1e6)}M.csv"

        with open(filename, 'w', newline='') as f:
            # Write metadata
            f.write(f"# Airfoil: {airfoil_name}\n")
            f.write(f"# Reynolds: {int(re)}\n")
            f.write(f"# Mach: 0.25\n")
            f.write(f"# Ncrit: 9.0\n")

            # Write data
            writer = csv.writer(f)
            writer.writerow(['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])

            for i in range(len(alpha)):
                writer.writerow([
                    f"{alpha[i]:.2f}",
                    f"{CL[i]:.6f}",
                    f"{CD[i]:.6f}",
                    f"{CDp[i]:.6f}",
                    f"{CM[i]:.6f}",
                    f"{Top_Xtr[i]:.4f}",
                    f"{Bot_Xtr[i]:.4f}"
                ])

        print(f"Created {filename}")

    print(f"\nExample database created in {output_path}")


# Example usage
if __name__ == '__main__':
    # Create example database
    print("Creating example XFOIL polar database...")
    create_example_database()

    # Load database
    print("\nLoading database...")
    loader = XFOILDatabaseLoader(database_dir='xfoil_data')
    db = loader.auto_load_airfoil('NACA_64-212')

    if db is not None:
        print(f"\nLoaded database for {db.airfoil_name}")
        print(f"Reynolds numbers: {db.reynolds_numbers / 1e6} million")
        print(f"Mach: {db.mach}")

        # Test interpolation
        print("\nTesting interpolation at Re=4.5e6, alpha=5°:")
        CL, CD, CM = db.interpolate_coefficients(4.5e6, 5.0)
        print(f"CL = {CL:.4f}")
        print(f"CD = {CD:.6f}")
        print(f"CM = {CM:.4f}")
        print(f"L/D = {CL/CD:.1f}")

        # Get CLmax
        print(f"\nCLmax at Re=4.5e6: {db.get_CLmax(4.5e6):.3f}")
        print(f"Alpha_0L at Re=4.5e6: {db.get_alpha_zero_lift(4.5e6):.2f}°")
