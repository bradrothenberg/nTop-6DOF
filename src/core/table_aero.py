"""
Table-based aerodynamic model using AVL-generated data.

Interpolates CL, CD, Cm from a 2D table of (alpha, elevator) values.
Adds lateral-directional derivatives for roll/yaw control.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


class TableAeroModel:
    """
    Aerodynamic model using 2D interpolation from AVL table data.

    Table provides: CL(alpha, elevator), CD(alpha, elevator), Cm(alpha, elevator)
    Linearized derivatives added for: lateral-directional dynamics
    """

    def __init__(self, table_file, S_ref, c_ref, b_ref, rho=0.002377):
        """
        Initialize table-based aero model.

        Parameters:
        -----------
        table_file : str
            Path to CSV file with columns: alpha, elevator, CL, CD, Cm
        S_ref : float
            Reference area (ft^2)
        c_ref : float
            Reference chord (ft)
        b_ref : float
            Reference span (ft)
        rho : float
            Air density (slug/ft^3)
        """
        self.S_ref = S_ref
        self.c_ref = c_ref
        self.b_ref = b_ref
        self.rho = rho

        # Load table data
        df = pd.read_csv(table_file)

        # Get unique alpha and elevator values (sorted)
        self.alphas = np.sort(df['alpha'].unique())
        self.elevators = np.sort(df['elevator'].unique())

        # Create 2D grids for interpolation
        # Need to pivot data into 2D arrays
        self.CL_grid = self._create_grid(df, 'CL')
        self.CD_grid = self._create_grid(df, 'CD')
        self.Cm_grid = self._create_grid(df, 'Cm')

        # Create interpolators (alpha and elevator in degrees)
        self.CL_interp = RegularGridInterpolator(
            (self.alphas, self.elevators), self.CL_grid,
            bounds_error=False, fill_value=None  # Extrapolate outside bounds
        )
        self.CD_interp = RegularGridInterpolator(
            (self.alphas, self.elevators), self.CD_grid,
            bounds_error=False, fill_value=None
        )
        self.Cm_interp = RegularGridInterpolator(
            (self.alphas, self.elevators), self.Cm_grid,
            bounds_error=False, fill_value=None
        )

        # Lateral-directional derivatives (from AVL linearized data)
        # These are less sensitive to alpha/elevator, so we use constant values
        self.CY_beta = -0.146    # Sideforce due to sideslip
        self.CY_p = -0.0312      # Sideforce due to roll rate
        self.CY_r = 0.126        # Sideforce due to yaw rate

        self.Cl_beta = -0.0296   # Roll moment due to sideslip (dihedral effect)
        self.Cl_p = -0.481       # Roll damping
        self.Cl_r = 0.0297       # Roll due to yaw rate
        self.Cl_da = 0.229       # Aileron effectiveness (per radian)
        self.Cl_dr = 0.00464     # Rudder roll coupling

        self.Cn_beta = 0.0533    # Directional stability (weathercock)
        self.Cn_p = -0.0159      # Yaw due to roll rate
        self.Cn_r = -0.126       # Yaw damping
        self.Cn_da = -0.00774    # Adverse yaw from aileron
        self.Cn_dr = -0.0414     # Rudder effectiveness (per radian)

        # Pitch rate derivative (from AVL)
        self.CL_q = 3.182        # Lift due to pitch rate
        self.Cm_q = -6.197       # Pitch damping (excellent!)

        print(f"Table-based aero model initialized:")
        print(f"  Alpha range: {self.alphas[0]:.1f} to {self.alphas[-1]:.1f} deg")
        print(f"  Elevator range: {self.elevators[0]:.1f} to {self.elevators[-1]:.1f} deg")
        print(f"  Table size: {len(self.alphas)} x {len(self.elevators)} = {len(self.alphas)*len(self.elevators)} points")

    def _create_grid(self, df, value_col):
        """Create 2D grid for interpolation."""
        # Pivot table to get 2D array
        pivot = df.pivot(index='alpha', columns='elevator', values=value_col)
        # Return as numpy array (rows=alpha, cols=elevator)
        return pivot.values

    def get_coefficients(self, alpha_deg, elevator_deg):
        """
        Get aerodynamic coefficients at given alpha and elevator.

        Parameters:
        -----------
        alpha_deg : float
            Angle of attack (degrees)
        elevator_deg : float
            Elevator deflection (degrees)

        Returns:
        --------
        CL, CD, Cm : float
            Interpolated coefficients
        """
        point = np.array([alpha_deg, elevator_deg])

        CL = float(self.CL_interp(point))
        CD = float(self.CD_interp(point))
        Cm = float(self.Cm_interp(point))

        return CL, CD, Cm

    def compute_forces_moments(self, state, controls):
        """
        Compute aerodynamic forces and moments.

        Parameters:
        -----------
        state : State
            Aircraft state
        controls : dict
            Control surface deflections (radians):
            - 'elevator': Elevator deflection
            - 'aileron': Aileron deflection
            - 'rudder': Rudder deflection

        Returns:
        --------
        forces : np.array (3,)
            Aerodynamic forces in body frame [Fx, Fy, Fz] (lbf)
        moments : np.array (3,)
            Aerodynamic moments in body frame [L, M, N] (ft-lbf)
        """
        # Get controls
        delta_e = controls.get('elevator', 0.0)
        delta_a = controls.get('aileron', 0.0)
        delta_r = controls.get('rudder', 0.0)

        # Airspeed and dynamic pressure
        V_body = state.velocity_body
        V = np.linalg.norm(V_body)

        if V < 1.0:
            return np.zeros(3), np.zeros(3)

        q_bar = 0.5 * self.rho * V**2

        # Angle of attack and sideslip (radians)
        alpha = np.arctan2(V_body[2], V_body[0])
        beta = np.arcsin(np.clip(V_body[1] / V, -1, 1))

        # Non-dimensional angular rates
        p, q, r = state.angular_rates
        p_hat = p * self.b_ref / (2 * V)
        q_hat = q * self.c_ref / (2 * V)
        r_hat = r * self.b_ref / (2 * V)

        # Get base coefficients from table (alpha and elevator in degrees)
        CL_base, CD_base, Cm_base = self.get_coefficients(
            np.degrees(alpha),
            np.degrees(delta_e)
        )

        # Add pitch rate effects
        CL = CL_base + self.CL_q * q_hat
        CD = CD_base  # Drag doesn't change much with pitch rate
        Cm = Cm_base + self.Cm_q * q_hat

        # Lateral-directional coefficients
        CY = self.CY_beta * beta + self.CY_p * p_hat + self.CY_r * r_hat

        Cl = (self.Cl_beta * beta + self.Cl_p * p_hat + self.Cl_r * r_hat +
              self.Cl_da * delta_a + self.Cl_dr * delta_r)

        Cn = (self.Cn_beta * beta + self.Cn_p * p_hat + self.Cn_r * r_hat +
              self.Cn_da * delta_a + self.Cn_dr * delta_r)

        # Convert to stability axes forces
        L_stab = -CD  # Drag (negative X in stability axes)
        Y_stab = CY   # Side force
        Z_stab = -CL  # Lift (negative Z in stability axes)

        # Rotate from stability to body axes
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)

        Fx = L_stab * cos_a - Z_stab * sin_a
        Fy = Y_stab
        Fz = L_stab * sin_a + Z_stab * cos_a

        # Scale by dynamic pressure and reference area
        forces = q_bar * self.S_ref * np.array([Fx, Fy, Fz])

        # Moments (already in body axes)
        moments = q_bar * self.S_ref * np.array([
            Cl * self.b_ref,  # Rolling moment
            Cm * self.c_ref,  # Pitching moment
            Cn * self.b_ref   # Yawing moment
        ])

        return forces, moments


if __name__ == "__main__":
    """Test table-based aero model."""

    from pathlib import Path
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.core.state import State

    # Create model
    table_file = Path("aero_tables/conventional_aero_table.csv")
    if not table_file.exists():
        print(f"ERROR: Table file not found: {table_file}")
        exit(1)

    model = TableAeroModel(
        table_file=str(table_file),
        S_ref=199.94,  # ft^2
        c_ref=26.689,  # ft
        b_ref=19.890,  # ft
        rho=0.002377   # slug/ft^3
    )

    # Test state
    state = State()
    state.position = np.array([0, 0, -5000])
    state.velocity_body = np.array([600, 0, 20])  # ~2 deg alpha
    state.set_euler_angles(0, np.radians(2), 0)
    state.angular_rates = np.array([0, 0, 0])

    # Test controls
    controls = {
        'elevator': np.radians(-5),
        'aileron': 0.0,
        'rudder': 0.0
    }

    print("\nTest case: 600 ft/s, alpha=2°, elevator=-5°")
    forces, moments = model.compute_forces_moments(state, controls)

    print(f"  Forces: [{forces[0]:.1f}, {forces[1]:.1f}, {forces[2]:.1f}] lbf")
    print(f"  Moments: [{moments[0]:.1f}, {moments[1]:.1f}, {moments[2]:.1f}] ft-lbf")

    # Compare with table lookup
    alpha_deg = np.degrees(np.arctan2(state.velocity_body[2], state.velocity_body[0]))
    CL, CD, Cm = model.get_coefficients(alpha_deg, np.degrees(controls['elevator']))

    print(f"\n  Table lookup: CL={CL:.4f}, CD={CD:.5f}, Cm={Cm:.4f}")
    print("\nTable-based aero model working!")
