"""
Aerodynamic models for 6-DOF flight dynamics.

Provides:
- Base aerodynamic model interface
- Table-based lookup for AVL-generated data
- Simple coefficient models for testing
"""

import numpy as np
from typing import Tuple, Dict
from abc import ABC, abstractmethod

# Handle imports
try:
    from .state import State
except ImportError:
    from state import State


class AeroModel(ABC):
    """
    Base class for aerodynamic models.

    Provides interface for computing forces and moments
    given aircraft state.
    """

    @abstractmethod
    def compute_forces_moments(self, state: State,
                                controls: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute aerodynamic forces and moments.

        Parameters:
        -----------
        state : State
            Current aircraft state
        controls : dict
            Control surface deflections (radians)
            e.g., {'elevator': 0.1, 'aileron': -0.05, 'rudder': 0.02}

        Returns:
        --------
        forces : np.ndarray, shape (3,)
            Forces in body frame [Fx, Fy, Fz] (lbf)
        moments : np.ndarray, shape (3,)
            Moments in body frame [L, M, N] (ft·lbf)
        """
        pass


class ConstantCoeffModel(AeroModel):
    """
    Simple aerodynamic model with constant coefficients.

    Good for basic testing and validation.
    """

    def __init__(self, CL: float = 0.5, CD: float = 0.05,
                 S_ref: float = 200.0, c_ref: float = 10.0, b_ref: float = 20.0,
                 rho: float = 0.002377):
        """
        Initialize constant coefficient model.

        Parameters:
        -----------
        CL : float
            Lift coefficient
        CD : float
            Drag coefficient
        S_ref : float
            Reference area (ft²)
        c_ref : float
            Reference chord (ft)
        b_ref : float
            Reference span (ft)
        rho : float
            Air density (slug/ft³)
        """
        self.CL = CL
        self.CD = CD
        self.S_ref = S_ref
        self.c_ref = c_ref
        self.b_ref = b_ref
        self.rho = rho

        # Simple stability derivatives
        self.Cm_0 = 0.0      # Pitch moment at zero alpha
        self.Cm_alpha = -0.5  # Pitch stiffness
        self.Cl_beta = -0.1   # Roll due to sideslip
        self.Cn_beta = 0.1    # Yaw stability

    def compute_forces_moments(self, state: State,
                                controls: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces and moments with constant coefficients."""
        # Dynamic pressure
        V = state.airspeed
        if V < 1.0:
            V = 1.0  # Avoid singularity
        q_bar = 0.5 * self.rho * V**2

        # Angle of attack and sideslip
        alpha = state.alpha
        beta = state.beta

        # Lift and drag
        L = q_bar * self.S_ref * self.CL
        D = q_bar * self.S_ref * self.CD

        # Transform to body frame
        Fx = -D * np.cos(alpha) + L * np.sin(alpha)
        Fz = -D * np.sin(alpha) - L * np.cos(alpha)
        Fy = 0.0  # Side force (simplified)

        forces = np.array([Fx, Fy, Fz])

        # Moments with simple stability derivatives
        Cm = self.Cm_0 + self.Cm_alpha * alpha
        Cl = self.Cl_beta * beta
        Cn = self.Cn_beta * beta

        M_pitch = q_bar * self.S_ref * self.c_ref * Cm
        L_roll = q_bar * self.S_ref * self.b_ref * Cl
        N_yaw = q_bar * self.S_ref * self.b_ref * Cn

        moments = np.array([L_roll, M_pitch, N_yaw])

        return forces, moments


class LinearAeroModel(AeroModel):
    """
    Linear aerodynamic model using stability derivatives.

    Suitable for small perturbations from trim conditions.
    Uses standard stability derivative notation.
    """

    def __init__(self, S_ref: float, c_ref: float, b_ref: float, rho: float = 0.002377):
        """
        Initialize linear aero model.

        Parameters:
        -----------
        S_ref : float
            Reference area (ft²)
        c_ref : float
            Reference chord (ft)
        b_ref : float
            Reference span (ft)
        rho : float
            Air density (slug/ft³)
        """
        self.S_ref = S_ref
        self.c_ref = c_ref
        self.b_ref = b_ref
        self.rho = rho

        # Force coefficients (to be set by user or from AVL)
        self.CL_0 = 0.0
        self.CL_alpha = 5.0
        self.CL_q = 0.0
        self.CL_de = 0.4  # elevator

        self.CD_0 = 0.02
        self.CD_alpha = 0.1
        self.CD_alpha2 = 0.5  # Induced drag

        self.CY_beta = -0.2
        self.CY_dr = 0.1  # rudder

        # Moment coefficients
        self.Cl_beta = -0.1
        self.Cl_p = -0.4
        self.Cl_r = 0.1
        self.Cl_da = 0.2  # aileron
        self.Cl_dr = 0.01

        self.Cm_0 = 0.0
        self.Cm_alpha = -0.5
        self.Cm_q = -10.0
        self.Cm_de = -1.0

        self.Cn_beta = 0.1
        self.Cn_p = -0.05
        self.Cn_r = -0.2
        self.Cn_da = -0.05
        self.Cn_dr = -0.1

    def compute_forces_moments(self, state: State,
                                controls: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces and moments using linear stability derivatives."""
        if controls is None:
            controls = {}

        # Get control deflections
        delta_e = controls.get('elevator', 0.0)
        delta_a = controls.get('aileron', 0.0)
        delta_r = controls.get('rudder', 0.0)

        # State variables
        V = state.airspeed
        if V < 1.0:
            V = 1.0
        q_bar = 0.5 * self.rho * V**2

        alpha = state.alpha
        beta = state.beta

        # Non-dimensional angular rates
        p, q, r = state.angular_rates
        p_hat = p * self.b_ref / (2 * V)
        q_hat = q * self.c_ref / (2 * V)
        r_hat = r * self.b_ref / (2 * V)

        # Force coefficients
        CL = self.CL_0 + self.CL_alpha * alpha + self.CL_q * q_hat + self.CL_de * delta_e
        CD = self.CD_0 + self.CD_alpha * alpha + self.CD_alpha2 * alpha**2
        CY = self.CY_beta * beta + self.CY_dr * delta_r

        # Moment coefficients
        Cl = self.Cl_beta * beta + self.Cl_p * p_hat + self.Cl_r * r_hat + \
             self.Cl_da * delta_a + self.Cl_dr * delta_r
        Cm = self.Cm_0 + self.Cm_alpha * alpha + self.Cm_q * q_hat + self.Cm_de * delta_e
        Cn = self.Cn_beta * beta + self.Cn_p * p_hat + self.Cn_r * r_hat + \
             self.Cn_da * delta_a + self.Cn_dr * delta_r

        # Forces in wind frame
        L_aero = q_bar * self.S_ref * CL
        D = q_bar * self.S_ref * CD
        Y = q_bar * self.S_ref * CY

        # Transform to body frame
        Fx = -D * np.cos(alpha) + L_aero * np.sin(alpha)
        Fy = Y
        Fz = -D * np.sin(alpha) - L_aero * np.cos(alpha)

        forces = np.array([Fx, Fy, Fz])

        # Moments
        L_moment = q_bar * self.S_ref * self.b_ref * Cl
        M_moment = q_bar * self.S_ref * self.c_ref * Cm
        N_moment = q_bar * self.S_ref * self.b_ref * Cn

        moments = np.array([L_moment, M_moment, N_moment])

        return forces, moments

    def set_derivatives_from_avl(self, avl_data: Dict):
        """
        Set stability derivatives from AVL output.

        Parameters:
        -----------
        avl_data : dict
            Dictionary containing AVL stability derivatives
        """
        # This will be populated when we have AVL interface working
        # For now, just a placeholder
        pass


class AVLTableModel(AeroModel):
    """
    Table-based aerodynamic model using AVL data.

    Interpolates forces and moments from pre-computed
    alpha/beta/control sweeps.
    """

    def __init__(self, S_ref: float, c_ref: float, b_ref: float,
                 alpha_table: np.ndarray, data_table: Dict):
        """
        Initialize table-based model.

        Parameters:
        -----------
        S_ref, c_ref, b_ref : float
            Reference dimensions
        alpha_table : np.ndarray
            Array of alpha values (radians)
        data_table : dict
            Dictionary with keys like 'CL', 'CD', 'Cm', etc.
            Each value is an array matching alpha_table
        """
        self.S_ref = S_ref
        self.c_ref = c_ref
        self.b_ref = b_ref
        self.alpha_table = alpha_table
        self.data_table = data_table

    def compute_forces_moments(self, state: State,
                                controls: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces/moments using table interpolation."""
        # Get alpha and interpolate coefficients
        alpha = state.alpha

        CL = np.interp(alpha, self.alpha_table, self.data_table['CL'])
        CD = np.interp(alpha, self.alpha_table, self.data_table['CD'])
        Cm = np.interp(alpha, self.alpha_table, self.data_table['Cm'])

        # Dynamic pressure
        V = state.airspeed
        if V < 1.0:
            V = 1.0
        # Use density at current altitude
        rho = self._compute_density(state.altitude)
        q_bar = 0.5 * rho * V**2

        # Forces
        L_aero = q_bar * self.S_ref * CL
        D = q_bar * self.S_ref * CD

        Fx = -D * np.cos(alpha) + L_aero * np.sin(alpha)
        Fz = -D * np.sin(alpha) - L_aero * np.cos(alpha)
        Fy = 0.0

        forces = np.array([Fx, Fy, Fz])

        # Moments
        M_pitch = q_bar * self.S_ref * self.c_ref * Cm
        moments = np.array([0.0, M_pitch, 0.0])

        return forces, moments

    def _compute_density(self, altitude: float) -> float:
        """Compute air density from US Standard Atmosphere."""
        # Simple approximation - could use full atmosphere model
        rho_sl = 0.002377  # slug/ft³ at sea level
        h = altitude
        if h < 36000:
            # Troposphere
            T = 518.67 - 0.00356616 * h  # Rankine
            rho = rho_sl * (T / 518.67)**4.256
        else:
            # Stratosphere (simplified)
            rho = 0.000706 * np.exp(-(h - 36000) / 20806)
        return rho


if __name__ == "__main__":
    # Test aerodynamic models
    print("=== Aerodynamic Model Tests ===\n")

    # Create test state
    state = State()
    state.altitude = 5000.0
    state.velocity_body = np.array([250.0, 0.0, 0.0])
    state.set_euler_angles(0, np.radians(5), 0)

    print("Test state:")
    print(f"  Altitude: {state.altitude:.0f} ft")
    print(f"  Airspeed: {state.airspeed:.1f} ft/s")
    print(f"  Alpha: {np.degrees(state.alpha):.2f}°")
    print(f"  Beta: {np.degrees(state.beta):.2f}°")
    print()

    # Test 1: Constant coefficient model
    print("1. Constant Coefficient Model:")
    model1 = ConstantCoeffModel(CL=0.5, CD=0.05, S_ref=199.94,
                                 c_ref=26.689, b_ref=19.890)
    forces, moments = model1.compute_forces_moments(state)
    print(f"  Forces: {forces} lbf")
    print(f"  Moments: {moments} ft·lbf")
    print()

    # Test 2: Linear model
    print("2. Linear Stability Derivative Model:")
    model2 = LinearAeroModel(S_ref=199.94, c_ref=26.689, b_ref=19.890)
    forces, moments = model2.compute_forces_moments(state,
                                                     controls={'elevator': np.radians(5)})
    print(f"  Forces: {forces} lbf")
    print(f"  Moments: {moments} ft·lbf")
    print()

    # Test 3: Table-based model
    print("3. Table-Based Model (from synthetic data):")
    alpha_table = np.radians(np.linspace(-5, 15, 21))
    CL_table = 0.2 + 5.0 * alpha_table
    CD_table = 0.02 + 0.05 * alpha_table**2
    Cm_table = 0.05 - 0.5 * alpha_table

    data_table = {
        'CL': CL_table,
        'CD': CD_table,
        'Cm': Cm_table
    }

    model3 = AVLTableModel(S_ref=199.94, c_ref=26.689, b_ref=19.890,
                           alpha_table=alpha_table, data_table=data_table)
    forces, moments = model3.compute_forces_moments(state)
    print(f"  Forces: {forces} lbf")
    print(f"  Moments: {moments} ft·lbf")
    print()

    print("All aerodynamic models working!")
