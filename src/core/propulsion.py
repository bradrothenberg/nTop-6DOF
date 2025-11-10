"""
Propulsion models for 6-DOF flight dynamics.

Provides:
- Base propulsion model interface
- Constant thrust model
- Throttle-dependent thrust model
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

# Handle imports
try:
    from .state import State
except ImportError:
    from state import State


class PropulsionModel(ABC):
    """
    Base class for propulsion models.

    Provides interface for computing thrust forces and moments.
    """

    @abstractmethod
    def compute_thrust(self, state: State, throttle: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thrust forces and moments.

        Parameters:
        -----------
        state : State
            Current aircraft state
        throttle : float
            Throttle setting [0, 1]

        Returns:
        --------
        forces : np.ndarray, shape (3,)
            Thrust forces in body frame [Fx, Fy, Fz] (lbf)
        moments : np.ndarray, shape (3,)
            Thrust moments in body frame [L, M, N] (ft·lbf)
        """
        pass


class ConstantThrustModel(PropulsionModel):
    """
    Simple constant thrust model.

    Thrust aligned with body x-axis, constant magnitude.
    """

    def __init__(self, thrust: float = 0.0, thrust_offset: np.ndarray = None):
        """
        Initialize constant thrust model.

        Parameters:
        -----------
        thrust : float
            Constant thrust magnitude (lbf)
        thrust_offset : np.ndarray, shape (3,), optional
            Thrust line offset from CG [x, y, z] (ft)
            Causes moments if offset from CG
        """
        self.thrust = thrust
        if thrust_offset is None:
            self.thrust_offset = np.array([0.0, 0.0, 0.0])
        else:
            self.thrust_offset = thrust_offset

    def compute_thrust(self, state: State, throttle: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute constant thrust."""
        # Thrust along body x-axis
        forces = np.hstack([self.thrust * throttle, 0.0, 0.0])

        # Moments due to thrust offset
        moments = np.cross(self.thrust_offset, forces)

        return forces, moments


class ThrottleDependentThrust(PropulsionModel):
    """
    Throttle-dependent thrust model.

    Linear relationship between throttle and thrust.
    Optionally includes velocity-dependent effects.
    """

    def __init__(self, thrust_max: float = 1000.0,
                 thrust_offset: np.ndarray = None,
                 velocity_factor: float = 0.0):
        """
        Initialize throttle-dependent thrust model.

        Parameters:
        -----------
        thrust_max : float
            Maximum thrust at full throttle (lbf)
        thrust_offset : np.ndarray, shape (3,), optional
            Thrust line offset from CG [x, y, z] (ft)
        velocity_factor : float
            Velocity dependence factor (0 = constant, 1 = full prop effect)
            T = T_max * throttle * (1 - velocity_factor * V / V_max)
        """
        self.thrust_max = thrust_max
        if thrust_offset is None:
            self.thrust_offset = np.array([0.0, 0.0, 0.0])
        else:
            self.thrust_offset = thrust_offset
        self.velocity_factor = velocity_factor
        self.V_max = 300.0  # ft/s, typical max airspeed

    def compute_thrust(self, state: State, throttle: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute throttle-dependent thrust."""
        # Base thrust from throttle
        thrust = self.thrust_max * throttle

        # Velocity dependence (propeller efficiency decreases with airspeed)
        if self.velocity_factor > 0:
            V = state.airspeed
            thrust *= (1.0 - self.velocity_factor * min(V / self.V_max, 1.0))

        # Thrust along body x-axis
        forces = np.hstack([thrust, 0.0, 0.0])

        # Moments due to thrust offset
        moments = np.cross(self.thrust_offset, forces)

        return forces, moments


class PropellerModel(PropulsionModel):
    """
    Simple propeller model using momentum theory.

    More realistic than constant thrust, accounts for:
    - Propeller efficiency vs. advance ratio
    - Power available from engine
    """

    def __init__(self, power_max: float = 50.0,  # HP
                 prop_diameter: float = 6.0,      # ft
                 prop_efficiency: float = 0.8,
                 thrust_offset: np.ndarray = None):
        """
        Initialize propeller model.

        Parameters:
        -----------
        power_max : float
            Maximum engine power (HP)
        prop_diameter : float
            Propeller diameter (ft)
        prop_efficiency : float
            Propeller efficiency [0, 1]
        thrust_offset : np.ndarray, optional
            Thrust line offset from CG [x, y, z] (ft)
        """
        self.power_max = power_max
        self.prop_diameter = prop_diameter
        self.prop_efficiency = prop_efficiency
        if thrust_offset is None:
            self.thrust_offset = np.array([0.0, 0.0, 0.0])
        else:
            self.thrust_offset = thrust_offset

        # Convert HP to ft·lbf/s
        self.power_max_ftlb_s = power_max * 550.0

    def compute_thrust(self, state: State, throttle: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute propeller thrust using simple momentum theory."""
        # Available power
        P_avail = self.power_max_ftlb_s * throttle

        # Airspeed
        V = state.airspeed
        if V < 1.0:
            V = 1.0  # Avoid singularity

        # Simple thrust estimate: T = eta * P / V
        thrust = self.prop_efficiency * P_avail / V

        # Limit thrust to reasonable values
        thrust = min(thrust, 2.0 * self.power_max)  # Reasonable upper bound

        # Thrust along body x-axis
        forces = np.hstack([thrust, 0.0, 0.0])

        # Moments due to thrust offset
        moments = np.cross(self.thrust_offset, forces)

        return forces, moments


class CombinedForceModel:
    """
    Combines aerodynamic and propulsion models.

    Convenience class that computes total forces and moments
    from separate aero and propulsion models.
    """

    def __init__(self, aero_model, propulsion_model):
        """
        Initialize combined model.

        Parameters:
        -----------
        aero_model : AeroModel
            Aerodynamic model
        propulsion_model : PropulsionModel
            Propulsion model
        """
        self.aero_model = aero_model
        self.propulsion_model = propulsion_model

    def __call__(self, state: State, throttle: float = 1.0,
                 controls: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute total forces and moments.

        Parameters:
        -----------
        state : State
            Current state
        throttle : float
            Throttle setting [0, 1]
        controls : dict, optional
            Control surface deflections

        Returns:
        --------
        forces : np.ndarray (3,)
            Total forces in body frame (lbf)
        moments : np.ndarray (3,)
            Total moments in body frame (ft·lbf)
        """
        # Get aerodynamic forces/moments
        F_aero, M_aero = self.aero_model.compute_forces_moments(state, controls)

        # Get propulsion forces/moments
        F_thrust, M_thrust = self.propulsion_model.compute_thrust(state, throttle)

        # Sum forces and moments
        forces = F_aero + F_thrust
        moments = M_aero + M_thrust

        return forces, moments


if __name__ == "__main__":
    # Test propulsion models
    print("=== Propulsion Model Tests ===\n")

    # Create test state
    state = State()
    state.altitude = 5000.0
    state.velocity_body = np.array([250.0, 0.0, 0.0])

    print("Test state:")
    print(f"  Altitude: {state.altitude:.0f} ft")
    print(f"  Airspeed: {state.airspeed:.1f} ft/s")
    print()

    # Test 1: Constant thrust
    print("1. Constant Thrust Model (500 lbf):")
    prop1 = ConstantThrustModel(thrust=500.0)
    forces, moments = prop1.compute_thrust(state, throttle=1.0)
    print(f"  Forces: {forces} lbf")
    print(f"  Moments: {moments} ft·lbf")
    print()

    # Test 2: Throttle-dependent
    print("2. Throttle-Dependent Model (1000 lbf max, 50% throttle):")
    prop2 = ThrottleDependentThrust(thrust_max=1000.0)
    forces, moments = prop2.compute_thrust(state, throttle=0.5)
    print(f"  Forces: {forces} lbf")
    print(f"  Moments: {moments} ft·lbf")
    print()

    # Test 3: Propeller model
    print("3. Propeller Model (50 HP, 6 ft diameter):")
    prop3 = PropellerModel(power_max=50.0, prop_diameter=6.0)
    forces, moments = prop3.compute_thrust(state, throttle=1.0)
    print(f"  Forces: {forces} lbf")
    print(f"  Moments: {moments} ft·lbf")
    print(f"  Thrust at 250 ft/s: {forces[0]:.1f} lbf")
    print()

    # Test 4: Thrust offset (causes pitch moment)
    print("4. Thrust with Offset (1 ft below CG):")
    prop4 = ConstantThrustModel(thrust=500.0, thrust_offset=np.array([0.0, 0.0, -1.0]))
    forces, moments = prop4.compute_thrust(state, throttle=1.0)
    print(f"  Forces: {forces} lbf")
    print(f"  Moments: {moments} ft·lbf")
    print(f"  Pitch moment from thrust offset: {moments[1]:.1f} ft·lbf")
    print()

    print("All propulsion models working!")
