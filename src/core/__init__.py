"""
Core 6-DOF flight dynamics components.

This module provides the fundamental building blocks for six-degree-of-freedom
aircraft flight simulation.
"""

from .aerodynamics import (
    AeroModel,
    ConstantCoeffModel,
    LinearAeroModel,
    FlyingWingAeroModel,
    AVLTableModel
)
from .dynamics import AircraftDynamics
from .state import State
from .quaternion import Quaternion
from .integrator import RK4Integrator, RK45Integrator
from .propulsion import ConstantThrustModel, PropellerModel, CombinedForceModel

__all__ = [
    'AeroModel',
    'ConstantCoeffModel',
    'LinearAeroModel',
    'FlyingWingAeroModel',
    'AVLTableModel',
    'AircraftDynamics',
    'State',
    'Quaternion',
    'RK4Integrator',
    'RK45Integrator',
    'ConstantThrustModel',
    'PropellerModel',
    'CombinedForceModel'
]
