"""
Control systems for flight simulation.

This module provides autopilot controllers and trim solvers.
"""

from .autopilot import PIDController, AltitudeHoldController, HeadingHoldController, AirspeedHoldController
from .trim import TrimSolver

__all__ = [
    'PIDController',
    'AltitudeHoldController',
    'HeadingHoldController',
    'AirspeedHoldController',
    'TrimSolver'
]
