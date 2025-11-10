"""
Trim Calculation

Finds equilibrium flight conditions by solving for control inputs
that result in zero state derivatives (steady flight).
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Callable, Dict, Optional, Tuple
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.state import State


class TrimSolver:
    """
    Trim solver for finding equilibrium flight conditions.

    Finds control settings and state parameters that result in steady flight
    (zero accelerations and angular accelerations).

    Parameters
    ----------
    dynamics_function : callable
        Function that computes state derivatives: f(state, controls) -> state_dot
    """

    def __init__(self, dynamics_function: Callable):
        """Initialize trim solver."""
        self.dynamics_function = dynamics_function

    def trim_straight_level(self,
                             altitude: float,
                             airspeed: float,
                             initial_guess: Optional[Dict] = None,
                             verbose: bool = False) -> Tuple[State, Dict, Dict]:
        """
        Find trim for straight and level flight.

        Parameters
        ----------
        altitude : float
            Target altitude (ft)
        airspeed : float
            Target airspeed (ft/s)
        initial_guess : dict, optional
            Initial guess for unknowns: {'alpha', 'theta', 'elevator', 'throttle'}
        verbose : bool, optional
            Print optimization progress

        Returns
        -------
        state_trim : State
            Trimmed state
        controls_trim : dict
            Trimmed control inputs
        info : dict
            Optimization info (success, residual, iterations)
        """
        # Default initial guess
        if initial_guess is None:
            initial_guess = {
                'alpha': np.radians(2),      # 2° angle of attack
                'theta': np.radians(2),      # 2° pitch angle
                'elevator': 0.0,             # 0° elevator
                'throttle': 0.5              # 50% throttle
            }

        # Pack unknowns into vector
        x0 = np.array([
            initial_guess['alpha'],
            initial_guess['theta'],
            initial_guess['elevator'],
            initial_guess['throttle']
        ])

        # Define residual function
        def residuals(x):
            alpha, theta, elevator, throttle = x

            # Build state
            state = State()
            state.position = np.array([0.0, 0.0, -altitude])

            # Velocity in body frame from airspeed and alpha
            u = airspeed * np.cos(alpha)
            w = airspeed * np.sin(alpha)
            state.velocity_body = np.array([u, 0.0, w])

            # Attitude (pitch = theta, roll = 0, yaw = 0)
            state.set_euler_angles(0.0, theta, 0.0)

            # Angular rates = 0 for steady flight
            state.angular_rates = np.array([0.0, 0.0, 0.0])

            # Controls
            controls = {
                'elevator': elevator,
                'aileron': 0.0,
                'rudder': 0.0,
                'throttle': throttle
            }

            # Compute state derivative
            state_dot = self.dynamics_function(state, controls)

            # Residuals: want u_dot, w_dot, q_dot, theta_dot ≈ 0
            # state_dot is [x_dot, y_dot, z_dot, u_dot, v_dot, w_dot,
            #                q0_dot, q1_dot, q2_dot, q3_dot, p_dot, q_dot, r_dot]
            residual = np.array([
                state_dot[3],   # u_dot (axial acceleration)
                state_dot[5],   # w_dot (vertical acceleration)
                state_dot[11],  # q_dot (pitch acceleration)
                state_dot[2]    # z_dot (climb rate, should be ~0 for level flight)
            ])

            return residual

        # Bounds
        bounds = [
            (-np.radians(15), np.radians(15)),  # alpha
            (-np.radians(15), np.radians(15)),  # theta
            (-np.radians(25), np.radians(25)),  # elevator
            (0.0, 1.0)                           # throttle
        ]

        # Solve using least squares
        result = least_squares(residuals, x0, bounds=np.array(bounds).T, verbose=2 if verbose else 0)

        # Extract solution
        alpha_trim, theta_trim, elevator_trim, throttle_trim = result.x

        # Build trimmed state
        state_trim = State()
        state_trim.position = np.array([0.0, 0.0, -altitude])
        u_trim = airspeed * np.cos(alpha_trim)
        w_trim = airspeed * np.sin(alpha_trim)
        state_trim.velocity_body = np.array([u_trim, 0.0, w_trim])
        state_trim.set_euler_angles(0.0, theta_trim, 0.0)
        state_trim.angular_rates = np.array([0.0, 0.0, 0.0])

        # Build trimmed controls
        controls_trim = {
            'elevator': elevator_trim,
            'aileron': 0.0,
            'rudder': 0.0,
            'throttle': throttle_trim
        }

        # Optimization info
        info = {
            'success': result.success,
            'residual_norm': np.linalg.norm(result.fun),
            'iterations': result.nfev,
            'message': result.message,
            'alpha_deg': np.degrees(alpha_trim),
            'theta_deg': np.degrees(theta_trim),
            'elevator_deg': np.degrees(elevator_trim),
            'throttle_pct': throttle_trim * 100
        }

        return state_trim, controls_trim, info

    def trim_coordinated_turn(self,
                               altitude: float,
                               airspeed: float,
                               turn_rate: float,
                               initial_guess: Optional[Dict] = None,
                               verbose: bool = False) -> Tuple[State, Dict, Dict]:
        """
        Find trim for coordinated turn.

        Parameters
        ----------
        altitude : float
            Target altitude (ft)
        airspeed : float
            Target airspeed (ft/s)
        turn_rate : float
            Turn rate (rad/s), positive = right turn
        initial_guess : dict, optional
            Initial guess for unknowns
        verbose : bool, optional
            Print optimization progress

        Returns
        -------
        state_trim : State
            Trimmed state
        controls_trim : dict
            Trimmed control inputs
        info : dict
            Optimization info
        """
        # Default initial guess
        if initial_guess is None:
            # Estimate bank angle from turn rate
            # For coordinated turn: tan(phi) ≈ V * r / g
            g = 32.174  # ft/s²
            phi_guess = np.arctan(airspeed * turn_rate / g)

            initial_guess = {
                'alpha': np.radians(2),
                'theta': np.radians(2),
                'phi': phi_guess,
                'elevator': 0.0,
                'aileron': 0.0,
                'rudder': 0.0,
                'throttle': 0.5
            }

        # Pack unknowns
        x0 = np.array([
            initial_guess['alpha'],
            initial_guess['theta'],
            initial_guess['phi'],
            initial_guess['elevator'],
            initial_guess['aileron'],
            initial_guess['rudder'],
            initial_guess['throttle']
        ])

        # Define residual function
        def residuals(x):
            alpha, theta, phi, elevator, aileron, rudder, throttle = x

            # Build state
            state = State()
            state.position = np.array([0.0, 0.0, -altitude])

            # Velocity in body frame
            u = airspeed * np.cos(alpha)
            w = airspeed * np.sin(alpha)
            state.velocity_body = np.array([u, 0.0, w])

            # Attitude
            state.set_euler_angles(phi, theta, 0.0)

            # Angular rates for coordinated turn
            # r = turn_rate in yaw axis (approximately)
            state.angular_rates = np.array([0.0, 0.0, turn_rate])

            # Controls
            controls = {
                'elevator': elevator,
                'aileron': aileron,
                'rudder': rudder,
                'throttle': throttle
            }

            # Compute state derivative
            state_dot = self.dynamics_function(state, controls)

            # Residuals
            residual = np.array([
                state_dot[3],   # u_dot
                state_dot[4],   # v_dot (side slip, want 0 for coordinated)
                state_dot[5],   # w_dot
                state_dot[2],   # z_dot (maintain altitude)
                state_dot[10],  # p_dot (roll acceleration)
                state_dot[11],  # q_dot (pitch acceleration)
                state_dot[12]   # r_dot (yaw acceleration, maintain turn rate)
            ])

            return residual

        # Bounds
        bounds = [
            (-np.radians(15), np.radians(15)),   # alpha
            (-np.radians(15), np.radians(15)),   # theta
            (-np.radians(60), np.radians(60)),   # phi (bank angle)
            (-np.radians(25), np.radians(25)),   # elevator
            (-np.radians(25), np.radians(25)),   # aileron
            (-np.radians(25), np.radians(25)),   # rudder
            (0.0, 1.0)                            # throttle
        ]

        # Solve
        result = least_squares(residuals, x0, bounds=np.array(bounds).T, verbose=2 if verbose else 0)

        # Extract solution
        alpha_trim, theta_trim, phi_trim, elevator_trim, aileron_trim, rudder_trim, throttle_trim = result.x

        # Build trimmed state
        state_trim = State()
        state_trim.position = np.array([0.0, 0.0, -altitude])
        u_trim = airspeed * np.cos(alpha_trim)
        w_trim = airspeed * np.sin(alpha_trim)
        state_trim.velocity_body = np.array([u_trim, 0.0, w_trim])
        state_trim.set_euler_angles(phi_trim, theta_trim, 0.0)
        state_trim.angular_rates = np.array([0.0, 0.0, turn_rate])

        # Build trimmed controls
        controls_trim = {
            'elevator': elevator_trim,
            'aileron': aileron_trim,
            'rudder': rudder_trim,
            'throttle': throttle_trim
        }

        # Optimization info
        info = {
            'success': result.success,
            'residual_norm': np.linalg.norm(result.fun),
            'iterations': result.nfev,
            'message': result.message,
            'alpha_deg': np.degrees(alpha_trim),
            'theta_deg': np.degrees(theta_trim),
            'phi_deg': np.degrees(phi_trim),
            'elevator_deg': np.degrees(elevator_trim),
            'aileron_deg': np.degrees(aileron_trim),
            'rudder_deg': np.degrees(rudder_trim),
            'throttle_pct': throttle_trim * 100
        }

        return state_trim, controls_trim, info


def test_trim():
    """Test trim solver with simple aircraft model."""
    print("=" * 60)
    print("Trim Solver Test")
    print("=" * 60)
    print()
    print("Note: This is a basic test. Full testing requires complete")
    print("      dynamics model (see Phase 3 integration tests)")
    print()


if __name__ == "__main__":
    test_trim()
