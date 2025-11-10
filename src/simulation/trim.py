"""
Trim solver for finding equilibrium flight conditions.

Finds the state and control inputs that result in zero acceleration (steady flight).
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.environment.atmosphere import StandardAtmosphere


def find_level_flight_trim(
    dynamics: AircraftDynamics,
    force_model,
    altitude: float,
    airspeed: float,
    heading: float = 0.0,
    initial_guess: Optional[Dict] = None
) -> Tuple[State, Dict[str, float], Dict]:
    """
    Find trim conditions for steady level flight.

    Solves for state and controls such that:
    - Linear acceleration = 0 (steady velocity)
    - Angular acceleration = 0 (steady attitude)
    - Altitude rate = 0 (level flight)

    Parameters:
    -----------
    dynamics : AircraftDynamics
        Aircraft dynamics model
    force_model : callable
        Function that returns (forces, moments) given state and controls
    altitude : float
        Desired altitude (ft, negative down)
    airspeed : float
        Desired airspeed (ft/s)
    heading : float
        Desired heading (rad)
    initial_guess : dict, optional
        Initial guess for optimization variables

    Returns:
    --------
    state : State
        Trimmed state
    controls : dict
        Trimmed control inputs
    info : dict
        Optimization information (success, residuals, etc.)
    """

    # Initial guess for optimization variables
    if initial_guess is None:
        initial_guess = {
            'alpha': np.radians(15),   # angle of attack (need high alpha for lift)
            'theta': np.radians(15),   # pitch angle
            'throttle': 0.4,           # throttle setting
            'elevator': 0.0,           # elevator deflection
        }

    # Pack into array
    x0 = np.array([
        initial_guess['alpha'],
        initial_guess['theta'],
        initial_guess['throttle'],
        initial_guess['elevator']
    ])

    def residuals(x):
        """
        Compute residuals for trim conditions.

        Residuals = [Fx_body, Fz_body, M_pitch, altitude_rate]
        Should all be zero for trim.
        """
        alpha, theta, throttle, elevator = x

        # Create state
        state = State()

        # Position
        state.position = np.array([0.0, 0.0, altitude])

        # Velocity in body frame
        # V_body = [V*cos(alpha), 0, V*sin(alpha)]
        state.velocity_body = np.array([
            airspeed * np.cos(alpha),
            0.0,
            airspeed * np.sin(alpha)
        ])

        # Attitude: pitch = theta, roll = 0, yaw = heading
        # For level flight: theta = alpha + gamma
        # For gamma = 0 (level): theta = alpha
        state.set_euler_angles(0.0, theta, heading)

        # Angular rates = 0 for steady flight
        state.angular_rates = np.array([0.0, 0.0, 0.0])

        # Control inputs
        controls = {
            'throttle': throttle,
            'elevator': elevator,
            'aileron': 0.0,
            'rudder': 0.0
        }

        # Compute forces and moments
        def force_func_with_controls(s):
            # force_model is either callable or has compute_forces_moments method
            if hasattr(force_model, 'compute_forces_moments'):
                F, M = force_model.compute_forces_moments(s, controls)
            else:
                # Assume callable (like CombinedForceModel)
                F, M = force_model(s, controls.get('throttle', 0.5), controls)
            return F, M

        # Get state derivative
        state_dot = dynamics.state_derivative(state, force_func_with_controls)

        # Extract accelerations
        # state_dot = [pos_dot, vel_dot, quat_dot, omega_dot]
        vel_dot = state_dot[3:6]      # Linear acceleration in body frame
        omega_dot = state_dot[10:13]  # Angular acceleration

        # Residuals we want to zero
        Fx_body_accel = vel_dot[0]    # Forward acceleration
        Fz_body_accel = vel_dot[2]    # Vertical acceleration in body
        M_pitch_accel = omega_dot[1]  # Pitch angular acceleration

        # Altitude rate (should be zero for level flight)
        # altitude_dot = -position_dot[2]
        altitude_dot = -state_dot[2]

        residuals = np.array([
            Fx_body_accel * 10.0,   # Scale to similar magnitude
            Fz_body_accel * 10.0,
            M_pitch_accel * 100.0,  # Scale angular accel
            altitude_dot              # Already in ft/s
        ])

        return residuals

    # Bounds for optimization variables
    bounds = (
        [np.radians(0), np.radians(0), 0.0, np.radians(-20)],    # Lower bounds
        [np.radians(30), np.radians(30), 1.0, np.radians(20)]    # Upper bounds
    )

    # Solve
    result = least_squares(
        residuals,
        x0,
        bounds=bounds,
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=1000,
        verbose=0
    )

    # Extract solution
    alpha_trim, theta_trim, throttle_trim, elevator_trim = result.x

    # Create trimmed state
    state_trim = State()
    state_trim.position = np.array([0.0, 0.0, altitude])
    state_trim.velocity_body = np.array([
        airspeed * np.cos(alpha_trim),
        0.0,
        airspeed * np.sin(alpha_trim)
    ])
    state_trim.set_euler_angles(0.0, theta_trim, heading)
    state_trim.angular_rates = np.array([0.0, 0.0, 0.0])

    controls_trim = {
        'throttle': throttle_trim,
        'elevator': elevator_trim,
        'aileron': 0.0,
        'rudder': 0.0
    }

    info = {
        'success': result.success,
        'message': result.message,
        'residuals': result.fun,
        'residual_norm': np.linalg.norm(result.fun),
        'alpha_deg': np.degrees(alpha_trim),
        'theta_deg': np.degrees(theta_trim),
        'throttle': throttle_trim,
        'elevator_deg': np.degrees(elevator_trim)
    }

    return state_trim, controls_trim, info


def verify_trim(
    state: State,
    controls: Dict[str, float],
    dynamics: AircraftDynamics,
    force_model,
    tolerance: float = 1e-3
) -> Dict:
    """
    Verify that a state/control combination is indeed trimmed.

    Parameters:
    -----------
    state : State
        State to verify
    controls : dict
        Control inputs
    dynamics : AircraftDynamics
        Dynamics model
    force_model : callable
        Force model
    tolerance : float
        Acceptable residual norm

    Returns:
    --------
    results : dict
        Verification results
    """

    def force_func(s):
        if hasattr(force_model, 'compute_forces_moments'):
            return force_model.compute_forces_moments(s, controls)
        else:
            return force_model(s, controls.get('throttle', 0.5), controls)

    # Get accelerations
    state_dot = dynamics.state_derivative(state, force_func)

    vel_dot = state_dot[3:6]
    omega_dot = state_dot[10:13]
    altitude_dot = -state_dot[2]

    # Compute norms
    linear_accel_norm = np.linalg.norm(vel_dot)
    angular_accel_norm = np.linalg.norm(omega_dot)

    is_trimmed = (
        linear_accel_norm < tolerance and
        angular_accel_norm < tolerance * 10 and
        abs(altitude_dot) < tolerance
    )

    results = {
        'is_trimmed': is_trimmed,
        'linear_accel': vel_dot,
        'linear_accel_norm': linear_accel_norm,
        'angular_accel': omega_dot,
        'angular_accel_norm': angular_accel_norm,
        'altitude_rate': altitude_dot,
        'tolerance': tolerance
    }

    return results


if __name__ == "__main__":
    """Test trim solver with flying wing."""

    from src.core.aerodynamics import LinearAeroModel
    from src.core.propulsion import PropellerModel, CombinedForceModel

    print("=" * 70)
    print("Trim Solver Test - Flying Wing")
    print("=" * 70)
    print()

    # Aircraft parameters
    mass = 228.924806  # slugs
    inertia = np.array([[19236.2914, 0, 0],
                        [0, 2251.0172, 0],
                        [0, 0, 21487.3086]])

    S_ref = 412.6370
    c_ref = 11.9555
    b_ref = 24.8630

    dynamics = AircraftDynamics(mass, inertia)

    aero = LinearAeroModel(S_ref, c_ref, b_ref, rho=0.002377)

    # Flying wing AVL derivatives
    aero.CL_0 = 0.000023
    aero.CL_alpha = 1.412241
    aero.CL_q = 1.282202
    aero.CL_de = 0.0  # No elevator on flying wing (elevons)

    aero.CD_0 = -0.000619
    aero.CD_alpha = 0.035509
    aero.CD_alpha2 = 0.5

    aero.Cm_0 = 0.000061
    aero.Cm_alpha = -0.079668
    aero.Cm_q = -0.347072
    aero.Cm_de = 0.0  # Symmetric elevon has no pitch effect

    aero.Cl_p = -0.109230
    aero.Cn_r = -0.001030

    prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)
    combined_model = CombinedForceModel(aero, prop)

    # Find trim at 5000 ft, 200 ft/s
    print("Finding trim for:")
    print(f"  Altitude: 5000 ft")
    print(f"  Airspeed: 200 ft/s")
    print(f"  Heading: 0 deg")
    print()

    state_trim, controls_trim, info = find_level_flight_trim(
        dynamics,
        combined_model,
        altitude=-5000.0,
        airspeed=200.0,
        heading=0.0
    )

    print("Trim Solution:")
    print(f"  Success: {info['success']}")
    print(f"  Message: {info['message']}")
    print(f"  Residual norm: {info['residual_norm']:.6e}")
    print()
    print(f"  Alpha: {info['alpha_deg']:.2f} deg")
    print(f"  Theta: {info['theta_deg']:.2f} deg")
    print(f"  Throttle: {info['throttle']:.3f}")
    print(f"  Elevator: {info['elevator_deg']:.2f} deg")
    print()

    # Verify trim
    print("Verifying trim...")
    verify_results = verify_trim(state_trim, controls_trim, dynamics, combined_model)

    print(f"  Is trimmed: {verify_results['is_trimmed']}")
    print(f"  Linear accel norm: {verify_results['linear_accel_norm']:.6e} ft/s²")
    print(f"  Angular accel norm: {verify_results['angular_accel_norm']:.6e} rad/s²")
    print(f"  Altitude rate: {verify_results['altitude_rate']:.6e} ft/s")
    print()

    if verify_results['is_trimmed']:
        print("CHECK Trim verified successfully!")
    else:
        print("X Trim verification failed")
        print(f"  Linear accel: {verify_results['linear_accel']}")
        print(f"  Angular accel: {verify_results['angular_accel']}")
