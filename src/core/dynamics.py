"""
6-DOF equations of motion for aircraft flight dynamics.

Implements the rigid body dynamics equations:
- Translational dynamics (Newton's 2nd law)
- Rotational dynamics (Euler's equations)
- Quaternion kinematics
"""

import numpy as np
from typing import Tuple, Callable

# Handle imports for both package and standalone usage
try:
    from .state import State
    from .quaternion import Quaternion
except ImportError:
    from state import State
    from quaternion import Quaternion


class AircraftDynamics:
    """
    6-DOF rigid body dynamics for aircraft.

    Equations of motion in body frame:
    - Forces: F = m * (v_dot + omega x v)
    - Moments: M = I * omega_dot + omega x (I * omega)
    - Kinematics: q_dot = 0.5 * Omega(omega) * q
    """

    def __init__(self, mass: float, inertia: np.ndarray):
        """
        Initialize aircraft dynamics.

        Parameters:
        -----------
        mass : float
            Aircraft mass (slugs)
        inertia : np.ndarray, shape (3, 3)
            Inertia tensor in body frame (slug*ft^2)
            [[Ixx, -Ixy, -Ixz],
             [-Ixy, Iyy, -Iyz],
             [-Ixz, -Iyz, Izz]]
        """
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)

        # Gravity constant (ft/s^2)
        self.g = 32.174

    def state_derivative(self, state: State, forces_moments: Callable) -> np.ndarray:
        """
        Compute state time derivative (state_dot).

        Parameters:
        -----------
        state : State
            Current aircraft state
        forces_moments : Callable
            Function that returns (forces, moments) given state
            forces: np.ndarray (3,) - [Fx, Fy, Fz] in body frame (lbf)
            moments: np.ndarray (3,) - [L, M, N] in body frame (ft*lbf)

        Returns:
        --------
        state_dot : np.ndarray, shape (13,)
            Time derivative of state vector
        """
        # Extract current state
        vel_body = state.velocity_body  # [u, v, w]
        omega = state.angular_rates     # [p, q, r]

        # Get forces and moments from external models
        forces, moments = forces_moments(state)

        # === Translational Dynamics ===
        # F = m * (v_dot + omega x v) + m * g * R^T * [0, 0, g]
        # Rearrange: v_dot = F/m - omega x v - g * R^T * [0, 0, g]

        # Gravity in inertial frame
        g_inertial = np.array([0, 0, self.g])

        # Transform gravity to body frame
        R_i_to_b = state.q.to_rotation_matrix()
        g_body = R_i_to_b @ g_inertial

        # Cross product: omega x v
        omega_cross_v = np.cross(omega, vel_body)

        # Velocity derivative in body frame
        vel_body_dot = forces / self.mass - omega_cross_v - g_body

        # === Rotational Dynamics ===
        # M = I * omega_dot + omega x (I * omega)
        # Rearrange: omega_dot = I^-1 * (M - omega x (I * omega))

        I_omega = self.inertia @ omega
        omega_cross_I_omega = np.cross(omega, I_omega)

        omega_dot = self.inertia_inv @ (moments - omega_cross_I_omega)

        # === Kinematics ===
        # Position derivative: x_dot = R^T * v_body
        R_b_to_i = R_i_to_b.T
        pos_dot = R_b_to_i @ vel_body

        # Quaternion derivative: q_dot = 0.5 * Omega(omega) * q
        p, q_rate, r = omega
        Omega = np.array([
            [0,  -p,  -q_rate,  -r],
            [p,   0,   r,  -q_rate],
            [q_rate,  -r,   0,   p],
            [r,   q_rate,  -p,   0]
        ])
        q_dot_array = 0.5 * Omega @ state.q.q

        # Construct state derivative
        state_dot = np.zeros(13)
        state_dot[0:3] = pos_dot           # Position derivative
        state_dot[3:6] = vel_body_dot      # Velocity derivative
        state_dot[6:10] = q_dot_array      # Quaternion derivative
        state_dot[10:13] = omega_dot       # Angular rate derivative

        return state_dot

    def propagate(self, state: State, dt: float, forces_moments: Callable) -> State:
        """
        Propagate state forward by dt using Euler integration.

        Parameters:
        -----------
        state : State
            Current state
        dt : float
            Time step (seconds)
        forces_moments : Callable
            Function returning (forces, moments)

        Returns:
        --------
        new_state : State
            State at t + dt
        """
        # Compute derivative
        state_dot = self.state_derivative(state, forces_moments)

        # Euler integration
        state_array = state.to_array()
        new_state_array = state_array + state_dot * dt

        # Create new state
        new_state = State()
        new_state.from_array(new_state_array)

        # Normalize quaternion to maintain unit constraint
        new_state.q.normalize()

        return new_state


class SimpleForceModel:
    """Simple force and moment model for testing."""

    def __init__(self, CL: float = 0.5, CD: float = 0.05, S_ref: float = 200.0, rho: float = 0.002377):
        """
        Initialize simple aerodynamic model.

        Parameters:
        -----------
        CL : float
            Constant lift coefficient
        CD : float
            Constant drag coefficient
        S_ref : float
            Reference area (ft^2)
        rho : float
            Air density (slug/ft^3)
        """
        self.CL = CL
        self.CD = CD
        self.S_ref = S_ref
        self.rho = rho

    def __call__(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forces and moments.

        Returns:
        --------
        forces : np.ndarray (3,)
            Forces in body frame [Fx, Fy, Fz] (lbf)
        moments : np.ndarray (3,)
            Moments in body frame [L, M, N] (ft*lbf)
        """
        # Dynamic pressure
        V = state.airspeed
        if V < 1.0:
            V = 1.0  # Avoid singularity
        q_bar = 0.5 * self.rho * V**2

        # Aerodynamic forces in wind frame
        L = q_bar * self.S_ref * self.CL
        D = q_bar * self.S_ref * self.CD

        # Transform to body frame (simplified - assumes small alpha)
        alpha = state.alpha
        Fx = -D * np.cos(alpha) + L * np.sin(alpha)
        Fz = -D * np.sin(alpha) - L * np.cos(alpha)
        Fy = 0.0

        forces = np.array([Fx, Fy, Fz])

        # Simple stability derivatives for moments
        # Pitch moment: Cm = Cm0 + Cm_alpha * alpha
        Cm0 = -0.05
        Cm_alpha = -0.5
        Cm = Cm0 + Cm_alpha * alpha

        c_ref = np.sqrt(self.S_ref)  # Rough estimate
        M_pitch = q_bar * self.S_ref * c_ref * Cm

        moments = np.array([0.0, M_pitch, 0.0])  # No roll or yaw for simplicity

        return forces, moments


if __name__ == "__main__":
    # Test dynamics
    print("=== Aircraft Dynamics Test ===\n")

    # Import integrator for proper time stepping
    try:
        from .integrator import RK4Integrator
    except ImportError:
        from integrator import RK4Integrator

    # Aircraft properties (from nTop UAV)
    mass = 234.8  # slugs
    inertia = np.array([
        [14908.4, 0, 0],
        [0, 2318.4, 0],
        [0, 0, 17226.9]
    ])  # slug*ft^2

    # Create dynamics model
    dynamics = AircraftDynamics(mass, inertia)

    # Create initial state - level flight at 250 ft/s
    state = State()
    state.altitude = 5000.0  # ft
    state.velocity_body = np.array([250.0, 0.0, 0.0])  # ft/s
    state.set_euler_angles(0, np.radians(2), 0)  # 2° pitch for level flight

    print("Initial state:")
    print(state)
    print()

    # Create simple force model
    force_model = SimpleForceModel(CL=0.5, CD=0.05, S_ref=199.94)

    # Compute derivative
    state_dot = dynamics.state_derivative(state, force_model)

    print("State derivative:")
    print(f"  Position rate: {state_dot[0:3]} ft/s")
    print(f"  Velocity rate: {state_dot[3:6]} ft/s^2")
    print(f"  Quaternion rate: {state_dot[6:10]}")
    print(f"  Angular accel: {state_dot[10:13]} rad/s^2")
    print()

    # Single RK4 step
    print("=== RK4 Integration Test ===\n")
    integrator = RK4Integrator(dt=0.01)

    # Create derivative function for integrator
    def derivative_func(s):
        return dynamics.state_derivative(s, force_model)

    state_rk4 = integrator.step(state, derivative_func)

    print(f"State after 0.01 seconds (RK4):")
    print(state_rk4)
    print()

    # Run short simulation with RK4
    print("=== 10 Second Simulation (RK4, dt=0.01) ===\n")
    t_hist, x_hist = integrator.integrate(state, (0, 10.0), derivative_func)

    # Reconstruct final state
    state_final = State()
    state_final.from_array(x_hist[-1, :])

    print(f"Final state after 10.0 seconds:")
    print(state_final)
    print()
    print(f"Altitude change: {state.altitude - state_final.altitude:.1f} ft")
    print(f"Airspeed change: {state_final.airspeed - state.airspeed:.1f} ft/s")
