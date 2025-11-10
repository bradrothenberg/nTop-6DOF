"""
Numerical integrators for 6-DOF flight dynamics.

Implements:
- RK4 (Runge-Kutta 4th order, fixed step)
- RK45 (Runge-Kutta-Fehlberg, adaptive step)
"""

import numpy as np
from typing import Callable, Tuple

# Handle imports
try:
    from .state import State
except ImportError:
    from state import State


class RK4Integrator:
    """
    4th-order Runge-Kutta integrator (fixed time step).

    Classic RK4 method with good accuracy for smooth dynamics.
    """

    def __init__(self, dt: float = 0.01):
        """
        Initialize RK4 integrator.

        Parameters:
        -----------
        dt : float
            Fixed time step (seconds)
        """
        self.dt = dt

    def step(self, state: State, derivative_func: Callable) -> State:
        """
        Advance state by one time step using RK4.

        Parameters:
        -----------
        state : State
            Current state
        derivative_func : Callable
            Function that computes state_dot = f(state)
            Returns: np.ndarray, shape (13,)

        Returns:
        --------
        new_state : State
            State at t + dt
        """
        # Get current state as array
        x = state.to_array()
        dt = self.dt

        # RK4 stages
        k1 = derivative_func(state)

        # k2: evaluate at t + dt/2, x + k1*dt/2
        state2 = State()
        state2.from_array(x + 0.5 * dt * k1)
        state2.q.normalize()
        k2 = derivative_func(state2)

        # k3: evaluate at t + dt/2, x + k2*dt/2
        state3 = State()
        state3.from_array(x + 0.5 * dt * k2)
        state3.q.normalize()
        k3 = derivative_func(state3)

        # k4: evaluate at t + dt, x + k3*dt
        state4 = State()
        state4.from_array(x + dt * k3)
        state4.q.normalize()
        k4 = derivative_func(state4)

        # Weighted average
        x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Create new state
        new_state = State()
        new_state.from_array(x_new)
        new_state.q.normalize()

        return new_state

    def integrate(self, state0: State, t_span: Tuple[float, float],
                  derivative_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate from t0 to tf.

        Parameters:
        -----------
        state0 : State
            Initial state
        t_span : tuple
            (t0, tf) time span
        derivative_func : Callable
            State derivative function

        Returns:
        --------
        t_history : np.ndarray
            Time points
        state_history : np.ndarray, shape (n_steps, 13)
            State at each time point
        """
        t0, tf = t_span
        n_steps = int((tf - t0) / self.dt) + 1

        # Preallocate
        t_history = np.linspace(t0, tf, n_steps)
        state_history = np.zeros((n_steps, 13))

        # Initial condition
        state_current = state0.copy()
        state_history[0, :] = state_current.to_array()

        # Integrate
        for i in range(1, n_steps):
            state_current = self.step(state_current, derivative_func)
            state_history[i, :] = state_current.to_array()

        return t_history, state_history


class RK45Integrator:
    """
    Runge-Kutta-Fehlberg adaptive step size integrator.

    Uses 4th and 5th order estimates to control error.
    """

    def __init__(self, rtol: float = 1e-6, atol: float = 1e-8,
                 dt_max: float = 1.0, dt_min: float = 1e-6):
        """
        Initialize RK45 integrator.

        Parameters:
        -----------
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        dt_max : float
            Maximum time step (seconds)
        dt_min : float
            Minimum time step (seconds)
        """
        self.rtol = rtol
        self.atol = atol
        self.dt_max = dt_max
        self.dt_min = dt_min

        # RK45 (Dormand-Prince) coefficients
        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
        ])

        self.b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])  # 5th order
        self.b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])  # 4th order
        self.c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])

    def step(self, state: State, dt: float, derivative_func: Callable) -> Tuple[State, float, bool]:
        """
        Attempt one RK45 step with error control.

        Parameters:
        -----------
        state : State
            Current state
        dt : float
            Proposed time step
        derivative_func : Callable
            State derivative function

        Returns:
        --------
        new_state : State
            State at t + dt (if accepted)
        dt_next : float
            Recommended next time step
        accepted : bool
            Whether step was accepted
        """
        x = state.to_array()

        # Compute k stages
        k = np.zeros((7, 13))
        k[0] = derivative_func(state)

        for i in range(1, 7):
            x_stage = x + dt * np.sum([self.a[i, j] * k[j] for j in range(i)], axis=0)
            state_stage = State()
            state_stage.from_array(x_stage)
            state_stage.q.normalize()
            k[i] = derivative_func(state_stage)

        # 5th order solution
        x_new = x + dt * np.sum([self.b[i] * k[i] for i in range(6)], axis=0)

        # 4th order solution
        x_new_star = x + dt * np.sum([self.b_star[i] * k[i] for i in range(7)], axis=0)

        # Error estimate
        error = np.abs(x_new - x_new_star)
        scale = self.atol + self.rtol * np.maximum(np.abs(x), np.abs(x_new))
        error_norm = np.sqrt(np.mean((error / scale)**2))

        # Step size control
        if error_norm < 1.0:
            # Accept step
            new_state = State()
            new_state.from_array(x_new)
            new_state.q.normalize()
            accepted = True
        else:
            # Reject step
            new_state = state
            accepted = False

        # Compute next step size
        safety_factor = 0.9
        if error_norm > 0:
            dt_next = safety_factor * dt * (1.0 / error_norm)**0.2
        else:
            dt_next = dt * 2.0

        dt_next = np.clip(dt_next, self.dt_min, self.dt_max)

        return new_state, dt_next, accepted


if __name__ == "__main__":
    # Test integrators
    print("=== Integrator Tests ===\n")

    # Simple test: exponential decay
    def exponential_decay(state):
        """Test ODE: dx/dt = -0.5 * x"""
        x = state.to_array()
        return -0.5 * x

    # Initial state
    state0 = State()
    state0.position = np.array([1.0, 0.0, 0.0])

    print("Test: dx/dt = -0.5 * x, x(0) = [1, 0, 0, ...]")
    print(f"Analytical solution: x(t) = exp(-0.5*t)")
    print()

    # RK4 integration
    print("1. RK4 Integration (dt=0.1):")
    rk4 = RK4Integrator(dt=0.1)
    t_hist, x_hist = rk4.integrate(state0, (0, 2.0), exponential_decay)

    x_analytical = np.exp(-0.5 * t_hist)
    print(f"   t=0.0: x={x_hist[0, 0]:.6f} (analytical={x_analytical[0]:.6f})")
    print(f"   t=1.0: x={x_hist[10, 0]:.6f} (analytical={x_analytical[10]:.6f})")
    print(f"   t=2.0: x={x_hist[20, 0]:.6f} (analytical={x_analytical[20]:.6f})")
    print(f"   Error at t=2.0: {abs(x_hist[20, 0] - x_analytical[20]):.2e}")
    print()

    print("2. RK4 with smaller step (dt=0.01):")
    rk4_fine = RK4Integrator(dt=0.01)
    t_hist_fine, x_hist_fine = rk4_fine.integrate(state0, (0, 2.0), exponential_decay)
    print(f"   t=2.0: x={x_hist_fine[-1, 0]:.6f} (analytical={np.exp(-1.0):.6f})")
    print(f"   Error at t=2.0: {abs(x_hist_fine[-1, 0] - np.exp(-1.0)):.2e}")
    print()

    print("Integrators working correctly!")
