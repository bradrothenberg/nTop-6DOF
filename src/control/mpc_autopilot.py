"""
Model Predictive Control (MPC) Autopilot for Flying Wing

Implements MPC to handle large altitude changes by:
1. Predicting future trajectory over horizon (N steps)
2. Optimizing control inputs to reach targets
3. Explicitly handling altitude-airspeed coupling
4. Satisfying control and state constraints

MPC solves at each timestep:
    minimize:   sum(||x - x_ref||^2 + ||u - u_ref||^2)
    subject to: x(k+1) = f(x(k), u(k))  (dynamics)
                u_min <= u <= u_max      (control limits)
                x_min <= x <= x_max      (state limits)
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict


class MPCAutopilot:
    """
    Model Predictive Control autopilot for flying wing.

    Uses simplified longitudinal dynamics model for prediction
    and optimization-based control synthesis.
    """

    def __init__(self,
                 # MPC parameters
                 horizon: int = 20,              # Prediction horizon steps
                 dt_mpc: float = 0.5,            # MPC timestep (seconds)
                 # Weights
                 Q_altitude: float = 1.0,        # Altitude tracking weight
                 Q_airspeed: float = 0.5,        # Airspeed tracking weight
                 Q_pitch: float = 0.1,           # Pitch angle weight
                 R_elevator: float = 0.1,        # Elevator control cost
                 R_throttle: float = 0.05,       # Throttle control cost
                 # Aircraft parameters
                 mass: float = 228.9,            # slugs
                 S_ref: float = 412.6,           # ft²
                 c_ref: float = 11.96,           # ft
                 # Trim values
                 elevon_trim: float = -0.0988,   # rad
                 throttle_trim: float = 0.794,
                 g: float = 32.174):             # ft/s²

        # MPC parameters
        self.horizon = horizon
        self.dt_mpc = dt_mpc

        # Weights
        self.Q_altitude = Q_altitude
        self.Q_airspeed = Q_airspeed
        self.Q_pitch = Q_pitch
        self.R_elevator = R_elevator
        self.R_throttle = R_throttle

        # Aircraft parameters
        self.mass = mass
        self.S_ref = S_ref
        self.c_ref = c_ref
        self.g = g

        # Trim
        self.elevon_trim = elevon_trim
        self.throttle_trim = throttle_trim

        # Aerodynamic derivatives (simplified for MPC model)
        self.CL_alpha = 1.412
        self.CD_0 = 0.006
        self.CD_alpha = 0.025
        self.CD_alpha2 = 0.05
        self.Cm_alpha = -0.0797
        self.Cm_q = -0.347
        self.Cm_de = -0.02

        # Targets
        self.target_altitude = 5000.0  # ft
        self.target_airspeed = 600.0   # ft/s

        # Control limits
        self.elevon_min = np.radians(-25.0)
        self.elevon_max = np.radians(25.0)
        self.throttle_min = 0.1
        self.throttle_max = 1.0

        # State limits
        self.alpha_max = np.radians(12.0)
        self.pitch_rate_max = np.radians(20.0)

        # Previous solution for warm start
        self.u_prev = None

    def set_targets(self, altitude: float, airspeed: float):
        """Set target altitude and airspeed."""
        self.target_altitude = altitude
        self.target_airspeed = airspeed

    def _simplified_dynamics(self, x: np.ndarray, u: np.ndarray, rho: float) -> np.ndarray:
        """
        Simplified longitudinal dynamics for MPC prediction.

        State: x = [h, V, gamma, q, theta]
            h: altitude (ft)
            V: airspeed (ft/s)
            gamma: flight path angle (rad)
            q: pitch rate (rad/s)
            theta: pitch angle (rad)

        Control: u = [elevon, throttle]
            elevon: elevator deflection (rad)
            throttle: throttle setting [0, 1]

        Returns: dx/dt
        """
        h, V, gamma, q, theta = x
        elevon, throttle = u

        # Angle of attack
        alpha = theta - gamma

        # Aerodynamic forces
        q_bar = 0.5 * rho * V**2
        CL = self.CL_alpha * alpha
        CD = self.CD_0 + self.CD_alpha * alpha + self.CD_alpha2 * alpha**2

        L = q_bar * self.S_ref * CL
        D = q_bar * self.S_ref * CD

        # Thrust (simplified - assume const with altitude for MPC model)
        T = 1700.0 * throttle  # lbf (approximate at 5k-10k ft)

        # Pitching moment
        q_hat = q * self.c_ref / (2 * V) if V > 1 else 0
        Cm = self.Cm_alpha * alpha + self.Cm_q * q_hat + self.Cm_de * elevon
        M = q_bar * self.S_ref * self.c_ref * Cm

        # Moment of inertia (approximate)
        Iyy = 2251.0  # slug-ft²

        # Equations of motion
        W = self.mass * self.g

        dh_dt = V * np.sin(gamma)
        dV_dt = (T - D) / self.mass - self.g * np.sin(gamma)
        dgamma_dt = (L - W * np.cos(gamma)) / (self.mass * V) if V > 1 else 0
        dq_dt = M / Iyy
        dtheta_dt = q

        return np.array([dh_dt, dV_dt, dgamma_dt, dq_dt, dtheta_dt])

    def _predict_trajectory(self, x0: np.ndarray, u_sequence: np.ndarray, rho: float) -> np.ndarray:
        """
        Predict trajectory over horizon given control sequence.

        Args:
            x0: Initial state [h, V, gamma, q, theta]
            u_sequence: Control sequence, shape (horizon, 2)
            rho: Air density

        Returns:
            X: Predicted states, shape (horizon+1, 5)
        """
        X = np.zeros((self.horizon + 1, 5))
        X[0] = x0

        for k in range(self.horizon):
            # Simple Euler integration (could use RK4 for better accuracy)
            x = X[k]
            u = u_sequence[k]
            dx = self._simplified_dynamics(x, u, rho)
            X[k+1] = x + dx * self.dt_mpc

            # Clamp states to reasonable bounds
            X[k+1, 0] = np.clip(X[k+1, 0], 0, 20000)       # altitude
            X[k+1, 1] = np.clip(X[k+1, 1], 100, 1000)      # airspeed
            X[k+1, 2] = np.clip(X[k+1, 2], -0.5, 0.5)      # gamma
            X[k+1, 3] = np.clip(X[k+1, 3], -0.3, 0.3)      # pitch rate
            X[k+1, 4] = np.clip(X[k+1, 4], -0.3, 0.3)      # pitch angle

        return X

    def _cost_function(self, u_flat: np.ndarray, x0: np.ndarray, rho: float) -> float:
        """
        MPC cost function to minimize.

        Args:
            u_flat: Flattened control sequence, shape (horizon * 2,)
            x0: Initial state
            rho: Air density

        Returns:
            cost: Total cost over horizon
        """
        # Reshape control sequence
        u_sequence = u_flat.reshape((self.horizon, 2))

        # Predict trajectory
        X = self._predict_trajectory(x0, u_sequence, rho)

        # Compute cost
        cost = 0.0

        for k in range(self.horizon + 1):
            h, V, gamma, q, theta = X[k]

            # State cost (tracking)
            altitude_error = h - self.target_altitude
            airspeed_error = V - self.target_airspeed
            pitch_error = theta  # Want small pitch deviations

            cost += self.Q_altitude * altitude_error**2
            cost += self.Q_airspeed * airspeed_error**2
            cost += self.Q_pitch * pitch_error**2

        # Control cost (effort)
        for k in range(self.horizon):
            elevon, throttle = u_sequence[k]
            elevon_effort = elevon - self.elevon_trim
            throttle_effort = throttle - self.throttle_trim

            cost += self.R_elevator * elevon_effort**2
            cost += self.R_throttle * throttle_effort**2

        return cost

    def solve_mpc(self, current_altitude: float, current_airspeed: float,
                  current_pitch: float, current_pitch_rate: float,
                  current_gamma: float, rho: float) -> Tuple[float, float]:
        """
        Solve MPC optimization problem.

        Args:
            current_altitude: Current altitude (ft)
            current_airspeed: Current airspeed (ft/s)
            current_pitch: Current pitch angle (rad)
            current_pitch_rate: Current pitch rate (rad/s)
            current_gamma: Current flight path angle (rad)
            rho: Current air density (slug/ft³)

        Returns:
            (elevon_cmd, throttle_cmd): Optimal control for current timestep
        """
        # Current state
        x0 = np.array([
            current_altitude,
            current_airspeed,
            current_gamma,
            current_pitch_rate,
            current_pitch
        ])

        # Initial guess for control sequence
        if self.u_prev is None:
            # Use trim values
            u0 = np.array([[self.elevon_trim, self.throttle_trim]] * self.horizon)
        else:
            # Warm start from previous solution (shift and append)
            u0 = np.vstack([self.u_prev[1:], self.u_prev[-1:]])

        u0_flat = u0.flatten()

        # Control bounds
        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.elevon_min, self.elevon_max))    # elevon
            bounds.append((self.throttle_min, self.throttle_max))  # throttle

        # Solve optimization
        result = minimize(
            self._cost_function,
            u0_flat,
            args=(x0, rho),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4}
        )

        # Extract solution
        u_opt = result.x.reshape((self.horizon, 2))
        self.u_prev = u_opt

        # Return first control action (receding horizon)
        elevon_cmd = u_opt[0, 0]
        throttle_cmd = u_opt[0, 1]

        return elevon_cmd, throttle_cmd

    def get_debug_info(self) -> Dict:
        """Get debug information."""
        return {
            'target_altitude': self.target_altitude,
            'target_airspeed': self.target_airspeed,
            'horizon': self.horizon
        }
