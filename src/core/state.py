"""
6-DOF state vector for aircraft flight dynamics.

State includes:
- Position (x, y, z) in NED inertial frame
- Velocity (u, v, w) in body frame
- Attitude quaternion (q0, q1, q2, q3)
- Angular rates (p, q, r) in body frame
"""

import numpy as np
from typing import Tuple

from archimedes import struct, field

# Handle imports for both package and standalone usage
try:
    from .quaternion import Quaternion
except ImportError:
    from quaternion import Quaternion


@struct(frozen=False)
class State:
    """
    Complete 6-DOF aircraft state vector.

    State variables (13 total):
    - Position: x_n, y_n, z_n (NED inertial frame, ft)
    - Velocity: u, v, w (body frame, ft/s)
    - Attitude: q0, q1, q2, q3 (quaternion)
    - Angular rates: p, q, r (body frame, rad/s)
    """

    # Position in NED frame (ft)
    x_n: float = 0.0  # North
    y_n: float = 0.0  # East
    z_n: float = 0.0  # Down (negative altitude)

    # Velocity in body frame (ft/s)
    u: float = 0.0  # Forward velocity
    v: float = 0.0  # Side velocity
    w: float = 0.0  # Vertical velocity

    # Attitude quaternion (inertial to body)
    q: Quaternion = field(default_factory=Quaternion)  # Identity (no rotation)

    # Angular rates in body frame (rad/s)
    p: float = 0.0  # Roll rate
    q_rate: float = 0.0  # Pitch rate
    r: float = 0.0  # Yaw rate

    @property
    def position(self) -> np.ndarray:
        """Get position vector in NED frame (ft)."""
        return np.hstack([self.x_n, self.y_n, self.z_n])

    @position.setter
    def position(self, pos: np.ndarray):
        """Set position vector."""
        self.x_n, self.y_n, self.z_n = pos

    @property
    def velocity_body(self) -> np.ndarray:
        """Get velocity vector in body frame (ft/s)."""
        return np.hstack([self.u, self.v, self.w])

    @velocity_body.setter
    def velocity_body(self, vel: np.ndarray):
        """Set velocity in body frame."""
        self.u, self.v, self.w = vel

    @property
    def angular_rates(self) -> np.ndarray:
        """Get angular rate vector in body frame (rad/s)."""
        return np.hstack([self.p, self.q_rate, self.r])

    @angular_rates.setter
    def angular_rates(self, omega: np.ndarray):
        """Set angular rates."""
        self.p, self.q_rate, self.r = omega

    @property
    def velocity_inertial(self) -> np.ndarray:
        """
        Get velocity in NED inertial frame (ft/s).

        Transforms body-frame velocity to inertial frame.
        """
        # Rotation matrix from body to inertial (transpose of body-to-inertial)
        R_b_to_i = self.q.to_rotation_matrix().T
        return R_b_to_i @ self.velocity_body

    @property
    def airspeed(self) -> float:
        """Get total airspeed magnitude (ft/s)."""
        return np.linalg.norm(self.velocity_body)

    @property
    def groundspeed(self) -> float:
        """Get groundspeed magnitude (ft/s)."""
        return np.linalg.norm(self.velocity_inertial)

    @property
    def altitude(self) -> float:
        """Get altitude above reference (ft, positive up)."""
        return -self.z_n

    @altitude.setter
    def altitude(self, alt: float):
        """Set altitude."""
        self.z_n = -alt

    @property
    def euler_angles(self) -> Tuple[float, float, float]:
        """
        Get Euler angles from quaternion (rad).

        Returns:
        --------
        phi : float
            Roll angle
        theta : float
            Pitch angle
        psi : float
            Yaw angle
        """
        return self.q.to_euler_angles()

    def set_euler_angles(self, phi: float, theta: float, psi: float):
        """
        Set attitude using Euler angles.

        Parameters:
        -----------
        phi : float
            Roll angle (rad)
        theta : float
            Pitch angle (rad)
        psi : float
            Yaw angle (rad)
        """
        self.q = Quaternion.from_euler_angles(phi, theta, psi)

    @property
    def alpha(self) -> float:
        """
        Get angle of attack (rad).

        alpha = atan(w / u)
        """
        if abs(self.u) < 1e-6:
            return 0.0
        return np.arctan2(self.w, self.u)

    @property
    def beta(self) -> float:
        """
        Get sideslip angle (rad).

        beta = asin(v / V)
        """
        V = self.airspeed
        if V < 1e-6:
            return 0.0
        return np.arcsin(np.clip(self.v / V, -1.0, 1.0))

    def to_array(self) -> np.ndarray:
        """
        Convert state to numpy array.

        Returns:
        --------
        x : np.ndarray, shape (13,)
            State vector [x_n, y_n, z_n, u, v, w, q0, q1, q2, q3, p, q, r]
        """
        return np.hstack([
            self.x_n, self.y_n, self.z_n,  # Position
            self.u, self.v, self.w,        # Velocity
            *self.q.q,                      # Quaternion (4 elements)
            self.p, self.q_rate, self.r    # Angular rates
        ])

    def from_array(self, x: np.ndarray):
        """
        Load state from numpy array.

        Parameters:
        -----------
        x : np.ndarray, shape (13,)
            State vector [x_n, y_n, z_n, u, v, w, q0, q1, q2, q3, p, q, r]
        """
        self.x_n, self.y_n, self.z_n = x[0:3]
        self.u, self.v, self.w = x[3:6]
        self.q = Quaternion(x[6:10])
        self.p, self.q_rate, self.r = x[10:13]

    def copy(self) -> 'State':
        """Create a deep copy of the state."""
        new_state = State()
        new_state.from_array(self.to_array())
        return new_state

    def __repr__(self) -> str:
        """String representation."""
        return f"State(pos={self.position}, vel={self.velocity_body}, omega={self.angular_rates})"

    def __str__(self) -> str:
        """Pretty print state."""
        phi, theta, psi = self.euler_angles
        V = self.airspeed

        return (
            f"6-DOF Aircraft State:\n"
            f"  Position (NED):   [{self.x_n:8.1f}, {self.y_n:8.1f}, {self.z_n:8.1f}] ft\n"
            f"  Altitude:         {self.altitude:8.1f} ft\n"
            f"  Velocity (body):  [{self.u:7.2f}, {self.v:7.2f}, {self.w:7.2f}] ft/s\n"
            f"  Airspeed:         {V:7.2f} ft/s\n"
            f"  Euler angles:     [{np.degrees(phi):6.2f}, {np.degrees(theta):6.2f}, {np.degrees(psi):6.2f}] deg\n"
            f"  Alpha, Beta:      [{np.degrees(self.alpha):6.2f}, {np.degrees(self.beta):6.2f}] deg\n"
            f"  Angular rates:    [{self.p:7.4f}, {self.q_rate:7.4f}, {self.r:7.4f}] rad/s"
        )


if __name__ == "__main__":
    # Test state functionality
    print("=== State Vector Tests ===\n")

    # Test 1: Default initialization
    state = State()
    print("1. Default state:")
    print(state)
    print()

    # Test 2: Set position and velocity
    state.position = np.array([1000.0, 500.0, -5000.0])  # 5000 ft altitude
    state.velocity_body = np.array([250.0, 0.0, 0.0])  # 250 ft/s forward
    state.set_euler_angles(np.radians(5), np.radians(3), np.radians(45))
    state.angular_rates = np.array([0.1, 0.05, 0.02])

    print("2. Modified state:")
    print(state)
    print()

    # Test 3: Array conversion
    x = state.to_array()
    print(f"3. State as array (shape {x.shape}):")
    print(x)
    print()

    # Test 4: Reconstruct from array
    state2 = State()
    state2.from_array(x)
    print("4. Reconstructed state:")
    print(state2)
    print()

    # Test 5: Verify airspeed and altitude
    print("5. Computed properties:")
    print(f"  Airspeed:      {state.airspeed:7.2f} ft/s")
    print(f"  Groundspeed:   {state.groundspeed:7.2f} ft/s")
    print(f"  Altitude:      {state.altitude:7.1f} ft")
    print(f"  Alpha:         {np.degrees(state.alpha):7.2f} deg")
    print(f"  Beta:          {np.degrees(state.beta):7.2f} deg")
