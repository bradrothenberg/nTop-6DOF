"""
Quaternion mathematics for attitude representation.

Quaternions provide a singularity-free representation of 3D rotations,
avoiding gimbal lock issues of Euler angles.

Convention: q = [q0, q1, q2, q3] = [scalar, vector]
            q = q0 + q1*i + q2*j + q3*k
"""

import numpy as np
from typing import Tuple

from archimedes import struct, field
from archimedes.spatial import quaternion_to_dcm


@struct(frozen=False)
class Quaternion:
    """
    Quaternion class for attitude representation and rotation.

    The quaternion represents rotation from inertial to body frame.
    Convention: q = [q0, q1, q2, q3] where q0 is scalar part.
    """

    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def normalize(self):
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(self.q)
        self.q = np.where(norm < 1e-10, np.array([1.0, 0.0, 0.0, 0.0]), self.q / norm)

    @property
    def scalar(self) -> float:
        """Get scalar part (q0)."""
        return self.q[0]

    @property
    def vector(self) -> np.ndarray:
        """Get vector part [q1, q2, q3]."""
        return self.q[1:4]

    def conjugate(self) -> 'Quaternion':
        """Return conjugate quaternion q* = [q0, -q1, -q2, -q3]."""
        q_conj = np.hstack([self.q[0], -self.q[1], -self.q[2], -self.q[3]])
        return Quaternion(q_conj)

    def inverse(self) -> 'Quaternion':
        """Return inverse quaternion (same as conjugate for unit quaternion)."""
        return self.conjugate()

    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        """
        Quaternion multiplication: self * other

        Parameters:
        -----------
        other : Quaternion
            Right operand

        Returns:
        --------
        result : Quaternion
            Product quaternion
        """
        q1 = self.q
        q2 = other.q

        # Quaternion multiplication formula
        q0 = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        q1_new = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        q2_new = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        q3_new = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

        return Quaternion(np.hstack([q0, q1_new, q2_new, q3_new]))

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Overload * operator for quaternion multiplication."""
        return self.multiply(other)

    def to_rotation_matrix(self) -> np.ndarray:
        """
        Convert quaternion to rotation matrix (DCM).

        Returns:
        --------
        R : np.ndarray, shape (3, 3)
            Direction cosine matrix (inertial to body)
        """
        return quaternion_to_dcm(self.q).T

    def to_euler_angles(self) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).

        Convention: ZYX Euler angles (yaw-pitch-roll sequence)

        Returns:
        --------
        phi : float
            Roll angle (radians)
        theta : float
            Pitch angle (radians)
        psi : float
            Yaw angle (radians)
        """
        q0, q1, q2, q3 = self.q

        # Roll (phi)
        phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))

        # Pitch (theta)
        sin_theta = 2*(q0*q2 - q3*q1)
        # Clamp to avoid numerical issues with arcsin
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        theta = np.arcsin(sin_theta)

        # Yaw (psi)
        psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

        return phi, theta, psi

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector from inertial to body frame using this quaternion.

        Parameters:
        -----------
        v : np.ndarray, shape (3,)
            Vector in inertial frame

        Returns:
        --------
        v_body : np.ndarray, shape (3,)
            Vector in body frame
        """
        # Use rotation matrix for efficiency
        R = self.to_rotation_matrix()
        return R @ v

    def integrate(self, omega: np.ndarray, dt: float) -> 'Quaternion':
        """
        Integrate quaternion forward in time given angular velocity.

        Uses first-order integration: q(t+dt) = q(t) + q_dot * dt

        Parameters:
        -----------
        omega : np.ndarray, shape (3,)
            Angular velocity in body frame [p, q, r] (rad/s)
        dt : float
            Time step (seconds)

        Returns:
        --------
        q_new : Quaternion
            Updated quaternion
        """
        p, q_rate, r = omega

        # Quaternion derivative matrix
        Omega = np.array([
            [0,  -p,  -q_rate,  -r],
            [p,   0,   r,  -q_rate],
            [q_rate,  -r,   0,   p],
            [r,   q_rate,  -p,   0]
        ])

        # Quaternion derivative: q_dot = 0.5 * Omega * q
        q_dot = 0.5 * Omega @ self.q

        # First-order Euler integration
        q_new = self.q + q_dot * dt

        return Quaternion(q_new)

    @staticmethod
    def from_euler_angles(phi: float, theta: float, psi: float) -> 'Quaternion':
        """
        Create quaternion from Euler angles.

        Parameters:
        -----------
        phi : float
            Roll angle (radians)
        theta : float
            Pitch angle (radians)
        psi : float
            Yaw angle (radians)

        Returns:
        --------
        q : Quaternion
            Quaternion representation
        """
        # Half angles
        phi_2 = phi / 2.0
        theta_2 = theta / 2.0
        psi_2 = psi / 2.0

        # Compute quaternion elements
        q0 = np.cos(phi_2)*np.cos(theta_2)*np.cos(psi_2) + np.sin(phi_2)*np.sin(theta_2)*np.sin(psi_2)
        q1 = np.sin(phi_2)*np.cos(theta_2)*np.cos(psi_2) - np.cos(phi_2)*np.sin(theta_2)*np.sin(psi_2)
        q2 = np.cos(phi_2)*np.sin(theta_2)*np.cos(psi_2) + np.sin(phi_2)*np.cos(theta_2)*np.sin(psi_2)
        q3 = np.cos(phi_2)*np.cos(theta_2)*np.sin(psi_2) - np.sin(phi_2)*np.sin(theta_2)*np.cos(psi_2)

        return Quaternion(np.array([q0, q1, q2, q3]))

    @staticmethod
    def from_rotation_matrix(R: np.ndarray) -> 'Quaternion':
        """
        Create quaternion from rotation matrix (DCM).

        Parameters:
        -----------
        R : np.ndarray, shape (3, 3)
            Direction cosine matrix

        Returns:
        --------
        q : Quaternion
            Quaternion representation
        """
        # Shepperd's method for numerical stability
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q0 = 0.25 / s
            q1 = (R[2, 1] - R[1, 2]) * s
            q2 = (R[0, 2] - R[2, 0]) * s
            q3 = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q0 = (R[2, 1] - R[1, 2]) / s
            q1 = 0.25 * s
            q2 = (R[0, 1] + R[1, 0]) / s
            q3 = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q0 = (R[0, 2] - R[2, 0]) / s
            q1 = (R[0, 1] + R[1, 0]) / s
            q2 = 0.25 * s
            q3 = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q0 = (R[1, 0] - R[0, 1]) / s
            q1 = (R[0, 2] + R[2, 0]) / s
            q2 = (R[1, 2] + R[2, 1]) / s
            q3 = 0.25 * s

        return Quaternion(np.array([q0, q1, q2, q3]))

    def __repr__(self) -> str:
        """String representation."""
        return f"Quaternion({self.q})"

    def __str__(self) -> str:
        """Pretty print."""
        phi, theta, psi = self.to_euler_angles()
        return (f"Quaternion: q={self.q}\n"
                f"  Roll:  {np.degrees(phi):7.2f}°\n"
                f"  Pitch: {np.degrees(theta):7.2f}°\n"
                f"  Yaw:   {np.degrees(psi):7.2f}°")


if __name__ == "__main__":
    # Test quaternion functionality
    print("=== Quaternion Tests ===\n")

    # Test 1: Identity quaternion
    q_identity = Quaternion()
    print("1. Identity quaternion:")
    print(q_identity)
    print()

    # Test 2: Create from Euler angles
    phi, theta, psi = np.radians([10, 5, 15])  # Roll, pitch, yaw
    q = Quaternion.from_euler_angles(phi, theta, psi)
    print("2. Quaternion from Euler angles (10°, 5°, 15°):")
    print(q)
    print()

    # Test 3: Convert back to Euler angles
    phi_back, theta_back, psi_back = q.to_euler_angles()
    print("3. Converted back to Euler:")
    print(f"  Roll:  {np.degrees(phi_back):7.2f}° (should be 10°)")
    print(f"  Pitch: {np.degrees(theta_back):7.2f}° (should be 5°)")
    print(f"  Yaw:   {np.degrees(psi_back):7.2f}° (should be 15°)")
    print()

    # Test 4: Rotation matrix
    R = q.to_rotation_matrix()
    print("4. Rotation matrix:")
    print(R)
    print()

    # Test 5: Rotate a vector
    v_inertial = np.array([1.0, 0.0, 0.0])
    v_body = q.rotate_vector(v_inertial)
    print(f"5. Rotate [1,0,0] from inertial to body:")
    print(f"  Result: {v_body}")
    print()

    # Test 6: Quaternion multiplication
    q2 = Quaternion.from_euler_angles(0, 0, np.radians(45))
    q_combined = q * q2
    print("6. Quaternion multiplication (compound rotation):")
    print(q_combined)
