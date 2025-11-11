"""
Check velocity direction in NED frame.
"""
import numpy as np
from src.core.state import State

alpha = np.radians(1.7531)
theta = alpha  # Level flight
V = 548.5

# Velocity in body frame
vx_body = V * np.cos(alpha)
vz_body = V * np.sin(alpha)
v_body = np.array([vx_body, 0, vz_body])

print("=" * 70)
print("Velocity Direction Check")
print("=" * 70)
print()

print(f"Body frame:")
print(f"  Alpha: {np.degrees(alpha):.4f} deg")
print(f"  Theta: {np.degrees(theta):.4f} deg")
print(f"  Velocity: [{vx_body:.2f}, 0, {vz_body:.2f}] ft/s")
print(f"  Magnitude: {np.linalg.norm(v_body):.2f} ft/s")
print()

# Transform to NED
state = State()
state.set_euler_angles(0, theta, 0)
R_body_to_ned = state.q.to_rotation_matrix().T
v_ned = R_body_to_ned @ v_body

print(f"NED frame:")
print(f"  Velocity: [{v_ned[0]:.2f}, {v_ned[1]:.2f}, {v_ned[2]:.2f}] ft/s")
print(f"  Magnitude: {np.linalg.norm(v_ned):.2f} ft/s")
print()

# Flight path angle
gamma = np.arctan2(v_ned[2], v_ned[0])
print(f"Flight path angle: {np.degrees(gamma):.4f} deg")
print()

if abs(gamma) < 0.1:
    print("OK: Flight path is nearly horizontal (level flight)")
else:
    print(f"WARNING: Flight path has {np.degrees(gamma):.2f} deg climb/dive")

print()
print("=" * 70)
