"""
Check how gravity transforms to body frame.
"""
import numpy as np
from src.core.state import State

# Create state at pitch angle theta
theta = np.radians(1.7531)

state = State()
state.set_euler_angles(0, theta, 0)

# Gravity in NED frame (down is positive)
g_ned = np.array([0, 0, 32.174])

# Transform to body frame
R = state.q.to_rotation_matrix()
g_body = R @ g_ned

print("=" * 70)
print("Gravity Transformation Check")
print("=" * 70)
print()
print(f"Pitch angle: {np.degrees(theta):.4f} deg")
print()
print(f"Gravity (NED frame): {g_ned}")
print(f"Gravity (body frame): {g_body}")
print()
print(f"  gx (forward): {g_body[0]:.4f} ft/s²")
print(f"  gy (right): {g_body[1]:.4f} ft/s²")
print(f"  gz (down): {g_body[2]:.4f} ft/s²")
print()

# For small angle approximation:
gx_approx = 32.174 * np.sin(theta)
gz_approx = 32.174 * np.cos(theta)

print("Small angle approximation:")
print(f"  gx ≈ g*sin(θ) = {gx_approx:.4f} ft/s²")
print(f"  gz ≈ g*cos(θ) = {gz_approx:.4f} ft/s²")
print()

# Check sign
if g_body[0] > 0:
    print("gx is POSITIVE: Gravity pulls forward in body frame")
    print("This is correct for nose-up pitch!")
    print("When pitched nose-up, gravity has a component pulling the aircraft forward (down the slope)")
else:
    print("gx is NEGATIVE: Gravity pulls backward in body frame")
    print("This would be for nose-down pitch")

print()
print("=" * 70)
