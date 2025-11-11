"""
Check acceleration in NED (inertial) frame.

For steady level flight, we need a_NED = 0, not a_body = 0!
"""
import numpy as np
from src.core.state import State

# Current "trim" values (from test_first_timestep.py)
theta = np.radians(1.7531)
ax_body = 1.9688  # ft/s²
az_body = -0.0207  # ft/s²

print("=" * 70)
print("NED Frame Acceleration Check")
print("=" * 70)
print()

print(f"Body frame:")
print(f"  Pitch: {np.degrees(theta):.4f} deg")
print(f"  ax (forward): {ax_body:.4f} ft/s²")
print(f"  az (down): {az_body:.4f} ft/s²")
print()

# Create state
state = State()
state.set_euler_angles(0, theta, 0)

# Transform to NED
R_body_to_ned = state.q.to_rotation_matrix().T  # Transpose for body->NED
a_body = np.array([ax_body, 0, az_body])
a_ned = R_body_to_ned @ a_body

print(f"NED frame:")
print(f"  ax (north): {a_ned[0]:.4f} ft/s²")
print(f"  ay (east): {a_ned[1]:.4f} ft/s²")
print(f"  az (down): {a_ned[2]:.4f} ft/s²")
print()

# For level flight, we want az_ned ≈ 0 (no vertical accel)
# and ax_ned ≈ 0 (no horizontal accel)

if abs(a_ned[2]) < 0.1:
    print("✓ Vertical acceleration near zero (good!)")
else:
    print(f"✗ Vertical acceleration: {a_ned[2]:.4f} ft/s² (should be ~0)")

if abs(a_ned[0]) < 0.1:
    print("✓ Horizontal acceleration near zero (good!)")
else:
    print(f"✗ Horizontal acceleration: {a_ned[0]:.4f} ft/s² (should be ~0)")

print()

# Let's also check what body acceleration would give zero NED acceleration
# For az_ned = 0:
# az_ned = ax_body * (-sin(θ)) + az_body * cos(θ) = 0
# az_body = ax_body * tan(θ)

print("For zero NED acceleration:")
az_body_needed = ax_body * np.tan(theta)
print(f"  If ax_body = {ax_body:.4f} ft/s²")
print(f"  Then az_body should be: {az_body_needed:.4f} ft/s²")
print(f"  Actual az_body: {az_body:.4f} ft/s²")
print(f"  Match? {abs(az_body - az_body_needed) < 0.1}")
print()

print("=" * 70)
