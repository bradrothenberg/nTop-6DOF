"""
Check rotation matrix convention.
"""
import numpy as np
from src.core.state import State

# Pitch up by 10 degrees (easier to visualize than 1.75 deg)
theta = np.radians(10)

state = State()
state.set_euler_angles(0, theta, 0)

# Test vector: point forward in NED frame
forward_ned = np.array([1, 0, 0])

# Transform to body frame
R = state.q.to_rotation_matrix()
forward_body = R @ forward_ned

print("=" * 70)
print("Rotation Matrix Convention Check")
print("=" * 70)
print()
print(f"Pitch angle: {np.degrees(theta):.1f} deg (nose up)")
print()
print("Test 1: Forward vector")
print(f"  NED frame: {forward_ned}")
print(f"  Body frame: {forward_body}")
print()

if forward_body[0] > 0 and forward_body[2] < 0:
    print("  Correct! Forward NED becomes forward+up in body (nose up)")
elif forward_body[0] > 0 and forward_body[2] > 0:
    print("  Hmm, forward NED becomes forward+down in body")
else:
    print(f"  Unexpected transformation")
print()

# Test vector: point down in NED frame (gravity)
down_ned = np.array([0, 0, 1])
down_body = R @ down_ned

print("Test 2: Down vector (gravity direction)")
print(f"  NED frame: {down_ned}")
print(f"  Body frame: {down_body}")
print()

# For nose-up pitch:
# - Gravity points down in NED
# - In body frame, should have negative X component (pulls back/down the nose)
# - And positive Z component (still mostly downward)

if down_body[0] < 0 and down_body[2] > 0:
    print("  Expected! Down NED becomes backward+down in body")
    print("  This means gravity pulls BACKWARD when nose is up")
elif down_body[0] > 0 and down_body[2] > 0:
    print("  Unexpected! Down NED becomes forward+down in body")
    print("  This would mean gravity pulls FORWARD when nose is up (wrong!)")
print()

# Let's manually compute the expected transformation
# Pitch rotation (nose up by theta, positive rotation about Y-axis):
# Standard aerospace convention (right-hand rule, Y points right):
# [ cos(θ)   0  sin(θ) ]
# [   0      1    0    ]
# [-sin(θ)   0  cos(θ) ]

R_expected = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])

print("Expected rotation matrix (standard aerospace):")
print(R_expected)
print()

print("Actual rotation matrix:")
print(R)
print()

print("Difference:")
print(R - R_expected)
print()

# Apply expected rotation to gravity
grav_ned = np.array([0, 0, 1])
grav_body_expected = R_expected @ grav_ned

print("Gravity transformation (expected):")
print(f"  NED: {grav_ned}")
print(f"  Body: {grav_body_expected}")
print(f"  gx = {grav_body_expected[0]:.4f}")
print(f"  gz = {grav_body_expected[2]:.4f}")
print()

if grav_body_expected[0] < 0:
    print("  gx is NEGATIVE: Gravity pulls backward (correct for nose-up!)")
elif grav_body_expected[0] > 0:
    print("  gx is POSITIVE: Gravity pulls forward (wrong for nose-up!)")

print()
print("=" * 70)
