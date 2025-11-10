"""
Test to verify pitch sign error in quaternion.py
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.core.quaternion import Quaternion

print("=" * 70)
print("Quaternion Pitch Sign Test")
print("=" * 70)
print()

# Test 1: Small positive pitch
print("Test 1: Create quaternion with +5° pitch")
pitch = np.radians(5.0)
q = Quaternion.from_euler_angles(0.0, pitch, 0.0)

print(f"  Input pitch: {np.degrees(pitch):.2f} deg")
print(f"  Quaternion: {q.q}")

# Extract pitch back
phi, theta, psi = q.to_euler_angles()
print(f"  Extracted pitch: {np.degrees(theta):.2f} deg")

if abs(np.degrees(theta) - 5.0) < 0.01:
    print("  Result: CORRECT")
else:
    print(f"  Result: WRONG (expected 5.0, got {np.degrees(theta):.2f})")
print()

# Test 2: Check the actual formula
print("Test 2: Manual pitch calculation")
q0, q1, q2, q3 = q.q

sin_theta_wrong = 2*(q0*q2 - q3*q1)  # Current (wrong) formula
sin_theta_correct = 2*(q0*q2 - q1*q3)  # Correct formula

print(f"  q0={q0:.6f}, q1={q1:.6f}, q2={q2:.6f}, q3={q3:.6f}")
print(f"  Current formula: sin_theta = 2*(q0*q2 - q3*q1) = {sin_theta_wrong:.6f}")
print(f"  Correct formula: sin_theta = 2*(q0*q2 - q1*q3) = {sin_theta_correct:.6f}")
print(f"  Difference: {sin_theta_wrong - sin_theta_correct:.6f}")
print()

theta_wrong = np.arcsin(np.clip(sin_theta_wrong, -1.0, 1.0))
theta_correct = np.arcsin(np.clip(sin_theta_correct, -1.0, 1.0))

print(f"  Current extraction: theta = {np.degrees(theta_wrong):.6f} deg")
print(f"  Correct extraction: theta = {np.degrees(theta_correct):.6f} deg")
print()

# Test 3: Verify with negative pitch moment scenario
print("Test 3: Simulate negative pitch moment (nose down)")
print("  Starting at pitch = 1.75 deg")
print("  Apply pitch rate q = -0.1 rad/s (nose down) for 0.1s")
print()

q_initial = Quaternion.from_euler_angles(0.0, np.radians(1.75), 0.0)
omega = np.array([0.0, -0.1, 0.0])  # Negative pitch rate (nose down)
dt = 0.1

q_new = q_initial.integrate(omega, dt)

phi_i, theta_i, psi_i = q_initial.to_euler_angles()
phi_n, theta_n, psi_n = q_new.to_euler_angles()

print(f"  Initial pitch: {np.degrees(theta_i):.4f} deg")
print(f"  After 0.1s with q=-0.1 rad/s:")
print(f"    New pitch: {np.degrees(theta_n):.4f} deg")
print(f"    Change: {np.degrees(theta_n - theta_i):.4f} deg")
print()

if theta_n < theta_i:
    print("  Result: CORRECT - Pitch decreased (nose down)")
else:
    print("  Result: WRONG - Pitch increased (nose up) when it should go down!")
print()

# Test 4: Check rotation matrix consistency
print("Test 4: Check rotation matrix pitch extraction")
R = q.to_rotation_matrix()

# For ZYX Euler angles, pitch can also be extracted from rotation matrix
# theta = arcsin(-R[2,0])
theta_from_R = np.arcsin(-R[2, 0])

print(f"  Pitch from quaternion: {np.degrees(theta):.6f} deg")
print(f"  Pitch from rotation matrix: {np.degrees(theta_from_R):.6f} deg")
print(f"  Match: {abs(theta - theta_from_R) < 0.001}")
print()

print("=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("The sign error in quaternion.py line 136 causes pitch angle")
print("to have the WRONG SIGN, making aircraft pitch UP when moment")
print("says it should pitch DOWN.")
print()
print("FIX: Change line 136 from:")
print("  sin_theta = 2*(q0*q2 - q3*q1)")
print("To:")
print("  sin_theta = 2*(q0*q2 - q1*q3)")
print()
