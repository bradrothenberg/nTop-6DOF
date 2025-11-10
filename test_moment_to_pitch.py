"""
Trace negative pitch moment through dynamics to verify sign propagation.

We know:
- Pitch moment M_y = -4832 lb-ft (NOSE DOWN)
- Expected: pitch rate should be negative (nose down)
- Expected: pitch angle should decrease

Let's trace through the full chain.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.quaternion import Quaternion

print("=" * 70)
print("Moment to Pitch Sign Propagation Test")
print("=" * 70)
print()

# Setup
mass = 228.924806
inertia = np.array([[19236.2914, 0, 0],
                    [0, 2251.0172, 0],
                    [0, 0, 21487.3086]])

dynamics = AircraftDynamics(mass, inertia)

# Initial state
state = State()
state.position = np.array([0.0, 0.0, -5000.0])
state.velocity_body = np.array([548.5, 0.0, 16.8])  # V=548.5, alpha=1.75°
state.set_euler_angles(0.0, np.radians(1.75), 0.0)
state.angular_rates = np.array([0.0, 0.0, 0.0])

print("Initial State:")
phi, theta, psi = state.euler_angles
print(f"  Pitch angle: {np.degrees(theta):.4f} deg")
print(f"  Pitch rate: {state.q_rate:.6f} rad/s")
print(f"  Quaternion: {state.q.q}")
print()

# Apply negative pitch moment
moments = np.array([0.0, -4832.0, 0.0])  # NOSE DOWN

print("Applied Moment:")
print(f"  M_y = {moments[1]:.1f} lb-ft (NEGATIVE = NOSE DOWN)")
print()

# Step 1: Compute omega_dot from moments
print("Step 1: Moments -> Angular Acceleration")
omega = state.angular_rates
I_omega = inertia @ omega
omega_cross_I_omega = np.cross(omega, I_omega)
omega_dot = np.linalg.inv(inertia) @ (moments - omega_cross_I_omega)

print(f"  omega = {omega}")
print(f"  I * omega = {I_omega}")
print(f"  omega x (I*omega) = {omega_cross_I_omega}")
print(f"  M - omega x (I*omega) = {moments - omega_cross_I_omega}")
print(f"  I^-1 = diag([{np.linalg.inv(inertia)[0,0]:.8f}, {np.linalg.inv(inertia)[1,1]:.8f}, {np.linalg.inv(inertia)[2,2]:.8f}])")
print(f"  omega_dot = I^-1 @ (M - ...) = {omega_dot}")
print(f"  pitch_accel (q_dot) = {omega_dot[1]:.6f} rad/s^2")
print()

if omega_dot[1] < 0:
    print("  Check: CORRECT sign (negative = nose down acceleration)")
else:
    print("  Check: WRONG sign (should be negative!)")
print()

# Step 2: Integrate omega for a small timestep
dt = 0.01
omega_new = omega + omega_dot * dt

print(f"Step 2: After dt={dt}s integration")
print(f"  omega_new = omega + omega_dot * dt")
print(f"  omega_new = {omega_new}")
print(f"  pitch_rate_new (q) = {omega_new[1]:.6f} rad/s")
print()

if omega_new[1] < 0:
    print("  Check: CORRECT sign (negative = pitching down)")
else:
    print("  Check: WRONG sign (should be negative!)")
print()

# Step 3: Compute quaternion derivative
print("Step 3: Angular Rates -> Quaternion Derivative")
p, q_rate, r = omega_new

Omega = np.array([
    [0,  -p,  -q_rate,  -r],
    [p,   0,   r,  -q_rate],
    [q_rate,  -r,   0,   p],
    [r,   q_rate,  -p,   0]
])

print(f"  Omega matrix:")
print(f"    {Omega[0]}")
print(f"    {Omega[1]}")
print(f"    {Omega[2]}")
print(f"    {Omega[3]}")
print()

q_dot_array = 0.5 * Omega @ state.q.q

print(f"  q_dot = 0.5 * Omega @ q = {q_dot_array}")
print()

# Step 4: Integrate quaternion
q_new_array = state.q.q + q_dot_array * dt
q_new = Quaternion(q_new_array)

print(f"Step 4: Quaternion Integration")
print(f"  q_new = q + q_dot * dt = {q_new.q}")
print()

# Step 5: Extract pitch angle
phi_new, theta_new, psi_new = q_new.to_euler_angles()

print(f"Step 5: Quaternion -> Euler Angles")
print(f"  Initial pitch: {np.degrees(theta):.6f} deg")
print(f"  New pitch: {np.degrees(theta_new):.6f} deg")
print(f"  Change: {np.degrees(theta_new - theta):.6f} deg")
print()

if theta_new < theta:
    print("  Check: CORRECT - Pitch decreased (nose down)")
else:
    print("  Check: WRONG - Pitch increased (should decrease!)")
print()

# Now let's check what the Euler angle extraction does
print("Step 6: Detailed Euler Angle Extraction")
q0, q1, q2, q3 = q_new.q

print(f"  q0={q0:.8f}, q1={q1:.8f}, q2={q2:.8f}, q3={q3:.8f}")

sin_theta = 2*(q0*q2 - q3*q1)
print(f"  sin_theta = 2*(q0*q2 - q3*q1) = 2*({q0:.8f}*{q2:.8f} - {q3:.8f}*{q1:.8f})")
print(f"  sin_theta = 2*({q0*q2:.8f} - {q3*q1:.8f})")
print(f"  sin_theta = {sin_theta:.8f}")

theta_extracted = np.arcsin(np.clip(sin_theta, -1.0, 1.0))
print(f"  theta = arcsin(sin_theta) = {np.degrees(theta_extracted):.6f} deg")
print()

print("=" * 70)
print("CONCLUSION:")
print("=" * 70)

if theta_new < theta:
    print("The sign propagation is CORRECT throughout the chain.")
    print("The issue must be elsewhere in the simulation loop.")
else:
    print("Found sign error! Pitch increases when it should decrease.")
    print("The error is in one of the steps above.")
print()
