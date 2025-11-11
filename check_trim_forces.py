"""
Quick check of force balance at trim condition.
"""
import numpy as np

# Trim values from FJ-44 test
alpha_deg = 1.7531
alpha = np.radians(alpha_deg)
V = 548.5  # ft/s
rho = 0.002745  # slug/ft^3
S = 412.6370  # ft^2
q_bar = 0.5 * rho * V**2

# Aerodynamic coefficients at trim
CL = 0.043235
CD = 0.000936

# Forces in wind frame
L = q_bar * S * CL
D = q_bar * S * CD

print("=" * 70)
print("Force Balance Check")
print("=" * 70)
print()
print(f"Alpha: {alpha_deg:.4f} deg = {alpha:.6f} rad")
print(f"q_bar: {q_bar:.2f} psf")
print()
print(f"CL: {CL:.6f}")
print(f"CD: {CD:.6f}")
print()
print(f"Lift: {L:.1f} lbf")
print(f"Drag: {D:.1f} lbf")
print()

# Transform to body frame (aero only)
Fx_aero = -D * np.cos(alpha) + L * np.sin(alpha)
Fz_aero = -D * np.sin(alpha) - L * np.cos(alpha)

print("Aerodynamic forces (body frame):")
print(f"  Fx_aero: {Fx_aero:.2f} lbf")
print(f"  Fz_aero: {Fz_aero:.2f} lbf")
print()

# For equilibrium in x-direction:
# Fx_total = Fx_aero + Thrust - m*g*sin(theta) ≈ 0
# At small angles with theta = alpha:
# Thrust ≈ D*cos(alpha) - L*sin(alpha) = -Fx_aero

Thrust_needed = -Fx_aero
print(f"Thrust needed for ax=0: {Thrust_needed:.2f} lbf")
print(f"But drag in wind frame is: {D:.2f} lbf")
print()

# The issue is that for trim, we need thrust = drag in WIND FRAME
# But we're setting thrust to match drag, then transforming to body frame
# where the lift component adds a forward force!

print("DIAGNOSIS:")
print(f"  In wind frame: T = D = {D:.1f} lbf (balanced)")
print(f"  In body frame: Fx_aero = {Fx_aero:.1f} lbf (forward)")
print(f"  In body frame: Thrust = {D:.1f} lbf (forward)")
print(f"  Total Fx = {Fx_aero + D:.1f} lbf (unbalanced!)")
print()

# For small angle approximation:
# Fx ≈ -D + L*alpha
Fx_approx = -D + L * alpha
print(f"Small angle approximation: Fx ≈ -D + L*alpha = {Fx_approx:.1f} lbf")
print()

# The correct approach: Thrust should equal drag, but we also need to account
# for the gravity term in the x-direction!
mass = 228.924806  # slugs
W = mass * 32.174  # lbf
theta = alpha  # level flight

# In body frame, x-direction force balance:
# Fx_aero + Thrust - m*g*sin(theta) = 0
# Thrust = -Fx_aero + m*g*sin(theta)

gravity_x = W * np.sin(theta)
Thrust_correct = -Fx_aero + gravity_x

print(f"Gravity component in x: {gravity_x:.2f} lbf")
print(f"Thrust needed (corrected): {Thrust_correct:.2f} lbf")
print()

print("=" * 70)
