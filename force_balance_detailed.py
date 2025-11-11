"""
Detailed force balance check in multiple reference frames.
"""
import numpy as np

# Trim values
alpha = np.radians(1.7531)
theta = alpha  # Level flight: gamma = 0
V = 548.5  # ft/s
mass = 228.924806  # slugs
g = 32.174  # ft/s²

# Aerodynamic forces (wind frame)
L = 7366.6  # lbf (up, perpendicular to velocity)
D = 159.5  # lbf (backward, opposing velocity)

# Weight
W = mass * g  # lbf

print("=" * 80)
print("Force Balance - Multiple Reference Frames")
print("=" * 80)
print()

# === Wind/Flight-Path Frame ===
print("WIND/FLIGHT-PATH FRAME (along and perp to velocity):")
print(f"  Along velocity: Thrust - Drag = T - {D:.1f} lbf")
print(f"  Perpendicular: Lift - Weight = {L:.1f} - {W:.1f} = {L-W:.1f} lbf")
print(f"  For equilibrium: T = {D:.1f} lbf")
print()

# === Body Frame ===
print("BODY FRAME (forward x, down z):")
print(f"  Alpha = {np.degrees(alpha):.4f} deg")
print()

# Transform aero forces from wind to body frame
Fx_aero = -D * np.cos(alpha) + L * np.sin(alpha)
Fz_aero = -D * np.sin(alpha) - L * np.cos(alpha)

print(f"  Aerodynamic forces:")
print(f"    Fx_aero = -D*cos(α) + L*sin(α) = {Fx_aero:.2f} lbf")
print(f"    Fz_aero = -D*sin(α) - L*cos(α) = {Fz_aero:.2f} lbf")
print()

# Thrust (assumed aligned with body x-axis)
print(f"  Thrust: Fx_thrust = T lbf (forward)")
print()

# === Inertial/NED Frame ===
print("INERTIAL (NED) FRAME:")
print(f"  Pitch angle θ = {np.degrees(theta):.4f} deg")
print()

# Gravity vector (NED: down is +z)
grav_ned = np.array([0, 0, W])
print(f"  Gravity: [0, 0, {W:.1f}] lbf (down)")
print()

# Transform gravity to body frame
# For pitch rotation (nose up by θ):
# xb = xn*cos(θ) + zn*sin(θ)
# zb = -xn*sin(θ) + zn*cos(θ)
Fx_grav_body = W * np.sin(theta)
Fz_grav_body = W * np.cos(theta)

print(f"  Gravity in body frame:")
print(f"    Fx_grav = W*sin(θ) = {Fx_grav_body:.2f} lbf (forward)")
print(f"    Fz_grav = W*cos(θ) = {Fz_grav_body:.2f} lbf (down)")
print()

# === Force Balance in Body Frame ===
print("FORCE BALANCE (Body Frame):")
print(f"  X-direction (forward):")
print(f"    Fx_aero + Fx_thrust + Fx_grav = ax * m")
print(f"    {Fx_aero:.2f} + T + {Fx_grav_body:.2f} = ax * {mass:.2f}")
print(f"    For ax=0: T = {-Fx_aero - Fx_grav_body:.2f} lbf")
print()
print(f"  Z-direction (down):")
print(f"    Fz_aero + Fz_grav = az * m")
print(f"    {Fz_aero:.2f} + {Fz_grav_body:.2f} = az * {mass:.2f}")
print(f"    Net Fz = {Fz_aero + Fz_grav_body:.2f} lbf")
print(f"    az = {(Fz_aero + Fz_grav_body)/mass:.4f} ft/s²")
print()

# === Wait, check dynamics equation ===
print("=" * 80)
print("CHECKING DYNAMICS IMPLEMENTATION")
print("=" * 80)
print()
print("The dynamics code uses:")
print("  vel_body_dot = forces/mass - omega_cross_v + g_body")
print()
print("Where 'forces' are aero + thrust, and 'g_body' is gravity ACCELERATION")
print("not gravity FORCE.")
print()

gx = g * np.sin(theta)
gz = g * np.cos(theta)

print(f"Gravity acceleration in body frame:")
print(f"  gx = g*sin(θ) = {gx:.4f} ft/s²")
print(f"  gz = g*cos(θ) = {gz:.4f} ft/s²")
print()

print(f"So the x-acceleration is:")
print(f"  ax = (Fx_aero + Fx_thrust)/m + gx")
print(f"  ax = ({Fx_aero:.2f} + T)/{mass:.2f} + {gx:.4f}")
print()

print(f"For ax=0:")
print(f"  (Fx_aero + T)/m + gx = 0")
print(f"  T = -m*gx - Fx_aero")
print(f"  T = -{mass:.2f}*{gx:.4f} - {Fx_aero:.2f}")
print(f"  T = {-mass*gx:.2f} - {Fx_aero:.2f}")
print(f"  T = {-mass*gx - Fx_aero:.2f} lbf")
print()

print("=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
if (-mass*gx - Fx_aero) < 0:
    print("Negative thrust required! This means:")
    print("1. The aerodynamic force Fx_aero is too large (too much lift*sin(alpha))")
    print("2. OR the gravity term is pulling us forward too much")
    print("3. OR there's an error in how forces are calculated/transformed")
print()

# Let me check what happens if we DON'T include gx (old dynamics before fix)
print("What if we ignored gravity (old buggy dynamics)?")
T_no_grav = -Fx_aero
print(f"  T = -Fx_aero = -{Fx_aero:.2f} = {T_no_grav:.2f} lbf")
print(f"  That's also negative!")
print()

# What if Fx_aero had the wrong sign?
print("What if Fx_aero sign is wrong?")
Fx_aero_flipped = -Fx_aero
T_flipped = -mass*gx - Fx_aero_flipped
print(f"  Fx_aero = {Fx_aero_flipped:.2f} lbf")
print(f"  T = {T_flipped:.2f} lbf")
print(f"  Close to D = {D:.1f} lbf? {abs(T_flipped - D) < 10}")
print()

print("=" * 80)
