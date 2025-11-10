"""
Debug trim solver - check if forces are being computed correctly.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import PropellerModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere

# Setup
mass = 228.924806
inertia = np.array([[19236.2914, 0, 0],
                    [0, 2251.0172, 0],
                    [0, 0, 21487.3086]])

S_ref = 412.6370
c_ref = 11.9555
b_ref = 24.8630

dynamics = AircraftDynamics(mass, inertia)
aero = LinearAeroModel(S_ref, c_ref, b_ref)

# Flying wing derivatives
aero.CL_0 = 0.000023
aero.CL_alpha = 1.412241

aero.CD_0 = -0.000619
aero.CD_alpha = 0.035509
aero.CD_alpha2 = 0.5

aero.Cm_0 = 0.000061
aero.Cm_alpha = -0.079668

prop = PropellerModel(power_max=50.0)
combined = CombinedForceModel(aero, prop)

# Test at Mach 0.5
altitude = -5000.0
airspeed = 548.5
alpha = np.radians(2.35)  # Expected trim alpha

atm = StandardAtmosphere(altitude)
aero.rho = atm.density

print("=" * 70)
print("Trim Solver Debug")
print("=" * 70)
print()
print(f"Altitude: {-altitude:.0f} ft")
print(f"Airspeed: {airspeed:.1f} ft/s (Mach 0.5)")
print(f"Alpha: {np.degrees(alpha):.2f} deg")
print(f"Density: {atm.density:.6f} slug/ft³")
print()

# Create state
state = State()
state.position = np.array([0.0, 0.0, altitude])
state.velocity_body = np.array([
    airspeed * np.cos(alpha),
    0.0,
    airspeed * np.sin(alpha)
])
state.set_euler_angles(0.0, alpha, 0.0)  # theta = alpha for level flight
state.angular_rates = np.array([0.0, 0.0, 0.0])

print(f"State:")
print(f"  velocity_body: {state.velocity_body}")
print(f"  alpha: {np.degrees(state.alpha):.2f} deg")
print(f"  beta: {np.degrees(state.beta):.2f} deg")
print(f"  airspeed: {state.airspeed:.1f} ft/s")
print()

# Compute forces
controls = {'throttle': 0.3, 'elevator': 0.0}
forces, moments = combined(state, controls['throttle'], controls)

print(f"Forces (body frame):")
print(f"  Fx: {forces[0]:.1f} lbf")
print(f"  Fy: {forces[1]:.1f} lbf")
print(f"  Fz: {forces[2]:.1f} lbf")
print()

# Calculate expected lift
q_bar = 0.5 * atm.density * airspeed**2
CL = aero.CL_0 + aero.CL_alpha * alpha
L_expected = q_bar * S_ref * CL
W = mass * 32.174

print(f"Lift calculation:")
print(f"  q_bar: {q_bar:.2f} psf")
print(f"  CL_0: {aero.CL_0:.6f}")
print(f"  CL_alpha * alpha: {aero.CL_alpha * alpha:.6f}")
print(f"  CL_total: {CL:.6f}")
print(f"  Lift expected: {L_expected:.0f} lbf")
print(f"  Weight: {W:.0f} lbf")
print(f"  Lift/Weight: {L_expected/W:.4f}")
print()

# Check accelerations
def force_func(s):
    return combined(s, controls['throttle'], controls)

state_dot = dynamics.state_derivative(state, force_func)
vel_dot = state_dot[3:6]

print(f"Accelerations:")
print(f"  ax (body): {vel_dot[0]:.2f} ft/s²")
print(f"  ay (body): {vel_dot[1]:.2f} ft/s²")
print(f"  az (body): {vel_dot[2]:.2f} ft/s² (should be near 0 for trim)")
print()

# Detailed breakdown
g_inertial = np.array([0, 0, 32.174])
R = state.q.to_rotation_matrix()
g_body = R @ g_inertial
omega_cross_v = np.cross(state.angular_rates, state.velocity_body)

accel_from_forces = forces / mass
accel_from_omega = -omega_cross_v
accel_from_gravity = -g_body

print(f"Acceleration breakdown:")
print(f"  From forces/mass: {accel_from_forces}")
print(f"  From omega x v: {accel_from_omega}")
print(f"  From gravity (-g_body): {accel_from_gravity}")
print(f"  Total: {vel_dot}")
print()

print(f"Gravity in body frame: {g_body}")
print(f"  (Should be near [0, 0, +32] for small pitch)")
print()

if abs(vel_dot[2]) < 1.0:
    print("SUCCESS: Vertical acceleration near zero!")
else:
    print("PROBLEM: Large vertical acceleration")
    print(f"  Expected: Fz/m - g_z ≈ 0")
    print(f"  Actual: {forces[2]/mass:.2f} - {g_body[2]:.2f} = {vel_dot[2]:.2f} ft/s²")
