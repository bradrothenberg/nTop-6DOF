"""
Debug the VERY FIRST timestep of the Mach 0.5 test.
Check what forces/moments are being computed and what accelerations result.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import PropellerModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere

print("=" * 70)
print("First Timestep Diagnostic - Mach 0.5 Trim Test")
print("=" * 70)
print()

# Aircraft parameters
mass = 228.924806
inertia = np.array([[19236.2914, 0, 0],
                    [0, 2251.0172, 0],
                    [0, 0, 21487.3086]])

S_ref = 412.6370
c_ref = 11.9555
b_ref = 24.8630

dynamics = AircraftDynamics(mass, inertia)
aero = LinearAeroModel(S_ref, c_ref, b_ref)

# Flying wing AVL derivatives
aero.CL_0 = 0.000023
aero.CL_alpha = 1.412241
aero.CL_q = 1.282202

aero.CD_0 = -0.000619
aero.CD_alpha = 0.035509
aero.CD_alpha2 = 0.5

aero.Cm_0 = 0.000061
aero.Cm_alpha = -0.079668
aero.Cm_q = -0.347072

prop = PropellerModel(power_max=50.0, prop_diameter=6.0, prop_efficiency=0.75)
combined_model = CombinedForceModel(aero, prop)

# Initial state at supposed trim
altitude = -5000.0
airspeed = 548.5
alpha_trim = np.radians(1.75)

atm = StandardAtmosphere(altitude)
aero.rho = atm.density

state = State()
state.position = np.array([0.0, 0.0, altitude])
state.velocity_body = np.array([
    airspeed * np.cos(alpha_trim),
    0.0,
    airspeed * np.sin(alpha_trim)
])
state.set_euler_angles(0.0, alpha_trim, 0.0)
state.angular_rates = np.array([0.0, 0.0, 0.0])

print("Initial State:")
print(f"  Position: {state.position}")
print(f"  Altitude: {-altitude} ft")
print(f"  Velocity (body): {state.velocity_body}")
print(f"  Airspeed: {state.airspeed:.2f} ft/s")
print(f"  Alpha: {np.degrees(state.alpha):.4f} deg")
print(f"  Pitch: {np.degrees(state.euler_angles[1]):.4f} deg")
print(f"  Angular rates: {state.angular_rates}")
print()

# Control inputs
throttle = 0.318
controls = {'throttle': throttle, 'elevator': 0.0}

# Compute forces and moments
forces, moments = combined_model(state, throttle, controls)

print("Forces and Moments:")
print(f"  Forces (body): {forces} lbf")
print(f"  Moments (body): {moments} ft-lbf")
print()

# Expected values
q_bar = 0.5 * atm.density * airspeed**2
CL = aero.CL_0 + aero.CL_alpha * alpha_trim
L_expected = q_bar * S_ref * CL
W = mass * 32.174

print(f"Expected vs Actual:")
print(f"  Weight: {W:.1f} lbf")
print(f"  Lift (expected): {L_expected:.1f} lbf")
print(f"  Fz (actual): {forces[2]:.1f} lbf")
print(f"  L/W: {L_expected/W:.6f}")
print()

# Compute state derivative
def force_func(s):
    return combined_model(s, throttle, controls)

state_dot = dynamics.state_derivative(state, force_func)

vel_dot = state_dot[3:6]
omega_dot = state_dot[10:13]

print("Accelerations:")
print(f"  Linear accel (body): {vel_dot} ft/s^2")
print(f"  ax: {vel_dot[0]:.4f} ft/s^2")
print(f"  ay: {vel_dot[1]:.4f} ft/s^2")
print(f"  az: {vel_dot[2]:.4f} ft/s^2")
print()
print(f"  Angular accel (body): {omega_dot} rad/s^2")
print(f"  p_dot: {omega_dot[0]:.6f} rad/s^2")
print(f"  q_dot: {omega_dot[1]:.6f} rad/s^2 = {np.degrees(omega_dot[1]):.4f} deg/s^2")
print(f"  r_dot: {omega_dot[2]:.6f} rad/s^2")
print()

# Check trim
if abs(vel_dot[2]) < 1.0 and abs(omega_dot[1]) < 0.01:
    print("STATUS: AT TRIM (accelerations near zero)")
else:
    print("STATUS: NOT AT TRIM!")
    if abs(vel_dot[2]) >= 1.0:
        print(f"  - Large vertical accel: {vel_dot[2]:.2f} ft/s^2")
    if abs(omega_dot[1]) >= 0.01:
        print(f"  - Large pitch accel: {np.degrees(omega_dot[1]):.2f} deg/s^2")
print()

# Check if pitch accel will cause pitch up or down
dt = 0.01
omega_new = state.angular_rates + omega_dot * dt
q_new = omega_new[1]

print(f"After dt = {dt}s:")
print(f"  Pitch rate: 0 -> {q_new:.6f} rad/s = {np.degrees(q_new):.4f} deg/s")

if q_new > 0:
    print("  Pitch rate is POSITIVE -> Aircraft will pitch UP (nose up)")
elif q_new < 0:
    print("  Pitch rate is NEGATIVE -> Aircraft will pitch DOWN (nose down)")
else:
    print("  Pitch rate is ZERO -> No change")
print()

# Detailed breakdown
print("Detailed Force Balance:")
g_inertial = np.array([0, 0, 32.174])
R = state.q.to_rotation_matrix()
g_body = R @ g_inertial
omega_cross_v = np.cross(state.angular_rates, state.velocity_body)

accel_from_forces = forces / mass
accel_from_omega = -omega_cross_v
accel_from_gravity = -g_body

print(f"  Gravity (body): {g_body}")
print(f"  From forces/mass: {accel_from_forces}")
print(f"  From -omega x v: {accel_from_omega}")
print(f"  From -g_body: {accel_from_gravity}")
print(f"  Total: {vel_dot}")
print()

# Check vertical force balance in body frame
print("Vertical Force Balance (Z-axis in body frame):")
print(f"  Fz (aero + thrust): {forces[2]:.1f} lbf")
print(f"  Weight in body frame: m*g_z = {mass * g_body[2]:.1f} lbf")
print(f"  Net force: {forces[2] - mass * g_body[2]:.1f} lbf")
print(f"  Resulting accel: {vel_dot[2]:.2f} ft/s^2")
print()

if abs(vel_dot[2]) > 1.0:
    print("DIAGNOSIS: Large vertical acceleration means NOT at trim!")
    print("Either:")
    print("  1. Lift is not equal to weight component")
    print("  2. Thrust vertical component is wrong")
    print("  3. Angle of attack is wrong")

print()
print("=" * 70)
