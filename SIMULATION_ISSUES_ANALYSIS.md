# Simulation Numerical Issues - Root Cause Analysis

## Problem Statement

The 6-DOF simulation shows instability despite AVL analysis confirming the aircraft is statically stable:
- **AVL Analysis**: Cm_alpha = -0.080 (stable), +5.6% static margin
- **Simulation**: Aircraft pitches to 90°, loses airspeed (200→10 ft/s), tumbles

This occurs even:
- Without autopilot control
- At calculated trim conditions (alpha ≈ 0°)
- With actual AVL stability derivatives

## Root Causes Identified

### 1. Missing Trim Condition Setup

**Issue**: Aircraft not in force/moment equilibrium at start.

For level flight, we need:
- **Lift = Weight**: `L = W` → `q_bar * S * CL = m * g`
- **Thrust = Drag**: `T = D` → `T = q_bar * S * CD`
- **Pitch Moment = 0**: `M = 0` → `Cm = 0`

**Current State**:
- Initial: V=200 ft/s, alpha=0°, throttle=0.25
- CL at alpha=0°: `CL_0 = 0.000023` (essentially zero)
- Lift generated: `L = 0.5 * 0.002377 * 200^2 * 412.64 * 0.000023 = 1.13 lbf`
- Weight: `W = 228.9 * 32.174 = 7,365 lbf`
- **Lift << Weight** → Aircraft descends and loses speed!

**Solution**: Implement trim solver to find (alpha, theta, throttle, elevator) for equilibrium.

### 2. Euler Angle Singularity (Gimbal Lock)

**Issue**: When pitch approaches ±90°, Euler angles become singular.

The rotation matrix from Euler angles has singularity at theta = ±90°:
```
R = Rz(yaw) * Ry(pitch) * Rx(roll)

At pitch = 90°:
- Roll and yaw axes align
- Infinite solutions for (roll, yaw) give same orientation
- Numerical instability in derivatives
```

**Current Evidence**:
- Pitch reaches 90° in all tests
- Roll jumps to ±180° (wrapping around singularity)
- Angular rates become erratic

**Solution**:
- Use quaternion-only dynamics (already implemented in State class)
- Add safeguards to prevent pitch from exceeding ±85°
- Switch to quaternion state vector in integration

### 3. Numerical Integration Issues

**Issue**: Simple Euler integration with dt=0.01s is insufficient for stiff dynamics.

**Current Implementation** (in dynamics.py):
```python
x_new = x + state_dot * dt  # Simple Euler method
```

**Problems**:
- Euler method is 1st order accurate: O(dt²) local error
- Unstable for stiff equations (high frequency modes)
- Accumulates errors rapidly

**Solution**: Implement RK4 (4th order Runge-Kutta) or adaptive integration.

### 4. Missing Gravity in Dynamics

**Issue**: Weight force may not be properly included in dynamics equations.

Need to verify that gravity appears correctly in force balance:
```python
# In body frame, weight has components based on orientation
W_body = q_transpose @ np.array([0, 0, m*g])
F_total = F_aero + F_prop + W_body
```

**Current State**: Need to check `AircraftDynamics.state_derivative()` implementation.

**Solution**: Ensure gravity is explicitly included in all force calculations.

### 5. Incorrect Initial Conditions

**Issue**: Starting far from trim causes transients that grow due to integration errors.

**Trim Calculation**:
```
For Cm = 0:
Cm_0 + Cm_alpha * alpha_trim = 0
alpha_trim = -0.000061 / (-0.079668) = 0.0008 rad ≈ 0.04°

For L = W:
CL_trim = W / (q_bar * S) = 7365 / (0.5 * 0.002377 * 200^2 * 412.64)
CL_trim = 7365 / 19604 = 0.376

But CL at alpha=0°: CL_0 = 0.000023 ≈ 0
Need: CL_0 + CL_alpha * alpha = 0.376
alpha_needed = (0.376 - 0.000023) / 1.412241 = 0.266 rad = 15.3°!
```

**This is the REAL problem**: At V=200 ft/s, the aircraft needs **15.3° angle of attack** to generate enough lift, not 0°!

**Solution**: Either increase initial velocity or accept higher trim alpha.

## Priority Fixes

### Priority 1: Fix Trim Condition (CRITICAL)

The aircraft needs to start in equilibrium. Two options:

**Option A**: Calculate proper trim speed for alpha ≈ 0°
```
For CL_trim = 0.376 and alpha = 0°:
q_bar_needed = W / (S * CL_0) = 7365 / (412.64 * 0.000023) = 776,000 psf
V_trim = sqrt(2 * q_bar / rho) = sqrt(2 * 776000 / 0.002377) = 808 ft/s!
```
This is too fast (Mach 0.72 at 5000 ft).

**Option B**: Use realistic speed with appropriate alpha
```
At V = 200 ft/s:
alpha_trim = 15.3° (for lift = weight)
theta_trim = alpha_trim + gamma (gamma ≈ 0 for level flight)
theta_trim ≈ 15.3°
```

**Recommended**: Start with V=200 ft/s, theta=15°, throttle adjusted for T=D.

### Priority 2: Implement RK4 Integration

Replace simple Euler with 4th order Runge-Kutta:
```python
def rk4_step(dynamics, state, force_func, dt):
    k1 = dynamics.state_derivative(state, force_func)

    x1 = state.to_array() + 0.5 * dt * k1
    state1 = State()
    state1.from_array(x1)
    k2 = dynamics.state_derivative(state1, force_func)

    x2 = state.to_array() + 0.5 * dt * k2
    state2 = State()
    state2.from_array(x2)
    k3 = dynamics.state_derivative(state2, force_func)

    x3 = state.to_array() + dt * k3
    state3 = State()
    state3.from_array(x3)
    k4 = dynamics.state_derivative(state3, force_func)

    x_new = state.to_array() + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_new
```

### Priority 3: Add Pitch Angle Safeguards

Prevent Euler angle singularity:
```python
# In state update
if abs(pitch) > np.radians(85):
    pitch = np.sign(pitch) * np.radians(85)
    # Recalculate quaternion
```

Or better: use quaternion-only state vector (no Euler angles until output).

### Priority 4: Verify Gravity Implementation

Check that weight force appears in dynamics correctly:
```python
# In AircraftDynamics.state_derivative()
# Should have:
F_gravity_body = q_transpose @ np.array([0, 0, mass * g])
F_total = F_aero + F_prop + F_gravity_body
```

## Implementation Plan

1. **Create trim solver** (`src/simulation/trim.py`)
   - Find equilibrium (V, alpha, theta, throttle, elevator)
   - Use optimization (scipy.optimize)
   - Validate trim is stable

2. **Implement RK4 integrator** (`src/simulation/integrator.py`)
   - 4th order Runge-Kutta
   - Adaptive timestep option
   - Quaternion normalization

3. **Add gravity verification** (check `src/core/dynamics.py`)
   - Ensure weight force is included
   - Validate sign conventions

4. **Create validated test case** (`examples/trimmed_flight_demo.py`)
   - Start from proper trim
   - Use RK4 integration
   - Verify stable trajectory

## Expected Results

After fixes:
- Aircraft maintains altitude (± 50 ft over 60s)
- Maintains airspeed (± 10 ft/s)
- Roll/pitch angles remain small (< 10° deviation)
- Smooth trajectory without divergence

## References

- Stevens & Lewis, "Aircraft Control and Simulation", 2nd ed.
- Cook, "Flight Dynamics Principles", 3rd ed.
- Quaternion attitude dynamics: Shuster (1993)
- RK4 integration: Numerical Recipes in C
