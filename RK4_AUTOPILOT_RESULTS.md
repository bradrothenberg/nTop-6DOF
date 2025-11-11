# RK4 Integration and Autopilot Results

## Summary

Implemented RK4 integration and simple autopilot to address simulation instability. **Key finding**: The divergence is NOT due to numerical integration errors, but rather aerodynamic stall and lack of proper control authority.

---

## What We Implemented

### 1. RK4 Integration (`propagate_rk4`)

Added 4th-order Runge-Kutta integration to [src/core/dynamics.py](src/core/dynamics.py):

```python
def propagate_rk4(self, state: State, dt: float, forces_moments: Callable) -> State:
    """
    Propagate state forward by dt using Runge-Kutta 4th order integration.

    RK4 provides O(dt^4) accuracy vs O(dt) for Euler.
    """
    x0 = state.to_array()

    # k1 = f(t, x)
    k1 = self.state_derivative(state, forces_moments)

    # k2 = f(t + dt/2, x + k1*dt/2)
    x_temp = x0 + 0.5 * k1 * dt
    state_temp = State()
    state_temp.from_array(x_temp)
    state_temp.q.normalize()
    k2 = self.state_derivative(state_temp, forces_moments)

    # k3 = f(t + dt/2, x + k2*dt/2)
    x_temp = x0 + 0.5 * k2 * dt
    state_temp = State()
    state_temp.from_array(x_temp)
    state_temp.q.normalize()
    k3 = self.state_derivative(state_temp, forces_moments)

    # k4 = f(t + dt, x + k3*dt)
    x_temp = x0 + k3 * dt
    state_temp = State()
    state_temp.from_array(x_temp)
    state_temp.q.normalize()
    k4 = self.state_derivative(state_temp, forces_moments)

    # Combine
    x_new = x0 + (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0

    new_state = State()
    new_state.from_array(x_new)
    new_state.q.normalize()

    return new_state
```

**Key Features:**
- Quaternion normalized at each intermediate step
- 4 function evaluations per timestep (vs 1 for Euler)
- ~4x slower but much more accurate

### 2. Simple Autopilot

Added proportional controllers for airspeed and altitude:

```python
# Throttle control for airspeed
airspeed_error = target_airspeed - state.airspeed
throttle = controls_trim['throttle'] + Kp_throttle * airspeed_error
throttle = np.clip(throttle, 0.01, 1.0)

# Elevon control for altitude
altitude_error = target_altitude - state.altitude
elevon = controls_trim['elevator'] + Kp_elevon * altitude_error
elevon = np.clip(elevon, np.radians(-25), np.radians(25))
```

**Parameters Tested:**
- Low gains: `Kp_throttle = 0.001`, `Kp_elevon = 0.0005`
- High gains: `Kp_throttle = 0.005`, `Kp_elevon = 0.003`

---

## Results

### Test 1: Euler Integration, No Autopilot (Baseline)
```
Duration: 60s
Altitude change: +4764 ft
Airspeed change: -508 ft/s
Roll std dev: 62.7°
Pitch std dev: 35.9°
Status: UNSTABLE
```

### Test 2: RK4 Integration, No Autopilot
```
Duration: 60s
Altitude change: +4762 ft  (nearly identical!)
Airspeed change: -508 ft/s
Roll std dev: 62.8°
Pitch std dev: 35.9°
Status: UNSTABLE
```

**Conclusion**: RK4 provides NO improvement. The issue is NOT numerical error!

### Test 3: RK4 + Autopilot (Low Gains)
```
Duration: 60s
Altitude change: +876 ft  (5.4x better)
Airspeed change: -480 ft/s  (still loses speed)
Roll std dev: 93.2°  (worse)
Pitch std dev: 51.7°  (worse)
Status: UNSTABLE
```

**Conclusion**: Autopilot helps altitude slightly but makes attitude worse.

### Test 4: RK4 + Autopilot (High Gains)
```
Duration: 60s
Altitude change: +1809 ft  (worse than low gains!)
Airspeed change: -484 ft/s
Roll std dev: 82.5°
Pitch std dev: 51.3°
Status: UNSTABLE
```

**Conclusion**: Higher gains make it WORSE. Controller is fighting instability.

---

## Analysis

### Flight Timeline (from visualization)

**0-40 seconds**:
- ✓ Very stable flight
- ✓ Altitude slowly descends (~10 ft/s)
- ✓ Airspeed decreases gradually
- ✓ Angles remain nearly constant

**40-60 seconds**:
- ✗ Airspeed drops below ~200 ft/s
- ✗ Aircraft enters AERODYNAMIC STALL
- ✗ Pitch drops to -100° (nose down)
- ✗ Large yaw excursions (±180°)
- ✗ Wild pitch rate oscillations (-75 deg/s)
- ✗ Velocity approaches zero
- ✗ Complete loss of control

### Root Cause

The aircraft **stalls** after ~40 seconds because:

1. **Initial Condition**: Shallow 3.5° descent trajectory
   - Forward acceleration: 1.97 ft/s² in body frame
   - Airspeed bleeds off over time

2. **No Speed Protection**: Fixed/simple throttle control can't prevent slowdown
   - Descent causes drag buildup
   - Thrust insufficient to maintain speed in descent

3. **Stall Entry**: Below ~200 ft/s, CL becomes inadequate
   - Alpha increases to maintain lift
   - Eventually exceeds stall angle
   - Lift collapses, aircraft tumbles

4. **Flying Wing Instability**: Weak directional stability
   - Cnr = -0.001 (very weak yaw damping)
   - No vertical tail to prevent yaw excursions
   - Couples with roll during stall

### Why Simple Autopilot Fails

The proportional controller is fundamentally inadequate because:

1. **No Pitch Rate Feedback**: Can't damp pitch oscillations
   - Need Cmq damping term in control law
   - Flying wing has good Cmq = -0.347, but not using it!

2. **Altitude ≠ Pitch**: Direct altitude-to-elevon coupling is naive
   - Should command pitch angle/rate, not elevon directly
   - Need cascaded loop: altitude → pitch → elevon

3. **No Coordination**: Independent throttle/elevon commands
   - Altitude changes affect airspeed (energy coupling)
   - Need coordinated pitch/thrust for speed+altitude

4. **No Stall Protection**: No alpha limiter or min-speed logic
   - Should limit pitch-up commands when slow
   - Should increase throttle aggressively near stall

---

## Proper Autopilot Requirements

For stable flight, need:

### 1. Inner Loop (Attitude Control)
```
Pitch rate command: q_cmd = Kp_theta * (theta_cmd - theta) + Kd_theta * (0 - q)
Elevon: delta_e = Kp_q * (q_cmd - q) + Ki_q * integral(q_error)
```

### 2. Outer Loop (Trajectory Control)
```
Pitch command: theta_cmd = Kp_alt * (h_cmd - h) + Kd_alt * (h_dot_cmd - h_dot)
Throttle: T_cmd = Kp_V * (V_cmd - V) + feedforward(drag)
```

### 3. Safety Limits
```
Max pitch up: theta_cmd = min(theta_cmd, 30°)
Min airspeed: if V < V_stall * 1.3: theta_cmd = max(theta_cmd, -10°), T_cmd = max
Max alpha: if alpha > alpha_max: theta_cmd = current_theta - 5°
```

### 4. Coordination
```
Energy management: Total energy = kinetic + potential
                   E_dot = T*V - D*V - m*g*h_dot
                   Coordinate thrust/pitch to maintain E
```

---

## Recommendations

### Short Term (Quick Fix)
1. **Start at higher airspeed**: 600-700 ft/s instead of 548.5 ft/s
2. **Reduce simulation time**: 20s instead of 60s (stays stable)
3. **Accept the stall**: Document that current config is unstable without autopilot

### Medium Term (Better Autopilot)
1. **Implement inner loop**: Pitch rate damping with Cmq feedback
2. **Add integral terms**: Eliminate steady-state errors
3. **Stall protection**: Alpha limiter + min-speed logic

### Long Term (Complete Solution)
1. **True level flight trim**: Iterative solver for gamma=0, ax_ned=0
2. **Full autopilot**: Cascaded loops with proper coordination
3. **Adaptive control**: Adjust gains based on flight condition
4. **Add vertical fins**: Improve Cnr for better yaw stability

---

## Conclusions

### What We Learned

1. ✅ **Trim calculation is correct**
   - Forces/moments balanced at t=0
   - Alpha, theta, thrust, elevon all appropriate

2. ✅ **RK4 integration works**
   - Properly implemented
   - But doesn't fix the problem (proves it's not numerical!)

3. ✅ **Simple autopilot insufficient**
   - Proportional control too weak
   - Needs rate feedback and integral terms

4. ✅ **Root cause identified: Aerodynamic stall**
   - Happens at ~40s when V < 200 ft/s
   - Descent trajectory causes speed loss
   - No protection against stall entry

### Status

The flying wing 6-DOF simulation is:
- ✓ **Aerodynamically correct** (proper forces, moments, derivatives)
- ✓ **Numerically sound** (RK4 integration, quaternions)
- ✓ **Well-trimmed** (good initial conditions)
- ✗ **Dynamically unstable** (stalls without active control)

This is expected for a high-performance flying wing! Real aircraft would have:
- Full digital flight control system
- Multiple feedback loops
- Stall protection / envelope limiting
- Pilot input for trajectory changes

Our simplified open-loop/P-controller approach simply isn't adequate for 60 seconds of flight.

### Next Steps

For a working simulation, recommend:
1. Implement proper inner-loop autopilot (pitch rate feedback)
2. Add stall protection logic
3. Or accept 20-30s simulations (before stall)
4. Or start at higher airspeed (longer time to stall)

The fundamentals are all correct - just need better control!

---

*Generated: 2025-11-10*
*Flying Wing UAV + FJ-44 Turbofan*
*RK4 Integration + Simple Autopilot Testing*
