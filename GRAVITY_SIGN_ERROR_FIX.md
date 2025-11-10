# Gravity Sign Error - Root Cause and Fix

## Summary

**FOUND AND FIXED**: Critical sign error in gravity force calculation causing complete simulation instability.

## The Bug

### Location
[src/core/dynamics.py:99](src/core/dynamics.py#L99) (now fixed)

### Original Code (WRONG)
```python
vel_body_dot = forces / self.mass - omega_cross_v - g_body
```

### Fixed Code (CORRECT)
```python
vel_body_dot = forces / self.mass - omega_cross_v + g_body
```

## Root Cause Analysis

### The Physics

In the NED (North-East-Down) reference frame:
- Gravity vector: `g_inertial = [0, 0, +32.174]` (down is positive)
- When transformed to body frame: `g_body = R @ g_inertial`

The force equation in body frame is:
```
F_total = F_aero + F_thrust + F_gravity
F_gravity = m * g_body

a = F_total / m = (F_aero + F_thrust) / m + g_body
```

Therefore, `g_body` should be **ADDED**, not subtracted!

### The Impact

With the wrong sign, gravity was being applied TWICE:

**Before Fix:**
```
az = (Fz / m) - omega_cross_v[2] - g_body[2]
   = (-7353.7 / 228.9) - 0 - 32.16
   = -32.12 - 32.16
   = -64.28 ft/s²  ← WRONG! Double gravity effect
```

**After Fix:**
```
az = (Fz / m) - omega_cross_v[2] + g_body[2]
   = (-7353.7 / 228.9) - 0 + 32.16
   = -32.12 + 32.16
   = 0.04 ft/s²  ← CORRECT! Nearly trimmed
```

### Symptoms

Before the fix, the aircraft experienced:
- **64 ft/s² downward acceleration** (2x gravity!)
- Lost 4738 ft altitude in 30 seconds
- Airspeed dropped from 548.5 ft/s to 37 ft/s
- Aircraft tumbled (pitch to 90°, roll to 180°)

After the fix:
- **0.04 ft/s² vertical acceleration** (nearly trimmed!)
- Vertical force balance is correct
- Still some instability due to incorrect trim angle (separate issue)

## Testing

### Test 1: First Timestep Diagnostic

**Before Fix:**
```
Accelerations:
  az: -64.2818 ft/s²  ← WRONG!

Vertical Force Balance:
  Fz (aero + thrust): -7353.7 lbf
  Weight in body: 7362.0 lbf
  Net force: -14715.7 lbf  ← Double counting!
```

**After Fix:**
```
Accelerations:
  az: 0.0362 ft/s²  ← CORRECT!

Vertical Force Balance:
  Fz: -7353.7 lbf
  Weight: 7362.0 lbf
  Net: 8.3 lbf  ← Nearly balanced!
```

### Test 2: Sign Propagation Chain

I also verified that the entire dynamics chain is correct:
1. **Moments → Angular Accelerations**: CORRECT (omega_dot = I^-1 @ M)
2. **Angular Accelerations → Quaternion Rates**: CORRECT (q_dot = 0.5 * Omega @ q)
3. **Quaternion → Euler Angles**: CORRECT (standard conversion)

The only error was in the **gravity force application**.

## Remaining Issues

### Issue: Incorrect Trim Angle

Even with the gravity fix, the manual trim at alpha = 1.75° is NOT stable because:

1. **Lift Trim**: Requires alpha ≈ 1.75° for L = W
2. **Moment Trim**: Requires alpha ≈ 0.044° for M = 0

These are INCOMPATIBLE! You cannot satisfy both L=W and M=0 with just alpha.

### Solution: Proper Trim Solver

Need to use the trim solver to find:
- `alpha` (for lift)
- `theta` (for flight path angle = 0)
- `throttle` (for thrust = drag)
- `elevator` (or elevons) (to balance pitch moment)

The trim solver optimizes all 4 variables simultaneously to satisfy:
```
Fx = 0  (thrust = drag)
Fz = 0  (lift = weight)
M = 0   (pitch moment balanced)
h_dot = 0  (altitude rate = 0)
```

## Status

- ✅ **Gravity sign error**: FIXED
- ✅ **Vertical force balance**: CORRECT (az ≈ 0 at supposed trim)
- ⏳ **Trim solver**: Needs work (currently hits bounds, doesn't converge)
- ⏳ **Simulation stability**: Pending proper trim

## Files Changed

1. **src/core/dynamics.py** - Fixed gravity sign (line 99)
2. **test_first_timestep.py** - Diagnostic tool (new)
3. **test_moment_to_pitch.py** - Sign propagation test (new)
4. **test_trim_debug.py** - Force/moment verification (existing)

## Next Steps

1. Fix trim solver convergence issues
2. Test simulation with properly trimmed initial conditions
3. Implement RK4 integration for better accuracy
4. Add safeguards for Euler angle singularities

## Key Insight

The original "sign error" was NOT in the quaternion or rotational dynamics as initially suspected. It was in the **translational dynamics** - specifically, the gravity force application. This caused:

- Aircraft to experience 2x gravity (64 ft/s² down)
- Complete loss of altitude and airspeed
- Secondary effects (pitch/roll instability due to low speed and high AoA)

With gravity fixed, the aircraft is MUCH closer to stable flight, but still needs proper trim solver to find equilibrium.

---

**Bottom Line**: One sign flip (`- g_body` → `+ g_body`) was the root cause of the entire "wobbly trajectory" issue!
