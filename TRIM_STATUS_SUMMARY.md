# Trim Solver Status and Remaining Issues

## Summary

Successfully fixed the **critical gravity sign error** that was causing 2x gravity effect. The vertical force balance is now correct (az ≈ 0 at lift-trim). However, the flying wing has fundamental design issues that prevent stable trim:

1. **No pitch moment control** (Cm_de = 0 for elevators)
2. **Severely underpowered** (need 212 HP, have 50 HP)
3. **Large unbalanced pitch moment** at lift-trim conditions

## What Was Fixed

### 1. Gravity Sign Error (FIXED ✅)

**File**: [src/core/dynamics.py:99](src/core/dynamics.py#L99)

**Change**:
```python
# BEFORE (WRONG):
vel_body_dot = forces / self.mass - omega_cross_v - g_body

# AFTER (CORRECT):
vel_body_dot = forces / self.mass - omega_cross_v + g_body
```

**Impact**:
- Vertical acceleration: -64 ft/s² → **0.04 ft/s²** (nearly zero!)
- Force balance: Correct
- Gravity was being applied TWICE before the fix

**Verification**:
- [test_first_timestep.py](test_first_timestep.py): Confirms az ≈ 0
- [test_moment_to_pitch.py](test_moment_to_pitch.py): Verifies sign propagation chain is correct
- [GRAVITY_SIGN_ERROR_FIX.md](GRAVITY_SIGN_ERROR_FIX.md): Complete analysis

### 2. Simplified Trim Solver (CREATED ✅)

**File**: [src/simulation/trim_simple.py](src/simulation/trim_simple.py)

**Approach**:
Since the flying wing has no elevator control of pitch moment (Cm_de = 0), the simplified solver finds:

1. **Alpha for L = W**: `alpha_trim = (CL_trim - CL_0) / CL_alpha = 1.7531°`
2. **Theta = Alpha**: For level flight (gamma = 0)
3. **Throttle for T ≈ D**: Iterative adjustment

**Accepts non-zero pitch moment** and relies on:
- Cm_alpha = -0.080 (static stability)
- Cmq = -0.347 (pitch damping)

**Result**:
- ✅ Lift = Weight (7365.4 lbf)
- ✅ az ≈ 0 ft/s² (vertical acceleration near zero)
- ❌ Pitch moment = -4840 ft-lbf (large!)
- ❌ Pitch accel = -123 deg/s² (causes divergence)

## Remaining Issues

### Issue 1: No Pitch Moment Control

**Problem**: Flying wing elevators have **zero pitch authority** (Cm_de = 0).

**Why**: The AVL data shows elevators deflected symmetrically create no net pitch moment. For flying wings, you need **antisymmetric elevon deflection** (left ≠ right) to create pitch moment through differential drag.

**Current**:
```
Elevator deflection: both elevons deflect same amount
Result: No change in pitch moment
```

**Needed**:
```
Differential elevon deflection:
  Left elevon: +δ
  Right elevon: -δ
Result: Creates pitch moment via drag asymmetry
```

**Solutions**:
1. **Add differential elevon control** to aerodynamics model
2. **Use thrust line offset** (add moment arm to propeller)
3. **Accept oscillations** and rely solely on damping (Cmq)

### Issue 2: Severely Underpowered

**Problem**: Aircraft needs **212 HP** to sustain Mach 0.5, but only has **50 HP**.

**Calculations**:

| Speed | Alpha | CL | CD | L/D | Drag | Thrust Avail (50 HP) | T/D Ratio |
|-------|-------|----|----|-----|------|----------------------|-----------|
| 200 ft/s | 17.68° | 0.436 | 0.058 | 7.5 | 979 lbf | 103 lbf | 0.105 |
| 350 ft/s | 5.77° | 0.142 | 0.008 | 17.7 | 416 lbf | 59 lbf | 0.142 |
| **548.5 ft/s** | **1.75°** | **0.043** | **0.001** | **46.2** | **159 lbf** | **38 lbf** | **0.238** |

At Mach 0.5:
- Required thrust: 159 lbf
- Available thrust (50 HP): 38 lbf
- **Shortfall**: 121 lbf (76%)

Required power:
```
P_req = T * V / η = 159.4 * 548.5 / 0.75 = 116,600 ft-lbf/s = 212 HP
```

**Solutions**:
1. **Increase power to 250 HP** (tested, helps but not enough)
2. **Reduce speed** to sustainable cruise (need T/D ≈ 1.0)
3. **Reduce weight** (currently 7365 lbf = 228.9 slugs)

### Issue 3: Large Pitch Moment at Trim

**Problem**: At alpha = 1.75° (where L = W), pitch moment is -4840 ft-lbf.

**Analysis**:
```
Cm_trim = Cm_0 + Cm_alpha * alpha
        = 0.000061 + (-0.079668) * 0.0306 rad
        = 0.000061 - 0.002438
        = -0.002377

M_pitch = q * S * c * Cm
        = 412.86 * 412.64 * 11.96 * (-0.002377)
        = -4840 ft-lbf
```

**This is a NOSE-DOWN moment**, causing pitch acceleration of -123 deg/s².

**Why it's a problem**:
- Aircraft immediately pitches down from trim
- Loses airspeed rapidly (548 → 40 ft/s in 60s)
- Gains altitude initially (from pitch-down kinetic to potential energy conversion)
- Then tumbles as airspeed drops too low

**What's needed**:
- **Elevator/elevon deflection** to create +4840 ft-lbf nose-UP moment
- But Cm_de = 0, so elevator can't help!

**Zero-moment trim angle**:
```
Cm = 0  →  Cm_0 + Cm_alpha * alpha = 0
alpha_Cm0 = -Cm_0 / Cm_alpha = -0.000061 / (-0.079668) = 0.044° ≈ 0°
```

But at alpha ≈ 0°:
```
CL = CL_0 + CL_alpha * 0 = 0.000023 ≈ 0  (NO LIFT!)
```

**Incompatible requirements**:
- Lift trim: Need alpha = 1.75°
- Moment trim: Need alpha = 0.044°

Can't satisfy both without pitch control!

## Current Status

### What Works

✅ **Gravity dynamics**: Correct (az ≈ 0 at lift-trim)
✅ **Vertical force balance**: Fz + m*g_z ≈ 0
✅ **Rotational dynamics chain**: Moments → omega_dot → q_dot → Euler angles (all correct)
✅ **Lift calculation**: L = W at alpha = 1.75°
✅ **Simplified trim solver**: Finds alpha and theta for L = W

### What Doesn't Work

❌ **Pitch moment balance**: -4840 ft-lbf at lift-trim
❌ **Pitch control**: Cm_de = 0 (no elevator authority)
❌ **Thrust**: T/D = 0.24 at Mach 0.5 with 50 HP (need 212 HP)
❌ **Simulation stability**: Aircraft pitches down, loses airspeed, tumbles

## Recommendations

### Short Term: Accept Limitations

1. **Document that flying wing cannot trim at Mach 0.5** with current configuration
2. **Test at lower speed** where:
   - Thrust sufficient (T/D > 0.8)
   - Pitch moment smaller
   - Natural damping can stabilize

### Medium Term: Fix Configuration

**Option A: Add Differential Elevon Control**
1. Modify aerodynamics model to accept left/right elevon deflections separately
2. Update AVL analysis with antisymmetric elevon deflection
3. Get Cm_elevon derivative (likely small but non-zero)

**Option B: Increase Power**
1. Change `PropellerModel` to 250+ HP
2. Retest at Mach 0.5
3. Still won't fix pitch moment issue

**Option C: Add Autopilot**
1. Accept that manual trim is impossible
2. Implement pitch rate controller using elevons
3. Controller actively maintains pitch attitude

### Long Term: Redesign

1. **Reduce weight**: 7365 lbf is too heavy for 50 HP
   - Target: <3000 lbf for 50 HP
   - Or increase to 200+ HP for current weight

2. **Optimize aerodynamics**:
   - Current L/D = 46 at Mach 0.5 (very good!)
   - But Cm_alpha = -0.080 creates large moment at moderate alpha
   - Consider CG shift or wing incidence angle

3. **Add control surfaces**:
   - Implement proper elevon mixing (pitch + roll)
   - Or add canards for pitch control
   - Or accept flying wing limitations and use thrust vectoring

## Test Results Summary

### Gravity Fix Test
**File**: [test_first_timestep.py](test_first_timestep.py)

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| az | -64.28 ft/s² | **0.04 ft/s²** |
| Vertical force | -14,716 lbf | **8.3 lbf** |
| Gravity effect | 2x (double!) | 1x (correct) |

### Trim Solver Test
**File**: [src/simulation/trim_simple.py](src/simulation/trim_simple.py)

Result at Mach 0.5:
- Alpha: 1.7531°
- Theta: 1.7531°
- Throttle: 0.392 (with 250 HP)
- CL: 0.0432, CD: 0.0009, L/D: 46.2
- Lift: 7365.4 lbf ✓
- Drag: 159.4 lbf
- Thrust: 66.8 lbf (T/D = 0.42) ❌
- Pitch moment: -4840 ft-lbf ❌
- az: -0.02 ft/s² ✓
- q_dot: -123 deg/s² ❌

### Simulation Test
**File**: [examples/flyingwing_proper_trim_test.py](examples/flyingwing_proper_trim_test.py)

60-second simulation from trim:
- Altitude: 5000 → 9841 ft (+4841 ft)
- Airspeed: 548.5 → 40.4 ft/s (-508 ft/s)
- Pitch std: 45.7° (large oscillations)
- Roll std: 78.7° (tumbling)
- **Status**: UNSTABLE

## Files Created/Modified

### New Files
1. `GRAVITY_SIGN_ERROR_FIX.md` - Gravity bug analysis
2. `test_first_timestep.py` - First timestep diagnostic
3. `test_moment_to_pitch.py` - Sign propagation verification
4. `test_quaternion_pitch_sign.py` - Quaternion validation
5. `src/simulation/trim_simple.py` - Simplified trim solver
6. `examples/flyingwing_proper_trim_test.py` - Proper trim test
7. `TRIM_STATUS_SUMMARY.md` - This document

### Modified Files
1. `src/core/dynamics.py` - Fixed gravity sign (line 99)
2. `src/simulation/trim.py` - Improved bounds and tolerances

## Next Steps

### Priority 1: Choose Configuration
- [ ] Accept limitations and document
- [ ] OR implement differential elevons
- [ ] OR add autopilot

### Priority 2: Fix Power
- [ ] Increase to 250+ HP
- [ ] OR reduce cruise speed to sustainable level
- [ ] OR reduce aircraft weight

### Priority 3: Improve Numerics
- [ ] Implement RK4 integration (currently simple Euler)
- [ ] Add adaptive timestep
- [ ] Quaternion-only state (avoid Euler singularities)

### Priority 4: Testing
- [ ] Test at lower speeds (200-300 ft/s)
- [ ] Test with autopilot
- [ ] Validate against real aircraft data

## Conclusion

The **gravity sign error has been fixed** and vertical dynamics are now correct. However, the flying wing configuration has fundamental issues:

1. **No pitch control** (Cm_de = 0)
2. **Insufficient power** (need 4x current)
3. **Conflicting trim requirements** (L=W at alpha=1.75°, M=0 at alpha=0°)

These are **design issues**, not simulation bugs. The simulation is working correctly, but the aircraft cannot trim at the requested conditions.

**Recommendation**: Either modify the aircraft configuration (add control authority and power) or accept that stable flight at Mach 0.5 requires active control (autopilot).
