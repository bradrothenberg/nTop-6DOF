# Flying Wing Autopilot - Parameter Tuning Guide

Practical guide for tuning PID gains for stable flight control.

---

## Overview

The `FlyingWingAutopilot` uses **triple-loop cascaded control** with PID controllers at each level. This guide provides a systematic approach to tuning the 9 gain parameters for optimal performance.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Tuning Strategy](#tuning-strategy)
3. [Inner Loop: Pitch Rate Damping](#inner-loop-pitch-rate-damping)
4. [Middle Loop: Pitch Attitude](#middle-loop-pitch-attitude)
5. [Outer Loop: Altitude Hold](#outer-loop-altitude-hold)
6. [Fine-Tuning](#fine-tuning)
7. [Common Issues](#common-issues)

---

## Quick Reference

### Recommended Starting Values

```python
# CONSERVATIVE (safest, works for most cases)
autopilot = FlyingWingAutopilot(
    # Inner loop (pitch rate damping)
    Kp_pitch_rate = 0.10,
    Ki_pitch_rate = 0.005,
    Kd_pitch_rate = 0.0,

    # Middle loop (pitch attitude)
    Kp_pitch = 0.50,
    Ki_pitch = 0.02,
    Kd_pitch = 0.10,

    # Outer loop (altitude hold)
    Kp_alt = 0.002,
    Ki_alt = 0.0001,
    Kd_alt = 0.005
)
```

### Tuned Values (Flying Wing UAV at Mach 0.54, 5000 ft)

```python
# TUNED (optimized for this specific aircraft)
autopilot = FlyingWingAutopilot(
    # Inner loop
    Kp_pitch_rate = 0.15,
    Ki_pitch_rate = 0.01,
    Kd_pitch_rate = 0.0,

    # Middle loop
    Kp_pitch = 0.8,
    Ki_pitch = 0.05,
    Kd_pitch = 0.15,

    # Outer loop
    Kp_alt = 0.003,
    Ki_alt = 0.0002,
    Kd_alt = 0.008
)
```

### Typical Ranges

| Parameter | Min | Nominal | Max | Units |
|-----------|-----|---------|-----|-------|
| **Kp_pitch_rate** | 0.05 | 0.15 | 0.30 | - |
| **Ki_pitch_rate** | 0.00 | 0.01 | 0.02 | - |
| **Kd_pitch_rate** | 0.00 | 0.00 | 0.05 | - |
| **Kp_pitch** | 0.30 | 0.80 | 1.50 | - |
| **Ki_pitch** | 0.01 | 0.05 | 0.10 | - |
| **Kd_pitch** | 0.05 | 0.15 | 0.30 | - |
| **Kp_alt** | 0.001 | 0.003 | 0.010 | rad/ft |
| **Ki_alt** | 0.0001 | 0.0002 | 0.001 | rad/(ft·s) |
| **Kd_alt** | 0.002 | 0.008 | 0.020 | rad·s/ft |

---

## Tuning Strategy

### The Golden Rule

**Always tune from innermost loop to outermost:**

1. ✅ **Inner loop first** (pitch rate damping)
2. ✅ **Middle loop second** (pitch attitude)
3. ✅ **Outer loop last** (altitude hold)

**Why?** Each outer loop depends on the inner loops being stable. Tuning out-of-order causes cascading instabilities.

### Evaluation Metrics

After each tuning iteration, check:

**Stability Metrics**:
- No elevon saturation (stays within ±25°)
- No limit cycle oscillations
- State variables remain bounded

**Performance Metrics**:
- Rise time (how fast to reach target)
- Overshoot (peak error)
- Settling time (time to within 5% of target)
- Steady-state error (final error after settling)

**Flying Wing Specific**:
- Pitch std dev < 5° (good damping)
- Altitude std dev < 200 ft (minimal phugoid)
- No stall protection activation

---

## Inner Loop: Pitch Rate Damping

**Purpose**: Damp pitch oscillations directly using elevon control.

### Starting Point

```python
Kp_pitch_rate = 0.10
Ki_pitch_rate = 0.00  # Start with zero
Kd_pitch_rate = 0.00  # Usually not needed
```

### Tuning Procedure

**Step 1: Find maximum Kp without oscillation**

```python
# Test with fixed pitch rate command
for Kp_test in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    autopilot.pitch_rate_controller.Kp = Kp_test

    # Run 10-second test
    # Command: 5 deg/s pitch rate

    # Check for:
    # - Elevon saturation (±25°) → too high
    # - Rapid oscillations (~1 Hz) → too high
    # - Slow response (>2 seconds) → too low
```

**Signs Kp is too high:**
- Elevon rapidly oscillates at ±25° limits
- High-frequency oscillations in pitch rate (~5-10 Hz)
- Pitch angle grows instead of stabilizing

**Signs Kp is too low:**
- Slow response to pitch rate commands (>3 seconds)
- Large steady-state error in pitch rate
- Insufficient damping of pitch oscillations

**Step 2: Add integral term (if needed)**

```python
# Only add Ki if there's steady-state error
Ki_pitch_rate = 0.005  # Start very small

# Increase gradually: 0.005 → 0.01 → 0.015
# Stop when steady-state error is eliminated
```

**Warning**: Too much integral gain causes:
- Overshoot
- Oscillations at medium frequency (~1-2 Hz)
- Sluggish response

**Step 3: Derivative (usually skip)**

Most flying wings don't need `Kd_pitch_rate` because:
- Aircraft has natural pitch damping (Cmq < 0)
- Adding more can amplify sensor noise
- Only use if pitch oscillations persist despite low Kp

### Validation Test

**Good Inner Loop**:
```python
# Command step change: 0 → 5 deg/s
# Expect:
# - Rise time: 0.3-0.5 seconds
# - Overshoot: < 20%
# - Settling time: < 1 second
# - Elevon: smooth, no saturation
```

---

## Middle Loop: Pitch Attitude

**Purpose**: Achieve commanded pitch angle by commanding pitch rate.

### Starting Point

```python
Kp_pitch = 0.50
Ki_pitch = 0.02
Kd_pitch = 0.10
```

### Tuning Procedure

**Step 1: Tune Kp with fixed pitch command**

```python
# Test with fixed pitch angle command
for Kp_test in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
    autopilot.pitch_controller.Kp = Kp_test

    # Run 15-second test
    # Command: 5° pitch up from level

    # Check for:
    # - Overshoot > 50% → too high
    # - Oscillations (period ~2-5s) → too high
    # - Slow response (>5 seconds) → too low
```

**Ziegler-Nichols Method** (optional):

1. Set `Ki = 0`, `Kd = 0`
2. Increase `Kp` until sustained oscillations appear (Kp_critical)
3. Measure oscillation period (T_critical)
4. Use formulas:
   ```
   Kp = 0.6 * Kp_critical
   Ki = 1.2 * Kp_critical / T_critical
   Kd = 0.075 * Kp_critical * T_critical
   ```

**Step 2: Add derivative for damping**

```python
# If overshoot > 30% or oscillations persist:
Kd_pitch = 0.15  # Start with this value

# Increase gradually: 0.10 → 0.15 → 0.20
# Stop when overshoot < 20%
```

**Step 3: Add integral to eliminate error**

```python
# If steady-state error > 0.5°:
Ki_pitch = 0.05

# Increase cautiously: 0.02 → 0.05 → 0.08
# Stop when error < 0.2°
```

### Validation Test

**Good Middle Loop**:
```python
# Command step: 0° → 5°
# Expect:
# - Rise time: 1-2 seconds
# - Overshoot: < 20% (< 6°)
# - Settling time: 3-5 seconds
# - No oscillations
# - Steady-state error: < 0.5°
```

---

## Outer Loop: Altitude Hold

**Purpose**: Maintain target altitude by commanding pitch angle.

### Starting Point

```python
Kp_alt = 0.002
Ki_alt = 0.0001
Kd_alt = 0.005
```

### Tuning Procedure

**Step 1: Tune Kp (most important)**

```python
# Test with altitude error
for Kp_test in [0.001, 0.002, 0.003, 0.005, 0.008, 0.010]:
    autopilot.altitude_controller.Kp = Kp_test

    # Run 30-second test
    # Start: 50 ft below target (or use 100 ft step)

    # Check for:
    # - Phugoid oscillations (period 20-40s) → too high
    # - Altitude overshoot > 100 ft → too high
    # - Slow convergence (>60 seconds) → too low
```

**Important**: Altitude loop is **very slow** compared to inner loops. Gains must be much smaller (typically 100-1000x smaller than pitch gains).

**Step 2: Add derivative to damp phugoid**

```python
# If altitude oscillates (phugoid mode):
Kd_alt = 0.008

# Increase gradually: 0.005 → 0.008 → 0.012 → 0.020
# Stop when oscillations damped (< 3 cycles)
```

The derivative term responds to climb/descent rate, preventing overshoot.

**Step 3: Add integral (carefully!)**

```python
# Only if steady-state error > 50 ft:
Ki_alt = 0.0002

# Increase slowly: 0.0001 → 0.0002 → 0.0005
# Too much Ki causes long-period oscillations
```

**Warning**: Altitude integral windup is common. Use anti-windup limits (already implemented in code).

### Validation Test

**Good Outer Loop**:
```python
# Command: 100 ft altitude increase
# Expect:
# - Rise time: 10-20 seconds
# - Overshoot: < 50 ft
# - Settling time: < 40 seconds
# - No phugoid oscillations
# - Steady-state error: < 20 ft
```

---

## Fine-Tuning

### Coordinated Adjustments

After initial tuning, make coordinated adjustments:

**For faster response** (more aggressive):
```python
# Increase all P gains by 30%
Kp_alt *= 1.3
Kp_pitch *= 1.3
Kp_pitch_rate *= 1.3

# Increase derivative gains by 50%
Kd_alt *= 1.5
Kd_pitch *= 1.5
```

**For smoother response** (more conservative):
```python
# Decrease all P gains by 30%
Kp_alt *= 0.7
Kp_pitch *= 0.7
Kp_pitch_rate *= 0.7

# Increase derivative gains slightly
Kd_alt *= 1.2
Kd_pitch *= 1.2
```

### Flight Condition Adaptation

**Higher airspeed** (more control authority):
```python
# Can use 30-50% higher gains
Kp_pitch_rate = 0.20  # was 0.15
Kp_pitch = 1.0        # was 0.8
```

**Lower airspeed** (less control authority):
```python
# Reduce gains by 30-50%
Kp_pitch_rate = 0.10  # was 0.15
Kp_pitch = 0.50       # was 0.8
```

**Heavy loading** (more inertia):
```python
# Reduce altitude gains (slower dynamics)
Kp_alt = 0.002   # was 0.003
Kd_alt = 0.005   # was 0.008

# Pitch gains can stay similar
```

### Safety Limits

Always set appropriate safety limits:

```python
# Pitch command limits (prevent extreme attitudes)
max_pitch_cmd = 12.0   # degrees (20° too aggressive for flying wing)
min_pitch_cmd = -8.0   # degrees

# Stall protection (critical!)
max_alpha = 12.0               # degrees (flying wing stalls ~15°)
stall_speed = 150.0            # ft/s
min_airspeed_margin = 1.3      # 30% above stall
```

---

## Common Issues

### Issue 1: Elevon Limit Cycles

**Symptoms**: Elevon rapidly oscillates ±25° at ~1 Hz

**Root cause**: `Kp_pitch_rate` too high

**Solution**:
```python
# Reduce inner loop gain by 50%
Kp_pitch_rate = 0.075  # was 0.15
```

### Issue 2: Phugoid Oscillations

**Symptoms**: Altitude and airspeed oscillate with 20-40 second period

**Root cause**: Insufficient damping in altitude loop

**Solutions**:
```python
# Option 1: Increase derivative gain
Kd_alt = 0.015  # was 0.008

# Option 2: Reduce proportional gain
Kp_alt = 0.002  # was 0.003

# Option 3: Coordinate throttle with altitude
# (see complete example in user guide)
```

### Issue 3: Altitude Overshoot

**Symptoms**: Altitude shoots past target by >100 ft

**Root cause**: `Kp_alt` too high or `Kd_alt` too low

**Solutions**:
```python
# Reduce proportional gain
Kp_alt = 0.002  # was 0.003 or higher

# Increase derivative gain
Kd_alt = 0.012  # was 0.008
```

### Issue 4: Slow Convergence

**Symptoms**: Takes >60 seconds to reach target altitude

**Root cause**: All gains too low

**Solutions**:
```python
# Increase proportional gains across the board
Kp_alt = 0.005         # was 0.002
Kp_pitch = 1.0         # was 0.5
Kp_pitch_rate = 0.20   # was 0.10
```

### Issue 5: Steady-State Error

**Symptoms**: Aircraft settles to altitude 20-50 ft from target

**Root cause**: No integral action or integral too small

**Solutions**:
```python
# Increase integral gain (carefully!)
Ki_alt = 0.0005  # was 0.0002

# Check integral limits aren't saturated
print(f"Integral value: {autopilot.altitude_controller.error_integral}")
# Should be within ±100 ft
```

### Issue 6: Stall Protection Always Active

**Symptoms**: `stall_protection_active = True` continuously

**Root causes**:
1. Airspeed too low (descending/losing energy)
2. Margin too tight
3. Aircraft actually near stall

**Solutions**:
```python
# Option 1: Start at higher airspeed
trim_airspeed = 650.0  # was 600.0

# Option 2: Widen margin
min_airspeed_margin = 1.4  # was 1.3

# Option 3: Improve throttle control
throttle_gain = 0.005  # was 0.002

# Option 4: Check if aircraft is actually slowing down
# Plot airspeed history to diagnose
```

---

## Tuning Checklist

Before declaring the autopilot "tuned":

- [ ] Inner loop stable for 10+ seconds with no saturation
- [ ] Middle loop achieves 5° pitch step in < 5 seconds
- [ ] Outer loop maintains altitude with < 200 ft oscillations
- [ ] 30-second simulation completes without divergence
- [ ] Elevon stays within ±20° (not hitting limits)
- [ ] Pitch std dev < 5°
- [ ] Altitude std dev < 250 ft
- [ ] No stall protection activation (unless intended)
- [ ] Airspeed stable (std dev < 15 ft/s)
- [ ] Roll remains near zero (std dev < 1°)

---

## Automated Tuning (Advanced)

For systematic tuning, consider implementing:

**Grid Search**:
```python
# Test all combinations
Kp_values = [0.1, 0.15, 0.2]
Ki_values = [0.005, 0.01, 0.015]

for Kp in Kp_values:
    for Ki in Ki_values:
        autopilot.pitch_rate_controller.Kp = Kp
        autopilot.pitch_rate_controller.Ki = Ki

        # Run simulation
        result = run_simulation(...)

        # Compute cost function
        cost = (result.pitch_std**2 +
                result.altitude_std**2 +
                result.saturation_events * 1000)

        # Store best
        if cost < best_cost:
            best_cost = cost
            best_gains = (Kp, Ki)
```

**Optimization** (scipy):
```python
from scipy.optimize import minimize

def cost_function(gains):
    Kp_alt, Kp_pitch, Kp_pitch_rate = gains

    # Set gains
    autopilot = FlyingWingAutopilot(
        Kp_alt=Kp_alt,
        Kp_pitch=Kp_pitch,
        Kp_pitch_rate=Kp_pitch_rate
    )

    # Run simulation
    result = run_simulation(autopilot)

    # Return cost (lower is better)
    return (result.altitude_error**2 +
            result.pitch_std**2 +
            result.control_effort)

# Optimize
result = minimize(
    cost_function,
    x0=[0.003, 0.8, 0.15],  # Initial guess
    bounds=[(0.001, 0.01), (0.3, 1.5), (0.05, 0.3)]
)

print(f"Optimal gains: {result.x}")
```

---

## Summary

**Key Principles**:
1. Tune inner loop first, outer loop last
2. Start conservative, increase gains gradually
3. Watch for saturation and oscillations
4. Flying wings need more damping than conventional aircraft
5. Altitude loop is 100-1000x slower than pitch loops

**Recommended Workflow**:
1. Start with conservative gains
2. Tune pitch rate loop (10s test)
3. Tune pitch attitude loop (15s test)
4. Tune altitude loop (30s test)
5. Fine-tune all three together (60s test)
6. Validate with longer simulation

**Quick Diagnostics**:
- Limit cycles → reduce `Kp_pitch_rate`
- Phugoid → increase `Kd_alt`
- Overshoot → reduce `Kp_alt`, increase `Kd_pitch`
- Slow response → increase all `Kp` gains by 30%

---

## References

1. Ziegler, J. G., & Nichols, N. B. (1942). "Optimum settings for automatic controllers." *Transactions of the ASME*, 64(11).
2. Åström, K. J., & Hägglund, T. (2006). *Advanced PID Control*. ISA.
3. Franklin, G. F., Powell, J. D., & Emami-Naeini, A. (2019). *Feedback Control of Dynamic Systems* (8th ed.). Pearson.

---

*Last updated: 2025-11-10*
*Flying Wing UAV - FJ-44 Turbofan Configuration*
