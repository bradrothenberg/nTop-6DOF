# AVL Analysis Results - Root Cause Found

## Summary

I extracted the **actual stability derivatives** from your `uav.avl` model using AVL. The results reveal the **root cause** of the trajectory instability.

## Critical Finding: Aircraft is Statically Unstable

### **Cm_alpha = +0.002710 (POSITIVE!)**

This is the **smoking gun**. A positive Cm_alpha means:
- When angle of attack increases → nose pitches UP (not down)
- This creates a **divergent feedback loop**
- Aircraft is **statically unstable** in pitch
- Cannot maintain trim without active control

**For stability, Cm_alpha MUST be negative** (typically -0.5 to -1.0)

---

## AVL Stability Derivatives (alpha = 2°)

### Force Derivatives
```
CL_alpha = 2.441430   (normal)
CD_alpha = 0.069800   (normal)
CY_beta  = -0.041976  (normal)
```

### Moment Derivatives (THE PROBLEMS)
```
Cm_alpha     = +0.002710   ← POSITIVE = UNSTABLE!
Cm_q         = -0.232245   ← VERY WEAK (should be -5 to -15)
Cm_elevator  = -0.002755   ← 100x WEAKER than estimated (-1.2)

Cl_beta      = -0.041339   (weak but acceptable)
Cl_p         = -0.173909   (weak)
Cl_flaperon  = -0.000189   ← ALMOST ZERO!

Cn_beta      = +0.013061   (weak yaw stability)
Cn_r         = -0.017481   (very weak yaw damping)
```

### Control Effectiveness (EXTREMELY WEAK)
```
Elevator:  Cm_de = -0.002755  (should be ~-1.0)
Flaperon:  Cl_da = -0.000189  (should be ~0.15)
Rudder:    Cn_dr = +0.000187  (should be ~-0.1)
```

---

## Why is the Aircraft Unstable?

Looking at your AVL geometry, I see the issue:

### **CG Location**
```
Xref =  12.9183 ft  (from AVL file)
Yref =   0.0011 ft
Zref =   0.0453 ft
```

### **Neutral Point**
```
Xnp = 12.8887 ft  (from AVL output)
```

###  **Static Margin = (Xnp - Xcg) / c_ref**
```
Static Margin = (12.8887 - 12.9183) / 26.6891
              = -0.0296 / 26.6891
              = -0.001 or -0.1%
```

**The CG is AHEAD of the neutral point!**

This is like trying to balance a broomstick with the heavy end at the top - it wants to tip over.

---

## Why are Controls So Weak?

### Elevator Hinge Line
From your AVL file:
```
CONTROL
 elevator  1.000   0.700   ...
```
- Hinge at 70% chord
- On horizontal tail starting at X = 22.42 ft

### Wing Flaperon Hinge Line
```
CONTROL
 flaperon  1.000   0.800   ...
```
- Hinge at 80% chord
- Only on outer 25% of span (sections 6-7)

**Problems**:
1. Elevator moment arm from CG is small (22.42 - 12.92 = 9.5 ft)
2. Flaperon is on trailing 20% of chord (very small surface)
3. Flaperon only on outer 25% of span (limited authority)

---

## Simulation Results with AVL Data

With **actual** derivatives, the simulation shows:

```
Final altitude:  29,153 ft  (target: 5,500 ft)
Final heading:   176°       (target: 20°)
Final airspeed:  1,246 ft/s (target: 205 ft/s)
Max pitch:       89.7°
Max roll:        180°
```

**Complete loss of control** - aircraft enters divergent climb/pitch-up.

---

## Solutions

### Option 1: Move CG Aft (RECOMMENDED)

Move CG from 12.92 ft to ~11.5 ft (1.4 ft forward)

**Effect**:
- Static margin = (12.89 - 11.5) / 26.69 = **+5.2%** (stable!)
- Cm_alpha becomes negative
- Aircraft becomes statically stable

**How**: Redistribute mass (move batteries, payload, fuel aft)

### Option 2: Increase Tail Size

Increase horizontal tail area by 50-100%

**Effect**:
- Moves neutral point aft
- Increases elevator effectiveness
- More pitch authority

**Drawback**: Adds weight and drag

### Option 3: Increase Elevator Size

Move hinge line forward (0.60 instead of 0.70)

**Effect**:
- Increases Cm_elevator by ~40%
- Better pitch control
- Still doesn't fix static instability

### Option 4: Use Canard Configuration

Add canard forward of CG

**Effect**:
- Moves NP forward
- CG can be further aft
- Inherently stable

**Drawback**: Major redesign

---

## Recommended Next Steps

### 1. **Fix CG Location** (Critical)

Update `uav.mass` file with CG moved aft:
```
# Current
234.836794   12.9183    0.0011    0.0453  ...

# Recommended (move 1.4 ft forward)
234.836794   11.5000    0.0011    0.0453  ...
```

### 2. **Re-run AVL Analysis**

```bash
cd avl_files
avl uav.avl
load
mass uav.mass
mset
0
oper
a a 2.0
x
st uav_stability_fixed.txt
```

Check that:
- Cm_alpha < 0 (negative)
- Static margin > 3%
- Neutral point Xnp < Xcg

### 3. **Test Simulation**

Run `uav_avl_demo.py` again with new derivatives

### 4. **If Still Unstable**: Increase Tail Size

Modify `uav.avl`:
- Scale horizontal tail chord by 1.5x
- Re-run AVL
- Check Cm_alpha and Cm_elevator

---

## Why Estimated Values Were Wrong

### Original Estimates
```
Cm_alpha = -0.6       (assumed stable)
Cm_elevator = -1.2    (assumed strong control)
Cl_aileron = 0.15     (assumed normal effectiveness)
```

### Actual AVL Values
```
Cm_alpha = +0.00271   (unstable!)
Cm_elevator = -0.00276 (100x weaker!)
Cl_flaperon = -0.000189 (1000x weaker!)
```

**The estimates assumed**:
- CG behind neutral point (stable)
- Normal control surface sizes
- Standard moment arms

**The reality**:
- CG ahead of neutral point (unstable)
- Very small control surfaces
- Poor moment arms

---

## Conclusion

The "wobbly" trajectory is caused by:

1. ✅ **Root Cause**: CG is 0.03 ft AHEAD of neutral point → Static instability
2. ✅ **Contributing**: Elevator effectiveness 100x weaker than estimated
3. ✅ **Contributing**: Flaperon effectiveness essentially zero
4. ✅ **Contributing**: Weak damping derivatives

**The aircraft CANNOT be stabilized by autopilot tuning alone.**

**You MUST either**:
- Move CG aft by ~1.4 ft, OR
- Increase horizontal tail size by ~50%, OR
- Redesign with canard

Once you fix the CG location and re-run AVL, the simulation should be stable!

---

## Files Created

1. `avl_files/uav_stability.txt` - AVL output with actual derivatives
2. `examples/uav_avl_demo.py` - Simulation using actual AVL data
3. `output/uav_avl_*.png` - Visualization showing instability

## Next Action

**Please decide**:
- **A)** Move CG aft and re-run AVL (fastest fix)
- **B)** Increase tail size in AVL model
- **C)** Both A and B for maximum stability

Let me know which approach you'd like to take!
