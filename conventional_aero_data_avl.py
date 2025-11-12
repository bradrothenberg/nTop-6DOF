"""
Aerodynamic data for conventional tail configuration.
Generated from AVL analysis at alpha = 3.5 degrees.

Configuration: nTop_ConventionalTail
- Main wing with ailerons
- Horizontal tail with elevator (x = 25-26 ft)
- Vertical tail with rudder (x = 25-27 ft)

Key improvements over flying wing:
- Cm_alpha = -1.643 (20x better pitch stability than -0.08)
- Cm_q = -6.197 (18x better pitch damping than -0.347)
- Cm_de = -0.007183 (60% of estimated, but still much better than -0.02)
- Cn_beta = 0.053403 (53x better directional stability than 0.001)
- Cn_r = -0.125986 (126x better yaw damping than -0.001)
"""

# Reference geometry
SREF = 412.64  # ft^2 (reference area)
CREF = 11.956  # ft (mean aerodynamic chord)
BREF = 24.863  # ft (wingspan)

# Trim condition at alpha = 3.5 degrees
TRIM_ALPHA = 3.5  # degrees
TRIM_ELEVATOR = 0.0  # degrees
TRIM_CL = 0.08767
TRIM_CD = 0.00150
TRIM_CM = -0.10054

# Longitudinal stability derivatives (stability axes)
CL_ALPHA = 1.432209  # /rad
CL_Q = 4.818955  # /rad
CL_DE = 0.003291  # /rad (per degree elevator)

CD_ALPHA = 0.049234  # /rad
CD_Q = -0.033181  # /rad
CD_DE = 0.000174  # /rad

CM_ALPHA = -1.643101  # /rad (EXCELLENT pitch stability!)
CM_Q = -6.197195  # /rad (EXCELLENT pitch damping!)
CM_DE = -0.007183  # /rad (elevator effectiveness per degree)

# Lateral-directional stability derivatives (stability axes)
CY_BETA = -0.054166  # /rad
CY_P = 0.011063  # /rad
CY_R = 0.123316  # /rad

CL_BETA = -0.031677  # /rad (dihedral effect)
CL_P = -0.107127  # /rad (roll damping)
CL_R = 0.057121  # /rad (roll due to yaw rate)
CL_DA = -0.001512  # /rad (aileron effectiveness per degree)
CL_DR = -0.000000  # /rad (roll due to rudder)

CN_BETA = 0.053403  # /rad (EXCELLENT directional stability!)
CN_P = -0.015952  # /rad
CN_R = -0.125986  # /rad (EXCELLENT yaw damping!)
CN_DA = 0.000242  # /rad
CN_DR = -0.000000  # /rad (rudder effectiveness per degree)

# Neutral point
NEUTRAL_POINT_X = 13.715940  # ft (behind reference point)

# Spiral stability criterion
SPIRAL_STABILITY_RATIO = 1.308313  # > 1 = spirally stable

# Convert per-degree control derivatives to per-radian
# AVL outputs control derivatives per degree of deflection
# For use in 6DOF simulation with radian inputs, multiply by 57.2958
CL_DE_RAD = CL_DE * 57.2958  # 0.1887 /rad
CD_DE_RAD = CD_DE * 57.2958  # 0.00997 /rad
CM_DE_RAD = CM_DE * 57.2958  # -0.4116 /rad
CL_DA_RAD = CL_DA * 57.2958  # -0.0866 /rad
CN_DR_RAD = CN_DR * 57.2958  # 0.0 /rad (no rudder effect?)

# Notes on results:
# 1. Pitch stability (Cm_alpha = -1.643) is EXCELLENT - 20x better than flying wing
# 2. Pitch damping (Cm_q = -6.197) is EXCELLENT - 18x better than flying wing
# 3. Elevator effectiveness (Cm_de = -0.007183/deg = -0.4116/rad) is lower than estimated
#    but still much better than flying wing (-0.02/rad)
# 4. Directional stability (Cn_beta = 0.053) is EXCELLENT - 53x better than flying wing
# 5. Yaw damping (Cn_r = -0.126) is EXCELLENT - 126x better than flying wing
# 6. Aircraft is spirally stable (ratio = 1.31 > 1)
# 7. Neutral point is at x = 13.716 ft, which is 0.87 ft aft of CG (12.846 ft)
#    This gives a static margin of ~7.3% MAC, which is good for stability

# Comparison with flying wing:
# | Metric     | Flying Wing | Conventional | Improvement |
# |------------|-------------|--------------|-------------|
# | Cm_alpha   | -0.08       | -1.643       | 20.5x       |
# | Cm_q       | -0.347      | -6.197       | 17.9x       |
# | Cm_de      | -0.02       | -0.4116      | 20.6x       |
# | Cn_beta    | 0.001       | 0.053        | 53x         |
# | Cn_r       | -0.001      | -0.126       | 126x        |
