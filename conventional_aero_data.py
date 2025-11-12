"""
Aerodynamic data for conventional tail configuration.
Estimated based on tail addition to flying wing.
"""

# Trim condition (estimated)
TRIM_ALPHA = 3.5  # degrees
TRIM_ELEVATOR = 0.0  # degrees
TRIM_CL = 0.45
TRIM_CD = 0.020

# Longitudinal stability derivatives
# Wing contributions similar to flying wing, plus strong tail contributions
CL_ALPHA = 4.8  # /rad (increased from 1.412 - wing + tail)
CL_Q = 3.5  # /rad (increased from 1.282)
CL_DE = 0.35  # /rad (elevator effectiveness)

CD_0 = 0.008  # Slightly higher due to tail drag
CD_ALPHA = 0.03
CD_ALPHA2 = 0.06

CM_0 = 0.0
CM_ALPHA = -0.85  # /rad (MUCH more negative - stable due to tail)
CM_Q = -12.0  # /rad (MUCH more negative - strong pitch damping from tail)
CM_DE = -1.2  # /rad (strong elevator authority)

# Lateral-directional stability derivatives
CY_BETA = -0.35  # /rad (increased from -0.2 due to vertical tail)
CL_BETA = -0.12  # /rad (similar to flying wing)
CL_P = -0.45  # /rad
CL_R = 0.15  # /rad
CL_DA = -0.08  # /rad (aileron effectiveness)

CN_BETA = 0.15  # /rad (MUCH more positive - directional stability from vertical tail)
CN_P = -0.08  # /rad
CN_R = -0.25  # /rad (strong yaw damping from vertical tail)
CN_DA = -0.01  # /rad (adverse yaw from ailerons)
CN_DR = -0.12  # /rad (rudder effectiveness)
