"""
Check what alpha results from the corrected velocity initialization.
"""
import numpy as np
from src.core.state import State

theta = np.radians(1.7531)
V = 548.5

# Old way (incorrect for level flight)
vx_old = V * np.cos(theta)
vz_old = V * np.sin(theta)

# New way (correct for level flight)
vx_new = V * np.cos(theta)
vz_new = -V * np.sin(theta)

print("=" * 70)
print("Alpha Check")
print("=" * 70)
print()

print(f"Theta: {np.degrees(theta):.4f} deg")
print()

print("Old initialization (descent):")
print(f"  vx: {vx_old:.2f} ft/s")
print(f"  vz: {vz_old:.2f} ft/s (positive = down in body)")
alpha_old = np.arctan2(vz_old, vx_old)
print(f"  alpha = atan2(vz, vx) = {np.degrees(alpha_old):.4f} deg")
print()

print("New initialization (level flight):")
print(f"  vx: {vx_new:.2f} ft/s")
print(f"  vz: {vz_new:.2f} ft/s (negative = up in body)")
alpha_new = np.arctan2(vz_new, vx_new)
print(f"  alpha = atan2(vz, vx) = {np.degrees(alpha_new):.4f} deg")
print()

print("But the trim solver calculated alpha for CL = CL_trim:")
alpha_trim_calc = np.radians(1.7531)
print(f"  alpha_trim_from_CL = {np.degrees(alpha_trim_calc):.4f} deg")
print()

print("PROBLEM:")
print("  The trim solver calculated alpha assuming vz > 0 (descent)")
print("  But we initialized with vz < 0 (level flight)")
print("  So the actual alpha doesn't match the calculated alpha!")
print()

print("The aerodynamic coefficients depend on alpha.")
print("With wrong alpha, lift will be wrong, causing vertical acceleration.")
print()

print("=" * 70)
