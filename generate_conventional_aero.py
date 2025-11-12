"""
Generate aerodynamic derivatives for conventional tail configuration using AVL.

This script:
1. Runs AVL to get stability derivatives
2. Finds trim condition for level flight
3. Saves aerodynamic data for simulation
"""

import sys
import os
from pathlib import Path
import subprocess
import re
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.aero.avl_interface import AVLInterface


def parse_st_file(st_file):
    """Parse AVL stability derivatives file (.st)."""

    derivatives = {}

    try:
        with open(st_file, 'r') as f:
            content = f.read()

            # Parse longitudinal derivatives
            patterns = {
                'CL_alpha': r'CLa\s*=\s*([-+]?\d*\.\d+)',
                'CL_q': r'CLq\s*=\s*([-+]?\d*\.\d+)',
                'CL_de': r'CLd1\s*=\s*([-+]?\d*\.\d+)',  # elevon/elevator
                'CD_0': r'CDff\s*=\s*([-+]?\d*\.\d+)',
                'Cm_alpha': r'Cma\s*=\s*([-+]?\d*\.\d+)',
                'Cm_q': r'Cmq\s*=\s*([-+]?\d*\.\d+)',
                'Cm_de': r'Cmd1\s*=\s*([-+]?\d*\.\d+)',

                # Lateral derivatives
                'CY_beta': r'CYb\s*=\s*([-+]?\d*\.\d+)',
                'Cl_beta': r'Clb\s*=\s*([-+]?\d*\.\d+)',
                'Cl_p': r'Clp\s*=\s*([-+]?\d*\.\d+)',
                'Cl_r': r'Clr\s*=\s*([-+]?\d*\.\d+)',
                'Cl_da': r'Cld1\s*=\s*([-+]?\d*\.\d+)',  # aileron
                'Cl_dr': r'Cld2\s*=\s*([-+]?\d*\.\d+)',  # rudder
                'Cn_beta': r'Cnb\s*=\s*([-+]?\d*\.\d+)',
                'Cn_p': r'Cnp\s*=\s*([-+]?\d*\.\d+)',
                'Cn_r': r'Cnr\s*=\s*([-+]?\d*\.\d+)',
                'Cn_da': r'Cnd1\s*=\s*([-+]?\d*\.\d+)',
                'Cn_dr': r'Cnd2\s*=\s*([-+]?\d*\.\d+)',
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    derivatives[key] = float(match.group(1))

    except Exception as e:
        print(f"Error parsing {st_file}: {e}")

    return derivatives


def find_trim_elevator(avl_exe, avl_file, mass_file, altitude=5000, airspeed=600):
    """
    Find elevator deflection for trim at specified flight condition.

    Returns:
    --------
    trim_results : dict
        Dictionary with trim values (alpha, elevator, CL, CD, CM)
    """

    print(f"\nFinding trim for:")
    print(f"  Altitude: {altitude} ft")
    print(f"  Airspeed: {airspeed} ft/s")
    print()

    # Create output directory
    output_dir = Path("avl_output")
    output_dir.mkdir(exist_ok=True)

    # Try different alpha values to find trim
    alphas = np.linspace(-5, 10, 31)

    best_trim = None
    min_cm = float('inf')

    for alpha in alphas:
        # Build AVL command sequence
        output_prefix = output_dir / f"trim_alpha_{alpha:.2f}"
        # Convert to absolute path for AVL
        ft_path = os.path.abspath(str(output_prefix) + ".ft")
        st_path = os.path.abspath(str(output_prefix) + ".st")

        commands = [
            "LOAD",
            avl_file,
            "MASS",
            mass_file,
            "OPER",
            "A",
            f"A {alpha}",
            "",  # Blank to confirm alpha
            "X",  # Execute
            "",  # Blank after execute - STAYS IN OPER
            "FT",
            ft_path,
            "",  # Blank after FT
            "ST",
            st_path,
            "",  # Blank after ST
            "QUIT",
            ""
        ]

        # Run AVL
        try:
            # Run from current directory so output files go to correct location
            result = subprocess.run(
                [avl_exe],
                input="\n".join(commands),
                capture_output=True,
                text=True,
                timeout=10
            )

            #Print AVL output for debugging
            if result.stdout:
                print(f"    AVL output: {result.stdout[:200]}")

            # Parse .ft file for CM
            if os.path.exists(ft_path):
                with open(ft_path, 'r') as f:
                    content = f.read()

                    match_cl = re.search(r'CLtot\s*=\s*([-+]?\d*\.\d+)', content)
                    match_cd = re.search(r'CDtot\s*=\s*([-+]?\d*\.\d+)', content)
                    match_cm = re.search(r'Cmtot\s*=\s*([-+]?\d*\.\d+)', content)

                    if match_cl and match_cd and match_cm:
                        CL = float(match_cl.group(1))
                        CD = float(match_cd.group(1))
                        CM = float(match_cm.group(1))

                        print(f"  alpha={alpha:6.2f}°  CL={CL:7.4f}  CD={CD:7.5f}  CM={CM:8.5f}")

                        if abs(CM) < abs(min_cm):
                            min_cm = CM
                            best_trim = {
                                'alpha': alpha,
                                'elevator': 0.0,  # No deflection needed at trim
                                'CL': CL,
                                'CD': CD,
                                'CM': CM
                            }

        except Exception as e:
            print(f"  Error at alpha={alpha}: {e}")
            continue

    if best_trim:
        print(f"\nBest trim found:")
        print(f"  Alpha: {best_trim['alpha']:.2f}°")
        print(f"  CL: {best_trim['CL']:.4f}")
        print(f"  CD: {best_trim['CD']:.5f}")
        print(f"  CM: {best_trim['CM']:.5f} (target: 0.0)")

    return best_trim


def main():
    print("=" * 70)
    print("Conventional Tail Configuration - AVL Analysis")
    print("=" * 70)
    print()

    # Paths
    avl_exe = r"C:\Users\bradrothenberg\OneDrive - nTop\Sync\AVL\avl.exe"
    avl_file = r"C:\Users\bradrothenberg\OneDrive - nTop\OUT\parts\nTopAVL\nTop6DOF\avl_files\uav_conventional.avl"
    mass_file = r"C:\Users\bradrothenberg\OneDrive - nTop\OUT\parts\nTopAVL\nTop6DOF\avl_files\uav_conventional.mass"

    if not os.path.exists(avl_exe):
        print(f"ERROR: AVL executable not found at {avl_exe}")
        return

    if not os.path.exists(avl_file):
        print(f"ERROR: AVL file not found at {avl_file}")
        return

    print(f"AVL executable: {avl_exe}")
    print(f"Geometry file: {avl_file}")
    print(f"Mass file: {mass_file}")
    print()

    # Find trim condition
    trim_results = find_trim_elevator(avl_exe, avl_file, mass_file,
                                      altitude=5000, airspeed=600)

    if not trim_results:
        print("\nERROR: Could not find trim condition")
        return

    # Generate stability derivatives at trim
    print("\nGenerating stability derivatives at trim...")
    output_dir = Path("avl_output")
    output_prefix = output_dir / "conventional_trim"

    # Absolute paths for AVL
    ft_path = os.path.abspath(str(output_prefix) + ".ft")
    st_path = os.path.abspath(str(output_prefix) + ".st")

    commands = [
        "LOAD",
        avl_file,
        "MASS",
        mass_file,
        "OPER",
        "A",
        f"A {trim_results['alpha']}",
        "",  # Blank to confirm alpha
        "X",  # Execute
        "",  # Blank after execute - STAYS IN OPER
        "FT",
        ft_path,
        "",  # Blank after FT
        "ST",
        st_path,
        "",  # Blank after ST
        "QUIT",
        ""
    ]

    result = subprocess.run(
        [avl_exe],
        input="\n".join(commands),
        capture_output=True,
        text=True,
        timeout=10
    )

    # Parse stability derivatives
    if os.path.exists(st_path):
        derivatives = parse_st_file(st_path)

        print("\nStability Derivatives:")
        print(f"  CL_alpha = {derivatives.get('CL_alpha', 0):.6f}")
        print(f"  CL_q     = {derivatives.get('CL_q', 0):.6f}")
        print(f"  CL_de    = {derivatives.get('CL_de', 0):.6f}")
        print(f"  Cm_alpha = {derivatives.get('Cm_alpha', 0):.6f}")
        print(f"  Cm_q     = {derivatives.get('Cm_q', 0):.6f}")
        print(f"  Cm_de    = {derivatives.get('Cm_de', 0):.6f}")
        print()
        print(f"  CY_beta  = {derivatives.get('CY_beta', 0):.6f}")
        print(f"  Cl_beta  = {derivatives.get('Cl_beta', 0):.6f}")
        print(f"  Cl_p     = {derivatives.get('Cl_p', 0):.6f}")
        print(f"  Cl_r     = {derivatives.get('Cl_r', 0):.6f}")
        print(f"  Cl_da    = {derivatives.get('Cl_da', 0):.6f}")
        print(f"  Cn_beta  = {derivatives.get('Cn_beta', 0):.6f}")
        print(f"  Cn_p     = {derivatives.get('Cn_p', 0):.6f}")
        print(f"  Cn_r     = {derivatives.get('Cn_r', 0):.6f}")
        print(f"  Cn_dr    = {derivatives.get('Cn_dr', 0):.6f}")

        # Save to Python file for easy import
        output_file = "conventional_aero_data.py"
        with open(output_file, 'w') as f:
            f.write('"""\n')
            f.write('Aerodynamic data for conventional tail configuration.\n')
            f.write('Generated by generate_conventional_aero.py\n')
            f.write('"""\n\n')
            f.write('# Trim condition\n')
            f.write(f'TRIM_ALPHA = {trim_results["alpha"]:.4f}  # degrees\n')
            f.write(f'TRIM_ELEVATOR = {trim_results["elevator"]:.4f}  # degrees\n')
            f.write(f'TRIM_CL = {trim_results["CL"]:.6f}\n')
            f.write(f'TRIM_CD = {trim_results["CD"]:.6f}\n\n')
            f.write('# Stability derivatives\n')
            for key, value in sorted(derivatives.items()):
                f.write(f'{key.upper()} = {value:.6f}\n')

        print(f"\nSaved aerodynamic data to: {output_file}")
    else:
        print(f"\nERROR: Stability derivatives file not found: {st_path}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
