"""
Run AVL using .run file approach for reliable execution.

This method is more robust than stdin piping.
"""

import subprocess
import os
import re
from pathlib import Path

def run_avl_with_runfile(avl_exe, avl_file, mass_file, run_file, output_prefix):
    """
    Run AVL using a .run file.

    The .run file specifies the run case (alpha, beta, controls, etc.)

    Parameters:
    -----------
    avl_exe : str
        Path to AVL executable
    avl_file : str
        Path to .avl geometry file
    mass_file : str
        Path to .mass file
    run_file : str
        Path to .run file with run case definition
    output_prefix : str
        Prefix for output files (.ft, .st)
    """

    # Create output directory
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    # AVL command file
    cmd_file = str(output_dir / "avl_commands.txt")

    # Build command sequence
    # Key insight: Use relative paths and run from geometry directory
    ft_file = os.path.basename(f"{output_prefix}.ft")
    st_file = os.path.basename(f"{output_prefix}.st")

    commands = [
        "LOAD",
        os.path.basename(avl_file),
        "MASS",
        os.path.basename(mass_file),
        "CASE",  # Load run case file
        os.path.basename(run_file),
        "OPER",
        "",  # Blank to acknowledge OPER menu display
        "X",  # Execute
        "FT",  # Write total forces
        ft_file,
        "ST",  # Write stability derivatives
        st_file,
        "",
        "QUIT"
    ]

    # Write command file
    with open(cmd_file, 'w') as f:
        f.write("\n".join(commands))

    print(f"Command file: {cmd_file}")
    print(f"Commands:")
    for cmd in commands:
        if cmd == "":
            print("  <blank>")
        else:
            print(f"  {cmd}")
    print()

    # Run AVL with command file as stdin
    print("Running AVL...")
    try:
        with open(cmd_file, 'r') as f:
            result = subprocess.run(
                [avl_exe],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(avl_file)  # Run from geometry directory
            )

        print("AVL completed")

        # Check for output files in geometry directory (where AVL writes them)
        geom_dir = os.path.dirname(avl_file)
        ft_basename = os.path.basename(f"{output_prefix}.ft")
        st_basename = os.path.basename(f"{output_prefix}.st")
        ft_file_local = os.path.join(geom_dir, ft_basename)
        st_file_local = os.path.join(geom_dir, st_basename)

        ft_exists = os.path.exists(ft_file_local)
        st_exists = os.path.exists(st_file_local)

        print(f"\nOutput files:")
        print(f"  {ft_file_local}: {'EXISTS' if ft_exists else 'NOT FOUND'}")
        print(f"  {st_file_local}: {'EXISTS' if st_exists else 'NOT FOUND'}")

        if not ft_exists or not st_exists:
            print("\nAVL stdout (last 2000 chars):")
            print(result.stdout[-2000:])
            print("\nAVL stderr:")
            print(result.stderr)
            return None

        # Parse results
        results = parse_avl_output(ft_file_local, st_file_local)
        return results

    except subprocess.TimeoutExpired:
        print("ERROR: AVL execution timed out")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def parse_avl_output(ft_file, st_file):
    """
    Parse AVL output files (.ft and .st).
    """

    results = {}

    # Parse forces file (.ft)
    if os.path.exists(ft_file):
        with open(ft_file, 'r') as f:
            content = f.read()

        patterns = {
            'CL': r'CLtot\s*=\s*([-+]?\d*\.\d+)',
            'CD': r'CDtot\s*=\s*([-+]?\d*\.\d+)',
            'CM': r'Cmtot\s*=\s*([-+]?\d*\.\d+)',
            'CY': r'CYtot\s*=\s*([-+]?\d*\.\d+)',
            'Cl': r'Cltot\s*=\s*([-+]?\d*\.\d+)',
            'Cn': r'Cntot\s*=\s*([-+]?\d*\.\d+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                results[key] = float(match.group(1))

    # Parse stability derivatives file (.st)
    if os.path.exists(st_file):
        with open(st_file, 'r') as f:
            content = f.read()

        patterns = {
            'CL_alpha': r'CLa\s*=\s*([-+]?\d*\.\d+)',
            'CL_q': r'CLq\s*=\s*([-+]?\d*\.\d+)',
            'CL_de': r'CLd1\s*=\s*([-+]?\d*\.\d+)',  # Control 1 = elevator
            'CD_0': r'CDff\s*=\s*([-+]?\d*\.\d+)',
            'Cm_alpha': r'Cma\s*=\s*([-+]?\d*\.\d+)',
            'Cm_q': r'Cmq\s*=\s*([-+]?\d*\.\d+)',
            'Cm_de': r'Cmd1\s*=\s*([-+]?\d*\.\d+)',
            'CY_beta': r'CYb\s*=\s*([-+]?\d*\.\d+)',
            'Cl_beta': r'Clb\s*=\s*([-+]?\d*\.\d+)',
            'Cl_p': r'Clp\s*=\s*([-+]?\d*\.\d+)',
            'Cl_r': r'Clr\s*=\s*([-+]?\d*\.\d+)',
            'Cl_da': r'Cld1\s*=\s*([-+]?\d*\.\d+)',  # Control 1 = aileron
            'Cl_dr': r'Cld2\s*=\s*([-+]?\d*\.\d+)',  # Control 2 = rudder
            'Cn_beta': r'Cnb\s*=\s*([-+]?\d*\.\d+)',
            'Cn_p': r'Cnp\s*=\s*([-+]?\d*\.\d+)',
            'Cn_r': r'Cnr\s*=\s*([-+]?\d*\.\d+)',
            'Cn_da': r'Cnd1\s*=\s*([-+]?\d*\.\d+)',
            'Cn_dr': r'Cnd2\s*=\s*([-+]?\d*\.\d+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                results[key] = float(match.group(1))

    return results


def main():
    print("=" * 70)
    print("AVL Analysis - Conventional Tail Configuration")
    print("Using .run file approach")
    print("=" * 70)
    print()

    # Paths
    avl_exe = r"C:\Users\bradrothenberg\OneDrive - nTop\Sync\AVL\avl.exe"
    avl_file = os.path.abspath("avl_files/uav_conventional.avl")
    mass_file = os.path.abspath("avl_files/uav_conventional.mass")
    run_file = os.path.abspath("avl_files/conventional_trim.run")
    output_prefix = os.path.abspath("avl_output/conventional_trim")

    if not os.path.exists(avl_exe):
        print(f"ERROR: AVL executable not found: {avl_exe}")
        return

    if not os.path.exists(avl_file):
        print(f"ERROR: AVL geometry file not found: {avl_file}")
        return

    if not os.path.exists(mass_file):
        print(f"ERROR: Mass file not found: {mass_file}")
        return

    if not os.path.exists(run_file):
        print(f"ERROR: Run file not found: {run_file}")
        return

    print(f"AVL executable: {avl_exe}")
    print(f"Geometry file:  {avl_file}")
    print(f"Mass file:      {mass_file}")
    print(f"Run file:       {run_file}")
    print(f"Output prefix:  {output_prefix}")
    print()

    # Run AVL
    results = run_avl_with_runfile(avl_exe, avl_file, mass_file, run_file, output_prefix)

    if results is None:
        print("\nERROR: AVL analysis failed")
        return

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("Total Forces/Moments:")
    for key in ['CL', 'CD', 'CM', 'CY', 'Cl', 'Cn']:
        if key in results:
            print(f"  {key:6s} = {results[key]:8.5f}")
    print()

    print("Longitudinal Stability Derivatives:")
    for key in ['CL_alpha', 'CL_q', 'CL_de', 'CD_0', 'Cm_alpha', 'Cm_q', 'Cm_de']:
        if key in results:
            print(f"  {key:10s} = {results[key]:8.5f}")
    print()

    print("Lateral-Directional Stability Derivatives:")
    for key in ['CY_beta', 'Cl_beta', 'Cl_p', 'Cl_r', 'Cl_da', 'Cl_dr',
                'Cn_beta', 'Cn_p', 'Cn_r', 'Cn_da', 'Cn_dr']:
        if key in results:
            print(f"  {key:10s} = {results[key]:8.5f}")
    print()

    # Save to Python file
    output_file = "conventional_aero_data_avl.py"
    with open(output_file, 'w') as f:
        f.write('"""\n')
        f.write('Aerodynamic data for conventional tail configuration.\n')
        f.write('Generated from AVL analysis using .run file.\n')
        f.write('"""\n\n')
        f.write('# Trim condition\n')
        f.write(f'TRIM_ALPHA = 3.5  # degrees\n')
        f.write(f'TRIM_ELEVATOR = 0.0  # degrees\n')
        f.write(f'TRIM_CL = {results.get("CL", 0.0):.6f}\n')
        f.write(f'TRIM_CD = {results.get("CD", 0.0):.6f}\n\n')
        f.write('# Stability derivatives\n')
        for key, value in sorted(results.items()):
            if key not in ['CL', 'CD', 'CM', 'CY', 'Cl', 'Cn']:
                f.write(f'{key.upper()} = {value:.6f}\n')

    print(f"Saved to: {output_file}")
    print()
    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
