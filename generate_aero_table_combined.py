"""
Generate complete aerodynamic table by running two AVL run files.

AVL has a limit of 25 run cases per file (NRMAX = 25).
To get 40 cases, we split into:
- Part 1: Cases 1-25 (alpha = -2, 0, 2, 4, 6 × elevator = -15, -10, -5, 0, 5)
- Part 2: Cases 1-15 (alpha = 8, 10, 12 × elevator = -15, -10, -5, 0, 5)

Combines results into single CSV table.
"""

import subprocess
import os
import numpy as np
import pandas as pd
from pathlib import Path

# AVL executable path
AVL_EXE = r"C:\Users\bradrothenberg\OneDrive - nTop\Sync\AVL\avl.exe"
AVL_DIR = Path("avl_files")
OUTPUT_DIR = Path("aero_tables")
OUTPUT_DIR.mkdir(exist_ok=True)

def parse_ft_file(ft_file_path):
    """Parse AVL .ft (total forces) file."""

    try:
        with open(ft_file_path, 'r') as f:
            content = f.read()

        # Extract alpha and elevator from file
        alpha = None
        elevator = None
        CL = None
        CD = None
        Cm = None

        lines = content.split('\n')
        for line in lines:
            if 'Alpha =' in line:
                alpha = float(line.split('=')[1].split()[0])
            if 'elevator' in line and '=' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        elevator = float(parts[-1].strip())
                    except:
                        pass
            if 'CLtot =' in line:
                CL = float(line.split('CLtot =')[1].strip())
            if 'CDtot =' in line:
                CD = float(line.split('CDtot =')[1].strip())
            if 'Cmtot =' in line:
                Cm = float(line.split('Cmtot =')[1].split()[0])

        if alpha is not None and elevator is not None and CL is not None and CD is not None and Cm is not None:
            return {
                'alpha': alpha,
                'elevator': elevator,
                'CL': CL,
                'CD': CD,
                'Cm': Cm
            }
        else:
            print(f"  WARNING: Missing data - alpha={alpha}, elevator={elevator}, CL={CL}, CD={CD}, Cm={Cm}")
            return None

    except Exception as e:
        print(f"  ERROR parsing {ft_file_path}: {e}")
        return None


def run_case(case_number, run_file_name, total_cases):
    """Run a specific case from the run file."""

    output_ft = f"{run_file_name}_case_{case_number}.ft"

    # Create batch commands
    commands = f"""CASE
{run_file_name}.run
OPER
{case_number}
X
FT
{output_ft}
O
QUIT

"""

    # Write batch file
    batch_file = AVL_DIR / f"batch_{run_file_name}_{case_number}.txt"
    with open(batch_file, 'w') as f:
        f.write(commands)

    # Run AVL
    try:
        result = subprocess.run(
            [AVL_EXE, "uav_conventional"],
            stdin=open(batch_file, 'r'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=AVL_DIR,
            timeout=10,
            text=True
        )

        # Parse output file
        output_file = AVL_DIR / output_ft
        if output_file.exists():
            data = parse_ft_file(output_file)
            # Clean up output file
            output_file.unlink()
            batch_file.unlink()
            return data
        else:
            print(f"  WARNING: Output file not created for case {case_number}")
            batch_file.unlink()
            return None

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for case {case_number}")
        if batch_file.exists():
            batch_file.unlink()
        return None
    except Exception as e:
        print(f"  ERROR for case {case_number}: {e}")
        if batch_file.exists():
            batch_file.unlink()
        return None


def generate_combined_table():
    """Generate complete aerodynamic table from both run files."""

    print("=" * 70)
    print("Generating Complete Aerodynamic Table (40 cases)")
    print("=" * 70)
    print()
    print("Part 1: uav_conventional_table.run (25 cases)")
    print("  Alpha: -2, 0, 2, 4, 6 deg")
    print("  Elevator: -15, -10, -5, 0, 5 deg")
    print()
    print("Part 2: uav_conventional_table_part2.run (15 cases)")
    print("  Alpha: 8, 10, 12 deg")
    print("  Elevator: -15, -10, -5, 0, 5 deg")
    print()

    results = []

    # Part 1: Cases 1-25
    print("=" * 70)
    print("PART 1: Running cases 1-25")
    print("=" * 70)
    for case_num in range(1, 26):
        print(f"[{case_num}/25] Running case {case_num}...", end='')

        data = run_case(case_num, "uav_conventional_table", 25)

        if data is not None:
            results.append(data)
            print(f" alpha={data['alpha']:+5.1f}°, elev={data['elevator']:+5.1f}° -> "
                  f"CL={data['CL']:+.4f}, CD={data['CD']:.5f}, Cm={data['Cm']:+.4f}")
        else:
            print(" FAILED")

    print()
    print(f"Part 1 complete: {len(results)}/25 cases successful")
    print()

    # Part 2: Cases 1-15 (renumbered from original 26-40)
    print("=" * 70)
    print("PART 2: Running cases 1-15 (alpha = 8, 10, 12)")
    print("=" * 70)
    part2_start = len(results)
    for case_num in range(1, 16):
        print(f"[{case_num}/15] Running case {case_num}...", end='')

        data = run_case(case_num, "uav_conventional_table_part2", 15)

        if data is not None:
            results.append(data)
            print(f" alpha={data['alpha']:+5.1f}°, elev={data['elevator']:+5.1f}° -> "
                  f"CL={data['CL']:+.4f}, CD={data['CD']:.5f}, Cm={data['Cm']:+.4f}")
        else:
            print(" FAILED")

    print()
    print(f"Part 2 complete: {len(results) - part2_start}/15 cases successful")
    print()
    print(f"TOTAL: {len(results)}/40 cases successful")
    print()

    if len(results) == 0:
        print("ERROR: No successful cases!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Remove any duplicates (shouldn't be any, but just in case)
    df_unique = df.drop_duplicates(subset=['alpha', 'elevator'])
    if len(df_unique) < len(df):
        print(f"WARNING: Removed {len(df) - len(df_unique)} duplicate entries")
        df = df_unique

    df = df.sort_values(['alpha', 'elevator'])

    # Save to CSV
    output_file = OUTPUT_DIR / "conventional_aero_table.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved aerodynamic table to: {output_file}")
    print()

    # Summary
    print("=" * 70)
    print("TABLE SUMMARY")
    print("=" * 70)
    print(f"Total unique points: {len(df)}")
    print(f"Alpha range: {df['alpha'].min():.0f} to {df['alpha'].max():.0f} deg ({len(df['alpha'].unique())} values)")
    print(f"Elevator range: {df['elevator'].min():.0f} to {df['elevator'].max():.0f} deg ({len(df['elevator'].unique())} values)")
    print(f"CL range: {df['CL'].min():+.4f} to {df['CL'].max():+.4f}")
    print(f"CD range: {df['CD'].min():.5f} to {df['CD'].max():.5f}")
    print(f"Cm range: {df['Cm'].min():+.4f} to {df['Cm'].max():+.4f}")
    print()

    # Show pivot table for easy viewing
    print("CL pivot table (rows=alpha, cols=elevator):")
    cl_pivot = df.pivot(index='alpha', columns='elevator', values='CL')
    print(cl_pivot.to_string())
    print()

    print("CD pivot table (rows=alpha, cols=elevator):")
    cd_pivot = df.pivot(index='alpha', columns='elevator', values='CD')
    print(cd_pivot.to_string())
    print()

    print("Cm pivot table (rows=alpha, cols=elevator):")
    cm_pivot = df.pivot(index='alpha', columns='elevator', values='Cm')
    print(cm_pivot.to_string())
    print()

    return df


if __name__ == "__main__":
    # Check prerequisites
    if not os.path.exists(AVL_EXE):
        print(f"ERROR: AVL executable not found at: {AVL_EXE}")
        exit(1)

    avl_file = AVL_DIR / "uav_conventional.avl"
    if not avl_file.exists():
        print(f"ERROR: AVL geometry file not found: {avl_file}")
        exit(1)

    run_file_1 = AVL_DIR / "uav_conventional_table.run"
    if not run_file_1.exists():
        print(f"ERROR: Run file 1 not found: {run_file_1}")
        exit(1)

    run_file_2 = AVL_DIR / "uav_conventional_table_part2.run"
    if not run_file_2.exists():
        print(f"ERROR: Run file 2 not found: {run_file_2}")
        exit(1)

    # Generate table
    df = generate_combined_table()

    if df is not None:
        print("=" * 70)
        print("Aerodynamic table generation COMPLETE!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("Aerodynamic table generation FAILED!")
        print("=" * 70)
