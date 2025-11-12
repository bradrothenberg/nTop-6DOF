"""
Generate aerodynamic table from AVL run file with multiple cases.

Uses the uav_conventional_table.run file which contains 35 run cases
covering alpha = [-2, 0, 2, 4, 6, 8, 10] and elevator = [-15, -10, -5, 0, 5].

For each run case, executes it in AVL and extracts CL, CD, Cm.
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

# Run file with all cases
RUN_FILE = "uav_conventional_table"

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


def run_case(case_number, total_cases):
    """Run a specific case from the run file."""

    output_ft = f"case_{case_number}.ft"

    # Create batch commands:
    # 1. CASE {runfile}.run - load run file with all cases
    # 2. OPER - enter operating point mode
    # 3. {case_number} - select case number
    # 4. X - execute
    # 5. FT - write total forces
    # 6. {output_file} - filename
    # 7. O - overwrite if file exists
    # 8. QUIT
    commands = f"""CASE
{RUN_FILE}.run
OPER
{case_number}
X
FT
{output_ft}
O
QUIT

"""

    # Write batch file
    batch_file = AVL_DIR / f"batch_{case_number}.txt"
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


def generate_table():
    """Generate full aerodynamic table from run file cases."""

    print("=" * 70)
    print("Generating Aerodynamic Table from AVL Run File")
    print("=" * 70)
    print()
    print(f"Run file: {RUN_FILE}.run")
    print(f"Total cases: 45 (9 alphas × 5 elevators)")
    print()

    results = []
    total_cases = 45

    for case_num in range(1, total_cases + 1):
        print(f"[{case_num}/{total_cases}] Running case {case_num}...", end='')

        data = run_case(case_num, total_cases)

        if data is not None:
            results.append(data)
            print(f" alpha={data['alpha']:+5.1f}°, elev={data['elevator']:+5.1f}° -> "
                  f"CL={data['CL']:+.4f}, CD={data['CD']:.5f}, Cm={data['Cm']:+.4f}")
        else:
            print(" FAILED")

    print()
    print(f"Successfully completed {len(results)}/{total_cases} cases")
    print()

    if len(results) == 0:
        print("ERROR: No successful cases!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(['elevator', 'alpha'])

    # Save to CSV
    output_file = OUTPUT_DIR / "conventional_aero_table.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved aerodynamic table to: {output_file}")
    print()

    # Summary
    print("=" * 70)
    print("TABLE SUMMARY")
    print("=" * 70)
    print(f"Total points: {len(df)}")
    print(f"Alpha range: {df['alpha'].min():.0f} to {df['alpha'].max():.0f} deg")
    print(f"Elevator range: {df['elevator'].min():.0f} to {df['elevator'].max():.0f} deg")
    print(f"CL range: {df['CL'].min():+.4f} to {df['CL'].max():+.4f}")
    print(f"CD range: {df['CD'].min():.5f} to {df['CD'].max():.5f}")
    print(f"Cm range: {df['Cm'].min():+.4f} to {df['Cm'].max():+.4f}")
    print()

    # Show pivot table for easy viewing
    print("CL pivot table (rows=elevator, cols=alpha):")
    cl_pivot = df.pivot(index='elevator', columns='alpha', values='CL')
    print(cl_pivot.to_string())
    print()

    print("CD pivot table (rows=elevator, cols=alpha):")
    cd_pivot = df.pivot(index='elevator', columns='alpha', values='CD')
    print(cd_pivot.to_string())
    print()

    print("Cm pivot table (rows=elevator, cols=alpha):")
    cm_pivot = df.pivot(index='elevator', columns='alpha', values='Cm')
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

    run_file = AVL_DIR / f"{RUN_FILE}.run"
    if not run_file.exists():
        print(f"ERROR: Run file not found: {run_file}")
        exit(1)

    # Generate table
    df = generate_table()

    if df is not None:
        print("=" * 70)
        print("Aerodynamic table generation COMPLETE!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("Aerodynamic table generation FAILED!")
        print("=" * 70)
