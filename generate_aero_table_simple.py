"""
Generate aerodynamic table by running AVL for each alpha/elevator combination.

Simple approach: For each case, create commands to set alpha, elevator, execute,
and write output. Then parse the .ft file.
"""

import subprocess
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time

# AVL executable path
AVL_EXE = r"C:\Users\bradrothenberg\OneDrive - nTop\Sync\AVL\avl.exe"
AVL_DIR = Path("avl_files")
OUTPUT_DIR = Path("aero_tables")
OUTPUT_DIR.mkdir(exist_ok=True)

# Reduced sweep for testing - can expand later
ALPHAS = [-2, 0, 2, 4, 6, 8, 10]  # degrees
ELEVATORS = [-15, -10, -5, 0, 5]  # degrees


def run_single_case(alpha_deg, elevator_deg):
    """Run AVL for a single alpha/elevator combination."""

    output_ft = f"case_a{alpha_deg}_e{elevator_deg}.ft"

    # Create batch commands:
    # 1. Enter OPER mode
    # 2. Set alpha value
    # 3. Set elevator value (D2 = control surface 2 = elevator)
    # 4. Execute (X)
    # 5. Write forces to file (FT)
    # 6. Quit
    commands = f"""OPER
A
{alpha_deg}

D2
{elevator_deg}

X
FT
{output_ft}

QUIT

"""

    # Write batch file
    batch_file = AVL_DIR / "temp_batch.txt"
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
            data = parse_ft_file(output_file, alpha_deg, elevator_deg)
            # Clean up output file
            output_file.unlink()
            return data
        else:
            print(f"  WARNING: Output file not created for alpha={alpha_deg}, elevator={elevator_deg}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for alpha={alpha_deg}, elevator={elevator_deg}")
        return None
    except Exception as e:
        print(f"  ERROR for alpha={alpha_deg}, elevator={elevator_deg}: {e}")
        return None
    finally:
        # Clean up batch file
        if batch_file.exists():
            batch_file.unlink()


def parse_ft_file(ft_file_path, alpha_deg, elevator_deg):
    """Parse AVL .ft (total forces) file."""

    try:
        with open(ft_file_path, 'r') as f:
            content = f.read()

        # Extract values using string search
        CL = None
        CD = None
        Cm = None

        # Look for "CLtot =   0.08767"
        if 'CLtot =' in content:
            line = [l for l in content.split('\n') if 'CLtot =' in l][0]
            CL = float(line.split('CLtot =')[1].strip())

        # Look for "CDtot =   0.00150"
        if 'CDtot =' in content:
            line = [l for l in content.split('\n') if 'CDtot =' in l][0]
            CD = float(line.split('CDtot =')[1].strip())

        # Look for "Cmtot" - it appears on same line as CYtot
        if 'Cmtot =' in content:
            line = [l for l in content.split('\n') if 'Cmtot =' in l][0]
            Cm = float(line.split('Cmtot =')[1].split()[0])

        if CL is not None and CD is not None and Cm is not None:
            return {
                'alpha': alpha_deg,
                'elevator': elevator_deg,
                'CL': CL,
                'CD': CD,
                'Cm': Cm
            }
        else:
            print(f"  WARNING: Could not parse all coefficients (CL={CL}, CD={CD}, Cm={Cm})")
            return None

    except Exception as e:
        print(f"  ERROR parsing {ft_file_path}: {e}")
        return None


def generate_table():
    """Generate full aerodynamic table."""

    print("=" * 70)
    print("Generating Aerodynamic Table - Simple Case-by-Case Approach")
    print("=" * 70)
    print()
    print(f"Alpha points: {ALPHAS}")
    print(f"Elevator points: {ELEVATORS}")
    print(f"Total cases: {len(ALPHAS) * len(ELEVATORS)}")
    print()

    results = []
    total_cases = len(ALPHAS) * len(ELEVATORS)
    case_num = 0

    for alpha in ALPHAS:
        for elevator in ELEVATORS:
            case_num += 1
            print(f"[{case_num}/{total_cases}] alpha={alpha:+3d}°, elevator={elevator:+3d}°", end='')

            data = run_single_case(alpha, elevator)

            if data is not None:
                results.append(data)
                print(f" -> CL={data['CL']:+.4f}, CD={data['CD']:.5f}, Cm={data['Cm']:+.4f}")
            else:
                print(" -> FAILED")

            time.sleep(0.1)  # Small delay to avoid issues

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

    # Show data
    print("Full table:")
    print(df.to_string(index=False))
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

    # Generate table
    df = generate_table()

    if df is not None:
        print("Aerodynamic table generation complete!")
    else:
        print("Aerodynamic table generation FAILED!")
