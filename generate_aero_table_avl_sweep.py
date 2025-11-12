"""
Generate full aerodynamic table using AVL's built-in sweep capabilities.

Uses AVL's parametric sweep feature (A A for alpha, D1 D1 for elevator) to
generate CL, CD, Cm tables across the flight envelope.

This is simpler and more reliable than manual batch scripting.
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

# Sweep parameters
ALPHA_MIN = -2.0   # degrees
ALPHA_MAX = 12.0   # degrees
ALPHA_STEP = 1.0   # degrees

ELEVATOR_MIN = -15.0  # degrees
ELEVATOR_MAX = 5.0    # degrees
ELEVATOR_STEP = 2.0   # degrees


def create_avl_sweep_commands(alpha_start, alpha_end, alpha_step,
                               elevator_start, elevator_end, elevator_step,
                               output_polar_file):
    """
    Create AVL batch commands using parametric sweep.

    AVL sweep commands:
    - A A alpha_start alpha_end alpha_step  (alpha sweep)
    - D1 D1 elevator_start elevator_end elevator_step  (elevator sweep, D1 = first control)
    """

    # Use AVL's parametric sweep feature
    # We'll sweep elevator at each alpha
    commands = f"""OPER

"""

    # For each elevator deflection, sweep alpha and save polar
    elevators = np.arange(elevator_start, elevator_end + elevator_step/2, elevator_step)

    for i, elev in enumerate(elevators):
        polar_file = f"polar_elev_{elev:.1f}.txt"

        commands += f"""D1 D1 {elev} {elev} 1
A A {alpha_start} {alpha_end} {alpha_step}
W
{polar_file}

"""

    commands += """QUIT

"""

    return commands


def parse_avl_polar_file(polar_file_path):
    """
    Parse AVL polar file (generated from A A sweep).

    Format:
    # Alpha     CL        CD       ...    Cm      ...
      -2.000   -0.1234   0.0120   ...   0.0567  ...
      -1.000   -0.0567   0.0115   ...   0.0234  ...
    """

    results = []

    with open(polar_file_path, 'r') as f:
        lines = f.readlines()

    # Find header line (starts with #)
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            header_idx = i
            break

    if header_idx is None:
        print(f"Warning: Could not find header in {polar_file_path}")
        return results

    # Parse data lines
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            parts = line.split()
            if len(parts) >= 10:  # Typical AVL polar has many columns
                alpha = float(parts[0])
                CL = float(parts[1])
                CD = float(parts[2])
                # Cm is typically in column 5 or 6
                Cm = float(parts[5]) if len(parts) > 5 else 0.0

                results.append({
                    'alpha': alpha,
                    'CL': CL,
                    'CD': CD,
                    'Cm': Cm
                })
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse line: {line}")
            continue

    return results


def generate_aero_table():
    """Generate complete aerodynamic table using AVL's sweep feature."""

    print("=" * 70)
    print("Generating Aerodynamic Table - AVL Parametric Sweep")
    print("=" * 70)
    print()

    # Generate sweep ranges
    alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP/2, ALPHA_STEP)
    elevators = np.arange(ELEVATOR_MIN, ELEVATOR_MAX + ELEVATOR_STEP/2, ELEVATOR_STEP)

    print(f"Alpha range: {ALPHA_MIN} to {ALPHA_MAX} deg ({len(alphas)} points)")
    print(f"Elevator range: {ELEVATOR_MIN} to {ELEVATOR_MAX} deg ({len(elevators)} points)")
    print(f"Total cases: {len(alphas) * len(elevators)}")
    print()

    # Create batch command file
    batch_file = AVL_DIR / "sweep_batch.txt"
    output_polar_base = "polar"

    batch_content = create_avl_sweep_commands(
        ALPHA_MIN, ALPHA_MAX, ALPHA_STEP,
        ELEVATOR_MIN, ELEVATOR_MAX, ELEVATOR_STEP,
        output_polar_base
    )

    with open(batch_file, 'w') as f:
        f.write(batch_content)

    print(f"Created batch file: {batch_file}")
    print()

    # Run AVL with batch commands
    print("Running AVL sweep...")
    try:
        result = subprocess.run(
            [AVL_EXE, "uav_conventional"],
            stdin=open(batch_file, 'r'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=AVL_DIR,
            timeout=60,
            text=True
        )

        print("AVL execution complete!")
        print()

        # Debug: print AVL output
        if result.stdout:
            print("AVL stdout (last 50 lines):")
            print('\n'.join(result.stdout.split('\n')[-50:]))
            print()

        if result.stderr:
            print("AVL stderr:")
            print(result.stderr)
            print()

    except subprocess.TimeoutExpired:
        print("ERROR: AVL execution timed out")
        return None
    except Exception as e:
        print(f"ERROR running AVL: {e}")
        return None

    # Parse polar files
    print("Parsing polar files...")
    all_results = []

    for elev in elevators:
        polar_file = AVL_DIR / f"polar_elev_{elev:.1f}.txt"

        if polar_file.exists():
            print(f"  Reading {polar_file.name}...")
            polar_data = parse_avl_polar_file(polar_file)

            # Add elevator value to each data point
            for data in polar_data:
                data['elevator'] = elev
                all_results.append(data)

            print(f"    -> {len(polar_data)} alpha points")
        else:
            print(f"  WARNING: {polar_file.name} not found")

    print()
    print(f"Total data points: {len(all_results)}")
    print()

    if len(all_results) == 0:
        print("ERROR: No data parsed from polar files!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Reorder columns
    df = df[['alpha', 'elevator', 'CL', 'CD', 'Cm']]

    # Sort by elevator then alpha
    df = df.sort_values(['elevator', 'alpha'])

    # Save to CSV
    output_file = OUTPUT_DIR / "conventional_aero_table.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved aerodynamic table to: {output_file}")
    print()

    # Print summary statistics
    print("=" * 70)
    print("TABLE SUMMARY")
    print("=" * 70)
    print(f"Total data points: {len(df)}")
    print(f"Alpha range: {df['alpha'].min():.1f} to {df['alpha'].max():.1f} deg")
    print(f"Elevator range: {df['elevator'].min():.1f} to {df['elevator'].max():.1f} deg")
    print(f"CL range: {df['CL'].min():.4f} to {df['CL'].max():.4f}")
    print(f"CD range: {df['CD'].min():.5f} to {df['CD'].max():.5f}")
    print(f"Cm range: {df['Cm'].min():.4f} to {df['Cm'].max():.4f}")
    print()

    # Show sample data
    print("Sample data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    print()

    return df


if __name__ == "__main__":
    # Check that AVL executable exists
    if not os.path.exists(AVL_EXE):
        print(f"ERROR: AVL executable not found at: {AVL_EXE}")
        print("Please update AVL_EXE path in this script")
        exit(1)

    # Check that AVL files exist
    avl_file = AVL_DIR / "uav_conventional.avl"
    if not avl_file.exists():
        print(f"ERROR: AVL geometry file not found: {avl_file}")
        exit(1)

    # Generate the table
    df = generate_aero_table()

    if df is not None:
        print("Aerodynamic table generation complete!")
    else:
        print("Aerodynamic table generation FAILED!")
