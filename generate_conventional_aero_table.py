"""
Generate full aerodynamic table for conventional tail configuration using AVL.

Sweeps alpha and elevator to build a comprehensive lookup table for:
- CL(alpha, elevator)
- CD(alpha, elevator)
- Cm(alpha, elevator)

This allows accurate force/moment prediction across the flight envelope.
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

def create_avl_run_case(alpha_deg, elevator_deg, case_name):
    """Create an AVL run case file for given alpha and elevator."""
    run_content = f"""-----------------------------------------------------------------------------
Run case  1:  {case_name}

alpha        ->  alpha       =   {alpha_deg:.5f}
beta         ->  beta        =   0.00000
pb/2V        ->  pb/2V       =   0.00000
qc/2V        ->  qc/2V       =   0.00000
rb/2V        ->  rb/2V       =   0.00000
aileron      ->  aileron     =   0.00000
elevator     ->  elevator    =   {elevator_deg:.5f}
rudder       ->  rudder      =   0.00000
"""
    return run_content

def create_avl_batch_commands(output_ft_file):
    """Create batch command file for AVL to execute and save results."""
    commands = f"""OPER
X
FT
{output_ft_file}

QUIT

"""
    return commands

def parse_avl_ft_file(ft_file_path):
    """Parse AVL .ft (total forces) file to extract CL, CD, Cm."""
    with open(ft_file_path, 'r') as f:
        lines = f.readlines()

    # Find the line with CLtot, CDtot
    CL = None
    CD = None
    Cm = None
    alpha = None
    elevator = None

    for i, line in enumerate(lines):
        if 'Alpha =' in line:
            alpha = float(line.split('=')[1].split()[0])
        if 'elevator' in line and '=' in line:
            parts = line.split('=')
            if len(parts) >= 2:
                try:
                    elevator = float(parts[1].strip())
                except:
                    pass
        if 'CLtot =' in line:
            CL = float(line.split('CLtot =')[1].strip())
        if 'CDtot =' in line:
            CD = float(line.split('CDtot =')[1].strip())
        if 'Cmtot =' in line:
            # Cmtot appears on same line as CYtot: "CYtot =   0.00000     Cmtot =  -0.10054"
            Cm = float(line.split('Cmtot =')[1].split()[0])

    return {
        'alpha': alpha,
        'elevator': elevator,
        'CL': CL,
        'CD': CD,
        'Cm': Cm
    }

def run_avl_case(alpha_deg, elevator_deg):
    """Run AVL for a specific alpha/elevator combination."""

    # Create temporary run file
    case_name = f"Alpha={alpha_deg:.1f}_Elev={elevator_deg:.1f}"
    run_file = AVL_DIR / "temp_sweep.run"
    batch_file = AVL_DIR / "temp_batch.txt"
    output_ft = "temp_sweep.ft"
    output_ft_path = AVL_DIR / output_ft

    # Write run file
    run_content = create_avl_run_case(alpha_deg, elevator_deg, case_name)
    with open(run_file, 'w') as f:
        f.write(run_content)

    # Write batch commands
    batch_content = create_avl_batch_commands(output_ft)
    with open(batch_file, 'w') as f:
        f.write(batch_content)

    # Run AVL
    try:
        result = subprocess.run(
            [AVL_EXE, "uav_conventional"],
            stdin=open(batch_file, 'r'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=AVL_DIR,
            timeout=10
        )

        # Parse results
        if output_ft_path.exists():
            data = parse_avl_ft_file(output_ft_path)
            # Clean up temp files
            output_ft_path.unlink()
            return data
        else:
            print(f"Warning: Output file not created for alpha={alpha_deg}, elevator={elevator_deg}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout for alpha={alpha_deg}, elevator={elevator_deg}")
        return None
    except Exception as e:
        print(f"Error running AVL: {e}")
        return None
    finally:
        # Clean up temp files
        if run_file.exists():
            run_file.unlink()
        if batch_file.exists():
            batch_file.unlink()

def generate_aero_table():
    """Generate complete aerodynamic table by sweeping alpha and elevator."""

    print("=" * 70)
    print("Generating Aerodynamic Table for Conventional Tail Configuration")
    print("=" * 70)
    print()

    # Generate sweep ranges
    alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP/2, ALPHA_STEP)
    elevators = np.arange(ELEVATOR_MIN, ELEVATOR_MAX + ELEVATOR_STEP/2, ELEVATOR_STEP)

    print(f"Alpha range: {ALPHA_MIN} to {ALPHA_MAX} deg ({len(alphas)} points)")
    print(f"Elevator range: {ELEVATOR_MIN} to {ELEVATOR_MAX} deg ({len(elevators)} points)")
    print(f"Total cases: {len(alphas) * len(elevators)}")
    print()

    # Run sweeps
    results = []
    total_cases = len(alphas) * len(elevators)
    case_num = 0

    for alpha in alphas:
        for elevator in elevators:
            case_num += 1
            print(f"Running case {case_num}/{total_cases}: alpha={alpha:.1f}°, elevator={elevator:.1f}°", end='')

            data = run_avl_case(alpha, elevator)

            if data is not None and data['CL'] is not None:
                results.append(data)
                print(f" -> CL={data['CL']:.4f}, CD={data['CD']:.5f}, Cm={data['Cm']:.4f}")
            else:
                print(" -> FAILED")

    print()
    print(f"Successfully completed {len(results)}/{total_cases} cases")
    print()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    output_file = OUTPUT_DIR / "conventional_aero_table.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved aerodynamic table to: {output_file}")

    # Print summary statistics
    print()
    print("=" * 70)
    print("TABLE SUMMARY")
    print("=" * 70)
    print(f"Alpha range: {df['alpha'].min():.1f} to {df['alpha'].max():.1f} deg")
    print(f"Elevator range: {df['elevator'].min():.1f} to {df['elevator'].max():.1f} deg")
    print(f"CL range: {df['CL'].min():.4f} to {df['CL'].max():.4f}")
    print(f"CD range: {df['CD'].min():.5f} to {df['CD'].max():.5f}")
    print(f"Cm range: {df['Cm'].min():.4f} to {df['Cm'].max():.4f}")
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

    print("Aerodynamic table generation complete!")
