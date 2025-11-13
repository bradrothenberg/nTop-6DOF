"""
Generate aerodynamic database for Flying Wing V2 using AVL.

This script:
1. Reads mass properties from DATA/mass.csv
2. Generates AVL .run file with flight conditions
3. Runs AVL alpha sweep
4. Creates aerodynamic database CSV for simulation
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.aero.avl_interface import AVLInterface
from src.aero.avl_run_cases import generate_run_cases, atmosphere_us_standard
from src.io.mass_properties import read_mass_csv


def main():
    """Generate flying wing V2 aerodynamic database."""

    print("\n" + "=" * 70)
    print("Flying Wing V2 - Aerodynamic Database Generation")
    print("=" * 70 + "\n")

    # File paths
    base_path = Path(__file__).parent
    avl_file = base_path / "avl_files" / "uav_flyingwing_v2.avl"
    mass_file = base_path / "DATA" / "mass.csv"
    output_dir = base_path / "aero_tables"
    output_dir.mkdir(exist_ok=True)

    # Check AVL file exists
    if not avl_file.exists():
        print(f"ERROR: AVL file not found: {avl_file}")
        print("Please run generate_flyingwing_v2.py first to create the geometry.")
        return

    # Check mass file exists
    if not mass_file.exists():
        print(f"ERROR: Mass file not found: {mass_file}")
        return

    # Read mass properties
    print("Reading mass properties...")
    mass_props = read_mass_csv(str(mass_file))

    print(f"  Mass:    {mass_props.mass_slugs:8.3f} slugs ({mass_props.mass_slugs * 32.174:.1f} lbm)")
    print(f"  CG:      ({mass_props.cg_ft[0]:8.4f}, {mass_props.cg_ft[1]:8.4f}, {mass_props.cg_ft[2]:8.4f}) ft")
    print(f"  Ixx:     {mass_props.inertia_slug_ft2[0]:12.1f} slug-ft^2")
    print(f"  Iyy:     {mass_props.inertia_slug_ft2[1]:12.1f} slug-ft^2")
    print(f"  Izz:     {mass_props.inertia_slug_ft2[2]:12.1f} slug-ft^2")
    print(f"  Ixz:     {mass_props.inertia_slug_ft2[4]:12.1f} slug-ft^2")

    # Generate .run file
    print("\nGenerating AVL .run file...")
    run_file = base_path / "avl_files" / "uav_flyingwing_v2.run"
    generate_run_cases(
        output_file=str(run_file),
        mass_slugs=mass_props.mass_slugs,
        inertia_slug_ft2=mass_props.inertia_slug_ft2
    )
    print(f"  Run file created: {run_file}")

    # Get reference geometry from AVL file (read from file)
    print("\nReading reference geometry from AVL file...")
    with open(avl_file, 'r') as f:
        lines = f.readlines()
        # Find Sref, Cref, Bref lines
        for i, line in enumerate(lines):
            if 'Sref' in line or '#Sref' in line:
                vals = lines[i+1].split()
                S_ref = float(vals[0])
                C_ref = float(vals[1])
                B_ref = float(vals[2])
                break

    print(f"  S_ref:   {S_ref:8.3f} ft^2")
    print(f"  C_ref:   {C_ref:8.3f} ft")
    print(f"  B_ref:   {B_ref:8.3f} ft")

    # Set up AVL interface
    print("\n" + "=" * 70)
    print("Running AVL Analysis...")
    print("=" * 70 + "\n")

    # Find AVL executable
    avl_exe = r"C:\Users\bradrothenberg\OneDrive - nTop\Sync\AVL\avl.exe"
    if not Path(avl_exe).exists():
        print(f"ERROR: AVL executable not found at: {avl_exe}")
        print("Please update the path in this script.")
        return

    avl = AVLInterface(avl_exe)

    # Define alpha sweep range
    # For flying wing: typically -5 to 15 degrees
    alpha_min = -5.0
    alpha_max = 15.0
    alpha_step = 1.0

    print(f"Alpha sweep: {alpha_min}° to {alpha_max}° in {alpha_step}° steps")
    print(f"Total cases: {int((alpha_max - alpha_min) / alpha_step) + 1}\n")

    # Run AVL sweep at cruise condition (Mach 0.3, 5000 ft - typical for flying wing)
    altitude = 5000.0  # ft
    mach = 0.3
    atm = atmosphere_us_standard(altitude)
    velocity = mach * atm['speed_of_sound']

    print(f"Flight condition:")
    print(f"  Altitude:  {altitude:8.1f} ft")
    print(f"  Mach:      {mach:8.3f}")
    print(f"  Velocity:  {velocity:8.1f} ft/s")
    print(f"  Density:   {atm['density']:.6f} slug/ft^3")
    print()

    # Run sweep
    print("Running AVL sweep (this may take a few minutes)...")
    results = avl.run_alpha_sweep(
        avl_file=str(avl_file),
        alpha_range=(alpha_min, alpha_max, alpha_step),
        beta=0.0,
        mach=mach
    )

    print(f"\nAVL sweep completed! Analyzed {len(results)} cases.\n")

    # Create aerodynamic database CSV
    print("=" * 70)
    print("Creating Aerodynamic Database...")
    print("=" * 70 + "\n")

    # Extract data into table format
    data = {
        'alpha_deg': [r.alpha for r in results],
        'CL': [r.CL for r in results],
        'CD': [r.CD for r in results],
        'Cm': [r.CM for r in results],
        'CY': [r.CY for r in results],
        'Cl': [r.Cl for r in results],
        'Cn': [r.Cn for r in results],
    }

    # Add stability derivatives if available
    if results[0].CLa is not None:
        data['CLa'] = [r.CLa for r in results]
        data['CMa'] = [r.CMa for r in results]

    df = pd.DataFrame(data)

    # Save to CSV
    output_csv = output_dir / "flyingwing_v2_aero.csv"
    df.to_csv(output_csv, index=False)

    print(f"Aerodynamic database saved to: {output_csv}")
    print()

    # Print summary statistics
    print("=" * 70)
    print("AERODYNAMIC SUMMARY")
    print("=" * 70)
    print()
    print("Coefficient Ranges:")
    print(f"  CL:   {df['CL'].min():8.4f} to {df['CL'].max():8.4f}")
    print(f"  CD:   {df['CD'].min():8.4f} to {df['CD'].max():8.4f}")
    print(f"  Cm:   {df['Cm'].min():8.4f} to {df['Cm'].max():8.4f}")
    print()

    # Find trim condition (Cm = 0)
    cm_zero_idx = np.argmin(np.abs(df['Cm'].values))
    alpha_trim = df.iloc[cm_zero_idx]['alpha_deg']
    CL_trim = df.iloc[cm_zero_idx]['CL']
    CD_trim = df.iloc[cm_zero_idx]['CD']
    L_D_trim = CL_trim / CD_trim if CD_trim > 0 else 0

    print(f"Trim Condition (Cm ≈ 0):")
    print(f"  Alpha:    {alpha_trim:8.2f}°")
    print(f"  CL:       {CL_trim:8.4f}")
    print(f"  CD:       {CD_trim:8.4f}")
    print(f"  L/D:      {L_D_trim:8.2f}")
    print()

    # Find max L/D
    L_D = df['CL'] / df['CD']
    max_ld_idx = L_D.idxmax()
    alpha_max_ld = df.iloc[max_ld_idx]['alpha_deg']
    CL_max_ld = df.iloc[max_ld_idx]['CL']
    CD_max_ld = df.iloc[max_ld_idx]['CD']
    max_ld = L_D.iloc[max_ld_idx]

    print(f"Maximum L/D:")
    print(f"  Alpha:    {alpha_max_ld:8.2f}°")
    print(f"  CL:       {CL_max_ld:8.4f}")
    print(f"  CD:       {CD_max_ld:8.4f}")
    print(f"  L/D:      {max_ld:8.2f}")
    print()

    # Stability derivatives (if available)
    if 'CLa' in df.columns:
        CLa_avg = df['CLa'].mean()
        CMa_avg = df['CMa'].mean()
        print(f"Stability Derivatives (average):")
        print(f"  CLa:      {CLa_avg:8.4f} /rad")
        print(f"  CMa:      {CMa_avg:8.4f} /rad")
        print(f"  Static margin: {-CMa_avg / CLa_avg * 100:8.2f}% MAC" if CLa_avg != 0 else "  N/A")
        print()

    print("=" * 70)
    print("SUCCESS! Aerodynamic database generated.")
    print("=" * 70)
    print()

    print("Next steps:")
    print(f"  1. Review the aerodynamic data in: {output_csv}")
    print(f"  2. Use AVLDatabase.from_avl_sweep() to load this data")
    print(f"  3. Create a simulation script using the database")
    print(f"  4. Run 6-DOF flight simulation with the new configuration")
    print()


if __name__ == "__main__":
    main()
