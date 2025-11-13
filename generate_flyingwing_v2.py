"""
Generate AVL geometry file for flying wing with winglets and elevons.

This script:
1. Reads wing, winglet, and elevon geometry from CSV files
2. Computes geometric properties
3. Generates AVL file with proper control surface definitions
4. Prepares for XFOIL/AVL hybrid aerodynamic analysis
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.io.geometry import (
    read_csv_points,
    compute_wing_geometry,
    compute_winglet_geometry,
    compute_elevon_geometry,
    print_geometry_summary
)
from src.aero.avl_geometry import AVLGeometryWriter
from src.io.mass_properties import read_mass_csv


def main():
    """Generate flying wing AVL file with winglets and elevons."""

    print("\n" + "=" * 70)
    print("Flying Wing V2 - AVL Geometry Generation")
    print("With Winglets and Elevons")
    print("=" * 70 + "\n")

    # File paths
    base_path = Path(__file__).parent
    data_path = base_path / "DATA"
    output_path = base_path / "avl_files"
    output_path.mkdir(exist_ok=True)

    le_file = data_path / "LEpts.csv"
    te_file = data_path / "TEpts.csv"
    winglet_file = data_path / "WINGLETpts.csv"
    elevon_file = data_path / "ELEVONpts.csv"
    mass_file = data_path / "mass.csv"

    # Check files exist
    for f in [le_file, te_file, winglet_file, elevon_file, mass_file]:
        if not f.exists():
            print(f"ERROR: Required file not found: {f}")
            return

    # Read geometry
    print("Reading geometry files...")
    le_points = read_csv_points(str(le_file), units='inches')
    te_points = read_csv_points(str(te_file), units='inches')
    winglet_points = read_csv_points(str(winglet_file), units='inches')
    elevon_points = read_csv_points(str(elevon_file), units='inches')

    # Compute wing geometry
    print("Computing wing geometry...")
    wing = compute_wing_geometry(le_points, te_points)

    # Compute winglet geometry
    print("Computing winglet geometry...")
    winglet = compute_winglet_geometry(winglet_points, wing)

    # Compute elevon geometry
    print("Computing elevon geometry...")
    elevon = compute_elevon_geometry(elevon_points, wing)

    # Read mass properties
    print("Reading mass properties...")
    mass_props = read_mass_csv(str(mass_file))

    # Print summary
    print("\n" + "=" * 70)
    print("GEOMETRY SUMMARY")
    print("=" * 70)

    print("\nWING:")
    print(f"  Span:              {wing.span:8.3f} ft")
    print(f"  Area:              {wing.area:8.3f} ft^2")
    print(f"  MAC:               {wing.mac:8.3f} ft")
    print(f"  Aspect Ratio:      {wing.aspect_ratio:8.3f}")
    print(f"  Taper Ratio:       {wing.taper_ratio:8.3f}")
    print(f"  LE Sweep:          {wing.sweep_le:8.2f} deg")
    print(f"  c/4 Sweep:         {wing.sweep_c4:8.2f} deg")
    print(f"  Dihedral:          {wing.dihedral:8.2f} deg")

    print("\nSPLIT WINGLETS:")
    print(f"  Attach Y:          {winglet.attach_y:8.3f} ft")
    print(f"  Attach Z:          {winglet.attach_z:8.3f} ft")
    print(f"\n  UPPER WINGLET:")
    print(f"    Height:          {winglet.upper_height:8.3f} ft")
    print(f"    Root Chord:      {winglet.upper_root_chord:8.3f} ft")
    print(f"    Tip Chord:       {winglet.upper_tip_chord:8.3f} ft")
    print(f"    Cant Angle:      {winglet.upper_cant_angle:8.2f} deg")
    print(f"    LE Sweep:        {winglet.upper_sweep_le:8.2f} deg")
    print(f"\n  LOWER WINGLET:")
    print(f"    Height:          {winglet.lower_height:8.3f} ft")
    print(f"    Root Chord:      {winglet.lower_root_chord:8.3f} ft")
    print(f"    Tip Chord:       {winglet.lower_tip_chord:8.3f} ft")
    print(f"    Cant Angle:      {winglet.lower_cant_angle:8.2f} deg")
    print(f"    LE Sweep:        {winglet.lower_sweep_le:8.2f} deg")

    print("\nELEVON:")
    print(f"  Y Inboard:         {elevon.y_inboard:8.3f} ft")
    print(f"  Y Outboard:        {elevon.y_outboard:8.3f} ft")
    print(f"  Span Coverage:     {elevon.y_outboard - elevon.y_inboard:8.3f} ft")
    print(f"  Hinge Chord Frac:  {elevon.chord_fraction:8.3f}")

    print("\nMASS PROPERTIES:")
    print(f"  Mass:              {mass_props.mass_slugs:8.3f} slugs")
    print(f"  CG:                ({mass_props.cg_ft[0]:8.4f}, {mass_props.cg_ft[1]:8.4f}, {mass_props.cg_ft[2]:8.4f}) ft")

    # Create AVL geometry writer
    print("\n" + "=" * 70)
    print("Generating AVL File...")
    print("=" * 70 + "\n")

    avl_writer = AVLGeometryWriter(name="nTop_FlyingWing_V2")

    # Set reference values
    # Note: Yref must be 0.0 for perfect symmetry in AVL analysis
    avl_writer.set_reference_values(
        s_ref=wing.area,
        c_ref=wing.mac,
        b_ref=wing.span,
        x_ref=mass_props.cg_ft[0],
        y_ref=0.0,  # Force to 0.0 for perfect symmetry
        z_ref=mass_props.cg_ft[2]
    )

    # Add main wing with elevons
    # Modify the wing to use proper elevon boundaries
    print("Adding main wing surface...")

    # For now, use the existing method (can be enhanced to use elevon geometry)
    avl_writer.add_wing_from_geometry(
        wing,
        airfoil="NACA 2412",
        flaperon_hinge=elevon.chord_fraction if elevon.chord_fraction > 0 else 0.75,
        flaperon_span=elevon.y_inboard / (wing.span / 2),  # Start elevon at inboard station
        name="Wing"
    )

    # Add split winglets (upper and lower)
    # Uses YDUPLICATE for perfect symmetry
    print("Adding split winglet surfaces...")
    avl_writer.add_split_winglets_from_geometry(
        winglet,
        airfoil="NACA 0012",
        elevon_hinge=0.75,
        has_elevon=True,
        use_yduplicate=True
    )

    # Write AVL file
    output_file = output_path / "uav_flyingwing_v2.avl"
    avl_writer.write_avl_file(str(output_file))

    print(f"\nAVL file written to: {output_file}")
    print("\n" + "=" * 70)
    print("SUCCESS: Flying wing V2 AVL geometry generated!")
    print("=" * 70 + "\n")

    print("Next steps:")
    print("  1. Visualize geometry in AVL")
    print("  2. Run AVL analysis to extract stability derivatives")
    print("  3. Run XFOIL for drag polars")
    print("  4. Generate combined aerodynamic tables")
    print("  5. Run 6-DOF simulation")
    print()


if __name__ == "__main__":
    main()
