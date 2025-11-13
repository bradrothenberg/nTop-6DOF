"""
AVL run case generator.

Creates .run files for different flight conditions.
"""

import numpy as np
from typing import Dict, List


def atmosphere_us_standard(altitude_ft: float) -> Dict[str, float]:
    """
    US Standard Atmosphere 1976 model.

    Parameters:
    -----------
    altitude_ft : float
        Geometric altitude in feet

    Returns:
    --------
    atm : dict
        Atmospheric properties (density, pressure, temperature, speed of sound)
    """
    # Constants
    R = 1716.0  # Gas constant for air, ft*lbf/(slug*R)
    gamma = 1.4  # Specific heat ratio
    g = 32.174  # Gravitational acceleration, ft/s^2

    # Sea level conditions
    T0 = 518.67  # Temperature at sea level, Rankine (59°F)
    P0 = 2116.22  # Pressure at sea level, lbf/ft^2
    rho0 = 0.002377  # Density at sea level, slug/ft^3

    # Troposphere (0 to 36,089 ft)
    if altitude_ft <= 36089:
        lapse_rate = -0.00356616  # °R/ft
        T = T0 + lapse_rate * altitude_ft
        P = P0 * (T / T0)**(-g / (lapse_rate * R))
        rho = P / (R * T)

    # Lower stratosphere (36,089 to 65,617 ft) - isothermal
    elif altitude_ft <= 65617:
        T = 389.97  # °R (constant)
        h1 = 36089.0
        T1 = T0 + (-0.00356616) * h1
        P1 = P0 * (T1 / T0)**(-g / (-0.00356616 * R))
        P = P1 * np.exp(-g * (altitude_ft - h1) / (R * T))
        rho = P / (R * T)

    else:
        # Upper stratosphere - approximation
        T = 389.97
        h1 = 36089.0
        T1 = T0 + (-0.00356616) * h1
        P1 = P0 * (T1 / T0)**(-g / (-0.00356616 * R))
        P = P1 * np.exp(-g * (65617 - h1) / (R * T))
        P = P * np.exp(-g * (altitude_ft - 65617) / (R * T))
        rho = P / (R * T)

    # Speed of sound
    a = np.sqrt(gamma * R * T)

    return {
        'temperature': T,  # Rankine
        'pressure': P,  # lbf/ft^2
        'density': rho,  # slug/ft^3
        'speed_of_sound': a  # ft/s
    }


def mach_to_velocity(mach: float, altitude_ft: float) -> float:
    """
    Convert Mach number to velocity in ft/s.

    Parameters:
    -----------
    mach : float
        Mach number
    altitude_ft : float
        Altitude in feet

    Returns:
    --------
    velocity : float
        Velocity in ft/s
    """
    atm = atmosphere_us_standard(altitude_ft)
    return mach * atm['speed_of_sound']


def create_run_case(name: str, alpha: float = 0.0, beta: float = 0.0,
                    pb_2v: float = 0.0, qc_2v: float = 0.0, rb_2v: float = 0.0,
                    elevator: float = 0.0, flaperon: float = 0.0, rudder: float = 0.0,
                    velocity: float = None, density: float = None,
                    gravity: float = 32.174, mass: float = None,
                    ix: float = None, iy: float = None, iz: float = None, iz_xz: float = None) -> str:
    """
    Create AVL run case string.

    Parameters:
    -----------
    name : str
        Case name
    alpha, beta : float
        Angle of attack and sideslip (degrees)
    pb_2v, qc_2v, rb_2v : float
        Normalized rotation rates
    elevator, flaperon, rudder : float
        Control deflections (degrees)
    velocity : float
        Flight velocity (ft/s)
    density : float
        Air density (slug/ft^3)
    gravity : float
        Gravitational acceleration (ft/s^2)
    mass : float
        Aircraft mass (slugs)
    ix, iy, iz, iz_xz : float
        Moments of inertia (slug*ft^2)

    Returns:
    --------
    run_case : str
        AVL run case string
    """
    lines = []
    lines.append(f" Run case  {name}:")
    lines.append(f"")
    lines.append(f" alpha        ->  alpha       =  {alpha:10.5f}")
    lines.append(f" beta         ->  beta        =  {beta:10.5f}")
    lines.append(f" pb/2V        ->  pb/2V       =  {pb_2v:10.5f}")
    lines.append(f" qc/2V        ->  qc/2V       =  {qc_2v:10.5f}")
    lines.append(f" rb/2V        ->  rb/2V       =  {rb_2v:10.5f}")

    if elevator != 0.0:
        lines.append(f" elevator     ->  Cm pitchmom =  {elevator:10.5f}")
    if flaperon != 0.0:
        lines.append(f" flaperon     ->  Cl rollmom  =  {flaperon:10.5f}")
    if rudder != 0.0:
        lines.append(f" rudder       ->  Cn yawmom   =  {rudder:10.5f}")

    if velocity is not None:
        lines.append(f"")
        lines.append(f" velocity     =  {velocity:10.3f}  ft/s")

    if density is not None:
        lines.append(f" density      =  {density:12.8f}  slugs/ft^3")

    if gravity is not None:
        lines.append(f" grav.acc.    =  {gravity:10.5f}  ft/s^2")

    if mass is not None:
        lines.append(f" mass         =  {mass:10.5f}  slugs")

    if ix is not None and iy is not None and iz is not None:
        lines.append(f" Ixx          =  {ix:12.4f}  slug-ft^2")
        lines.append(f" Iyy          =  {iy:12.4f}  slug-ft^2")
        lines.append(f" Izz          =  {iz:12.4f}  slug-ft^2")
        if iz_xz is not None:
            lines.append(f" Izx          =  {iz_xz:12.4f}  slug-ft^2")

    return "\n".join(lines)


def generate_run_cases(output_file: str, mass_slugs: float, inertia_slug_ft2: np.ndarray):
    """
    Generate AVL run cases for cruise, climb, and landing.

    Parameters:
    -----------
    output_file : str
        Output .run file path
    mass_slugs : float
        Aircraft mass in slugs
    inertia_slug_ft2 : np.ndarray
        Inertia tensor [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    """

    cases = []

    # Cruise condition: Mach 0.25 at 20,000 ft
    altitude_cruise = 20000.0
    mach_cruise = 0.25
    atm_cruise = atmosphere_us_standard(altitude_cruise)
    velocity_cruise = mach_cruise * atm_cruise['speed_of_sound']

    case_cruise = create_run_case(
        name="Cruise_M025_20kft",
        alpha=0.0,
        beta=0.0,
        velocity=velocity_cruise,
        density=atm_cruise['density'],
        gravity=32.174,
        mass=mass_slugs,
        ix=inertia_slug_ft2[0],
        iy=inertia_slug_ft2[1],
        iz=inertia_slug_ft2[2],
        iz_xz=inertia_slug_ft2[4]
    )
    cases.append(case_cruise)

    # Climb condition: Lower speed, higher alpha
    altitude_climb = 10000.0
    mach_climb = 0.20
    atm_climb = atmosphere_us_standard(altitude_climb)
    velocity_climb = mach_climb * atm_climb['speed_of_sound']

    case_climb = create_run_case(
        name="Climb_M020_10kft",
        alpha=5.0,  # Higher alpha for climb
        beta=0.0,
        velocity=velocity_climb,
        density=atm_climb['density'],
        gravity=32.174,
        mass=mass_slugs,
        ix=inertia_slug_ft2[0],
        iy=inertia_slug_ft2[1],
        iz=inertia_slug_ft2[2],
        iz_xz=inertia_slug_ft2[4]
    )
    cases.append(case_climb)

    # Landing condition: Low speed, high alpha, sea level
    altitude_landing = 0.0
    mach_landing = 0.15
    atm_landing = atmosphere_us_standard(altitude_landing)
    velocity_landing = mach_landing * atm_landing['speed_of_sound']

    case_landing = create_run_case(
        name="Landing_M015_SL",
        alpha=8.0,  # Higher alpha for landing
        beta=0.0,
        velocity=velocity_landing,
        density=atm_landing['density'],
        gravity=32.174,
        mass=mass_slugs,
        ix=inertia_slug_ft2[0],
        iy=inertia_slug_ft2[1],
        iz=inertia_slug_ft2[2],
        iz_xz=inertia_slug_ft2[4]
    )
    cases.append(case_landing)

    # Write run file (AVL format requires blank line, separator, then cases numbered)
    with open(output_file, 'w') as f:
        f.write("\n")  # Start with blank line
        for i, case in enumerate(cases, 1):
            f.write(" " + "-" * 70 + "\n")
            # Replace "Run case  Name:" with "Run case  #:  Name"
            case_lines = case.split('\n')
            if case_lines[0].startswith(' Run case  '):
                # Extract name from " Run case  Name:"
                name_part = case_lines[0].replace(' Run case  ', '').rstrip(':')
                case_lines[0] = f" Run case  {i}:  {name_part}"
            f.write('\n'.join(case_lines))
            f.write("\n\n")

    # Print summary
    print(f"Generated {len(cases)} run cases:")
    print(f"  1. Cruise:  Mach {mach_cruise:.2f} at {altitude_cruise:,.0f} ft, V = {velocity_cruise:.1f} ft/s")
    print(f"  2. Climb:   Mach {mach_climb:.2f} at {altitude_climb:,.0f} ft, V = {velocity_climb:.1f} ft/s")
    print(f"  3. Landing: Mach {mach_landing:.2f} at {altitude_landing:,.0f} ft, V = {velocity_landing:.1f} ft/s")


if __name__ == "__main__":
    import os
    import sys

    # Add parent directory for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from src.io.mass_properties import read_mass_csv

    # Paths
    base_path = r"C:\Users\bradrothenberg\OneDrive - nTop\OUT\parts\nTopAVL\nTop6DOF"
    data_path = os.path.join(base_path, "Data")
    output_path = os.path.join(base_path, "avl_files")

    mass_file = os.path.join(data_path, "mass.csv")
    output_file = os.path.join(output_path, "uav.run")

    # Read mass properties
    mass_props = read_mass_csv(mass_file)

    # Generate run cases
    generate_run_cases(output_file, mass_props.mass_slugs, mass_props.inertia_slug_ft2)

    print(f"\nAVL run file written to: {output_file}")
