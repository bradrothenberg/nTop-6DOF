"""
Geometry parser for nTop-exported wing and tail surfaces.

This module reads LE/TE point data from CSV files and computes
geometric properties for AVL input generation.

Units: Assumes input in inches, converts to feet for AVL
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class WingGeometry:
    """Container for wing geometric properties."""
    le_points: np.ndarray  # Leading edge points (N x 3) in feet
    te_points: np.ndarray  # Trailing edge points (N x 3) in feet
    span: float            # Total span in feet
    area: float            # Planform area in ft^2
    mac: float             # Mean aerodynamic chord in feet
    taper_ratio: float     # Tip chord / root chord
    aspect_ratio: float    # b^2 / S
    sweep_le: float        # Leading edge sweep angle in degrees
    sweep_c4: float        # Quarter-chord sweep angle in degrees
    root_chord: float      # Root chord in feet
    tip_chord: float       # Tip chord in feet
    mac_y: float           # Spanwise location of MAC in feet
    mac_le_x: float        # X-location of MAC leading edge in feet
    dihedral: float        # Dihedral angle in degrees


@dataclass
class WingletGeometry:
    """Container for winglet geometric properties."""
    points: np.ndarray     # All winglet points (N x 3) in feet
    upper_root_le: np.ndarray    # Upper root leading edge position (3,) in feet
    upper_root_te: np.ndarray    # Upper root trailing edge position (3,) in feet
    upper_tip_le: np.ndarray     # Upper tip leading edge position (3,) in feet
    upper_tip_te: np.ndarray     # Upper tip trailing edge position (3,) in feet
    lower_root_le: np.ndarray    # Lower root leading edge position (3,) in feet
    lower_root_te: np.ndarray    # Lower root trailing edge position (3,) in feet
    lower_tip_le: np.ndarray     # Lower tip leading edge position (3,) in feet
    lower_tip_te: np.ndarray     # Lower tip trailing edge position (3,) in feet
    upper_root_chord: float      # Upper winglet root chord in feet
    upper_tip_chord: float       # Upper winglet tip chord in feet
    upper_height: float          # Upper winglet height (span) in feet
    lower_root_chord: float      # Lower winglet root chord in feet
    lower_tip_chord: float       # Lower winglet tip chord in feet
    lower_height: float          # Lower winglet height (span) in feet
    upper_cant_angle: float      # Upper winglet cant angle in degrees
    lower_cant_angle: float      # Lower winglet cant angle in degrees
    upper_sweep_le: float        # Upper winglet LE sweep in degrees
    lower_sweep_le: float        # Lower winglet LE sweep in degrees
    attach_y: float              # Attachment Y station on main wing in feet
    attach_z: float              # Attachment Z position on main wing in feet


@dataclass
class ElevonGeometry:
    """Container for elevon control surface geometry."""
    points: np.ndarray     # Elevon boundary points (4 x 3) in feet
    y_inboard: float       # Inboard span station in feet
    y_outboard: float      # Outboard span station in feet
    hinge_line_x: callable # Function: y -> x position of hinge line
    chord_fraction: float  # Average hinge position as fraction of chord


def read_csv_points(filepath: str, units: str = 'inches') -> np.ndarray:
    """
    Read LE or TE points from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to CSV file with x,y,z columns
    units : str
        Input units ('inches' or 'feet')

    Returns:
    --------
    points : np.ndarray
        (N x 3) array of points in feet
    """
    df = pd.read_csv(filepath)
    points = df[['x', 'y', 'z']].values

    # Convert to feet if needed
    if units.lower() == 'inches':
        points = points / 12.0

    return points


def compute_wing_geometry(le_points: np.ndarray, te_points: np.ndarray) -> WingGeometry:
    """
    Compute wing geometric properties from LE/TE points.

    Parameters:
    -----------
    le_points : np.ndarray
        Leading edge points (N x 3) in feet
    te_points : np.ndarray
        Trailing edge points (N x 3) in feet

    Returns:
    --------
    geom : WingGeometry
        Container with all geometric properties
    """

    # Sort by spanwise position (y-coordinate)
    le_sorted = le_points[np.argsort(le_points[:, 1])]
    te_sorted = te_points[np.argsort(te_points[:, 1])]

    # Extract coordinates
    y_stations = le_sorted[:, 1]
    le_x = le_sorted[:, 0]
    le_z = le_sorted[:, 2]
    te_x = te_sorted[:, 0]
    te_z = te_sorted[:, 2]

    # Compute local chord at each station
    chords = te_x - le_x

    # Span (tip to tip)
    span = y_stations[-1] - y_stations[0]

    # Root and tip chords
    root_idx = np.argmin(np.abs(y_stations))
    root_chord = chords[root_idx]
    tip_chord = (chords[0] + chords[-1]) / 2.0  # Average both tips

    # Taper ratio
    taper_ratio = tip_chord / root_chord

    # Planform area (use full span integration)
    area = np.trapezoid(chords, y_stations)

    # Mean aerodynamic chord (MAC) and its location
    # MAC = (2/S) * integral(c^2 * dy)
    mac = (2.0 / area) * np.trapezoid(chords**2, y_stations)

    # MAC spanwise location: y_mac = (2/S) * integral(c * y * dy)
    mac_y = (2.0 / area) * np.trapezoid(chords * y_stations, y_stations)

    # MAC leading edge x-location (interpolate)
    mac_le_x = np.interp(mac_y, y_stations, le_x)

    # Aspect ratio
    aspect_ratio = span**2 / area

    # Leading edge sweep angle (using right half-span)
    half_span_idx = np.where(y_stations >= 0)[0]
    y_half = y_stations[half_span_idx]
    le_x_half = le_x[half_span_idx]

    if len(y_half) > 1:
        le_sweep_fit = np.polyfit(y_half, le_x_half, 1)
        sweep_le = np.degrees(np.arctan(le_sweep_fit[0]))
    else:
        sweep_le = 0.0

    # Quarter-chord sweep angle
    x_c4 = le_x + 0.25 * chords
    x_c4_half = x_c4[half_span_idx]

    if len(y_half) > 1:
        c4_sweep_fit = np.polyfit(y_half, x_c4_half, 1)
        sweep_c4 = np.degrees(np.arctan(c4_sweep_fit[0]))
    else:
        sweep_c4 = 0.0

    # Dihedral angle (from z vs y on right half-span)
    le_z_half = le_z[half_span_idx]

    if len(y_half) > 1:
        dihedral_fit = np.polyfit(y_half, le_z_half, 1)
        dihedral = np.degrees(np.arctan(dihedral_fit[0]))
    else:
        dihedral = 0.0

    return WingGeometry(
        le_points=le_sorted,
        te_points=te_sorted,
        span=span,
        area=area,
        mac=mac,
        taper_ratio=taper_ratio,
        aspect_ratio=aspect_ratio,
        sweep_le=sweep_le,
        sweep_c4=sweep_c4,
        root_chord=root_chord,
        tip_chord=tip_chord,
        mac_y=mac_y,
        mac_le_x=mac_le_x,
        dihedral=dihedral
    )


def estimate_tail_geometry(wing: WingGeometry,
                          v_h: float = 0.6,
                          v_v: float = 0.05,
                          tail_arm_factor: float = 2.5) -> Tuple[Dict, Dict]:
    """
    Estimate horizontal and vertical tail geometry using volume coefficients.

    Parameters:
    -----------
    wing : WingGeometry
        Wing geometric properties
    v_h : float
        Horizontal tail volume coefficient (typical: 0.5-0.7)
    v_v : float
        Vertical tail volume coefficient (typical: 0.04-0.06)
    tail_arm_factor : float
        Tail moment arm as multiple of wing MAC

    Returns:
    --------
    h_tail : dict
        Horizontal tail geometry (area, span, chord, etc.)
    v_tail : dict
        Vertical tail geometry (area, span, chord, etc.)
    """

    # Tail moment arms
    l_h = tail_arm_factor * wing.mac
    l_v = tail_arm_factor * wing.mac

    # Horizontal tail area: S_h = (V_h * S * MAC) / l_h
    s_h = (v_h * wing.area * wing.mac) / l_h

    # Assume horizontal tail aspect ratio similar to wing (scaled down)
    ar_h = wing.aspect_ratio * 0.8
    b_h = np.sqrt(s_h * ar_h)
    c_h = s_h / b_h

    # Vertical tail area: S_v = (V_v * S * b) / l_v
    s_v = (v_v * wing.area * wing.span) / l_v

    # Assume vertical tail aspect ratio (height/chord)
    ar_v = 1.5
    h_v = np.sqrt(s_v * ar_v)
    c_v = s_v / h_v

    # Horizontal tail position - start at root trailing edge
    # Find root chord trailing edge location
    root_idx = np.argmin(np.abs(wing.le_points[:, 1]))
    root_te_x = wing.te_points[root_idx, 0]

    x_h = root_te_x  # Tail starts at root TE

    # Vertical tail position - same as horizontal
    x_v = root_te_x

    h_tail = {
        'area': s_h,
        'span': b_h,
        'chord': c_h,
        'aspect_ratio': ar_h,
        'taper_ratio': 0.8,  # Assume slight taper
        'sweep_c4': 0.0,     # Assume no sweep
        'x_position': x_h,
        'y_position': 0.0,
        'z_position': 0.0,   # Assume on centerline
        'moment_arm': l_h
    }

    v_tail = {
        'area': s_v,
        'height': h_v,
        'chord': c_v,
        'aspect_ratio': ar_v,
        'taper_ratio': 0.7,  # Assume taper
        'sweep_c4': 5.0,     # Slight sweep for stability
        'x_position': x_v,
        'y_position': 0.0,
        'z_position': 0.0,
        'moment_arm': l_v
    }

    return h_tail, v_tail


def print_geometry_summary(wing: WingGeometry, h_tail: Dict = None, v_tail: Dict = None):
    """Print formatted summary of geometric properties."""

    print("=" * 60)
    print("WING GEOMETRY SUMMARY")
    print("=" * 60)
    print(f"Span:              {wing.span:8.3f} ft")
    print(f"Area:              {wing.area:8.3f} ft^2")
    print(f"Mean Aero Chord:   {wing.mac:8.3f} ft")
    print(f"Root Chord:        {wing.root_chord:8.3f} ft")
    print(f"Tip Chord:         {wing.tip_chord:8.3f} ft")
    print(f"Taper Ratio:       {wing.taper_ratio:8.3f}")
    print(f"Aspect Ratio:      {wing.aspect_ratio:8.3f}")
    print(f"LE Sweep:          {wing.sweep_le:8.2f} deg")
    print(f"c/4 Sweep:         {wing.sweep_c4:8.2f} deg")
    print(f"Dihedral:          {wing.dihedral:8.2f} deg")
    print(f"MAC Y-location:    {wing.mac_y:8.3f} ft")
    print(f"MAC LE X-location: {wing.mac_le_x:8.3f} ft")

    if h_tail:
        print("\n" + "=" * 60)
        print("HORIZONTAL TAIL GEOMETRY (ESTIMATED)")
        print("=" * 60)
        print(f"Area:              {h_tail['area']:8.3f} ft^2")
        print(f"Span:              {h_tail['span']:8.3f} ft")
        print(f"Chord:             {h_tail['chord']:8.3f} ft")
        print(f"Aspect Ratio:      {h_tail['aspect_ratio']:8.3f}")
        print(f"Taper Ratio:       {h_tail['taper_ratio']:8.3f}")
        print(f"X Position:        {h_tail['x_position']:8.3f} ft")
        print(f"Moment Arm:        {h_tail['moment_arm']:8.3f} ft")

    if v_tail:
        print("\n" + "=" * 60)
        print("VERTICAL TAIL GEOMETRY (ESTIMATED)")
        print("=" * 60)
        print(f"Area:              {v_tail['area']:8.3f} ft^2")
        print(f"Height:            {v_tail['height']:8.3f} ft")
        print(f"Chord:             {v_tail['chord']:8.3f} ft")
        print(f"Aspect Ratio:      {v_tail['aspect_ratio']:8.3f}")
        print(f"Taper Ratio:       {v_tail['taper_ratio']:8.3f}")
        print(f"X Position:        {v_tail['x_position']:8.3f} ft")
        print(f"Moment Arm:        {v_tail['moment_arm']:8.3f} ft")

    print("=" * 60)


def compute_winglet_geometry(winglet_points: np.ndarray, wing: WingGeometry) -> WingletGeometry:
    """
    Compute split winglet geometric properties from point cloud.

    The winglet points define upper and lower winglet surfaces that extend
    vertically (in Z) both above and below the wing tip attachment point.

    Parameters:
    -----------
    winglet_points : np.ndarray
        Winglet boundary points (N x 3) in feet
    wing : WingGeometry
        Main wing geometry for attachment point reference

    Returns:
    --------
    geom : WingletGeometry
        Container with upper and lower winglet properties
    """

    # Winglet extends in Z direction (vertically above and below wing)
    z_vals = winglet_points[:, 2]
    y_vals = winglet_points[:, 1]

    # Find attachment point (middle Z values)
    z_min = np.min(z_vals)
    z_max = np.max(z_vals)
    z_mid = (z_min + z_max) / 2.0

    # Attachment Y location (average Y coordinate)
    attach_y = np.mean(y_vals)

    # Separate upper winglet (Z > z_mid) and lower winglet (Z < z_mid)
    upper_mask = z_vals > z_mid
    lower_mask = z_vals < z_mid

    upper_pts = winglet_points[upper_mask]
    lower_pts = winglet_points[lower_mask]

    # UPPER WINGLET
    # Root: points at minimum Z in upper section (attachment level)
    z_upper_min = np.min(upper_pts[:, 2])
    z_upper_max = np.max(upper_pts[:, 2])

    upper_root_mask = np.abs(upper_pts[:, 2] - z_upper_min) < 0.3
    upper_root_pts = upper_pts[upper_root_mask]
    upper_root_le_idx = np.argmin(upper_root_pts[:, 0])
    upper_root_te_idx = np.argmax(upper_root_pts[:, 0])
    upper_root_le = upper_root_pts[upper_root_le_idx]
    upper_root_te = upper_root_pts[upper_root_te_idx]

    # Tip: points at maximum Z
    upper_tip_mask = np.abs(upper_pts[:, 2] - z_upper_max) < 0.3
    upper_tip_pts = upper_pts[upper_tip_mask]
    upper_tip_le_idx = np.argmin(upper_tip_pts[:, 0])
    upper_tip_te_idx = np.argmax(upper_tip_pts[:, 0])
    upper_tip_le = upper_tip_pts[upper_tip_le_idx]
    upper_tip_te = upper_tip_pts[upper_tip_te_idx]

    upper_root_chord = np.linalg.norm(upper_root_te - upper_root_le)
    upper_tip_chord = np.linalg.norm(upper_tip_te - upper_tip_le)
    upper_height = z_upper_max - z_upper_min

    # Upper cant angle
    dy_upper = np.abs(np.mean(upper_tip_pts[:, 1])) - np.abs(np.mean(upper_root_pts[:, 1]))
    dz_upper = upper_height
    if dz_upper > 0.01:
        upper_cant_angle = np.degrees(np.arctan(dy_upper / dz_upper))
    else:
        upper_cant_angle = 0.0

    # Upper LE sweep
    dx_upper = upper_tip_le[0] - upper_root_le[0]
    if dz_upper > 0.01:
        upper_sweep_le = np.degrees(np.arctan(dx_upper / dz_upper))
    else:
        upper_sweep_le = 0.0

    # LOWER WINGLET
    # Root: points at maximum Z in lower section (attachment level)
    z_lower_min = np.min(lower_pts[:, 2])
    z_lower_max = np.max(lower_pts[:, 2])

    lower_root_mask = np.abs(lower_pts[:, 2] - z_lower_max) < 0.3
    lower_root_pts = lower_pts[lower_root_mask]
    lower_root_le_idx = np.argmin(lower_root_pts[:, 0])
    lower_root_te_idx = np.argmax(lower_root_pts[:, 0])
    lower_root_le = lower_root_pts[lower_root_le_idx]
    lower_root_te = lower_root_pts[lower_root_te_idx]

    # Tip: points at minimum Z
    lower_tip_mask = np.abs(lower_pts[:, 2] - z_lower_min) < 0.3
    lower_tip_pts = lower_pts[lower_tip_mask]
    lower_tip_le_idx = np.argmin(lower_tip_pts[:, 0])
    lower_tip_te_idx = np.argmax(lower_tip_pts[:, 0])
    lower_tip_le = lower_tip_pts[lower_tip_le_idx]
    lower_tip_te = lower_tip_pts[lower_tip_te_idx]

    lower_root_chord = np.linalg.norm(lower_root_te - lower_root_le)
    lower_tip_chord = np.linalg.norm(lower_tip_te - lower_tip_le)
    lower_height = z_lower_max - z_lower_min

    # Lower cant angle
    dy_lower = np.abs(np.mean(lower_tip_pts[:, 1])) - np.abs(np.mean(lower_root_pts[:, 1]))
    dz_lower = lower_height
    if dz_lower > 0.01:
        lower_cant_angle = np.degrees(np.arctan(dy_lower / dz_lower))
    else:
        lower_cant_angle = 0.0

    # Lower LE sweep
    dx_lower = lower_tip_le[0] - lower_root_le[0]
    if dz_lower > 0.01:
        lower_sweep_le = np.degrees(np.arctan(dx_lower / dz_lower))
    else:
        lower_sweep_le = 0.0

    # Attachment Z is the average of upper root and lower root
    attach_z = (upper_root_le[2] + lower_root_le[2]) / 2.0

    return WingletGeometry(
        points=winglet_points,
        upper_root_le=upper_root_le,
        upper_root_te=upper_root_te,
        upper_tip_le=upper_tip_le,
        upper_tip_te=upper_tip_te,
        lower_root_le=lower_root_le,
        lower_root_te=lower_root_te,
        lower_tip_le=lower_tip_le,
        lower_tip_te=lower_tip_te,
        upper_root_chord=upper_root_chord,
        upper_tip_chord=upper_tip_chord,
        upper_height=upper_height,
        lower_root_chord=lower_root_chord,
        lower_tip_chord=lower_tip_chord,
        lower_height=lower_height,
        upper_cant_angle=upper_cant_angle,
        lower_cant_angle=lower_cant_angle,
        upper_sweep_le=upper_sweep_le,
        lower_sweep_le=lower_sweep_le,
        attach_y=attach_y,
        attach_z=attach_z
    )


def compute_elevon_geometry(elevon_points: np.ndarray, wing: WingGeometry) -> ElevonGeometry:
    """
    Compute elevon control surface boundaries from corner points.

    The elevon points define the quadrilateral control surface boundary.
    Typically 4 corners: inboard LE, inboard TE, outboard TE, outboard LE.

    Parameters:
    -----------
    elevon_points : np.ndarray
        Elevon boundary points (4 x 3) in feet
    wing : WingGeometry
        Main wing geometry for chord reference

    Returns:
    --------
    geom : ElevonGeometry
        Container with elevon properties
    """

    # Identify inboard/outboard stations (min/max Y)
    y_vals = elevon_points[:, 1]
    y_inboard = np.min(y_vals)
    y_outboard = np.max(y_vals)

    # Find LE and TE points at inboard and outboard
    inboard_mask = np.abs(y_vals - y_inboard) < 0.1
    outboard_mask = np.abs(y_vals - y_outboard) < 0.1

    inboard_pts = elevon_points[inboard_mask]
    outboard_pts = elevon_points[outboard_mask]

    # LE is min X, TE is max X
    inboard_le_x = np.min(inboard_pts[:, 0])
    inboard_te_x = np.max(inboard_pts[:, 0])
    outboard_le_x = np.min(outboard_pts[:, 0])
    outboard_te_x = np.max(outboard_pts[:, 0])

    # Hinge line is the LE of the elevon (leading edge of control surface)
    # Create linear interpolation function: y -> x
    hinge_y = np.array([y_inboard, y_outboard])
    hinge_x = np.array([inboard_le_x, outboard_le_x])

    # Polynomial fit (linear)
    hinge_poly = np.poly1d(np.polyfit(hinge_y, hinge_x, 1))
    hinge_line_x = lambda y: hinge_poly(y)

    # Compute average chord fraction
    # Interpolate wing LE and TE at elevon stations
    wing_le_x_inboard = np.interp(y_inboard, wing.le_points[:, 1], wing.le_points[:, 0])
    wing_te_x_inboard = np.interp(y_inboard, wing.te_points[:, 1], wing.te_points[:, 0])
    wing_le_x_outboard = np.interp(y_outboard, wing.le_points[:, 1], wing.le_points[:, 0])
    wing_te_x_outboard = np.interp(y_outboard, wing.te_points[:, 1], wing.te_points[:, 0])

    chord_inboard = wing_te_x_inboard - wing_le_x_inboard
    chord_outboard = wing_te_x_outboard - wing_le_x_outboard

    # Hinge as fraction of chord
    hinge_frac_inboard = (inboard_le_x - wing_le_x_inboard) / chord_inboard
    hinge_frac_outboard = (outboard_le_x - wing_le_x_outboard) / chord_outboard
    chord_fraction = (hinge_frac_inboard + hinge_frac_outboard) / 2.0

    return ElevonGeometry(
        points=elevon_points,
        y_inboard=y_inboard,
        y_outboard=y_outboard,
        hinge_line_x=hinge_line_x,
        chord_fraction=chord_fraction
    )


if __name__ == "__main__":
    # Test with current geometry
    import os

    base_path = r"C:\Users\bradrothenberg\OneDrive - nTop\OUT\parts\nTopAVL\nTop6DOF\DATA"
    le_file = os.path.join(base_path, "LEpts.csv")
    te_file = os.path.join(base_path, "TEpts.csv")
    winglet_file = os.path.join(base_path, "WINGLETpts.csv")
    elevon_file = os.path.join(base_path, "ELEVONpts.csv")

    # Read and compute wing geometry
    le_points = read_csv_points(le_file, units='inches')
    te_points = read_csv_points(te_file, units='inches')
    wing = compute_wing_geometry(le_points, te_points)

    print_geometry_summary(wing)

    # Read and compute winglet geometry
    if os.path.exists(winglet_file):
        winglet_points = read_csv_points(winglet_file, units='inches')
        winglet = compute_winglet_geometry(winglet_points, wing)

        print("\n" + "=" * 60)
        print("SPLIT WINGLET GEOMETRY")
        print("=" * 60)
        print(f"Attachment Y:      {winglet.attach_y:8.3f} ft")
        print(f"Attachment Z:      {winglet.attach_z:8.3f} ft")
        print()
        print("UPPER WINGLET:")
        print(f"  Root Chord:      {winglet.upper_root_chord:8.3f} ft")
        print(f"  Tip Chord:       {winglet.upper_tip_chord:8.3f} ft")
        print(f"  Height:          {winglet.upper_height:8.3f} ft")
        print(f"  Cant Angle:      {winglet.upper_cant_angle:8.2f} deg")
        print(f"  LE Sweep:        {winglet.upper_sweep_le:8.2f} deg")
        print()
        print("LOWER WINGLET:")
        print(f"  Root Chord:      {winglet.lower_root_chord:8.3f} ft")
        print(f"  Tip Chord:       {winglet.lower_tip_chord:8.3f} ft")
        print(f"  Height:          {winglet.lower_height:8.3f} ft")
        print(f"  Cant Angle:      {winglet.lower_cant_angle:8.2f} deg")
        print(f"  LE Sweep:        {winglet.lower_sweep_le:8.2f} deg")
        print("=" * 60)

    # Read and compute elevon geometry
    if os.path.exists(elevon_file):
        elevon_points = read_csv_points(elevon_file, units='inches')
        elevon = compute_elevon_geometry(elevon_points, wing)

        print("\n" + "=" * 60)
        print("ELEVON GEOMETRY")
        print("=" * 60)
        print(f"Y Inboard:         {elevon.y_inboard:8.3f} ft")
        print(f"Y Outboard:        {elevon.y_outboard:8.3f} ft")
        print(f"Span Coverage:     {elevon.y_outboard - elevon.y_inboard:8.3f} ft")
        print(f"Hinge Chord Frac:  {elevon.chord_fraction:8.3f}")
        print("=" * 60)
