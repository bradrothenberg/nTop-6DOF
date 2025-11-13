"""
AVL geometry file generator.

Creates AVL input files from wing/tail geometric data.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.io.geometry import WingGeometry


class AVLGeometryWriter:
    """Generate AVL geometry files from wing/tail data."""

    def __init__(self, name: str = "UAV"):
        """
        Initialize AVL geometry writer.

        Parameters:
        -----------
        name : str
            Aircraft name
        """
        self.name = name
        self.mach = 0.0
        self.symmetry = "0"  # XZ-plane symmetry (default for aircraft)
        self.s_ref = 0.0
        self.c_ref = 0.0
        self.b_ref = 0.0
        self.x_ref = 0.0
        self.y_ref = 0.0
        self.z_ref = 0.0
        self.cd_p = 0.0  # Parasitic drag coefficient

        self.surfaces = []

    def set_reference_values(self, s_ref: float, c_ref: float, b_ref: float,
                           x_ref: float, y_ref: float, z_ref: float):
        """
        Set aircraft reference values.

        Parameters:
        -----------
        s_ref : float
            Reference area (ft^2)
        c_ref : float
            Reference chord (ft)
        b_ref : float
            Reference span (ft)
        x_ref, y_ref, z_ref : float
            Moment reference point (ft)
        """
        self.s_ref = s_ref
        self.c_ref = c_ref
        self.b_ref = b_ref
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.z_ref = z_ref

    def add_wing_from_geometry(self, wing: WingGeometry, airfoil: str = "NACA 64-212",
                               flaperon_hinge: float = 0.8, flaperon_span: float = 0.75,
                               name: str = "Wing"):
        """
        Add wing surface from WingGeometry object.

        Parameters:
        -----------
        wing : WingGeometry
            Wing geometric properties
        airfoil : str
            Airfoil designation (NACA 4/5/6-digit or file path)
        flaperon_hinge : float
            Flaperon hinge line as fraction of chord (0.8 = 80% chord)
        flaperon_span : float
            Flaperon extent as fraction of semi-span (0.75 = 75% semispan)
        name : str
            Surface name
        """
        # AVL uses right half-span only with symmetry
        # Extract right half data
        y_stations = wing.le_points[:, 1]
        right_idx = np.where(y_stations >= 0)[0]

        le_points_right = wing.le_points[right_idx, :]
        te_points_right = wing.te_points[right_idx, :]

        # Sort by increasing y
        sort_idx = np.argsort(le_points_right[:, 1])
        le_points_right = le_points_right[sort_idx, :]
        te_points_right = te_points_right[sort_idx, :]

        # Determine flaperon break station
        y_max = le_points_right[-1, 1]
        y_flaperon_start = y_max * flaperon_span

        # Build sections
        sections = []

        for i in range(len(le_points_right)):
            le = le_points_right[i, :]
            te = te_points_right[i, :]
            chord = te[0] - le[0]

            # Determine if this section has flaperon control
            has_flaperon = (le[1] >= y_flaperon_start)

            section = {
                'x_le': le[0],
                'y_le': le[1],
                'z_le': le[2],
                'chord': chord,
                'ainc': 0.0,  # Incidence angle
                'airfoil': airfoil,
                'controls': []
            }

            if has_flaperon:
                # Add flaperon control surface
                section['controls'].append({
                    'name': 'flaperon',
                    'gain': 1.0,
                    'xhinge': flaperon_hinge,
                    'hinge_vec': [0, 0, 0],  # Default hinge vector
                    'sign_dup': -1.0  # Antisymmetric for roll control
                })

            sections.append(section)

        surface = {
            'name': name,
            'n_chord': 12,
            'c_space': 1.0,  # Cosine spacing
            'n_span': 20,  # Spanwise panels
            's_space': 1.0,  # Cosine spacing
            'component': 1,
            'y_duplicate': 0.0,  # Reflect about XZ plane
            'sections': sections
        }

        self.surfaces.append(surface)

    def add_horizontal_tail(self, h_tail: Dict, airfoil: str = "NACA 0012",
                           elevator_hinge: float = 0.7, name: str = "Horizontal Tail"):
        """
        Add horizontal tail surface.

        Parameters:
        -----------
        h_tail : dict
            Horizontal tail geometry dictionary
        airfoil : str
            Airfoil designation
        elevator_hinge : float
            Elevator hinge line as fraction of chord
        name : str
            Surface name
        """
        # Simple trapezoidal horizontal tail
        x_root = h_tail['x_position']
        y_root = 0.0
        z_root = h_tail['z_position']
        c_root = h_tail['chord']

        y_tip = h_tail['span'] / 2.0
        c_tip = c_root * h_tail['taper_ratio']

        # Quarter-chord sweep
        sweep_c4_rad = np.radians(h_tail['sweep_c4'])
        x_tip = x_root + (y_tip - y_root) * np.tan(sweep_c4_rad) - 0.25 * (c_tip - c_root)

        sections = [
            {
                'x_le': x_root,
                'y_le': y_root,
                'z_le': z_root,
                'chord': c_root,
                'ainc': 0.0,
                'airfoil': airfoil,
                'controls': [{
                    'name': 'elevator',
                    'gain': 1.0,
                    'xhinge': elevator_hinge,
                    'hinge_vec': [0, 0, 0],
                    'sign_dup': 1.0  # Symmetric
                }]
            },
            {
                'x_le': x_tip,
                'y_le': y_tip,
                'z_le': z_root,  # Assume no dihedral
                'chord': c_tip,
                'ainc': 0.0,
                'airfoil': airfoil,
                'controls': [{
                    'name': 'elevator',
                    'gain': 1.0,
                    'xhinge': elevator_hinge,
                    'hinge_vec': [0, 0, 0],
                    'sign_dup': 1.0
                }]
            }
        ]

        surface = {
            'name': name,
            'n_chord': 8,
            'c_space': 1.0,
            'n_span': 12,  # Spanwise panels for horizontal tail
            's_space': 1.0,
            'component': 2,
            'y_duplicate': 0.0,
            'sections': sections
        }

        self.surfaces.append(surface)

    def add_vertical_tail(self, v_tail: Dict, airfoil: str = "NACA 0012",
                         rudder_hinge: float = 0.7, name: str = "Vertical Tail"):
        """
        Add vertical tail surface.

        Parameters:
        -----------
        v_tail : dict
            Vertical tail geometry dictionary
        airfoil : str
            Airfoil designation
        rudder_hinge : float
            Rudder hinge line as fraction of chord
        name : str
            Surface name
        """
        # Vertical tail (no y_duplicate, no symmetry for this surface)
        x_root = v_tail['x_position']
        y_root = 0.0
        z_root = 0.0
        c_root = v_tail['chord']

        y_tip = 0.0  # Vertical tail stays at centerline
        z_tip = v_tail['height']
        c_tip = c_root * v_tail['taper_ratio']

        # Quarter-chord sweep
        sweep_c4_rad = np.radians(v_tail['sweep_c4'])
        x_tip = x_root + z_tip * np.tan(sweep_c4_rad) - 0.25 * (c_tip - c_root)

        sections = [
            {
                'x_le': x_root,
                'y_le': y_root,
                'z_le': z_root,
                'chord': c_root,
                'ainc': 0.0,
                'airfoil': airfoil,
                'controls': [{
                    'name': 'rudder',
                    'gain': 1.0,
                    'xhinge': rudder_hinge,
                    'hinge_vec': [0, 0, 0],
                    'sign_dup': 1.0
                }]
            },
            {
                'x_le': x_tip,
                'y_le': y_tip,
                'z_le': z_tip,
                'chord': c_tip,
                'ainc': 0.0,
                'airfoil': airfoil,
                'controls': [{
                    'name': 'rudder',
                    'gain': 1.0,
                    'xhinge': rudder_hinge,
                    'hinge_vec': [0, 0, 0],
                    'sign_dup': 1.0
                }]
            }
        ]

        surface = {
            'name': name,
            'n_chord': 8,
            'c_space': 1.0,
            'n_span': 10,  # Spanwise panels for vertical tail
            's_space': 1.0,
            'component': 3,
            'y_duplicate': None,  # No duplication for vertical tail
            'sections': sections
        }

        self.surfaces.append(surface)

    def add_winglet_from_geometry(self, winglet: 'WingletGeometry', airfoil: str = "NACA 0012",
                                  elevon_hinge: float = 0.75, has_elevon: bool = True,
                                  side: str = "right", name: str = "Winglet"):
        """
        Add winglet surface from WingletGeometry object.

        Parameters:
        -----------
        winglet : WingletGeometry
            Winglet geometric properties
        airfoil : str
            Airfoil designation
        elevon_hinge : float
            Elevon hinge line as fraction of chord
        has_elevon : bool
            Whether this winglet section has elevon control
        side : str
            "right" or "left" winglet
        name : str
            Surface name
        """
        # Winglets are vertical surfaces extending in Z
        # Need to create sections at different Z stations

        # Create 3 sections: root, mid, tip
        n_sections = 3
        z_stations = np.linspace(winglet.root_le[2], winglet.tip_le[2], n_sections)

        sections = []

        for z in z_stations:
            # Interpolate X, Y, chord at this Z station
            t = (z - winglet.root_le[2]) / (winglet.tip_le[2] - winglet.root_le[2]) if winglet.height > 0.01 else 0

            # Linear interpolation between root and tip
            x_le = winglet.root_le[0] + t * (winglet.tip_le[0] - winglet.root_le[0])
            y_le = winglet.root_le[1] + t * (winglet.tip_le[1] - winglet.root_le[1])
            chord = winglet.root_chord + t * (winglet.tip_chord - winglet.root_chord)

            section = {
                'x_le': x_le,
                'y_le': y_le if side == "right" else -y_le,  # Mirror for left winglet
                'z_le': z,
                'chord': chord,
                'ainc': 0.0,
                'airfoil': airfoil,
                'controls': []
            }

            # Add elevon control to outer sections
            if has_elevon and t > 0.3:  # Elevon on outer 70% of winglet
                section['controls'].append({
                    'name': 'elevon',
                    'gain': 1.0,
                    'xhinge': elevon_hinge,
                    'hinge_vec': [0, 0, 0],
                    'sign_dup': -1.0  # Antisymmetric for roll control
                })

            sections.append(section)

        surface = {
            'name': name,
            'n_chord': 8,
            'c_space': 1.0,
            'n_span': 8,  # Spanwise panels along winglet height
            's_space': 1.0,
            'component': 4 if side == "right" else 5,  # Separate components
            'y_duplicate': None,  # No duplication - explicit left/right
            'sections': sections
        }

        self.surfaces.append(surface)

    def write_avl_file(self, filepath: str):
        """
        Write AVL geometry file.

        Parameters:
        -----------
        filepath : str
            Output file path
        """
        with open(filepath, 'w') as f:
            # Header
            f.write(f"{self.name}\n")
            f.write(f"#Mach\n")
            f.write(f" {self.mach:8.3f}\n")
            f.write(f"#IYsym   IZsym   Zsym\n")
            f.write(f" {self.symmetry}      0       0.0\n")
            f.write(f"#Sref    Cref    Bref\n")
            f.write(f" {self.s_ref:10.4f}  {self.c_ref:10.4f}  {self.b_ref:10.4f}\n")
            f.write(f"#Xref    Yref    Zref\n")
            f.write(f" {self.x_ref:10.4f}  {self.y_ref:10.4f}  {self.z_ref:10.4f}\n")
            f.write(f"#CDp (optional)\n")
            f.write(f" {self.cd_p:10.5f}\n")
            f.write(f"#\n")

            # Surfaces
            for surface in self.surfaces:
                f.write(f"#{'=' * 60}\n")
                f.write(f"SURFACE\n")
                f.write(f"{surface['name']}\n")
                f.write(f"#Nchordwise  Cspace  [Nspanwise  Sspace]\n")
                if surface['n_span'] is not None:
                    f.write(f" {surface['n_chord']:3d}        {surface['c_space']:5.2f}     {surface['n_span']:3d}       {surface['s_space']:5.2f}\n")
                else:
                    f.write(f" {surface['n_chord']:3d}        {surface['c_space']:5.2f}\n")
                f.write(f"#\n")
                f.write(f"COMPONENT\n")
                f.write(f" {surface['component']}\n")
                f.write(f"#\n")

                if surface['y_duplicate'] is not None:
                    f.write(f"YDUPLICATE\n")
                    f.write(f" {surface['y_duplicate']:10.4f}\n")
                    f.write(f"#\n")

                # Sections
                for i, section in enumerate(surface['sections']):
                    f.write(f"#-----------------\n")
                    f.write(f"SECTION\n")
                    f.write(f"#Xle     Yle      Zle      Chord    Ainc\n")
                    f.write(f" {section['x_le']:8.4f} {section['y_le']:8.4f} {section['z_le']:8.4f} ")
                    f.write(f"{section['chord']:8.4f} {section['ainc']:6.2f}\n")
                    f.write(f"#\n")

                    # Airfoil
                    if section['airfoil']:
                        f.write(f"AFIL\n")
                        f.write(f" {section['airfoil']}\n")
                        f.write(f"#\n")

                    # Control surfaces
                    for control in section['controls']:
                        f.write(f"CONTROL\n")
                        f.write(f"#name     gain    Xhinge  XYZhvec  SgnDup\n")
                        f.write(f" {control['name']:8s} {control['gain']:6.3f}  {control['xhinge']:6.3f}  ")
                        f.write(f"{control['hinge_vec'][0]:4.1f} {control['hinge_vec'][1]:4.1f} {control['hinge_vec'][2]:4.1f}  ")
                        f.write(f"{control['sign_dup']:6.2f}\n")
                        f.write(f"#\n")

                f.write(f"\n")


def generate_avl_geometry_from_csv(le_file: str, te_file: str, mass_file: str,
                                   output_file: str, aircraft_name: str = "nTop_UAV"):
    """
    Generate complete AVL geometry file from CSV inputs.

    Parameters:
    -----------
    le_file : str
        Path to leading edge CSV file
    te_file : str
        Path to trailing edge CSV file
    mass_file : str
        Path to mass properties CSV file
    output_file : str
        Path to output AVL file
    aircraft_name : str
        Aircraft name
    """
    from src.io.geometry import read_csv_points, compute_wing_geometry, estimate_tail_geometry
    from src.io.mass_properties import read_mass_csv

    # Read geometry
    le_points = read_csv_points(le_file, units='inches')
    te_points = read_csv_points(te_file, units='inches')
    wing = compute_wing_geometry(le_points, te_points)

    # Estimate tail geometry
    h_tail, v_tail = estimate_tail_geometry(wing, v_h=0.6, v_v=0.05, tail_arm_factor=2.5)

    # Read mass properties for CG location
    mass_props = read_mass_csv(mass_file)

    # Create AVL writer
    avl_writer = AVLGeometryWriter(name=aircraft_name)

    # Set reference values
    avl_writer.set_reference_values(
        s_ref=wing.area,
        c_ref=wing.mac,
        b_ref=wing.span,
        x_ref=mass_props.cg_ft[0],
        y_ref=mass_props.cg_ft[1],
        z_ref=mass_props.cg_ft[2]
    )

    # Add surfaces
    # Using NACA 4-digit airfoils that AVL can generate internally
    avl_writer.add_wing_from_geometry(wing, airfoil="NACA 2412",
                                      flaperon_hinge=0.8, flaperon_span=0.75)
    avl_writer.add_horizontal_tail(h_tail, airfoil="NACA 0012", elevator_hinge=0.7)
    avl_writer.add_vertical_tail(v_tail, airfoil="NACA 0012", rudder_hinge=0.7)

    # Write file
    avl_writer.write_avl_file(output_file)

    return wing, h_tail, v_tail, mass_props


if __name__ == "__main__":
    import os

    # Paths
    base_path = r"C:\Users\bradrothenberg\OneDrive - nTop\OUT\parts\nTopAVL\nTop6DOF"
    data_path = os.path.join(base_path, "Data")
    output_path = os.path.join(base_path, "avl_files")

    le_file = os.path.join(data_path, "LEpts.csv")
    te_file = os.path.join(data_path, "TEpts.csv")
    mass_file = os.path.join(data_path, "mass.csv")
    output_file = os.path.join(output_path, "uav.avl")

    # Generate AVL file
    wing, h_tail, v_tail, mass_props = generate_avl_geometry_from_csv(
        le_file, te_file, mass_file, output_file, aircraft_name="nTop_UAV"
    )

    print(f"\nAVL geometry file written to: {output_file}")
    print(f"\nSummary:")
    print(f"  Wing area:  {wing.area:8.3f} ft^2")
    print(f"  Wing span:  {wing.span:8.3f} ft")
    print(f"  Wing MAC:   {wing.mac:8.3f} ft")
    print(f"  CG:         ({mass_props.cg_ft[0]:8.4f}, {mass_props.cg_ft[1]:8.4f}, {mass_props.cg_ft[2]:8.4f}) ft")
