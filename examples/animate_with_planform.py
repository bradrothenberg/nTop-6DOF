"""
Animate aircraft trajectory with 3D planform visualization.

Shows the actual wing and tail geometry as the aircraft flies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
from pathlib import Path

def rotation_matrix_from_euler(phi, theta, psi):
    """Create rotation matrix from Euler angles (roll, pitch, yaw)."""
    # Roll (phi) - rotation about x-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    # Pitch (theta) - rotation about y-axis
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # Yaw (psi) - rotation about z-axis
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    # Combined rotation: R = R_z @ R_y @ R_x
    return R_z @ R_y @ R_x


def create_conventional_aircraft_geometry():
    """
    Create 3D geometry for conventional tail aircraft.
    Uses actual dimensions from AVL file.

    Returns vertices for:
    - Main wing (tapered)
    - Horizontal tail
    - Vertical tail
    - Fuselage (simple box)
    """

    # Wing geometry (from AVL file uav_conventional.avl)
    # Highly tapered wing with 7 sections
    # Root: Xle=0.022, chord=22.4
    # Tip: Xle=18.37, Yle=12.43, chord=0.25
    # Bref = 24.86 ft (half-span = 12.43 ft)

    # Simplified to trapezoid using root, mid, and tip sections
    wing_root_le = 0.022
    wing_root_chord = 22.4
    wing_mid_le = 6.49  # Section at y=1.74
    wing_mid_chord = 15.0
    wing_mid_y = 1.74
    wing_tip_le = 18.37
    wing_tip_chord = 0.25
    wing_tip_y = 12.43

    wing_vertices = np.array([
        # Left wing (3 sections for better representation)
        [wing_root_le, 0, 0],  # Root LE
        [wing_root_le + wing_root_chord, 0, 0],  # Root TE
        [wing_mid_le + wing_mid_chord, -wing_mid_y, 0],  # Mid TE
        [wing_mid_le, -wing_mid_y, 0],  # Mid LE
        [wing_tip_le + wing_tip_chord, -wing_tip_y, 0],  # Tip TE
        [wing_tip_le, -wing_tip_y, 0],  # Tip LE
        # Right wing
        [wing_root_le, 0, 0],  # Root LE
        [wing_root_le + wing_root_chord, 0, 0],  # Root TE
        [wing_mid_le + wing_mid_chord, wing_mid_y, 0],  # Mid TE
        [wing_mid_le, wing_mid_y, 0],  # Mid LE
        [wing_tip_le + wing_tip_chord, wing_tip_y, 0],  # Tip TE
        [wing_tip_le, wing_tip_y, 0],  # Tip LE
    ])

    # Horizontal tail (from AVL file)
    # Root: Xle=25.0, chord=4.5
    # Tip: Xle=26.0, Yle=4.5, chord=3.5
    htail_root_le = 25.0
    htail_root_chord = 4.5
    htail_tip_le = 26.0
    htail_tip_chord = 3.5
    htail_span = 4.5  # Half-span
    htail_z = 0.0  # At fuselage centerline

    htail_vertices = np.array([
        # Left htail
        [htail_root_le, 0, htail_z],
        [htail_root_le + htail_root_chord, 0, htail_z],
        [htail_tip_le + htail_tip_chord, -htail_span, htail_z],
        [htail_tip_le, -htail_span, htail_z],
        # Right htail
        [htail_root_le, 0, htail_z],
        [htail_root_le + htail_root_chord, 0, htail_z],
        [htail_tip_le + htail_tip_chord, htail_span, htail_z],
        [htail_tip_le, htail_span, htail_z],
    ])

    # Vertical tail (from AVL file)
    # Root: Xle=25.0, Z=0, chord=4.5
    # Tip: Xle=27.0, Z=3.5, chord=3.0
    vtail_root_le = 25.0
    vtail_root_chord = 4.5
    vtail_tip_le = 27.0
    vtail_tip_chord = 3.0
    vtail_height = 3.5

    vtail_vertices = np.array([
        [vtail_root_le, 0, 0],  # Bottom LE
        [vtail_root_le + vtail_root_chord, 0, 0],  # Bottom TE
        [vtail_tip_le + vtail_tip_chord, 0, vtail_height],  # Top TE
        [vtail_tip_le, 0, vtail_height],  # Top LE
    ])

    # Simple fuselage (box)
    fuse_front_x = 0.0
    fuse_rear_x = 24.0
    fuse_width = 1.0
    fuse_height = 1.2

    fuselage_vertices = np.array([
        # Front
        [fuse_front_x, -fuse_width/2, -fuse_height/2],
        [fuse_front_x, fuse_width/2, -fuse_height/2],
        [fuse_front_x, fuse_width/2, fuse_height/2],
        [fuse_front_x, -fuse_width/2, fuse_height/2],
        # Rear
        [fuse_rear_x, -fuse_width/2, -fuse_height/2],
        [fuse_rear_x, fuse_width/2, -fuse_height/2],
        [fuse_rear_x, fuse_width/2, fuse_height/2],
        [fuse_rear_x, -fuse_width/2, fuse_height/2],
    ])

    return {
        'wing': wing_vertices,
        'htail': htail_vertices,
        'vtail': vtail_vertices,
        'fuselage': fuselage_vertices
    }


def transform_geometry(geometry, position, attitude):
    """
    Transform aircraft geometry to world coordinates.

    Parameters:
    -----------
    geometry : dict
        Dictionary of component vertices
    position : np.array (3,)
        Aircraft position [x, y, z]
    attitude : np.array (3,)
        Euler angles [roll, pitch, yaw]

    Returns:
    --------
    transformed : dict
        Dictionary of transformed vertices
    """
    phi, theta, psi = attitude
    R = rotation_matrix_from_euler(phi, theta, psi)

    transformed = {}
    for component, vertices in geometry.items():
        # Rotate each vertex about aircraft origin (CG)
        rotated = vertices @ R.T
        # Translate to world position
        transformed[component] = rotated + position

    return transformed


def create_animation(history_file, output_file):
    """Create animation with aircraft planform."""

    print(f"Loading trajectory from: {history_file}")
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    times = history['time']
    positions = history['position']
    attitudes = history['attitude']

    print(f"Loaded {len(times)} time steps")
    print(f"Duration: {times[-1]:.1f} seconds")

    # Create base aircraft geometry
    base_geometry = create_conventional_aircraft_geometry()

    # Downsample for animation (every 5th frame)
    skip = 5
    times_anim = times[::skip]
    positions_anim = positions[::skip]
    attitudes_anim = attitudes[::skip]

    print(f"Animation frames: {len(times_anim)}")

    # Set up the figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot full trajectory as reference (invert Z for altitude-up display)
    ax.plot(positions[:, 0], positions[:, 1], -positions[:, 2],
            'b-', alpha=0.3, linewidth=1, label='Flight Path')

    # Initialize collections for aircraft components
    wing_collection = Poly3DCollection([], facecolors='cyan',
                                       edgecolors='black', linewidths=1, alpha=0.8)
    htail_collection = Poly3DCollection([], facecolors='yellow',
                                        edgecolors='black', linewidths=1, alpha=0.8)
    vtail_collection = Poly3DCollection([], facecolors='red',
                                        edgecolors='black', linewidths=1, alpha=0.8)
    fuselage_collection = Poly3DCollection([], facecolors='gray',
                                           edgecolors='black', linewidths=1, alpha=0.6)

    ax.add_collection3d(wing_collection)
    ax.add_collection3d(htail_collection)
    ax.add_collection3d(vtail_collection)
    ax.add_collection3d(fuselage_collection)

    # Time text
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    # Dynamic axis limits - will be updated each frame to follow aircraft
    # Start with full trajectory view (invert Z for altitude-up display)
    x_min, x_max = positions[:, 0].min() - 50, positions[:, 0].max() + 50
    y_min, y_max = positions[:, 1].min() - 50, positions[:, 1].max() + 50
    z_min, z_max = -positions[:, 2].max() - 50, -positions[:, 2].min() + 50

    # Camera zoom distance (smaller = closer zoom)
    zoom_distance = 100  # ft - shows aircraft nicely with surrounding context

    ax.set_xlabel('X (ft)', fontsize=10)
    ax.set_ylabel('Y (ft)', fontsize=10)
    ax.set_zlabel('Altitude (ft)', fontsize=10)
    ax.set_title('Conventional Tail Aircraft - 3D Trajectory', fontsize=14, fontweight='bold')

    # Set view angle
    ax.view_init(elev=20, azim=-60)

    ax.grid(True, alpha=0.3)

    def create_surface_from_vertices(vertices):
        """Create surface polygons from vertices."""
        if len(vertices) == 4:
            # Quad (like vtail)
            return [vertices]
        elif len(vertices) == 8:
            # Two quads (like htail)
            return [vertices[:4], vertices[4:]]
        elif len(vertices) == 12:
            # Tapered wing - 6 vertices per side, create triangular sections
            # Left wing: vertices 0-5
            # Right wing: vertices 6-11
            left_surfaces = [
                vertices[[0, 1, 2, 3]],  # Root to mid
                vertices[[3, 2, 4, 5]],  # Mid to tip
            ]
            right_surfaces = [
                vertices[[6, 7, 8, 9]],  # Root to mid
                vertices[[9, 8, 10, 11]],  # Mid to tip
            ]
            return left_surfaces + right_surfaces
        else:
            return []

    def create_fuselage_surfaces(vertices):
        """Create box surfaces from 8 vertices."""
        # Indices for 6 faces of a box
        faces = [
            [0, 1, 2, 3],  # Front face
            [4, 5, 6, 7],  # Rear face
            [0, 1, 5, 4],  # Bottom face
            [2, 3, 7, 6],  # Top face
            [0, 3, 7, 4],  # Left face
            [1, 2, 6, 5],  # Right face
        ]
        return [vertices[face] for face in faces]

    def update(frame):
        """Update function for animation."""
        # Get current state
        pos = positions_anim[frame]
        att = attitudes_anim[frame]
        t = times_anim[frame]

        # Transform geometry (invert Z for altitude-up display)
        pos_display = pos.copy()
        pos_display[2] = -pos_display[2]  # Invert Z axis
        transformed = transform_geometry(base_geometry, pos_display, att)

        # Update wing
        wing_surfaces = create_surface_from_vertices(transformed['wing'])
        wing_collection.set_verts(wing_surfaces)

        # Update htail
        htail_surfaces = create_surface_from_vertices(transformed['htail'])
        htail_collection.set_verts(htail_surfaces)

        # Update vtail
        vtail_surfaces = create_surface_from_vertices(transformed['vtail'])
        vtail_collection.set_verts(vtail_surfaces)

        # Update fuselage
        fuselage_surfaces = create_fuselage_surfaces(transformed['fuselage'])
        fuselage_collection.set_verts(fuselage_surfaces)

        # Update camera to follow aircraft (zoom in, with inverted Z)
        ax.set_xlim(pos_display[0] - zoom_distance, pos_display[0] + zoom_distance)
        ax.set_ylim(pos_display[1] - zoom_distance, pos_display[1] + zoom_distance)
        ax.set_zlim(pos_display[2] - zoom_distance, pos_display[2] + zoom_distance)

        # Update time text
        roll_deg = np.degrees(att[0])
        pitch_deg = np.degrees(att[1])
        alt_ft = -pos[2]  # Convert to positive altitude
        time_text.set_text(f't = {t:.1f}s\nAlt = {alt_ft:.0f} ft\nRoll = {roll_deg:.1f}°\nPitch = {pitch_deg:.1f}°')

        return wing_collection, htail_collection, vtail_collection, fuselage_collection, time_text

    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update, frames=len(times_anim),
                        interval=50, blit=False, repeat=True)

    # Save as GIF
    print(f"Saving animation to: {output_file}")
    writer = PillowWriter(fps=20)
    anim.save(output_file, writer=writer, dpi=100)

    print("Animation complete!")


if __name__ == "__main__":
    # Find the most recent history file
    history_file = Path("output/flight_history.pkl")

    if not history_file.exists():
        print(f"ERROR: History file not found: {history_file}")
        print("Run a simulation first to generate trajectory data")
        exit(1)

    output_file = Path("output/conventional_planform_animation.gif")
    create_animation(history_file, output_file)

    print(f"\nAnimation saved to: {output_file}")
