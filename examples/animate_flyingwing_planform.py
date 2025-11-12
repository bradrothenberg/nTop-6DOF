"""
Animate flying wing trajectory with 3D planform visualization.

Shows the actual wing geometry as the aircraft flies.
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


def create_flyingwing_geometry():
    """
    Create 3D geometry for flying wing aircraft.
    Uses actual dimensions from AVL file (uav_flyingwing.avl).

    Returns vertices for:
    - Wing (highly tapered, 7 sections)
    - Small fuselage pod
    """

    # Wing geometry from AVL file
    # 7 sections with highly tapered planform
    # Span: 24.863 ft (Bref), highly swept

    wing_sections = [
        # [Xle, Yle, Chord]
        [0.0219, 0.0000, 22.4008],     # Root
        [1.5078, 0.5789, 20.7604],     # Section 2
        [3.3144, 1.1579, 18.9268],     # Section 3
        [6.4911, 1.7368, 14.9830],     # Section 4
        [12.8889, 5.9671, 6.0971],     # Section 5
        [17.9008, 11.8672, 2.0784],    # Section 6
        [18.3743, 12.4315, 0.2455],    # Tip
    ]

    # Build wing vertices (left and right halves)
    n_sections = len(wing_sections)
    n_verts_per_side = n_sections * 2  # LE and TE for each section

    wing_vertices_left = []
    wing_vertices_right = []

    for xle, yle, chord in wing_sections:
        # Left wing (negative Y)
        wing_vertices_left.append([xle, -yle, 0])          # Leading edge
        wing_vertices_left.append([xle + chord, -yle, 0])  # Trailing edge

        # Right wing (positive Y)
        wing_vertices_right.append([xle, yle, 0])          # Leading edge
        wing_vertices_right.append([xle + chord, yle, 0])  # Trailing edge

    wing_vertices = np.array(wing_vertices_left + wing_vertices_right)

    # Small fuselage pod at center
    fuse_length = 8.0
    fuse_width = 1.2
    fuse_height = 1.0
    fuse_x_start = 10.0

    fuselage_vertices = np.array([
        # Front
        [fuse_x_start, -fuse_width/2, -fuse_height/2],
        [fuse_x_start, fuse_width/2, -fuse_height/2],
        [fuse_x_start, fuse_width/2, fuse_height/2],
        [fuse_x_start, -fuse_width/2, fuse_height/2],
        # Rear
        [fuse_x_start + fuse_length, -fuse_width/2, -fuse_height/2],
        [fuse_x_start + fuse_length, fuse_width/2, -fuse_height/2],
        [fuse_x_start + fuse_length, fuse_width/2, fuse_height/2],
        [fuse_x_start + fuse_length, -fuse_width/2, fuse_height/2],
    ])

    return {
        'wing': wing_vertices,
        'fuselage': fuselage_vertices,
        'n_sections': n_sections
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
        if component == 'n_sections':
            continue
        # Rotate each vertex about aircraft origin (CG)
        rotated = vertices @ R.T
        # Translate to world position
        transformed[component] = rotated + position

    return transformed


def create_animation(history_file, output_file):
    """Create animation with flying wing planform."""

    print(f"Loading trajectory from: {history_file}")
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    times = history['time']
    positions = history['position']
    attitudes = history['attitude']

    print(f"Loaded {len(times)} time steps")
    print(f"Duration: {times[-1]:.1f} seconds")

    # Create base aircraft geometry
    base_geometry = create_flyingwing_geometry()
    n_sections = base_geometry['n_sections']

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
    fuselage_collection = Poly3DCollection([], facecolors='gray',
                                           edgecolors='black', linewidths=1, alpha=0.6)

    ax.add_collection3d(wing_collection)
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
    ax.set_title('Flying Wing - 3D Trajectory (Hybrid XFOIL+AVL)', fontsize=14, fontweight='bold')

    # Set view angle
    ax.view_init(elev=20, azim=-60)

    ax.grid(True, alpha=0.3)

    def create_wing_surfaces(vertices, n_sections):
        """Create wing surface polygons from vertices."""
        # Vertices are organized as [LE, TE, LE, TE, ...] for each section
        # Left wing: indices 0 to 2*n_sections-1
        # Right wing: indices 2*n_sections to 4*n_sections-1

        surfaces = []

        # Left wing panels
        for i in range(n_sections - 1):
            # Panel between section i and i+1
            idx_base = i * 2
            # Quadrilateral: [LE_i, TE_i, TE_i+1, LE_i+1]
            panel = vertices[[idx_base, idx_base+1, idx_base+3, idx_base+2]]
            surfaces.append(panel)

        # Right wing panels
        offset = n_sections * 2
        for i in range(n_sections - 1):
            idx_base = offset + i * 2
            panel = vertices[[idx_base, idx_base+1, idx_base+3, idx_base+2]]
            surfaces.append(panel)

        return surfaces

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
        wing_surfaces = create_wing_surfaces(transformed['wing'], n_sections)
        wing_collection.set_verts(wing_surfaces)

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

        return wing_collection, fuselage_collection, time_text

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
    # Find the hybrid flight history file
    history_file = Path("output/flyingwing_hybrid_history.pkl")

    if not history_file.exists():
        print(f"ERROR: History file not found: {history_file}")
        print("Run flyingwing_hybrid_aero.py first to generate trajectory data")
        exit(1)

    output_file = Path("output/flyingwing_hybrid_planform_animation.gif")
    create_animation(history_file, output_file)

    print(f"\nAnimation saved to: {output_file}")
