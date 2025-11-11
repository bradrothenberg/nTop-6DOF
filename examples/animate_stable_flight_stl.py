"""
Animate Flying Wing Stable Flight with STL Model

Creates an animated GIF showing the flying wing's 3D STL model
following its trajectory during stable flight demonstration.
"""

import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import TurbofanModel
from src.environment.atmosphere import StandardAtmosphere
from src.control.autopilot import FlyingWingAutopilot


def load_stl(filename):
    """
    Load STL file and return vertices and faces.

    Returns:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of vertex indices for each triangle
    """
    from stl import mesh

    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(filename)

    # Extract vertices (each triangle has 3 vertices)
    # STL format stores vertices per triangle, so we need to reshape
    vertices = stl_mesh.vectors.reshape(-1, 3)

    # Create face indices
    n_triangles = len(stl_mesh.vectors)
    faces = np.arange(n_triangles * 3).reshape(-1, 3)

    return vertices, faces


def rotate_mesh(vertices, roll, pitch, yaw):
    """
    Rotate mesh vertices by Euler angles.

    Args:
        vertices: Nx3 array of vertices
        roll, pitch, yaw: Rotation angles in radians

    Returns:
        rotated_vertices: Nx3 array of rotated vertices
    """
    # Rotation matrices
    # Roll (rotation about x-axis)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Pitch (rotation about y-axis)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Yaw (rotation about z-axis)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation (ZYX order - yaw, pitch, roll)
    R = Rz @ Ry @ Rx

    # Apply rotation
    rotated = (R @ vertices.T).T

    return rotated


def translate_mesh(vertices, position):
    """Translate mesh by position vector."""
    return vertices + position


def main():
    """Run stable flight simulation and create STL animation."""

    print("=" * 70)
    print("Flying Wing - STL Model Animation")
    print("=" * 70)
    print()

    # Check if STL file exists
    stl_path = Path(__file__).parent.parent / "Data" / "uav.stl"
    if not stl_path.exists():
        print(f"ERROR: STL file not found at {stl_path}")
        print("Please ensure Data/uav.stl exists")
        return

    print(f"Loading STL model from: {stl_path}")

    # Try to import numpy-stl
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        print("ERROR: numpy-stl not installed")
        print("Please install: pip install numpy-stl")
        return

    # Load STL mesh
    print("Loading 3D model...")
    aircraft_mesh = stl_mesh.Mesh.from_file(str(stl_path))
    print(f"  Loaded {len(aircraft_mesh.vectors)} triangles")

    # Get mesh bounds for scaling
    mesh_min = aircraft_mesh.vectors.min(axis=(0, 1))
    mesh_max = aircraft_mesh.vectors.max(axis=(0, 1))
    mesh_size = mesh_max - mesh_min
    print(f"  Mesh dimensions: {mesh_size[0]:.1f} x {mesh_size[1]:.1f} x {mesh_size[2]:.1f}")

    # Center the mesh
    mesh_center = (mesh_min + mesh_max) / 2
    aircraft_mesh.vectors -= mesh_center

    # Scale mesh to reasonable size (assume it's in inches, scale to feet)
    # Typical flying wing: ~25 ft span
    target_span = 25.0  # feet
    current_span = mesh_size[1]  # y-axis is typically span
    scale_factor = target_span / current_span if current_span > 0 else 1.0
    aircraft_mesh.vectors *= scale_factor
    print(f"  Scaled by {scale_factor:.3f}x to {target_span:.1f} ft span")
    print()

    # Aircraft configuration (from working example)
    mass = 228.924806
    inertia = np.array([[19236.2914, 0.0, 0.0],
                        [0.0, 2251.0172, 0.0],
                        [0.0, 0.0, 21487.3086]])

    S_ref = 412.6370
    c_ref = 11.9555
    b_ref = 24.8630

    # Create aerodynamic model
    aero = LinearAeroModel(S_ref, c_ref, b_ref)
    aero.CL_0 = 0.000023
    aero.CL_alpha = 1.412241
    aero.CL_q = 1.282202
    aero.CL_de = 0.0
    aero.CD_0 = 0.006
    aero.CD_alpha = 0.025
    aero.CD_alpha2 = 0.05
    aero.Cm_0 = 0.000061
    aero.Cm_alpha = -0.079668
    aero.Cm_q = -0.347
    aero.Cm_de = -0.02
    aero.Cl_beta = -0.1
    aero.Cl_p = -0.4
    aero.Cl_r = 0.1
    aero.Cl_da = -0.001536
    aero.Cn_beta = 0.05
    aero.Cn_p = -0.05
    aero.Cn_r = -0.1
    aero.CY_beta = -0.2

    turbofan = TurbofanModel(thrust_max=1900.0, altitude_lapse_rate=0.7)
    dynamics = AircraftDynamics(mass, inertia)

    # Create autopilot
    autopilot = FlyingWingAutopilot(
        Kp_alt=0.005,
        Ki_alt=0.0005,
        Kd_alt=0.012,
        Kp_pitch=0.8,
        Ki_pitch=0.05,
        Kd_pitch=0.15,
        Kp_pitch_rate=0.15,
        Ki_pitch_rate=0.01
    )
    autopilot.set_trim(np.radians(-5.66))
    autopilot.set_target_altitude(-5000.0)

    # Initial state
    state = State()
    state.position = np.array([0.0, 0.0, -5000.0])
    state.velocity_body = np.array([600.0, 0.0, 0.0])
    state.set_euler_angles(0.0, np.radians(1.4649), 0.0)
    state.angular_rates = np.array([0.0, 0.0, 0.0])

    print("Running stable flight simulation (30 seconds)...")

    # Simulation parameters
    dt = 0.05
    max_time = 30.0

    # Storage
    positions = []
    euler_angles = []

    t = 0.0
    step = 0
    while t <= max_time:
        positions.append(state.position.copy())
        euler_angles.append(np.array(state.euler_angles))

        altitude = -state.position[2]
        airspeed = state.airspeed

        # Get autopilot command
        elevon = autopilot.update(
            current_altitude=altitude,
            current_pitch=state.euler_angles[1],
            current_pitch_rate=state.angular_rates[1],
            current_airspeed=airspeed,
            current_alpha=state.alpha,
            dt=dt
        )

        # Throttle control
        throttle = 0.80 + 0.015 * (600.0 - airspeed)
        throttle = np.clip(throttle, 0.5, 1.0)

        # Force function
        controls = {
            'elevator': elevon,
            'aileron': 0.0,
            'rudder': 0.0,
            'throttle': throttle
        }

        def force_func(s):
            atm = StandardAtmosphere(s.altitude)
            aero.rho = atm.density
            aero_forces, aero_moments = aero.compute_forces_moments(s, controls)
            prop_forces, prop_moments = turbofan.compute_thrust(s, controls['throttle'])
            return aero_forces + prop_forces, aero_moments + prop_moments

        # RK4 integration
        state_dot = dynamics.state_derivative(state, force_func)
        state_array = state.to_array()

        k1 = state_dot
        state_temp = State()
        state_temp.from_array(state_array + 0.5 * dt * k1)
        k2 = dynamics.state_derivative(state_temp, force_func)

        state_temp.from_array(state_array + 0.5 * dt * k2)
        k3 = dynamics.state_derivative(state_temp, force_func)

        state_temp.from_array(state_array + dt * k3)
        k4 = dynamics.state_derivative(state_temp, force_func)

        state_new = state_array + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state.from_array(state_new)

        t += dt
        step += 1

    positions = np.array(positions)
    euler_angles = np.array(euler_angles)

    print(f"Simulation complete! Generated {len(positions)} frames")
    print()

    # Create animation
    print("Creating 3D STL animation...")
    os.makedirs('output', exist_ok=True)

    # Subsample for animation (every 5th frame for smoother playback)
    skip = 5
    positions_anim = positions[::skip]
    euler_angles_anim = euler_angles[::skip]

    print(f"  Animation frames: {len(positions_anim)}")

    # Setup figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Convert positions to miles for plotting
    x_traj = positions[:, 0] / 5280.0
    y_traj = positions[:, 1] / 5280.0
    z_traj = -positions[:, 2]

    # Set axis limits
    x_range = [x_traj.min() - 0.1, x_traj.max() + 0.1]
    y_range = [-0.1, 0.1]
    z_range = [4800, 5200]

    def update(frame):
        ax.clear()

        # Get current position and orientation
        pos = positions_anim[frame]
        euler = euler_angles_anim[frame]

        roll, pitch, yaw = euler

        # Position in feet
        x_ft = pos[0]
        y_ft = pos[1]
        z_ft = -pos[2]  # Altitude (positive up)

        # Rotate mesh
        rotated_vectors = []
        for triangle in aircraft_mesh.vectors:
            # Rotate each vertex
            rotated_tri = []
            for vertex in triangle:
                # Apply rotation (note: NED to plotting coordinates)
                # NED: x=north, y=east, z=down
                # Plot: x=east, y=north, z=up
                v_rotated = rotate_mesh(vertex.reshape(1, -1), roll, pitch, yaw)[0]

                # Translate to aircraft position
                # Swap coordinates: NED to plot frame
                v_world = np.array([
                    y_ft + v_rotated[1],  # y = east
                    x_ft + v_rotated[0],  # x = north
                    z_ft - v_rotated[2]   # z = up (NED z is down)
                ])
                rotated_tri.append(v_world)

            rotated_vectors.append(rotated_tri)

        # Create 3D polygon collection
        poly = Poly3DCollection(rotated_vectors,
                               facecolors='lightblue',
                               edgecolors='black',
                               linewidths=0.1,
                               alpha=0.9)
        ax.add_collection3d(poly)

        # Plot trajectory up to current point
        current_idx = frame * skip
        ax.plot(y_traj[:current_idx], x_traj[:current_idx], z_traj[:current_idx],
               'r-', linewidth=2, alpha=0.5, label='Flight Path')

        # Mark start position
        ax.scatter(y_traj[0], x_traj[0], z_traj[0],
                  c='g', s=100, marker='o', label='Start')

        # Set labels and title
        ax.set_xlabel('East (ft)', fontsize=10)
        ax.set_ylabel('North (miles)', fontsize=10)
        ax.set_zlabel('Altitude (ft)', fontsize=10)
        ax.set_title(f'Flying Wing Stable Flight - t={frame*skip*dt:.1f}s\n' +
                    f'Alt: {z_ft:.0f} ft, Pitch: {np.degrees(pitch):.1f}°',
                    fontsize=12)

        # Set axis limits
        ax.set_xlim([-50, 50])
        ax.set_ylim(x_range)
        ax.set_zlim(z_range)

        # Better viewing angle
        ax.view_init(elev=20, azim=frame * 2)  # Rotate view slowly

        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Create animation
    print("  Rendering animation...")
    anim = FuncAnimation(fig, update, frames=len(positions_anim),
                        interval=50, blit=False)

    # Save as GIF
    output_path = 'output/flyingwing_stl_animation.gif'
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer)

    print(f"  Saved: {output_path}")
    print()
    print("=" * 70)
    print("Animation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
