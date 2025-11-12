"""
Animate Flying Wing Stable Flight with STL Wireframe

Creates a fast-rendering animated GIF using wireframe representation
of the STL model during stable flight.
"""

import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import TurbofanModel
from src.environment.atmosphere import StandardAtmosphere
from src.control.autopilot import FlyingWingAutopilot


def load_stl_simplified(filename, simplify_factor=10):
    """
    Load STL file and return simplified edge representation.

    Args:
        filename: Path to STL file
        simplify_factor: Keep every Nth triangle

    Returns:
        edges: List of edge vertices for wireframe
    """
    from stl import mesh as stl_mesh

    # Load the STL file
    aircraft_mesh = stl_mesh.Mesh.from_file(filename)

    # Get mesh bounds and center
    mesh_min = aircraft_mesh.vectors.min(axis=(0, 1))
    mesh_max = aircraft_mesh.vectors.max(axis=(0, 1))
    mesh_center = (mesh_min + mesh_max) / 2

    # Center the mesh
    aircraft_mesh.vectors -= mesh_center

    # Scale to reasonable size
    mesh_size = mesh_max - mesh_min
    target_span = 25.0  # feet
    current_span = mesh_size[1]
    scale_factor = target_span / current_span if current_span > 0 else 1.0
    aircraft_mesh.vectors *= scale_factor

    # Extract edges (simplified)
    edges = []
    for i, triangle in enumerate(aircraft_mesh.vectors):
        if i % simplify_factor == 0:  # Simplify by taking every Nth triangle
            # Each triangle has 3 edges
            edges.append([triangle[0], triangle[1]])
            edges.append([triangle[1], triangle[2]])
            edges.append([triangle[2], triangle[0]])

    return edges, mesh_size * scale_factor


def rotate_point(point, roll, pitch, yaw):
    """Rotate a single point by Euler angles."""
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation
    R = Rz @ Ry @ Rx
    return R @ point


def main():
    """Run stable flight simulation and create fast STL wireframe animation."""

    print("=" * 70)
    print("Flying Wing - STL Wireframe Animation (Fast)")
    print("=" * 70)
    print()

    # Check if STL file exists
    stl_path = Path(__file__).parent.parent / "Data" / "uav.stl"
    if not stl_path.exists():
        print(f"ERROR: STL file not found at {stl_path}")
        return

    print(f"Loading STL model from: {stl_path}")

    try:
        from stl import mesh as stl_mesh
    except ImportError:
        print("ERROR: numpy-stl not installed")
        return

    # Load simplified STL
    print("Loading and simplifying 3D model...")
    edges, mesh_size = load_stl_simplified(str(stl_path), simplify_factor=20)
    print(f"  Loaded {len(edges)} edges (simplified)")
    print(f"  Model size: {mesh_size[0]:.1f} x {mesh_size[1]:.1f} x {mesh_size[2]:.1f} ft")
    print()

    # Aircraft configuration
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

    # Simulation
    dt = 0.05
    max_time = 30.0

    positions = []
    euler_angles = []

    t = 0.0
    while t <= max_time:
        positions.append(state.position.copy())
        euler_angles.append(np.array(state.euler_angles))

        altitude = -state.position[2]
        airspeed = state.airspeed

        elevon = autopilot.update(
            current_altitude=altitude,
            current_pitch=state.euler_angles[1],
            current_pitch_rate=state.angular_rates[1],
            current_airspeed=airspeed,
            current_alpha=state.alpha,
            dt=dt
        )

        throttle = 0.80 + 0.015 * (600.0 - airspeed)
        throttle = np.clip(throttle, 0.5, 1.0)

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

    positions = np.array(positions)
    euler_angles = np.array(euler_angles)

    print(f"Simulation complete! {len(positions)} frames")
    print()

    # Create animation
    print("Creating STL wireframe animation...")
    os.makedirs('output', exist_ok=True)

    # Subsample for smooth animation
    skip = 3
    positions_anim = positions[::skip]
    euler_angles_anim = euler_angles[::skip]

    print(f"  Animation frames: {len(positions_anim)}")

    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory
    x_traj = positions[:, 0] / 5280.0
    y_traj = positions[:, 1] / 5280.0
    z_traj = -positions[:, 2]

    def update(frame):
        ax.clear()

        pos = positions_anim[frame]
        euler = euler_angles_anim[frame]
        roll, pitch, yaw = euler

        x_ft = pos[0]
        y_ft = pos[1]
        z_ft = -pos[2]

        # Draw aircraft wireframe
        for edge in edges:
            # Rotate and translate both endpoints
            p1_rot = rotate_point(edge[0], roll, pitch, yaw)
            p2_rot = rotate_point(edge[1], roll, pitch, yaw)

            # NED to plot coordinates
            p1_world = np.array([y_ft + p1_rot[1], x_ft + p1_rot[0], z_ft - p1_rot[2]])
            p2_world = np.array([y_ft + p2_rot[1], x_ft + p2_rot[0], z_ft - p2_rot[2]])

            ax.plot([p1_world[0], p2_world[0]],
                   [p1_world[1], p2_world[1]],
                   [p1_world[2], p2_world[2]],
                   'b-', linewidth=0.5, alpha=0.6)

        # Plot trajectory
        current_idx = frame * skip
        ax.plot(y_traj[:current_idx], x_traj[:current_idx], z_traj[:current_idx],
               'r-', linewidth=2, alpha=0.7, label='Flight Path')

        ax.scatter(y_traj[0], x_traj[0], z_traj[0],
                  c='g', s=150, marker='o', label='Start')

        # Labels
        ax.set_xlabel('East (ft)', fontsize=11)
        ax.set_ylabel('North (miles)', fontsize=11)
        ax.set_zlabel('Altitude (ft)', fontsize=11)
        ax.set_title(f'Flying Wing Stable Flight (STL Model)\n' +
                    f't={frame*skip*dt:.1f}s | Alt: {z_ft:.0f} ft | Pitch: {np.degrees(pitch):.1f}°',
                    fontsize=13, fontweight='bold')

        # Set limits
        ax.set_xlim([-50, 50])
        ax.set_ylim([x_traj.min()-0.05, x_traj.max()+0.05])
        ax.set_zlim([4900, 5100])

        # Rotate view
        ax.view_init(elev=25, azim=45 + frame)

        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Create animation
    print("  Rendering frames...")
    anim = FuncAnimation(fig, update, frames=len(positions_anim),
                        interval=50, blit=False)

    # Save
    output_path = 'output/flyingwing_stl_wireframe.gif'
    writer = PillowWriter(fps=20)
    print("  Saving GIF (this may take a minute)...")
    anim.save(output_path, writer=writer)

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({file_size:.2f} MB)")
    print()
    print("=" * 70)
    print("Animation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
