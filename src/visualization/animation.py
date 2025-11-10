"""
Animation Module

Provides animation capabilities for flight dynamics visualization.
Includes animated 3D trajectories and aircraft attitude visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Callable
import warnings


class TrajectoryAnimation:
    """
    Animate aircraft trajectory in 3D space.

    Parameters
    ----------
    positions : np.ndarray
        Position array of shape (N, 3) containing [x, y, z] coordinates
    attitudes : Optional[np.ndarray]
        Euler angles array of shape (N, 3) containing [roll, pitch, yaw] in radians
    time : Optional[np.ndarray]
        Time array of shape (N,). If None, frame indices are used
    dt : float, optional
        Time step between frames (seconds). Default is 0.1
    """

    def __init__(
        self,
        positions: np.ndarray,
        attitudes: Optional[np.ndarray] = None,
        time: Optional[np.ndarray] = None,
        dt: float = 0.1
    ):
        self.positions = positions
        self.attitudes = attitudes
        self.time = time if time is not None else np.arange(len(positions)) * dt
        self.dt = dt
        self.n_frames = len(positions)

        # Animation objects (created during animate())
        self.fig = None
        self.ax = None
        self.trajectory_line = None
        self.aircraft_marker = None
        self.attitude_vectors = None

    def setup_figure(self, figsize: Tuple[float, float] = (10, 8)) -> Tuple[Figure, Axes3D]:
        """
        Set up the figure and 3D axes for animation.

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches

        Returns
        -------
        Tuple[Figure, Axes3D]
            Figure and 3D axes objects
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set up axes
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        z = self.positions[:, 2]

        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5

        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.ax.set_xlabel('East (ft)', fontsize=11)
        self.ax.set_ylabel('North (ft)', fontsize=11)
        self.ax.set_zlabel('Altitude (ft)', fontsize=11)
        self.ax.set_title('Flight Trajectory Animation', fontsize=13, fontweight='bold')
        self.ax.grid(True, alpha=0.3)

        return self.fig, self.ax

    def init_animation(self):
        """Initialize animation elements (called by FuncAnimation)."""
        # Trajectory line (initially empty)
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')

        # Aircraft marker
        self.aircraft_marker, = self.ax.plot([], [], [], 'ro', markersize=10, label='Aircraft')

        # Attitude vectors (if attitudes are provided)
        if self.attitudes is not None:
            self.attitude_vectors = {
                'roll': self.ax.quiver(0, 0, 0, 0, 0, 0, color='r', length=100, normalize=True, label='Roll'),
                'pitch': self.ax.quiver(0, 0, 0, 0, 0, 0, color='g', length=100, normalize=True, label='Pitch'),
                'yaw': self.ax.quiver(0, 0, 0, 0, 0, 0, color='b', length=100, normalize=True, label='Yaw')
            }

        self.ax.legend(loc='upper right')

        return self.trajectory_line, self.aircraft_marker

    def update_frame(self, frame: int):
        """
        Update animation for given frame.

        Parameters
        ----------
        frame : int
            Current frame index
        """
        # Update trajectory (show path up to current frame)
        self.trajectory_line.set_data(self.positions[:frame+1, 0], self.positions[:frame+1, 1])
        self.trajectory_line.set_3d_properties(self.positions[:frame+1, 2])

        # Update aircraft position
        current_pos = self.positions[frame]
        self.aircraft_marker.set_data([current_pos[0]], [current_pos[1]])
        self.aircraft_marker.set_3d_properties([current_pos[2]])

        # Update attitude vectors if available
        if self.attitudes is not None and self.attitude_vectors is not None:
            roll, pitch, yaw = self.attitudes[frame]

            # Compute body frame vectors
            vector_length = (self.positions[:, 0].max() - self.positions[:, 0].min()) * 0.1

            # Roll axis (x-body, forward)
            x_body = np.array([
                np.cos(yaw) * np.cos(pitch),
                np.sin(yaw) * np.cos(pitch),
                -np.sin(pitch)
            ]) * vector_length

            # Pitch axis (y-body, right wing)
            y_body = np.array([
                np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
                np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
                np.cos(pitch) * np.sin(roll)
            ]) * vector_length

            # Yaw axis (z-body, down)
            z_body = np.array([
                np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll),
                np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll),
                np.cos(pitch) * np.cos(roll)
            ]) * vector_length

            # Remove old vectors and add new ones
            for key in self.attitude_vectors:
                self.attitude_vectors[key].remove()

            self.attitude_vectors['roll'] = self.ax.quiver(
                current_pos[0], current_pos[1], current_pos[2],
                x_body[0], x_body[1], x_body[2],
                color='r', arrow_length_ratio=0.3
            )
            self.attitude_vectors['pitch'] = self.ax.quiver(
                current_pos[0], current_pos[1], current_pos[2],
                y_body[0], y_body[1], y_body[2],
                color='g', arrow_length_ratio=0.3
            )
            self.attitude_vectors['yaw'] = self.ax.quiver(
                current_pos[0], current_pos[1], current_pos[2],
                z_body[0], z_body[1], z_body[2],
                color='b', arrow_length_ratio=0.3
            )

        # Update title with time
        self.ax.set_title(f'Flight Trajectory Animation (t = {self.time[frame]:.2f} s)',
                         fontsize=13, fontweight='bold')

        artists = [self.trajectory_line, self.aircraft_marker]
        if self.attitude_vectors:
            artists.extend(self.attitude_vectors.values())

        return artists

    def animate(
        self,
        interval: int = 50,
        repeat: bool = True,
        figsize: Tuple[float, float] = (10, 8)
    ) -> FuncAnimation:
        """
        Create animation.

        Parameters
        ----------
        interval : int, optional
            Delay between frames in milliseconds (default 50ms = 20 fps)
        repeat : bool, optional
            Whether to repeat animation
        figsize : Tuple[float, float], optional
            Figure size in inches

        Returns
        -------
        FuncAnimation
            Matplotlib animation object
        """
        self.setup_figure(figsize=figsize)

        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_animation,
            frames=self.n_frames,
            interval=interval,
            repeat=repeat,
            blit=False  # blit=True doesn't work well with 3D
        )

        return anim

    def save(
        self,
        filename: str,
        fps: int = 20,
        dpi: int = 100,
        interval: int = 50,
        figsize: Tuple[float, float] = (10, 8),
        writer: str = 'pillow'
    ):
        """
        Save animation to file.

        Parameters
        ----------
        filename : str
            Output filename (extension determines format: .gif, .mp4)
        fps : int, optional
            Frames per second for video output
        dpi : int, optional
            Resolution (dots per inch)
        interval : int, optional
            Delay between frames in milliseconds
        figsize : Tuple[float, float], optional
            Figure size in inches
        writer : str, optional
            Animation writer: 'pillow' for GIF, 'ffmpeg' for MP4
        """
        anim = self.animate(interval=interval, repeat=False, figsize=figsize)

        # Select writer based on file extension
        if filename.endswith('.gif'):
            writer_obj = PillowWriter(fps=fps)
        elif filename.endswith('.mp4'):
            try:
                writer_obj = FFMpegWriter(fps=fps)
            except Exception:
                warnings.warn("FFMpeg not available, falling back to Pillow (GIF)")
                filename = filename.replace('.mp4', '.gif')
                writer_obj = PillowWriter(fps=fps)
        else:
            writer_obj = PillowWriter(fps=fps)

        anim.save(filename, writer=writer_obj, dpi=dpi)
        print(f"Animation saved to {filename}")


def animate_trajectory(
    positions: np.ndarray,
    attitudes: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    dt: float = 0.1,
    interval: int = 50,
    save_path: Optional[str] = None,
    fps: int = 20,
    dpi: int = 100,
    figsize: Tuple[float, float] = (10, 8)
) -> Optional[FuncAnimation]:
    """
    Convenience function to create and optionally save trajectory animation.

    Parameters
    ----------
    positions : np.ndarray
        Position array of shape (N, 3)
    attitudes : Optional[np.ndarray]
        Euler angles array of shape (N, 3)
    time : Optional[np.ndarray]
        Time array of shape (N,)
    dt : float, optional
        Time step between frames
    interval : int, optional
        Delay between frames in milliseconds
    save_path : Optional[str], optional
        Path to save animation (if None, animation is displayed)
    fps : int, optional
        Frames per second for saved video
    dpi : int, optional
        Resolution for saved video
    figsize : Tuple[float, float], optional
        Figure size in inches

    Returns
    -------
    Optional[FuncAnimation]
        Animation object (or None if saved to file)
    """
    animator = TrajectoryAnimation(positions, attitudes, time, dt)

    if save_path:
        animator.save(save_path, fps=fps, dpi=dpi, interval=interval, figsize=figsize)
        return None
    else:
        anim = animator.animate(interval=interval, figsize=figsize)
        plt.show()
        return anim


def create_comparison_animation(
    positions_list: list,
    labels: list,
    colors: Optional[list] = None,
    time: Optional[np.ndarray] = None,
    dt: float = 0.1,
    interval: int = 50,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> Optional[FuncAnimation]:
    """
    Create animation comparing multiple trajectories.

    Parameters
    ----------
    positions_list : list
        List of position arrays, each of shape (N, 3)
    labels : list
        List of labels for each trajectory
    colors : Optional[list], optional
        List of colors for each trajectory
    time : Optional[np.ndarray], optional
        Time array (must be same length as each position array)
    dt : float, optional
        Time step between frames
    interval : int, optional
        Delay between frames in milliseconds
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Optional[str], optional
        Path to save animation

    Returns
    -------
    Optional[FuncAnimation]
        Animation object (or None if saved)
    """
    n_trajectories = len(positions_list)

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_trajectories))

    # Get frame count (minimum across all trajectories)
    n_frames = min(len(pos) for pos in positions_list)

    if time is None:
        time = np.arange(n_frames) * dt

    # Set up figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Compute bounds across all trajectories
    all_positions = np.vstack(positions_list)
    x_all = all_positions[:, 0]
    y_all = all_positions[:, 1]
    z_all = all_positions[:, 2]

    max_range = np.array([x_all.max()-x_all.min(), y_all.max()-y_all.min(), z_all.max()-z_all.min()]).max() / 2.0
    mid_x = (x_all.max()+x_all.min()) * 0.5
    mid_y = (y_all.max()+y_all.min()) * 0.5
    mid_z = (z_all.max()+z_all.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('East (ft)', fontsize=11)
    ax.set_ylabel('North (ft)', fontsize=11)
    ax.set_zlabel('Altitude (ft)', fontsize=11)
    ax.set_title('Trajectory Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Initialize trajectory lines and markers
    lines = []
    markers = []
    for i, (positions, label, color) in enumerate(zip(positions_list, labels, colors)):
        line, = ax.plot([], [], [], '-', linewidth=2, color=color, label=label)
        marker, = ax.plot([], [], [], 'o', markersize=8, color=color)
        lines.append(line)
        markers.append(marker)

    ax.legend(loc='upper right')

    def init():
        for line, marker in zip(lines, markers):
            line.set_data([], [])
            line.set_3d_properties([])
            marker.set_data([], [])
            marker.set_3d_properties([])
        return lines + markers

    def update(frame):
        for i, (positions, line, marker) in enumerate(zip(positions_list, lines, markers)):
            if frame < len(positions):
                # Update trajectory
                line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
                line.set_3d_properties(positions[:frame+1, 2])

                # Update marker
                marker.set_data([positions[frame, 0]], [positions[frame, 1]])
                marker.set_3d_properties([positions[frame, 2]])

        ax.set_title(f'Trajectory Comparison (t = {time[frame]:.2f} s)',
                    fontsize=13, fontweight='bold')

        return lines + markers

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=interval,
        repeat=True,
        blit=False
    )

    if save_path:
        if save_path.endswith('.gif'):
            writer = PillowWriter(fps=int(1000/interval))
        else:
            writer = PillowWriter(fps=int(1000/interval))

        anim.save(save_path, writer=writer, dpi=100)
        print(f"Comparison animation saved to {save_path}")
        return None
    else:
        plt.show()
        return anim
