"""
Standard Plotting Functions

Provides visualization capabilities for flight dynamics simulation data.
Includes trajectory plots, state histories, control deflections, and trim envelopes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def plot_trajectory_3d(
    positions: np.ndarray,
    title: str = "3D Flight Trajectory",
    axes_labels: Tuple[str, str, str] = ("East (ft)", "North (ft)", "Altitude (ft)"),
    show_markers: bool = True,
    marker_interval: int = 50,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot 3D flight trajectory.

    Parameters
    ----------
    positions : np.ndarray
        Position array of shape (N, 3) containing [x, y, z] coordinates
    title : str, optional
        Plot title
    axes_labels : Tuple[str, str, str], optional
        Labels for x, y, z axes
    show_markers : bool, optional
        Whether to show position markers along trajectory
    marker_interval : int, optional
        Interval between markers (if show_markers=True)
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Optional[str], optional
        Path to save figure (if None, figure is not saved)

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates (negate z to convert NED altitude to positive up)
    x = positions[:, 0]
    y = positions[:, 1]
    z = -positions[:, 2]  # Negate to convert NED (down positive) to altitude (up positive)

    # Plot trajectory line
    ax.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')

    # Add markers if requested
    if show_markers and len(positions) > marker_interval:
        marker_indices = np.arange(0, len(positions), marker_interval)
        ax.scatter(x[marker_indices], y[marker_indices], z[marker_indices],
                  c='r', marker='o', s=30, label='Waypoints')

    # Mark start and end points
    ax.scatter(x[0], y[0], z[0], c='g', marker='o', s=100, label='Start', edgecolors='k')
    ax.scatter(x[-1], y[-1], z[-1], c='r', marker='s', s=100, label='End', edgecolors='k')

    # Labels and formatting
    ax.set_xlabel(axes_labels[0], fontsize=11)
    ax.set_ylabel(axes_labels[1], fontsize=11)
    ax.set_zlabel(axes_labels[2], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio for better visualization
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_states_vs_time(
    time: np.ndarray,
    states: Dict[str, np.ndarray],
    title: str = "State Variables vs Time",
    figsize: Tuple[float, float] = (12, 10),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot state variables (position, velocity, attitude) vs time.

    Parameters
    ----------
    time : np.ndarray
        Time array (N,)
    states : Dict[str, np.ndarray]
        Dictionary containing state arrays:
        - 'position': (N, 3) - [x, y, z] or [pn, pe, altitude]
        - 'velocity': (N, 3) - [u, v, w] body frame velocity
        - 'euler_angles': (N, 3) - [roll, pitch, yaw] in radians
        - 'angular_rates': (N, 3) - [p, q, r] body frame rates
    title : str, optional
        Main plot title
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Optional[str], optional
        Path to save figure

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Position
    if 'position' in states:
        pos = states['position']
        axes[0].plot(time, pos[:, 0], 'r-', label='X (East)', linewidth=1.5)
        axes[0].plot(time, pos[:, 1], 'g-', label='Y (North)', linewidth=1.5)
        axes[0].plot(time, pos[:, 2], 'b-', label='Z (Altitude)', linewidth=1.5)
        axes[0].set_ylabel('Position (ft)', fontsize=11)
        axes[0].legend(loc='best', ncol=3)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Position', fontsize=11, fontweight='bold')

    # Velocity
    if 'velocity' in states:
        vel = states['velocity']
        axes[1].plot(time, vel[:, 0], 'r-', label='u (Forward)', linewidth=1.5)
        axes[1].plot(time, vel[:, 1], 'g-', label='v (Right)', linewidth=1.5)
        axes[1].plot(time, vel[:, 2], 'b-', label='w (Down)', linewidth=1.5)
        axes[1].set_ylabel('Velocity (ft/s)', fontsize=11)
        axes[1].legend(loc='best', ncol=3)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Body Frame Velocity', fontsize=11, fontweight='bold')

    # Euler Angles
    if 'euler_angles' in states:
        angles = states['euler_angles']
        axes[2].plot(time, np.degrees(angles[:, 0]), 'r-', label='Roll', linewidth=1.5)
        axes[2].plot(time, np.degrees(angles[:, 1]), 'g-', label='Pitch', linewidth=1.5)
        axes[2].plot(time, np.degrees(angles[:, 2]), 'b-', label='Yaw', linewidth=1.5)
        axes[2].set_ylabel('Angle (deg)', fontsize=11)
        axes[2].legend(loc='best', ncol=3)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Euler Angles', fontsize=11, fontweight='bold')

    # Angular Rates
    if 'angular_rates' in states:
        rates = states['angular_rates']
        axes[3].plot(time, np.degrees(rates[:, 0]), 'r-', label='p (Roll rate)', linewidth=1.5)
        axes[3].plot(time, np.degrees(rates[:, 1]), 'g-', label='q (Pitch rate)', linewidth=1.5)
        axes[3].plot(time, np.degrees(rates[:, 2]), 'b-', label='r (Yaw rate)', linewidth=1.5)
        axes[3].set_ylabel('Rate (deg/s)', fontsize=11)
        axes[3].set_xlabel('Time (s)', fontsize=11)
        axes[3].legend(loc='best', ncol=3)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title('Angular Rates', fontsize=11, fontweight='bold')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_controls_vs_time(
    time: np.ndarray,
    controls: Dict[str, np.ndarray],
    title: str = "Control Inputs vs Time",
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot control surface deflections and throttle vs time.

    Parameters
    ----------
    time : np.ndarray
        Time array (N,)
    controls : Dict[str, np.ndarray]
        Dictionary containing control arrays:
        - 'elevator': (N,) - Elevator deflection (radians or normalized)
        - 'aileron': (N,) - Aileron deflection
        - 'rudder': (N,) - Rudder deflection
        - 'throttle': (N,) - Throttle setting (0-1)
    title : str, optional
        Main plot title
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Optional[str], optional
        Path to save figure

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Count available controls
    n_controls = len(controls)

    fig, axes = plt.subplots(n_controls, 1, figsize=figsize, sharex=True)

    # Make axes iterable if only one subplot
    if n_controls == 1:
        axes = [axes]

    idx = 0

    # Elevator
    if 'elevator' in controls:
        elevator = controls['elevator']
        # Check if in radians (typical range -0.5 to 0.5) or normalized
        if np.abs(elevator).max() < 1.0:
            axes[idx].plot(time, np.degrees(elevator), 'b-', linewidth=2)
            axes[idx].set_ylabel('Elevator (deg)', fontsize=11)
        else:
            axes[idx].plot(time, elevator, 'b-', linewidth=2)
            axes[idx].set_ylabel('Elevator', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title('Elevator', fontsize=11, fontweight='bold')
        idx += 1

    # Aileron
    if 'aileron' in controls:
        aileron = controls['aileron']
        if np.abs(aileron).max() < 1.0:
            axes[idx].plot(time, np.degrees(aileron), 'g-', linewidth=2)
            axes[idx].set_ylabel('Aileron (deg)', fontsize=11)
        else:
            axes[idx].plot(time, aileron, 'g-', linewidth=2)
            axes[idx].set_ylabel('Aileron', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title('Aileron', fontsize=11, fontweight='bold')
        idx += 1

    # Rudder
    if 'rudder' in controls:
        rudder = controls['rudder']
        if np.abs(rudder).max() < 1.0:
            axes[idx].plot(time, np.degrees(rudder), 'r-', linewidth=2)
            axes[idx].set_ylabel('Rudder (deg)', fontsize=11)
        else:
            axes[idx].plot(time, rudder, 'r-', linewidth=2)
            axes[idx].set_ylabel('Rudder', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title('Rudder', fontsize=11, fontweight='bold')
        idx += 1

    # Throttle
    if 'throttle' in controls:
        throttle = controls['throttle']
        axes[idx].plot(time, throttle, 'k-', linewidth=2)
        axes[idx].set_ylabel('Throttle', fontsize=11)
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title('Throttle', fontsize=11, fontweight='bold')
        idx += 1

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_forces_moments(
    time: np.ndarray,
    forces: np.ndarray,
    moments: np.ndarray,
    title: str = "Forces and Moments vs Time",
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot forces and moments time histories.

    Parameters
    ----------
    time : np.ndarray
        Time array (N,)
    forces : np.ndarray
        Force array of shape (N, 3) containing [Fx, Fy, Fz] in body frame
    moments : np.ndarray
        Moment array of shape (N, 3) containing [L, M, N] in body frame
    title : str, optional
        Main plot title
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Optional[str], optional
        Path to save figure

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Forces
    axes[0].plot(time, forces[:, 0], 'r-', label='Fx (Forward)', linewidth=1.5)
    axes[0].plot(time, forces[:, 1], 'g-', label='Fy (Right)', linewidth=1.5)
    axes[0].plot(time, forces[:, 2], 'b-', label='Fz (Down)', linewidth=1.5)
    axes[0].set_ylabel('Force (lbf)', fontsize=11)
    axes[0].legend(loc='best', ncol=3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Body Frame Forces', fontsize=11, fontweight='bold')

    # Moments
    axes[1].plot(time, moments[:, 0], 'r-', label='L (Roll)', linewidth=1.5)
    axes[1].plot(time, moments[:, 1], 'g-', label='M (Pitch)', linewidth=1.5)
    axes[1].plot(time, moments[:, 2], 'b-', label='N (Yaw)', linewidth=1.5)
    axes[1].set_ylabel('Moment (ft-lbf)', fontsize=11)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].legend(loc='best', ncol=3)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Body Frame Moments', fontsize=11, fontweight='bold')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_trim_envelope(
    trim_results: List[Dict[str, Any]],
    x_var: str,
    y_var: str,
    title: str = "Trim Envelope",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot trim envelope showing relationships between trim variables.

    Parameters
    ----------
    trim_results : List[Dict[str, Any]]
        List of trim solution dictionaries, each containing:
        - 'alpha': Angle of attack (rad)
        - 'theta': Pitch angle (rad)
        - 'elevator': Elevator deflection (rad)
        - 'throttle': Throttle setting
        - 'velocity': Airspeed (ft/s)
        - etc.
    x_var : str
        Variable name for x-axis (e.g., 'velocity', 'alpha')
    y_var : str
        Variable name for y-axis (e.g., 'elevator', 'throttle')
    title : str, optional
        Plot title
    xlabel : Optional[str], optional
        X-axis label (auto-generated if None)
    ylabel : Optional[str], optional
        Y-axis label (auto-generated if None)
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Optional[str], optional
        Path to save figure

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Extract data
    x_data = np.array([result.get(x_var, np.nan) for result in trim_results])
    y_data = np.array([result.get(y_var, np.nan) for result in trim_results])

    # Remove NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]

    # Convert angles to degrees if needed
    angle_vars = ['alpha', 'theta', 'phi', 'beta', 'elevator', 'aileron', 'rudder']
    if x_var in angle_vars:
        x_data = np.degrees(x_data)
    if y_var in angle_vars:
        y_data = np.degrees(y_data)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_data, y_data, 'bo-', linewidth=2, markersize=6)
    ax.grid(True, alpha=0.3)

    # Labels
    if xlabel is None:
        xlabel = x_var.replace('_', ' ').title()
        if x_var in angle_vars:
            xlabel += ' (deg)'
        elif x_var == 'velocity':
            xlabel += ' (ft/s)'
        elif x_var == 'altitude':
            xlabel += ' (ft)'

    if ylabel is None:
        ylabel = y_var.replace('_', ' ').title()
        if y_var in angle_vars:
            ylabel += ' (deg)'
        elif y_var == 'throttle':
            ylabel += ' (0-1)'

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def setup_plotting_style():
    """
    Set up default matplotlib plotting style for consistent appearance.

    Call this function once at the start of your script for consistent styling.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
