"""
Phase 6 Tests: Visualization

Tests for plotting and animation functionality.
"""

import pytest
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.plotting import (
    plot_trajectory_3d,
    plot_states_vs_time,
    plot_controls_vs_time,
    plot_forces_moments,
    plot_trim_envelope,
    setup_plotting_style
)

from src.visualization.animation import (
    TrajectoryAnimation,
    animate_trajectory,
    create_comparison_animation
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt


class TestPlotting:
    """Test standard plotting functions."""

    def test_plot_trajectory_3d_basic(self):
        """Test basic 3D trajectory plotting."""
        # Create simple trajectory
        t = np.linspace(0, 10, 100)
        positions = np.column_stack([
            100 * t,  # x
            50 * np.sin(0.5 * t),  # y
            5000 + 100 * t  # z
        ])

        fig = plot_trajectory_3d(positions)

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_trajectory_3d_with_markers(self):
        """Test 3D trajectory with markers."""
        t = np.linspace(0, 10, 100)
        positions = np.column_stack([
            100 * t,
            50 * np.sin(0.5 * t),
            5000 + 100 * t
        ])

        fig = plot_trajectory_3d(positions, show_markers=True, marker_interval=20)

        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory_3d_save(self):
        """Test saving 3D trajectory plot."""
        t = np.linspace(0, 10, 50)
        positions = np.column_stack([
            100 * t,
            50 * np.sin(0.5 * t),
            5000 + 100 * t
        ])

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            fig = plot_trajectory_3d(positions, save_path=temp_path)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            plt.close(fig)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_plot_states_vs_time(self):
        """Test state variable plotting."""
        t = np.linspace(0, 10, 100)

        states = {
            'position': np.column_stack([
                100 * t,
                50 * np.sin(0.5 * t),
                5000 + 10 * t
            ]),
            'velocity': np.column_stack([
                200 + 5 * np.sin(t),
                10 * np.cos(t),
                5 * np.sin(2 * t)
            ]),
            'euler_angles': np.column_stack([
                np.radians(5 * np.sin(0.3 * t)),
                np.radians(3 + 2 * np.cos(0.2 * t)),
                np.radians(45 * t / 10)
            ]),
            'angular_rates': np.column_stack([
                np.radians(2 * np.sin(0.5 * t)),
                np.radians(1 * np.cos(0.3 * t)),
                np.radians(0.5 * np.sin(0.4 * t))
            ])
        }

        fig = plot_states_vs_time(t, states)

        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_plot_states_partial(self):
        """Test state plotting with partial data."""
        t = np.linspace(0, 10, 100)

        states = {
            'position': np.column_stack([100 * t, 50 * t, 5000 * np.ones_like(t)]),
            'velocity': np.column_stack([200 * np.ones_like(t), np.zeros_like(t), np.zeros_like(t)])
        }

        fig = plot_states_vs_time(t, states)

        assert fig is not None
        plt.close(fig)

    def test_plot_controls_vs_time(self):
        """Test control input plotting."""
        t = np.linspace(0, 10, 100)

        controls = {
            'elevator': np.radians(5 * np.sin(0.5 * t)),
            'aileron': np.radians(3 * np.cos(0.3 * t)),
            'rudder': np.radians(2 * np.sin(0.2 * t)),
            'throttle': 0.7 + 0.1 * np.sin(0.1 * t)
        }

        fig = plot_controls_vs_time(t, controls)

        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_plot_controls_throttle_only(self):
        """Test control plotting with throttle only."""
        t = np.linspace(0, 10, 100)

        controls = {
            'throttle': 0.5 + 0.2 * np.sin(0.1 * t)
        }

        fig = plot_controls_vs_time(t, controls)

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_forces_moments(self):
        """Test force and moment plotting."""
        t = np.linspace(0, 10, 100)

        forces = np.column_stack([
            500 + 50 * np.sin(0.5 * t),
            10 * np.cos(0.3 * t),
            -100 + 20 * np.sin(0.2 * t)
        ])

        moments = np.column_stack([
            100 * np.sin(0.4 * t),
            200 * np.cos(0.3 * t),
            50 * np.sin(0.5 * t)
        ])

        fig = plot_forces_moments(t, forces, moments)

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_trim_envelope(self):
        """Test trim envelope plotting."""
        # Create sample trim results
        velocities = np.linspace(150, 300, 20)
        trim_results = []

        for v in velocities:
            trim_results.append({
                'velocity': v,
                'alpha': np.radians(2 + 0.01 * (300 - v)),
                'elevator': np.radians(-2 - 0.005 * (300 - v)),
                'throttle': 0.3 + 0.002 * v
            })

        fig = plot_trim_envelope(trim_results, 'velocity', 'alpha')

        assert fig is not None
        plt.close(fig)

    def test_plot_trim_envelope_throttle(self):
        """Test trim envelope with throttle."""
        altitudes = np.linspace(0, 10000, 15)
        trim_results = []

        for h in altitudes:
            trim_results.append({
                'altitude': h,
                'throttle': 0.4 + 0.00003 * h,
                'velocity': 220 - 0.002 * h
            })

        fig = plot_trim_envelope(trim_results, 'altitude', 'throttle')

        assert fig is not None
        plt.close(fig)

    def test_setup_plotting_style(self):
        """Test plotting style setup."""
        setup_plotting_style()

        # Check that rcParams were modified
        assert plt.rcParams['figure.facecolor'] == 'white'
        assert plt.rcParams['axes.grid'] == True


class TestAnimation:
    """Test animation functionality."""

    def test_trajectory_animation_init(self):
        """Test trajectory animation initialization."""
        t = np.linspace(0, 10, 50)
        positions = np.column_stack([
            100 * t,
            50 * np.sin(0.5 * t),
            5000 + 100 * t
        ])

        animator = TrajectoryAnimation(positions)

        assert animator.n_frames == len(positions)
        assert animator.positions.shape == (50, 3)

    def test_trajectory_animation_with_attitudes(self):
        """Test trajectory animation with attitude data."""
        t = np.linspace(0, 10, 50)
        positions = np.column_stack([
            100 * t,
            50 * np.sin(0.5 * t),
            5000 + 100 * t
        ])

        attitudes = np.column_stack([
            np.radians(5 * np.sin(0.3 * t)),
            np.radians(3 * np.ones_like(t)),
            np.radians(45 * t / 10)
        ])

        animator = TrajectoryAnimation(positions, attitudes=attitudes)

        assert animator.attitudes is not None
        assert animator.attitudes.shape == (50, 3)

    def test_trajectory_animation_setup_figure(self):
        """Test animation figure setup."""
        t = np.linspace(0, 10, 50)
        positions = np.column_stack([
            100 * t,
            50 * np.sin(0.5 * t),
            5000 + 100 * t
        ])

        animator = TrajectoryAnimation(positions)
        fig, ax = animator.setup_figure()

        assert fig is not None
        assert ax is not None
        assert ax.name == '3d'
        plt.close(fig)

    def test_trajectory_animation_save_gif(self):
        """Test saving animation to GIF."""
        t = np.linspace(0, 2, 20)  # Short animation
        positions = np.column_stack([
            100 * t,
            50 * np.sin(t),
            5000 + 50 * t
        ])

        animator = TrajectoryAnimation(positions)

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            temp_path = f.name

        try:
            animator.save(temp_path, fps=10, dpi=50)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            plt.close('all')

    def test_animate_trajectory_function(self):
        """Test animate_trajectory convenience function."""
        t = np.linspace(0, 5, 25)
        positions = np.column_stack([
            100 * t,
            50 * np.sin(0.5 * t),
            5000 + 50 * t
        ])

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            temp_path = f.name

        try:
            animate_trajectory(positions, save_path=temp_path, fps=10, dpi=50)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            plt.close('all')

    def test_create_comparison_animation(self):
        """Test comparison animation."""
        t = np.linspace(0, 5, 25)

        # Two trajectories
        pos1 = np.column_stack([
            100 * t,
            50 * np.sin(0.5 * t),
            5000 + 50 * t
        ])

        pos2 = np.column_stack([
            100 * t,
            30 * np.cos(0.5 * t),
            5000 + 40 * t
        ])

        positions_list = [pos1, pos2]
        labels = ['Trajectory 1', 'Trajectory 2']

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            temp_path = f.name

        try:
            create_comparison_animation(
                positions_list,
                labels,
                save_path=temp_path
            )
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            plt.close('all')


class TestIntegration:
    """Integration tests for visualization."""

    def test_complete_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Simulate flight data
        t = np.linspace(0, 20, 200)

        positions = np.column_stack([
            200 * t,
            100 * np.sin(0.3 * t),
            5000 + 200 * np.sin(0.1 * t)
        ])

        states = {
            'position': positions,
            'velocity': np.column_stack([
                200 + 20 * np.sin(0.2 * t),
                30 * np.cos(0.3 * t),
                20 * np.cos(0.1 * t)
            ]),
            'euler_angles': np.column_stack([
                np.radians(10 * np.sin(0.3 * t)),
                np.radians(5 + 3 * np.cos(0.2 * t)),
                np.radians(90 * t / 20)
            ]),
            'angular_rates': np.column_stack([
                np.radians(3 * np.sin(0.4 * t)),
                np.radians(2 * np.cos(0.3 * t)),
                np.radians(1 * np.sin(0.5 * t))
            ])
        }

        controls = {
            'elevator': np.radians(3 * np.sin(0.2 * t)),
            'aileron': np.radians(2 * np.cos(0.3 * t)),
            'rudder': np.radians(1 * np.sin(0.1 * t)),
            'throttle': 0.7 + 0.1 * np.sin(0.15 * t)
        }

        forces = np.column_stack([
            500 + 50 * np.sin(0.2 * t),
            20 * np.cos(0.3 * t),
            -150 + 30 * np.sin(0.1 * t)
        ])

        moments = np.column_stack([
            100 * np.sin(0.3 * t),
            200 * np.cos(0.2 * t),
            50 * np.sin(0.4 * t)
        ])

        # Create all plots
        fig1 = plot_trajectory_3d(positions)
        fig2 = plot_states_vs_time(t, states)
        fig3 = plot_controls_vs_time(t, controls)
        fig4 = plot_forces_moments(t, forces, moments)

        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None
        assert fig4 is not None

        plt.close('all')

    def test_visualization_with_simulation_data(self):
        """Test visualization with simulation-like data."""
        from src.core.state import State
        from src.core.quaternion import Quaternion

        # Create states
        n_steps = 100
        states_list = []

        for i in range(n_steps):
            state = State()
            state.position_ned = np.array([i * 10, i * 5, -5000 - i * 2])
            state.velocity_body = np.array([200, 0, 0])
            state.orientation = Quaternion.from_euler_angles(
                np.radians(5 * np.sin(0.1 * i)),
                np.radians(3),
                np.radians(i * 0.5)
            )
            states_list.append(state)

        # Extract data
        positions = np.array([s.position_ned for s in states_list])

        # Plot
        fig = plot_trajectory_3d(positions)

        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
