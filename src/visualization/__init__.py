"""
Visualization Module

Provides plotting and animation capabilities for flight dynamics data.
"""

from .plotting import (
    plot_trajectory_3d,
    plot_states_vs_time,
    plot_controls_vs_time,
    plot_forces_moments,
    plot_trim_envelope,
    setup_plotting_style
)

from .animation import (
    TrajectoryAnimation,
    animate_trajectory,
    create_comparison_animation
)

__all__ = [
    'plot_trajectory_3d',
    'plot_states_vs_time',
    'plot_controls_vs_time',
    'plot_forces_moments',
    'plot_trim_envelope',
    'setup_plotting_style',
    'TrajectoryAnimation',
    'animate_trajectory',
    'create_comparison_animation'
]
