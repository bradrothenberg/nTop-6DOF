"""
Total Energy Control System (TECS) for Flying Wing

Implements industry-standard total energy control for decoupled
altitude and airspeed management during large flight envelope changes.

Theory:
    Total Energy (E) = Kinetic Energy + Potential Energy
                     = (1/2) * m * V² + m * g * h

    Energy Rate (dE/dt) = Thrust - Drag (power balance)

    Energy Distribution = Balance between KE and PE

Control Strategy:
    - Throttle controls TOTAL ENERGY RATE (add/remove energy from system)
    - Pitch controls ENERGY DISTRIBUTION (trade altitude for speed)

This decouples altitude/speed control and prevents phugoid oscillations.
"""

import numpy as np
from typing import Tuple


class TotalEnergyAutopilot:
    """
    Total Energy Control System for flying wing.

    Uses energy-based control laws to handle large altitude and airspeed changes
    without the coupling issues of traditional altitude/speed controllers.
    """

    def __init__(self,
                 # Total energy rate gains (throttle)
                 Kp_energy_rate: float = 0.003,
                 Ki_energy_rate: float = 0.0005,
                 Kd_energy_rate: float = 0.002,
                 # Energy distribution gains (pitch)
                 Kp_distribution: float = 0.8,
                 Ki_distribution: float = 0.05,
                 Kd_distribution: float = 0.15,
                 # Pitch rate damping (inner loop)
                 Kp_pitch_rate: float = 0.15,
                 Ki_pitch_rate: float = 0.01,
                 # Limits
                 max_pitch_cmd: float = 15.0,
                 min_pitch_cmd: float = -10.0,
                 max_throttle: float = 1.0,
                 min_throttle: float = 0.1):

        # Gains
        self.Kp_energy_rate = Kp_energy_rate
        self.Ki_energy_rate = Ki_energy_rate
        self.Kd_energy_rate = Kd_energy_rate
        self.Kp_distribution = Kp_distribution
        self.Ki_distribution = Ki_distribution
        self.Kd_distribution = Kd_distribution
        self.Kp_pitch_rate = Kp_pitch_rate
        self.Ki_pitch_rate = Ki_pitch_rate

        # Limits
        self.max_pitch_cmd = np.radians(max_pitch_cmd)
        self.min_pitch_cmd = np.radians(min_pitch_cmd)
        self.max_throttle = max_throttle
        self.min_throttle = min_throttle

        # Control state
        self.energy_rate_integrator = 0.0
        self.distribution_integrator = 0.0
        self.pitch_rate_integrator = 0.0
        self.prev_energy_rate_error = 0.0
        self.prev_distribution_error = 0.0

        # Targets
        self.target_altitude = 0.0  # ft (positive up)
        self.target_airspeed = 600.0  # ft/s
        self.elevon_trim = 0.0
        self.throttle_trim = 0.8

        # Constants
        self.g = 32.174  # ft/s² (gravity)

        # Safety limits
        self.max_alpha = np.radians(12.0)
        self.stall_speed = 150.0
        self.min_airspeed = 195.0

    def set_trim(self, elevon_trim: float, throttle_trim: float):
        """Set trim values."""
        self.elevon_trim = elevon_trim
        self.throttle_trim = throttle_trim

    def set_targets(self, altitude: float, airspeed: float):
        """
        Set target altitude and airspeed.

        Args:
            altitude: Target altitude (ft, positive up)
            airspeed: Target airspeed (ft/s)
        """
        self.target_altitude = altitude
        self.target_airspeed = airspeed

    def _compute_specific_energy(self, altitude: float, airspeed: float) -> float:
        """
        Compute specific energy (energy per unit mass).

        E_specific = (V²/2) + g*h

        Args:
            altitude: Current altitude (ft, positive up)
            airspeed: Current airspeed (ft/s)

        Returns:
            Specific energy (ft²/s²)
        """
        kinetic = 0.5 * airspeed**2
        potential = self.g * altitude
        return kinetic + potential

    def _compute_energy_distribution(self, altitude: float, airspeed: float,
                                     target_altitude: float, target_airspeed: float) -> float:
        """
        Compute energy distribution error.

        This represents the balance between kinetic and potential energy.
        Positive error means we want more altitude (less speed).
        Negative error means we want more speed (less altitude).

        Args:
            altitude: Current altitude (ft)
            airspeed: Current airspeed (ft/s)
            target_altitude: Target altitude (ft)
            target_airspeed: Target airspeed (ft/s)

        Returns:
            Energy distribution error (dimensionless, normalized)
        """
        # Altitude error contribution (normalized by target airspeed)
        alt_error = target_altitude - altitude
        alt_contribution = self.g * alt_error / (self.target_airspeed**2)

        # Airspeed error contribution (normalized)
        speed_error = target_airspeed - airspeed
        speed_contribution = speed_error / self.target_airspeed

        # Energy distribution error: positive = need more altitude, negative = need more speed
        distribution_error = alt_contribution - speed_contribution

        return distribution_error

    def update(self, current_altitude: float, current_airspeed: float,
               current_pitch: float, current_pitch_rate: float,
               current_alpha: float, dt: float) -> Tuple[float, float]:
        """
        Update total energy controller.

        Args:
            current_altitude: Current altitude (ft, positive up)
            current_airspeed: Current airspeed (ft/s)
            current_pitch: Current pitch angle (rad)
            current_pitch_rate: Current pitch rate (rad/s)
            current_alpha: Current angle of attack (rad)
            dt: Time step (s)

        Returns:
            (elevon_cmd, throttle_cmd): Control commands
        """
        # ===== TOTAL ENERGY RATE CONTROL (Throttle) =====
        # Compute specific energies
        current_energy = self._compute_specific_energy(current_altitude, current_airspeed)
        target_energy = self._compute_specific_energy(self.target_altitude, self.target_airspeed)

        # Energy error and rate
        energy_error = target_energy - current_energy

        # PID for energy rate (throttle controls how fast we add/remove energy)
        self.energy_rate_integrator += energy_error * dt
        self.energy_rate_integrator = np.clip(self.energy_rate_integrator, -5000.0, 5000.0)
        energy_derivative = (energy_error - self.prev_energy_rate_error) / dt if dt > 0 else 0.0
        self.prev_energy_rate_error = energy_error

        throttle_cmd = (self.throttle_trim +
                       self.Kp_energy_rate * energy_error +
                       self.Ki_energy_rate * self.energy_rate_integrator +
                       self.Kd_energy_rate * energy_derivative)

        throttle_cmd = np.clip(throttle_cmd, self.min_throttle, self.max_throttle)

        # ===== ENERGY DISTRIBUTION CONTROL (Pitch) =====
        # Compute how we want to distribute energy between altitude and speed
        distribution_error = self._compute_energy_distribution(
            current_altitude, current_airspeed,
            self.target_altitude, self.target_airspeed
        )

        # PID for energy distribution (pitch controls altitude vs speed trade-off)
        self.distribution_integrator += distribution_error * dt
        self.distribution_integrator = np.clip(self.distribution_integrator, -1.0, 1.0)
        distribution_derivative = (distribution_error - self.prev_distribution_error) / dt if dt > 0 else 0.0
        self.prev_distribution_error = distribution_error

        pitch_cmd = (self.Kp_distribution * distribution_error +
                    self.Ki_distribution * self.distribution_integrator +
                    self.Kd_distribution * distribution_derivative)

        # Apply stall protection
        if current_airspeed < self.min_airspeed or current_alpha > self.max_alpha:
            pitch_cmd = min(pitch_cmd, 0.0)  # Don't pitch up in stall

        pitch_cmd = np.clip(pitch_cmd, self.min_pitch_cmd, self.max_pitch_cmd)

        # ===== PITCH RATE DAMPING (Inner Loop) =====
        # Simple pitch rate feedback for damping
        pitch_rate_error = pitch_cmd - current_pitch
        self.pitch_rate_integrator += pitch_rate_error * dt
        self.pitch_rate_integrator = np.clip(self.pitch_rate_integrator, -0.2, 0.2)

        # Pitch rate command
        pitch_rate_cmd = self.Kp_pitch_rate * (pitch_cmd - current_pitch)

        # Pitch rate damping
        pitch_rate_error_inner = pitch_rate_cmd - current_pitch_rate

        elevon_cmd = (self.Kp_pitch_rate * pitch_rate_error_inner +
                     self.Ki_pitch_rate * self.pitch_rate_integrator +
                     self.elevon_trim)

        # Limit elevon deflection
        elevon_cmd = np.clip(elevon_cmd, np.radians(-25.0), np.radians(25.0))

        return elevon_cmd, throttle_cmd

    def reset_integrators(self):
        """Reset all integrators."""
        self.energy_rate_integrator = 0.0
        self.distribution_integrator = 0.0
        self.pitch_rate_integrator = 0.0
        self.prev_energy_rate_error = 0.0
        self.prev_distribution_error = 0.0

    def get_debug_info(self, current_altitude: float, current_airspeed: float) -> dict:
        """Get debug information for analysis."""
        current_energy = self._compute_specific_energy(current_altitude, current_airspeed)
        target_energy = self._compute_specific_energy(self.target_altitude, self.target_airspeed)
        distribution_error = self._compute_energy_distribution(
            current_altitude, current_airspeed,
            self.target_altitude, self.target_airspeed
        )

        return {
            'current_energy': current_energy,
            'target_energy': target_energy,
            'energy_error': target_energy - current_energy,
            'distribution_error': distribution_error,
            'energy_rate_integrator': self.energy_rate_integrator,
            'distribution_integrator': self.distribution_integrator
        }
