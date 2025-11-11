"""
Gain-Scheduled Autopilot for Flying Wing

Implements gain scheduling to handle large altitude changes during mission profiles.
Different PID gains are used for different flight phases (cruise, climb, descent).
"""

import numpy as np
from typing import Dict, Optional


class GainSet:
    """Container for a complete set of autopilot gains."""

    def __init__(self, name: str,
                 Kp_alt: float, Ki_alt: float, Kd_alt: float,
                 Kp_pitch: float, Ki_pitch: float, Kd_pitch: float,
                 Kp_pitch_rate: float, Ki_pitch_rate: float,
                 max_pitch_cmd: float, min_pitch_cmd: float):
        self.name = name
        self.Kp_alt = Kp_alt
        self.Ki_alt = Ki_alt
        self.Kd_alt = Kd_alt
        self.Kp_pitch = Kp_pitch
        self.Ki_pitch = Ki_pitch
        self.Kd_pitch = Kd_pitch
        self.Kp_pitch_rate = Kp_pitch_rate
        self.Ki_pitch_rate = Ki_pitch_rate
        self.max_pitch_cmd = max_pitch_cmd
        self.min_pitch_cmd = min_pitch_cmd


class GainScheduledAutopilot:
    """
    Flying wing autopilot with gain scheduling for different flight phases.

    Uses different PID gains for cruise, climb, and descent to handle
    large altitude changes while maintaining stability.
    """

    def __init__(self):
        # Define gain sets for different flight modes
        self.gain_sets = {
            'cruise': GainSet(
                name='cruise',
                # Tight altitude hold - proven stable gains
                Kp_alt=0.005,
                Ki_alt=0.0005,
                Kd_alt=0.012,
                Kp_pitch=0.8,
                Ki_pitch=0.05,
                Kd_pitch=0.15,
                Kp_pitch_rate=0.15,
                Ki_pitch_rate=0.01,
                max_pitch_cmd=12.0,
                min_pitch_cmd=-8.0
            ),
            'climb': GainSet(
                name='climb',
                # More aggressive altitude response for climb
                Kp_alt=0.008,      # 60% increase
                Ki_alt=0.001,      # 100% increase
                Kd_alt=0.020,      # 67% increase
                Kp_pitch=1.0,      # 25% increase
                Ki_pitch=0.08,     # 60% increase
                Kd_pitch=0.20,     # 33% increase
                Kp_pitch_rate=0.18,  # 20% increase
                Ki_pitch_rate=0.015, # 50% increase
                max_pitch_cmd=15.0,  # Allow more pitch up
                min_pitch_cmd=-8.0
            ),
            'descent': GainSet(
                name='descent',
                # Moderate gains for controlled descent
                Kp_alt=0.006,      # 20% increase from cruise
                Ki_alt=0.0008,     # 60% increase
                Kd_alt=0.015,      # 25% increase
                Kp_pitch=0.9,      # 12.5% increase
                Ki_pitch=0.06,     # 20% increase
                Kd_pitch=0.18,     # 20% increase
                Kp_pitch_rate=0.16,  # 7% increase
                Ki_pitch_rate=0.012, # 20% increase
                max_pitch_cmd=10.0,
                min_pitch_cmd=-12.0  # Allow more pitch down
            )
        }

        # Current active gains (start with cruise)
        self.current_gains = self.gain_sets['cruise']
        self.target_gains = self.gain_sets['cruise']

        # Transition state
        self.transition_time = 5.0  # seconds to blend between gain sets
        self.transition_timer = 0.0
        self.transitioning = False

        # Control state
        self.alt_integrator = 0.0
        self.pitch_integrator = 0.0
        self.pitch_rate_integrator = 0.0
        self.prev_alt_error = 0.0
        self.prev_pitch_error = 0.0

        # Targets
        self.target_altitude = 0.0
        self.elevon_trim = 0.0

        # Safety limits
        self.max_alpha = np.radians(12.0)
        self.stall_speed = 150.0
        self.min_airspeed_margin = 1.3
        self.min_airspeed = self.stall_speed * self.min_airspeed_margin

    def set_trim(self, elevon_trim: float):
        """Set trim elevon deflection."""
        self.elevon_trim = elevon_trim

    def set_target_altitude(self, altitude: float):
        """Set target altitude (NED frame, negative up)."""
        self.target_altitude = altitude

    def set_mode(self, mode: str, dt: float):
        """
        Change flight mode and initiate gain transition.

        Args:
            mode: One of 'cruise', 'climb', 'descent'
            dt: Time step for transition
        """
        if mode not in self.gain_sets:
            raise ValueError(f"Unknown mode: {mode}. Must be 'cruise', 'climb', or 'descent'")

        if self.target_gains != self.gain_sets[mode]:
            self.target_gains = self.gain_sets[mode]
            self.transition_timer = 0.0
            self.transitioning = True

    def _blend_gains(self, alpha: float) -> GainSet:
        """
        Blend between current and target gains.

        Args:
            alpha: Blend factor (0 = current, 1 = target)
        """
        def lerp(a, b, t):
            return a + (b - a) * t

        return GainSet(
            name=f"blend_{alpha:.2f}",
            Kp_alt=lerp(self.current_gains.Kp_alt, self.target_gains.Kp_alt, alpha),
            Ki_alt=lerp(self.current_gains.Ki_alt, self.target_gains.Ki_alt, alpha),
            Kd_alt=lerp(self.current_gains.Kd_alt, self.target_gains.Kd_alt, alpha),
            Kp_pitch=lerp(self.current_gains.Kp_pitch, self.target_gains.Kp_pitch, alpha),
            Ki_pitch=lerp(self.current_gains.Ki_pitch, self.target_gains.Ki_pitch, alpha),
            Kd_pitch=lerp(self.current_gains.Kd_pitch, self.target_gains.Kd_pitch, alpha),
            Kp_pitch_rate=lerp(self.current_gains.Kp_pitch_rate, self.target_gains.Kp_pitch_rate, alpha),
            Ki_pitch_rate=lerp(self.current_gains.Ki_pitch_rate, self.target_gains.Ki_pitch_rate, alpha),
            max_pitch_cmd=lerp(self.current_gains.max_pitch_cmd, self.target_gains.max_pitch_cmd, alpha),
            min_pitch_cmd=lerp(self.current_gains.min_pitch_cmd, self.target_gains.min_pitch_cmd, alpha)
        )

    def update(self, current_altitude: float, current_pitch: float,
               current_pitch_rate: float, current_airspeed: float,
               current_alpha: float, dt: float) -> float:
        """
        Update autopilot and return elevon command.

        Args:
            current_altitude: Current altitude (positive up, ft)
            current_pitch: Current pitch angle (rad)
            current_pitch_rate: Current pitch rate (rad/s)
            current_airspeed: Current airspeed (ft/s)
            current_alpha: Current angle of attack (rad)
            dt: Time step (s)

        Returns:
            Elevon deflection command (rad)
        """
        # Update gain transition
        if self.transitioning:
            self.transition_timer += dt
            alpha = min(1.0, self.transition_timer / self.transition_time)
            gains = self._blend_gains(alpha)

            if alpha >= 1.0:
                self.current_gains = self.target_gains
                self.transitioning = False
        else:
            gains = self.current_gains

        # Stall protection
        stall_protection_active = False
        if current_airspeed < self.min_airspeed or current_alpha > self.max_alpha:
            stall_protection_active = True

        # Outer loop: Altitude hold
        alt_error = self.target_altitude - (-current_altitude)  # NED frame
        self.alt_integrator += alt_error * dt
        self.alt_integrator = np.clip(self.alt_integrator, -500.0, 500.0)  # Anti-windup
        alt_derivative = (alt_error - self.prev_alt_error) / dt if dt > 0 else 0.0
        self.prev_alt_error = alt_error

        pitch_cmd = (gains.Kp_alt * alt_error +
                    gains.Ki_alt * self.alt_integrator +
                    gains.Kd_alt * alt_derivative)

        # Apply stall protection
        if stall_protection_active:
            pitch_cmd = min(pitch_cmd, 0.0)  # Don't pitch up in stall

        pitch_cmd = np.clip(pitch_cmd, np.radians(gains.min_pitch_cmd),
                           np.radians(gains.max_pitch_cmd))

        # Middle loop: Pitch attitude
        pitch_error = pitch_cmd - current_pitch
        self.pitch_integrator += pitch_error * dt
        self.pitch_integrator = np.clip(self.pitch_integrator, -0.5, 0.5)  # Anti-windup
        pitch_derivative = (pitch_error - self.prev_pitch_error) / dt if dt > 0 else 0.0
        self.prev_pitch_error = pitch_error

        pitch_rate_cmd = (gains.Kp_pitch * pitch_error +
                         gains.Ki_pitch * self.pitch_integrator +
                         gains.Kd_pitch * pitch_derivative)

        pitch_rate_cmd = np.clip(pitch_rate_cmd, -0.5, 0.5)  # Limit commanded rate

        # Inner loop: Pitch rate damping
        pitch_rate_error = pitch_rate_cmd - current_pitch_rate
        self.pitch_rate_integrator += pitch_rate_error * dt
        self.pitch_rate_integrator = np.clip(self.pitch_rate_integrator, -0.2, 0.2)

        elevon_cmd = (gains.Kp_pitch_rate * pitch_rate_error +
                     gains.Ki_pitch_rate * self.pitch_rate_integrator +
                     self.elevon_trim)

        # Limit elevon deflection
        elevon_cmd = np.clip(elevon_cmd, np.radians(-25.0), np.radians(25.0))

        return elevon_cmd

    def get_current_mode(self) -> str:
        """Get the current or target gain mode."""
        return self.target_gains.name if self.transitioning else self.current_gains.name

    def reset_integrators(self):
        """Reset all integrators (useful for large mode changes)."""
        self.alt_integrator = 0.0
        self.pitch_integrator = 0.0
        self.pitch_rate_integrator = 0.0
        self.prev_alt_error = 0.0
        self.prev_pitch_error = 0.0
