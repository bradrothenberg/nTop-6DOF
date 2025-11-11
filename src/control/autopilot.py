"""
Autopilot Controllers

Provides PID-based flight control systems:
- Generic PID controller
- Altitude hold
- Heading hold
- Airspeed hold
"""

import numpy as np
from typing import Optional


class PIDController:
    """
    Generic PID (Proportional-Integral-Derivative) controller.

    Parameters
    ----------
    Kp : float
        Proportional gain
    Ki : float
        Integral gain
    Kd : float
        Derivative gain
    output_limits : tuple of float, optional
        (min, max) output saturation limits
    integral_limits : tuple of float, optional
        (min, max) integral windup limits

    Attributes
    ----------
    error_integral : float
        Accumulated integral error
    error_prev : float
        Previous error for derivative calculation
    """

    def __init__(self,
                 Kp: float,
                 Ki: float,
                 Kd: float,
                 output_limits: Optional[tuple] = None,
                 integral_limits: Optional[tuple] = None):
        """Initialize PID controller."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits

        self.error_integral = 0.0
        self.error_prev = 0.0
        self.first_call = True

    def update(self, error: float, dt: float) -> float:
        """
        Compute control output for given error.

        Parameters
        ----------
        error : float
            Control error (setpoint - measured)
        dt : float
            Time step (seconds)

        Returns
        -------
        float
            Control output
        """
        # Proportional term
        P = self.Kp * error

        # Integral term with anti-windup
        self.error_integral += error * dt
        if self.integral_limits is not None:
            self.error_integral = np.clip(self.error_integral,
                                           self.integral_limits[0],
                                           self.integral_limits[1])
        I = self.Ki * self.error_integral

        # Derivative term (skip on first call to avoid spike)
        if self.first_call:
            D = 0.0
            self.first_call = False
        else:
            error_rate = (error - self.error_prev) / dt
            D = self.Kd * error_rate

        self.error_prev = error

        # Total output
        output = P + I + D

        # Apply output limits
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output

    def reset(self):
        """Reset controller state."""
        self.error_integral = 0.0
        self.error_prev = 0.0
        self.first_call = True


class AltitudeHoldController:
    """
    Altitude hold autopilot.

    Controls elevator to maintain desired altitude using cascaded PID loops:
    - Outer loop: altitude error → pitch angle command
    - Inner loop: pitch error → elevator deflection

    Parameters
    ----------
    Kp_alt : float
        Proportional gain for altitude loop (rad/ft)
    Ki_alt : float
        Integral gain for altitude loop
    Kd_alt : float
        Derivative gain for altitude loop
    Kp_pitch : float
        Proportional gain for pitch loop (rad/rad)
    Ki_pitch : float
        Integral gain for pitch loop
    Kd_pitch : float
        Derivative gain for pitch loop
    pitch_limit : float, optional
        Maximum pitch angle command (radians), default: 30 deg
    elevator_limit : float, optional
        Maximum elevator deflection (radians), default: 25 deg
    """

    def __init__(self,
                 Kp_alt: float = 0.001,
                 Ki_alt: float = 0.0001,
                 Kd_alt: float = 0.005,
                 Kp_pitch: float = 2.0,
                 Ki_pitch: float = 0.5,
                 Kd_pitch: float = 0.1,
                 pitch_limit: float = np.radians(30),
                 elevator_limit: float = np.radians(25)):
        """Initialize altitude hold controller."""

        # Outer loop: altitude → pitch command
        self.altitude_pid = PIDController(
            Kp=Kp_alt,
            Ki=Ki_alt,
            Kd=Kd_alt,
            output_limits=(-pitch_limit, pitch_limit),
            integral_limits=(-pitch_limit / Ki_alt if Ki_alt > 0 else None,
                              pitch_limit / Ki_alt if Ki_alt > 0 else None)
        )

        # Inner loop: pitch → elevator
        self.pitch_pid = PIDController(
            Kp=Kp_pitch,
            Ki=Ki_pitch,
            Kd=Kd_pitch,
            output_limits=(-elevator_limit, elevator_limit),
            integral_limits=(-elevator_limit / Ki_pitch if Ki_pitch > 0 else None,
                              elevator_limit / Ki_pitch if Ki_pitch > 0 else None)
        )

        self.altitude_target = 0.0

    def set_target_altitude(self, altitude: float):
        """
        Set target altitude.

        Parameters
        ----------
        altitude : float
            Desired altitude (ft)
        """
        self.altitude_target = altitude

    def update(self, current_altitude: float, current_pitch: float, dt: float) -> float:
        """
        Compute elevator command to maintain altitude.

        Parameters
        ----------
        current_altitude : float
            Current altitude (ft)
        current_pitch : float
            Current pitch angle (radians)
        dt : float
            Time step (seconds)

        Returns
        -------
        float
            Elevator deflection command (radians), positive = trailing edge down
        """
        # Outer loop: altitude error → pitch command
        altitude_error = self.altitude_target - current_altitude
        pitch_command = self.altitude_pid.update(altitude_error, dt)

        # Inner loop: pitch error → elevator
        pitch_error = pitch_command - current_pitch
        elevator_command = self.pitch_pid.update(pitch_error, dt)

        return elevator_command

    def reset(self):
        """Reset controller state."""
        self.altitude_pid.reset()
        self.pitch_pid.reset()


class FlyingWingAutopilot:
    """
    Enhanced autopilot for flying wing aircraft with pitch rate damping and stall protection.

    Uses triple-loop architecture:
    - Outer loop: Altitude → Pitch angle command
    - Middle loop: Pitch angle → Pitch rate command
    - Inner loop: Pitch rate → Elevon deflection

    Includes stall protection that limits pitch commands and increases throttle
    when airspeed is critically low or angle of attack is too high.

    Parameters
    ----------
    Kp_alt, Ki_alt, Kd_alt : float
        Altitude hold PID gains
    Kp_pitch, Ki_pitch, Kd_pitch : float
        Pitch attitude PID gains
    Kp_pitch_rate, Ki_pitch_rate, Kd_pitch_rate : float
        Pitch rate PID gains
    max_pitch_cmd : float
        Maximum pitch up command (degrees)
    min_pitch_cmd : float
        Maximum pitch down command (degrees)
    max_alpha : float
        Maximum angle of attack for stall protection (degrees)
    stall_speed : float
        Stall speed (ft/s)
    min_airspeed_margin : float
        Safety margin above stall speed (multiplier)
    """

    def __init__(self,
                 # Altitude hold gains
                 Kp_alt: float = 0.02,
                 Ki_alt: float = 0.001,
                 Kd_alt: float = 0.05,
                 # Pitch attitude gains
                 Kp_pitch: float = 1.5,
                 Ki_pitch: float = 0.1,
                 Kd_pitch: float = 0.3,
                 # Pitch rate gains
                 Kp_pitch_rate: float = 0.5,
                 Ki_pitch_rate: float = 0.05,
                 Kd_pitch_rate: float = 0.0,
                 # Safety limits
                 max_pitch_cmd: float = 20.0,
                 min_pitch_cmd: float = -15.0,
                 max_alpha: float = 12.0,
                 stall_speed: float = 150.0,
                 min_airspeed_margin: float = 1.3):
        """Initialize flying wing autopilot."""

        # Outer loop: altitude → pitch command
        self.altitude_controller = PIDController(
            Kp=Kp_alt, Ki=Ki_alt, Kd=Kd_alt,
            output_limits=(np.radians(min_pitch_cmd), np.radians(max_pitch_cmd)),
            integral_limits=(-100.0, 100.0)  # ft
        )

        # Middle loop: pitch angle → pitch rate command
        self.pitch_controller = PIDController(
            Kp=Kp_pitch, Ki=Ki_pitch, Kd=Kd_pitch,
            output_limits=(np.radians(-30), np.radians(30)),  # deg/s
            integral_limits=(np.radians(-5), np.radians(5))  # rad
        )

        # Inner loop: pitch rate → elevon command
        self.pitch_rate_controller = PIDController(
            Kp=Kp_pitch_rate, Ki=Ki_pitch_rate, Kd=Kd_pitch_rate,
            output_limits=(np.radians(-25), np.radians(25)),
            integral_limits=(np.radians(-10), np.radians(10))  # rad
        )

        # Safety limits
        self.max_pitch_cmd = np.radians(max_pitch_cmd)
        self.min_pitch_cmd = np.radians(min_pitch_cmd)
        self.max_alpha = np.radians(max_alpha)
        self.min_airspeed = stall_speed * min_airspeed_margin
        self.stall_speed = stall_speed

        # Trim values
        self.elevon_trim = 0.0
        self.altitude_target = 0.0

        # Diagnostics
        self.stall_protection_active = False
        self.alpha_protection_active = False

    def set_trim(self, elevon_trim: float):
        """Set trim elevon deflection for feedforward."""
        self.elevon_trim = elevon_trim

    def set_target_altitude(self, altitude: float):
        """Set target altitude (NED frame, negative = up)."""
        self.altitude_target = altitude

    def update(self, current_altitude: float, current_pitch: float,
               current_pitch_rate: float, current_airspeed: float,
               current_alpha: float, dt: float) -> float:
        """
        Update autopilot and compute elevon command.

        Parameters
        ----------
        current_altitude : float
            Current altitude (NED, negative = up)
        current_pitch : float
            Current pitch angle (radians)
        current_pitch_rate : float
            Current pitch rate (rad/s)
        current_airspeed : float
            Current airspeed (ft/s)
        current_alpha : float
            Current angle of attack (radians)
        dt : float
            Time step (seconds)

        Returns
        -------
        float
            Elevon deflection command (radians)
        """
        # Reset protection flags
        self.stall_protection_active = False
        self.alpha_protection_active = False

        # === OUTER LOOP: Altitude → Pitch Command ===
        altitude_error = self.altitude_target - current_altitude
        pitch_cmd = self.altitude_controller.update(altitude_error, dt)

        # Limit pitch command based on envelope
        pitch_cmd = np.clip(pitch_cmd, self.min_pitch_cmd, self.max_pitch_cmd)

        # === STALL PROTECTION ===
        # If too slow, limit pitch up
        if current_airspeed < self.min_airspeed:
            # Force pitch down if critically slow
            pitch_cmd = min(pitch_cmd, current_pitch - np.radians(5))
            self.stall_protection_active = True

        # If alpha too high, limit pitch up
        if current_alpha > self.max_alpha:
            pitch_cmd = min(pitch_cmd, current_pitch - np.radians(2))
            self.alpha_protection_active = True

        # === MIDDLE LOOP: Pitch Attitude → Pitch Rate Command ===
        pitch_error = pitch_cmd - current_pitch
        pitch_rate_cmd = self.pitch_controller.update(pitch_error, dt)

        # === INNER LOOP: Pitch Rate → Elevon ===
        pitch_rate_error = pitch_rate_cmd - current_pitch_rate
        elevon = self.pitch_rate_controller.update(pitch_rate_error, dt)

        # Add trim for feedforward
        elevon += self.elevon_trim

        # Saturate elevon
        elevon = np.clip(elevon, np.radians(-25), np.radians(25))

        return elevon

    def reset(self):
        """Reset all controller states."""
        self.altitude_controller.reset()
        self.pitch_controller.reset()
        self.pitch_rate_controller.reset()
        self.stall_protection_active = False
        self.alpha_protection_active = False


class HeadingHoldController:
    """
    Heading hold autopilot.

    Controls aileron to maintain desired heading using cascaded PID loops:
    - Outer loop: heading error → roll angle command
    - Inner loop: roll error → aileron deflection

    Parameters
    ----------
    Kp_heading : float
        Proportional gain for heading loop (rad/rad)
    Ki_heading : float
        Integral gain for heading loop
    Kd_heading : float
        Derivative gain for heading loop
    Kp_roll : float
        Proportional gain for roll loop (rad/rad)
    Ki_roll : float
        Integral gain for roll loop
    Kd_roll : float
        Derivative gain for roll loop
    roll_limit : float, optional
        Maximum roll angle command (radians), default: 30 deg
    aileron_limit : float, optional
        Maximum aileron deflection (radians), default: 25 deg
    """

    def __init__(self,
                 Kp_heading: float = 0.5,
                 Ki_heading: float = 0.05,
                 Kd_heading: float = 0.1,
                 Kp_roll: float = 2.0,
                 Ki_roll: float = 0.3,
                 Kd_roll: float = 0.1,
                 roll_limit: float = np.radians(30),
                 aileron_limit: float = np.radians(25)):
        """Initialize heading hold controller."""

        # Outer loop: heading → roll command
        self.heading_pid = PIDController(
            Kp=Kp_heading,
            Ki=Ki_heading,
            Kd=Kd_heading,
            output_limits=(-roll_limit, roll_limit),
            integral_limits=(-roll_limit / Ki_heading if Ki_heading > 0 else None,
                              roll_limit / Ki_heading if Ki_heading > 0 else None)
        )

        # Inner loop: roll → aileron
        self.roll_pid = PIDController(
            Kp=Kp_roll,
            Ki=Ki_roll,
            Kd=Kd_roll,
            output_limits=(-aileron_limit, aileron_limit),
            integral_limits=(-aileron_limit / Ki_roll if Ki_roll > 0 else None,
                              aileron_limit / Ki_roll if Ki_roll > 0 else None)
        )

        self.heading_target = 0.0

    def set_target_heading(self, heading: float):
        """
        Set target heading.

        Parameters
        ----------
        heading : float
            Desired heading (radians), 0 = North, positive = clockwise
        """
        self.heading_target = heading

    def update(self, current_heading: float, current_roll: float, dt: float) -> float:
        """
        Compute aileron command to maintain heading.

        Parameters
        ----------
        current_heading : float
            Current heading (radians)
        current_roll : float
            Current roll angle (radians)
        dt : float
            Time step (seconds)

        Returns
        -------
        float
            Aileron deflection command (radians), positive = right aileron down
        """
        # Compute heading error with wrap-around
        heading_error = self.heading_target - current_heading

        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Outer loop: heading error → roll command
        roll_command = self.heading_pid.update(heading_error, dt)

        # Inner loop: roll error → aileron
        roll_error = roll_command - current_roll
        aileron_command = self.roll_pid.update(roll_error, dt)

        return aileron_command

    def reset(self):
        """Reset controller state."""
        self.heading_pid.reset()
        self.roll_pid.reset()


class AirspeedHoldController:
    """
    Airspeed hold autopilot.

    Controls throttle to maintain desired airspeed using PID control.

    Parameters
    ----------
    Kp : float
        Proportional gain (1/(ft/s))
    Ki : float
        Integral gain
    Kd : float
        Derivative gain
    throttle_min : float, optional
        Minimum throttle (0-1), default: 0.0
    throttle_max : float, optional
        Maximum throttle (0-1), default: 1.0
    """

    def __init__(self,
                 Kp: float = 0.01,
                 Ki: float = 0.001,
                 Kd: float = 0.05,
                 throttle_min: float = 0.0,
                 throttle_max: float = 1.0):
        """Initialize airspeed hold controller."""

        self.pid = PIDController(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            output_limits=(throttle_min, throttle_max),
            integral_limits=(-1.0 / Ki if Ki > 0 else None,
                              1.0 / Ki if Ki > 0 else None)
        )

        self.airspeed_target = 0.0

    def set_target_airspeed(self, airspeed: float):
        """
        Set target airspeed.

        Parameters
        ----------
        airspeed : float
            Desired airspeed (ft/s)
        """
        self.airspeed_target = airspeed

    def update(self, current_airspeed: float, dt: float) -> float:
        """
        Compute throttle command to maintain airspeed.

        Parameters
        ----------
        current_airspeed : float
            Current airspeed (ft/s)
        dt : float
            Time step (seconds)

        Returns
        -------
        float
            Throttle command (0-1)
        """
        airspeed_error = self.airspeed_target - current_airspeed
        throttle_command = self.pid.update(airspeed_error, dt)

        return throttle_command

    def reset(self):
        """Reset controller state."""
        self.pid.reset()


def test_pid():
    """Test PID controller with simple step response."""
    print("=" * 60)
    print("PID Controller Test")
    print("=" * 60)
    print()

    # Create PID controller
    pid = PIDController(Kp=2.0, Ki=0.5, Kd=0.1, output_limits=(-10, 10))

    # Simulate step response
    setpoint = 10.0
    measured = 0.0
    dt = 0.01

    print("Step response simulation:")
    print(f"{'Time (s)':<10} {'Measured':<12} {'Error':<12} {'Output':<12}")
    print("-" * 50)

    for i in range(200):
        t = i * dt
        error = setpoint - measured
        output = pid.update(error, dt)

        # Simple first-order plant: dy/dt = output
        measured += output * dt

        if i % 20 == 0:
            print(f"{t:<10.2f} {measured:<12.4f} {error:<12.4f} {output:<12.4f}")

    print()
    print(f"Final value: {measured:.4f} (target: {setpoint:.4f})")
    print()


if __name__ == "__main__":
    test_pid()
