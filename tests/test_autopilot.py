"""
Unit tests for autopilot controllers.

Tests PIDController and FlyingWingAutopilot classes.
"""

import pytest
import numpy as np
from src.control.autopilot import (
    PIDController,
    FlyingWingAutopilot,
    AltitudeHoldController,
    HeadingHoldController,
    AirspeedHoldController
)


class TestPIDController:
    """Test generic PID controller."""

    def test_initialization(self):
        """Test PID controller initialization."""
        pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.01)

        assert pid.Kp == 1.0
        assert pid.Ki == 0.1
        assert pid.Kd == 0.01
        assert pid.error_integral == 0.0
        assert pid.error_prev == 0.0
        assert pid.first_call is True

    def test_proportional_only(self):
        """Test pure proportional control (P controller)."""
        pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0)

        error = 10.0
        dt = 0.01
        output = pid.update(error, dt)

        # Output should be exactly Kp * error
        assert output == pytest.approx(2.0 * 10.0)

    def test_integral_accumulation(self):
        """Test integral term accumulates error over time."""
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0)

        error = 5.0
        dt = 0.1

        # First update
        output1 = pid.update(error, dt)
        assert output1 == pytest.approx(5.0 * 0.1)  # Ki * error * dt

        # Second update (integral should accumulate)
        output2 = pid.update(error, dt)
        assert output2 == pytest.approx(5.0 * 0.2)  # Ki * 2 * error * dt

    def test_derivative_term(self):
        """Test derivative term responds to error rate."""
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0)

        dt = 0.1

        # First call (derivative should be zero)
        output1 = pid.update(error=10.0, dt=dt)
        assert output1 == 0.0

        # Second call (error decreasing)
        output2 = pid.update(error=5.0, dt=dt)
        error_rate = (5.0 - 10.0) / dt
        assert output2 == pytest.approx(1.0 * error_rate)

    def test_output_limits(self):
        """Test output saturation."""
        pid = PIDController(Kp=10.0, Ki=0.0, Kd=0.0, output_limits=(-5.0, 5.0))

        # Large error should saturate output
        output = pid.update(error=100.0, dt=0.01)
        assert output == 5.0  # Should clip at upper limit

        # Negative error should saturate at lower limit
        output = pid.update(error=-100.0, dt=0.01)
        assert output == -5.0

    def test_integral_windup_limits(self):
        """Test integral anti-windup."""
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0, integral_limits=(-10.0, 10.0))

        error = 100.0
        dt = 0.1

        # Accumulate large integral
        for _ in range(100):
            output = pid.update(error, dt)

        # Integral should be clamped
        assert pid.error_integral == 10.0  # Should clip at upper limit
        assert output == 10.0

    def test_reset(self):
        """Test controller reset."""
        pid = PIDController(Kp=1.0, Ki=1.0, Kd=1.0)

        # Update to accumulate state
        pid.update(error=10.0, dt=0.1)
        assert pid.error_integral > 0
        assert pid.error_prev != 0
        assert pid.first_call is False

        # Reset
        pid.reset()
        assert pid.error_integral == 0.0
        assert pid.error_prev == 0.0
        assert pid.first_call is True


class TestFlyingWingAutopilot:
    """Test flying wing autopilot with triple-loop architecture."""

    def test_initialization_default(self):
        """Test autopilot initialization with default parameters."""
        autopilot = FlyingWingAutopilot()

        # Check controllers exist
        assert autopilot.altitude_controller is not None
        assert autopilot.pitch_controller is not None
        assert autopilot.pitch_rate_controller is not None

        # Check default limits
        assert autopilot.max_pitch_cmd == pytest.approx(np.radians(20.0))
        assert autopilot.min_pitch_cmd == pytest.approx(np.radians(-15.0))
        assert autopilot.max_alpha == pytest.approx(np.radians(12.0))
        assert autopilot.min_airspeed == pytest.approx(150.0 * 1.3)

        # Check initial state
        assert autopilot.elevon_trim == 0.0
        assert autopilot.altitude_target == 0.0
        assert autopilot.stall_protection_active is False
        assert autopilot.alpha_protection_active is False

    def test_initialization_custom(self):
        """Test autopilot initialization with custom parameters."""
        autopilot = FlyingWingAutopilot(
            Kp_alt=0.005,
            Ki_alt=0.0005,
            Kd_alt=0.01,
            max_pitch_cmd=15.0,
            min_pitch_cmd=-10.0,
            max_alpha=10.0,
            stall_speed=140.0,
            min_airspeed_margin=1.4
        )

        assert autopilot.max_pitch_cmd == pytest.approx(np.radians(15.0))
        assert autopilot.min_pitch_cmd == pytest.approx(np.radians(-10.0))
        assert autopilot.max_alpha == pytest.approx(np.radians(10.0))
        assert autopilot.stall_speed == 140.0
        assert autopilot.min_airspeed == pytest.approx(140.0 * 1.4)

    def test_set_trim(self):
        """Test setting trim elevon deflection."""
        autopilot = FlyingWingAutopilot()

        trim_value = np.radians(-6.81)
        autopilot.set_trim(trim_value)

        assert autopilot.elevon_trim == pytest.approx(trim_value)

    def test_set_target_altitude(self):
        """Test setting target altitude."""
        autopilot = FlyingWingAutopilot()

        target_alt = 5000.0
        autopilot.set_target_altitude(target_alt)

        assert autopilot.altitude_target == target_alt

    def test_update_nominal_conditions(self):
        """Test autopilot update under nominal conditions (no stall protection)."""
        autopilot = FlyingWingAutopilot(
            Kp_alt=0.003,
            Kp_pitch=0.8,
            Kp_pitch_rate=0.15
        )

        autopilot.set_target_altitude(5000.0)
        autopilot.set_trim(np.radians(-6.81))

        # Nominal flight conditions
        current_altitude = 4950.0  # 50 ft below target
        current_pitch = np.radians(2.0)
        current_pitch_rate = 0.0
        current_airspeed = 600.0  # Well above stall
        current_alpha = np.radians(2.0)  # Safe AOA
        dt = 0.01

        elevon = autopilot.update(
            current_altitude,
            current_pitch,
            current_pitch_rate,
            current_airspeed,
            current_alpha,
            dt
        )

        # Check output is reasonable
        assert np.isfinite(elevon)
        assert -np.radians(25) <= elevon <= np.radians(25)

        # Check no stall protection triggered
        assert autopilot.stall_protection_active is False
        assert autopilot.alpha_protection_active is False

    def test_elevon_saturation(self):
        """Test elevon output stays within ±25° limits."""
        autopilot = FlyingWingAutopilot(
            Kp_alt=10.0,  # Very high gain to force saturation
            Kp_pitch=10.0,
            Kp_pitch_rate=10.0
        )

        autopilot.set_target_altitude(5000.0)

        # Large altitude error
        current_altitude = 3000.0  # 2000 ft below target
        current_pitch = 0.0
        current_pitch_rate = 0.0
        current_airspeed = 600.0
        current_alpha = 0.0
        dt = 0.01

        elevon = autopilot.update(
            current_altitude,
            current_pitch,
            current_pitch_rate,
            current_airspeed,
            current_alpha,
            dt
        )

        # Should saturate at ±25°
        assert -np.radians(25) <= elevon <= np.radians(25)

    def test_stall_protection_low_airspeed(self):
        """Test stall protection activates at low airspeed."""
        autopilot = FlyingWingAutopilot(
            stall_speed=150.0,
            min_airspeed_margin=1.3
        )

        autopilot.set_target_altitude(5000.0)

        # Low airspeed (below 1.3 × stall speed)
        current_altitude = 4950.0
        current_pitch = np.radians(5.0)
        current_pitch_rate = 0.0
        current_airspeed = 180.0  # Below 195 ft/s (150 × 1.3)
        current_alpha = np.radians(5.0)
        dt = 0.01

        elevon = autopilot.update(
            current_altitude,
            current_pitch,
            current_pitch_rate,
            current_airspeed,
            current_alpha,
            dt
        )

        # Stall protection should be active
        assert autopilot.stall_protection_active is True
        assert np.isfinite(elevon)

    def test_alpha_protection_high_aoa(self):
        """Test alpha protection activates at high angle of attack."""
        autopilot = FlyingWingAutopilot(max_alpha=12.0)

        autopilot.set_target_altitude(5000.0)

        # High angle of attack
        current_altitude = 4950.0
        current_pitch = np.radians(15.0)
        current_pitch_rate = 0.0
        current_airspeed = 600.0
        current_alpha = np.radians(13.0)  # Above 12° limit
        dt = 0.01

        elevon = autopilot.update(
            current_altitude,
            current_pitch,
            current_pitch_rate,
            current_airspeed,
            current_alpha,
            dt
        )

        # Alpha protection should be active
        assert autopilot.alpha_protection_active is True
        assert np.isfinite(elevon)

    def test_both_protections_active(self):
        """Test both stall and alpha protection can activate simultaneously."""
        autopilot = FlyingWingAutopilot(
            stall_speed=150.0,
            min_airspeed_margin=1.3,
            max_alpha=12.0
        )

        autopilot.set_target_altitude(5000.0)

        # Critical conditions: low speed AND high alpha
        current_altitude = 4950.0
        current_pitch = np.radians(15.0)
        current_pitch_rate = 0.0
        current_airspeed = 180.0  # Low
        current_alpha = np.radians(13.0)  # High
        dt = 0.01

        elevon = autopilot.update(
            current_altitude,
            current_pitch,
            current_pitch_rate,
            current_airspeed,
            current_alpha,
            dt
        )

        # Both protections should be active
        assert autopilot.stall_protection_active is True
        assert autopilot.alpha_protection_active is True
        assert np.isfinite(elevon)

    def test_trim_feedforward(self):
        """Test trim value is added to control output."""
        autopilot = FlyingWingAutopilot(
            Kp_alt=0.0,  # Zero gains for this test
            Ki_alt=0.0,
            Kd_alt=0.0,
            Kp_pitch=0.0,
            Ki_pitch=0.0,
            Kd_pitch=0.0,
            Kp_pitch_rate=0.0,
            Ki_pitch_rate=0.0,
            Kd_pitch_rate=0.0
        )

        trim_value = np.radians(-6.81)
        autopilot.set_trim(trim_value)
        autopilot.set_target_altitude(5000.0)

        # Zero errors (at trim)
        elevon = autopilot.update(
            current_altitude=5000.0,
            current_pitch=0.0,
            current_pitch_rate=0.0,
            current_airspeed=600.0,
            current_alpha=0.0,
            dt=0.01
        )

        # With zero gains and zero errors, output should equal trim
        assert elevon == pytest.approx(trim_value, abs=1e-6)

    def test_reset(self):
        """Test controller reset."""
        autopilot = FlyingWingAutopilot()
        autopilot.set_target_altitude(5000.0)

        # Run several updates to accumulate state
        for _ in range(10):
            autopilot.update(
                current_altitude=4900.0,
                current_pitch=np.radians(2.0),
                current_pitch_rate=0.0,
                current_airspeed=600.0,
                current_alpha=np.radians(2.0),
                dt=0.01
            )

        # Should have non-zero integral terms
        assert autopilot.altitude_controller.error_integral != 0.0

        # Reset
        autopilot.reset()

        # All states should be cleared
        assert autopilot.altitude_controller.error_integral == 0.0
        assert autopilot.pitch_controller.error_integral == 0.0
        assert autopilot.pitch_rate_controller.error_integral == 0.0
        assert autopilot.stall_protection_active is False
        assert autopilot.alpha_protection_active is False

    def test_multi_step_consistency(self):
        """Test autopilot produces consistent outputs over multiple steps."""
        autopilot = FlyingWingAutopilot(
            Kp_alt=0.003,
            Kp_pitch=0.8,
            Kp_pitch_rate=0.15
        )

        autopilot.set_target_altitude(5000.0)
        autopilot.set_trim(np.radians(-6.81))

        # Run 100 timesteps
        dt = 0.01
        elevon_history = []

        for i in range(100):
            elevon = autopilot.update(
                current_altitude=4950.0 + i * 0.5,  # Slowly approaching target
                current_pitch=np.radians(2.0),
                current_pitch_rate=0.0,
                current_airspeed=600.0,
                current_alpha=np.radians(2.0),
                dt=dt
            )
            elevon_history.append(elevon)

            # Each output should be valid
            assert np.isfinite(elevon)
            assert -np.radians(25) <= elevon <= np.radians(25)

        # Outputs should vary as altitude approaches target
        assert len(set(elevon_history)) > 1  # Not all the same


class TestAltitudeHoldController:
    """Test altitude hold controller."""

    def test_initialization(self):
        """Test altitude hold controller initialization."""
        controller = AltitudeHoldController()

        assert controller.altitude_pid is not None
        assert controller.pitch_pid is not None
        assert controller.altitude_target == 0.0

    def test_set_target_altitude(self):
        """Test setting target altitude."""
        controller = AltitudeHoldController()
        controller.set_target_altitude(5000.0)

        assert controller.altitude_target == 5000.0

    def test_update(self):
        """Test altitude hold update."""
        controller = AltitudeHoldController()
        controller.set_target_altitude(5000.0)

        elevator = controller.update(
            current_altitude=4950.0,
            current_pitch=np.radians(2.0),
            dt=0.01
        )

        # Should produce valid elevator command
        assert np.isfinite(elevator)
        assert -np.radians(25) <= elevator <= np.radians(25)

    def test_reset(self):
        """Test controller reset."""
        controller = AltitudeHoldController()
        controller.set_target_altitude(5000.0)

        # Run updates
        for _ in range(10):
            controller.update(4900.0, 0.0, 0.01)

        controller.reset()

        assert controller.altitude_pid.error_integral == 0.0
        assert controller.pitch_pid.error_integral == 0.0


class TestHeadingHoldController:
    """Test heading hold controller."""

    def test_initialization(self):
        """Test heading hold controller initialization."""
        controller = HeadingHoldController()

        assert controller.heading_pid is not None
        assert controller.roll_pid is not None
        assert controller.heading_target == 0.0

    def test_set_target_heading(self):
        """Test setting target heading."""
        controller = HeadingHoldController()
        controller.set_target_heading(np.radians(90.0))

        assert controller.heading_target == pytest.approx(np.radians(90.0))

    def test_heading_wrap_around(self):
        """Test heading error wraps around ±180°."""
        controller = HeadingHoldController()

        # Target: 350° (near north)
        controller.set_target_heading(np.radians(350.0))

        # Current: 10° (also near north, should command small turn)
        aileron = controller.update(
            current_heading=np.radians(10.0),
            current_roll=0.0,
            dt=0.01
        )

        # Should produce valid aileron command
        assert np.isfinite(aileron)

    def test_reset(self):
        """Test controller reset."""
        controller = HeadingHoldController()
        controller.set_target_heading(np.radians(90.0))

        for _ in range(10):
            controller.update(np.radians(45.0), 0.0, 0.01)

        controller.reset()

        assert controller.heading_pid.error_integral == 0.0
        assert controller.roll_pid.error_integral == 0.0


class TestAirspeedHoldController:
    """Test airspeed hold controller."""

    def test_initialization(self):
        """Test airspeed hold controller initialization."""
        controller = AirspeedHoldController()

        assert controller.pid is not None
        assert controller.airspeed_target == 0.0

    def test_set_target_airspeed(self):
        """Test setting target airspeed."""
        controller = AirspeedHoldController()
        controller.set_target_airspeed(600.0)

        assert controller.airspeed_target == 600.0

    def test_update(self):
        """Test airspeed hold update."""
        controller = AirspeedHoldController()
        controller.set_target_airspeed(600.0)

        throttle = controller.update(
            current_airspeed=580.0,
            dt=0.01
        )

        # Should produce valid throttle command
        assert np.isfinite(throttle)
        assert 0.0 <= throttle <= 1.0

    def test_throttle_saturation(self):
        """Test throttle stays within 0-1 limits."""
        controller = AirspeedHoldController(Kp=10.0)
        controller.set_target_airspeed(600.0)

        # Very low airspeed
        throttle = controller.update(current_airspeed=100.0, dt=0.01)
        assert 0.0 <= throttle <= 1.0

    def test_reset(self):
        """Test controller reset."""
        controller = AirspeedHoldController()
        controller.set_target_airspeed(600.0)

        for _ in range(10):
            controller.update(500.0, 0.01)

        controller.reset()

        assert controller.pid.error_integral == 0.0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
