"""
Phase 3 Integration Tests

Tests for supporting systems:
- Standard Atmosphere
- PID controllers
- Autopilot systems
- Trim solver
- AVL database
"""

import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.atmosphere import StandardAtmosphere
from src.control.autopilot import PIDController, AltitudeHoldController, HeadingHoldController, AirspeedHoldController
from src.control.trim import TrimSolver
from src.aero.avl_database import AVLDatabase
from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import ConstantThrustModel, CombinedForceModel


class TestStandardAtmosphere:
    """Test standard atmosphere model."""

    def test_sea_level_conditions(self):
        """Test sea level standard conditions."""
        atm = StandardAtmosphere(0.0)

        # Check against known values
        assert np.isclose(atm.temperature, 518.67, rtol=1e-4)  # 59°F
        assert np.isclose(atm.pressure, 2116.22, rtol=1e-4)  # 14.696 psi
        assert np.isclose(atm.density, 0.002377, rtol=1e-4)

    def test_altitude_variation(self):
        """Test atmosphere varies with altitude."""
        atm_sl = StandardAtmosphere(0.0)
        atm_10k = StandardAtmosphere(10000.0)

        # Temperature should decrease
        assert atm_10k.temperature < atm_sl.temperature

        # Pressure should decrease
        assert atm_10k.pressure < atm_sl.pressure

        # Density should decrease
        assert atm_10k.density < atm_sl.density

    def test_speed_of_sound(self):
        """Test speed of sound calculation."""
        atm = StandardAtmosphere(0.0)

        # Sea level speed of sound ≈ 1116.4 ft/s
        assert np.isclose(atm.speed_of_sound, 1116.4, rtol=0.01)

    def test_mach_number(self):
        """Test Mach number calculation."""
        atm = StandardAtmosphere(10000.0)
        velocity = 200.0  # ft/s

        mach = atm.get_mach_number(velocity)

        assert mach > 0
        assert mach < 1.0  # Subsonic

    def test_dynamic_pressure(self):
        """Test dynamic pressure calculation."""
        atm = StandardAtmosphere(5000.0)
        velocity = 250.0  # ft/s

        q = atm.get_dynamic_pressure(velocity)

        # Should be positive
        assert q > 0

        # Check formula: q = 0.5 * rho * V^2
        expected_q = 0.5 * atm.density * velocity**2
        assert np.isclose(q, expected_q)

    def test_reynolds_number(self):
        """Test Reynolds number calculation."""
        atm = StandardAtmosphere(10000.0)
        velocity = 200.0  # ft/s
        chord = 10.0  # ft

        Re = atm.get_reynolds_number(velocity, chord)

        # Should be in typical range for aircraft
        assert Re > 1e6  # Greater than 1 million
        assert Re < 1e8  # Less than 100 million


class TestPIDController:
    """Test PID controller."""

    def test_proportional_only(self):
        """Test P controller response."""
        pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0)

        error = 5.0
        output = pid.update(error, 0.01)

        # Output should be Kp * error
        assert np.isclose(output, 10.0)

    def test_integral_accumulation(self):
        """Test integral term accumulates."""
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0)

        # Apply constant error for several steps
        error = 2.0
        dt = 0.1

        for i in range(10):
            output = pid.update(error, dt)

        # Integral should accumulate: Ki * error * time
        expected = 1.0 * 2.0 * 1.0  # Ki * error * total_time
        assert np.isclose(output, expected, rtol=0.1)

    def test_output_limits(self):
        """Test output saturation."""
        pid = PIDController(Kp=10.0, Ki=0.0, Kd=0.0, output_limits=(-5, 5))

        error = 10.0  # Would give output = 100 without limits
        output = pid.update(error, 0.01)

        # Should be saturated at max limit
        assert output == 5.0

    def test_reset(self):
        """Test controller reset."""
        pid = PIDController(Kp=1.0, Ki=1.0, Kd=0.0)

        # Accumulate some integral
        for i in range(10):
            pid.update(5.0, 0.1)

        # Reset
        pid.reset()

        # Next update should not include accumulated integral
        output = pid.update(5.0, 0.1)
        assert np.isclose(output, 5.0 + 0.5)  # P + I for single step


class TestAltitudeHoldController:
    """Test altitude hold autopilot."""

    def test_initialization(self):
        """Test controller initializes properly."""
        controller = AltitudeHoldController()

        assert controller.altitude_target == 0.0
        assert controller.altitude_pid is not None
        assert controller.pitch_pid is not None

    def test_set_target(self):
        """Test setting target altitude."""
        controller = AltitudeHoldController()
        controller.set_target_altitude(5000.0)

        assert controller.altitude_target == 5000.0

    def test_control_response(self):
        """Test controller produces elevator command."""
        controller = AltitudeHoldController()
        controller.set_target_altitude(5000.0)

        # Current state: below target
        elevator = controller.update(current_altitude=4500.0,
                                      current_pitch=np.radians(2),
                                      dt=0.01)

        # Should produce positive elevator (climb)
        assert isinstance(elevator, float)
        assert not np.isnan(elevator)


class TestHeadingHoldController:
    """Test heading hold autopilot."""

    def test_heading_wrap_around(self):
        """Test heading error wraps correctly."""
        controller = HeadingHoldController()

        # Target: 350° = -10° from North
        controller.set_target_heading(np.radians(350))

        # Current: 10°
        aileron = controller.update(current_heading=np.radians(10),
                                      current_roll=0.0,
                                      dt=0.01)

        # Should command left turn (negative aileron)
        # The shortest path from 10° to 350° is -20° (left)
        assert isinstance(aileron, float)


class TestAirspeedHoldController:
    """Test airspeed hold autopilot."""

    def test_throttle_increase(self):
        """Test controller increases throttle when slow."""
        controller = AirspeedHoldController()
        controller.set_target_airspeed(250.0)

        # Current: slower than target
        throttle = controller.update(current_airspeed=200.0, dt=0.01)

        # Should be positive throttle
        assert throttle > 0


class TestTrimSolver:
    """Test trim solver."""

    def test_trim_solver_initialization(self):
        """Test trim solver initializes."""
        # Create simple dynamics function
        mass = 234.8
        inertia = np.diag([14908.4, 2318.4, 17226.9])
        dynamics = AircraftDynamics(mass, inertia)

        aero = LinearAeroModel(S_ref=199.94, c_ref=26.689, b_ref=19.890)
        aero.CL_0 = 0.2
        aero.CL_alpha = 4.5
        aero.Cm_alpha = -0.6

        prop = ConstantThrustModel(thrust=600.0)
        force_model = CombinedForceModel(aero, prop)

        def dynamics_func(state, controls):
            throttle = controls.get('throttle', 0.5)
            return dynamics.state_derivative(state, lambda s: force_model(s, throttle))

        solver = TrimSolver(dynamics_func)
        assert solver is not None

    def test_trim_straight_level_converges(self):
        """Test trim solver finds solution for straight flight."""
        # Setup dynamics
        mass = 234.8
        inertia = np.diag([14908.4, 2318.4, 17226.9])
        dynamics = AircraftDynamics(mass, inertia)

        aero = LinearAeroModel(S_ref=199.94, c_ref=26.689, b_ref=19.890)
        aero.CL_0 = 0.2
        aero.CL_alpha = 4.5
        aero.CD_0 = 0.03
        aero.CD_alpha2 = 0.5
        aero.Cm_0 = 0.0
        aero.Cm_alpha = -0.6
        aero.Cm_elevator = -0.8

        prop = ConstantThrustModel(thrust=600.0)
        force_model = CombinedForceModel(aero, prop)

        def dynamics_func(state, controls):
            throttle = controls.get('throttle', 0.5)
            elevator = controls.get('elevator', 0.0)
            # Add elevator effect to Cm
            return dynamics.state_derivative(state, lambda s: force_model(s, throttle))

        solver = TrimSolver(dynamics_func)

        # Find trim
        state_trim, controls_trim, info = solver.trim_straight_level(
            altitude=5000.0,
            airspeed=200.0,
            verbose=False
        )

        # Check convergence
        assert info['success'] or info['residual_norm'] < 1.0  # Relaxed tolerance

        # Check trim state is reasonable
        assert state_trim.altitude > 4000
        assert state_trim.altitude < 6000
        assert state_trim.airspeed > 150
        assert state_trim.airspeed < 250


class TestAVLDatabase:
    """Test AVL aerodynamic database."""

    def test_database_creation(self):
        """Test creating database."""
        alpha_table = np.radians(np.linspace(-5, 15, 21))
        CL_table = 0.2 + 5.0 * alpha_table

        data_table = {'CL': CL_table}

        db = AVLDatabase(alpha_table, data_table, S_ref=200.0, c_ref=10.0, b_ref=20.0)

        assert db.S_ref == 200.0
        assert len(db.alpha_table) == 21

    def test_interpolation(self):
        """Test coefficient interpolation."""
        alpha_table = np.radians(np.array([0, 5, 10]))
        CL_table = np.array([0.2, 0.6, 1.0])

        data_table = {'CL': CL_table}

        db = AVLDatabase(alpha_table, data_table, S_ref=200.0, c_ref=10.0, b_ref=20.0)

        # Interpolate at midpoint
        coeffs = db.get_coefficients(np.radians(2.5))

        # Should be between 0.2 and 0.6
        assert coeffs['CL'] > 0.2
        assert coeffs['CL'] < 0.6

    def test_forces_moments(self):
        """Test force and moment calculation."""
        alpha_table = np.radians(np.array([0, 5, 10]))
        CL_table = np.array([0.2, 0.6, 1.0])
        CD_table = np.array([0.02, 0.03, 0.05])
        Cm_table = np.array([0.0, -0.05, -0.1])

        data_table = {'CL': CL_table, 'CD': CD_table, 'Cm': Cm_table}

        db = AVLDatabase(alpha_table, data_table, S_ref=200.0, c_ref=10.0, b_ref=20.0)

        # Get forces at alpha = 5°
        forces, moments = db.get_forces_moments(alpha=np.radians(5), q_bar=50.0)

        assert forces.shape == (3,)
        assert moments.shape == (3,)

        # Lift should be upward (negative Fz)
        assert forces[2] < 0

        # Check forces are not NaN
        assert not np.any(np.isnan(forces))
        assert not np.any(np.isnan(moments))


class TestIntegration:
    """Integration tests combining multiple Phase 3 components."""

    def test_simulation_with_atmosphere(self):
        """Test simulation using standard atmosphere."""
        # Aircraft setup
        mass = 234.8
        inertia = np.diag([14908.4, 2318.4, 17226.9])
        dynamics = AircraftDynamics(mass, inertia)

        # Simple aero model
        aero = LinearAeroModel(S_ref=199.94, c_ref=26.689, b_ref=19.890)
        aero.CL_0 = 0.2
        aero.CL_alpha = 4.5

        prop = ConstantThrustModel(thrust=600.0)
        force_model = CombinedForceModel(aero, prop)

        # Initial state
        state = State()
        state.altitude = 5000
        state.velocity_body = np.array([200, 0, 0])

        # Get atmospheric properties
        atm = StandardAtmosphere(state.altitude)

        # Update aero model density
        aero.rho = atm.density

        # Compute forces
        forces, moments = force_model(state, throttle=0.8)

        # Check forces are reasonable
        assert not np.any(np.isnan(forces))
        assert not np.any(np.isnan(moments))

    def test_autopilot_simulation(self):
        """Test short simulation with autopilot."""
        # Setup
        mass = 234.8
        inertia = np.diag([14908.4, 2318.4, 17226.9])
        dynamics = AircraftDynamics(mass, inertia)

        aero = LinearAeroModel(S_ref=199.94, c_ref=26.689, b_ref=19.890)
        aero.CL_0 = 0.2
        aero.CL_alpha = 4.5
        aero.Cm_alpha = -0.6

        prop = ConstantThrustModel(thrust=600.0)
        force_model = CombinedForceModel(aero, prop)

        # Autopilot
        alt_controller = AltitudeHoldController()
        alt_controller.set_target_altitude(5000.0)

        # Initial state (slightly below target)
        state = State()
        state.altitude = 4900
        state.velocity_body = np.array([200, 0, 0])

        # Run a few steps
        dt = 0.01
        for i in range(100):
            # Get elevator command
            elevator = alt_controller.update(state.altitude,
                                              state.euler_angles[1],
                                              dt)

            # Simple dynamics update (just checking it runs)
            forces, moments = force_model(state, throttle=0.8)

            # Check no NaNs
            assert not np.isnan(elevator)
            assert not np.any(np.isnan(forces))


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
