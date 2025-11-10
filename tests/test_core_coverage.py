"""
Additional Core Module Tests for Coverage

Tests to improve coverage for:
- Quaternion operations
- Dynamics edge cases
- Integrators
- Propulsion models
- Atmosphere edge cases
"""

import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.quaternion import Quaternion
from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.integrator import RK4Integrator, RK45Integrator
from src.core.propulsion import ConstantThrustModel, PropellerModel, CombinedForceModel
from src.environment.atmosphere import StandardAtmosphere


class TestQuaternionCoverage:
    """Additional quaternion tests for coverage."""

    def test_quaternion_properties(self):
        """Test quaternion scalar and vector properties."""
        q = Quaternion(np.array([0.7071, 0.7071, 0, 0]))

        assert np.isclose(q.scalar, 0.7071, atol=1e-4)
        assert np.allclose(q.vector, [0.7071, 0, 0], atol=1e-4)

    def test_quaternion_conjugate(self):
        """Test quaternion conjugate."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q_conj = q.conjugate()

        # Conjugate should reverse rotation
        assert np.isclose(q_conj.scalar, q.scalar)
        assert np.allclose(q_conj.vector, -q.vector)

    def test_quaternion_inverse(self):
        """Test quaternion inverse."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q_inv = q.inverse()

        # q * q_inv should be identity
        q_identity = q * q_inv
        assert np.allclose(q_identity.q, [1, 0, 0, 0], atol=1e-10)

    def test_quaternion_rotate_vector(self):
        """Test rotating a vector with quaternion."""
        # 90 degree rotation about z-axis
        q = Quaternion.from_euler_angles(0, 0, np.pi/2)
        v = np.array([1, 0, 0])
        v_rot = q.rotate_vector(v)

        # [1,0,0] rotated 90° about z should give [0,1,0]
        assert np.allclose(v_rot, [0, 1, 0], atol=1e-10)

    def test_quaternion_integrate(self):
        """Test quaternion integration."""
        q = Quaternion()  # Identity
        omega = np.array([0, 0, 0.1])  # Slow rotation about z
        dt = 0.1

        q_new = q.integrate(omega, dt)

        # Should have small rotation
        assert isinstance(q_new, Quaternion)
        assert np.isclose(np.linalg.norm(q_new.q), 1.0)

    def test_quaternion_from_rotation_matrix(self):
        """Test creating quaternion from rotation matrix."""
        # Create rotation matrix for 90° yaw
        R = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 1]], dtype=float)

        q = Quaternion.from_rotation_matrix(R)

        # Convert back to check
        R2 = q.to_rotation_matrix()
        assert np.allclose(R, R2, atol=1e-10)

    def test_quaternion_string_representation(self):
        """Test string representations."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)

        repr_str = repr(q)
        str_str = str(q)

        assert 'Quaternion' in repr_str
        assert 'Roll' in str_str or 'Pitch' in str_str

    def test_quaternion_degenerate_normalization(self):
        """Test normalization of near-zero quaternion."""
        q = Quaternion(np.array([1e-12, 0, 0, 0]))

        # Should reset to identity
        assert np.allclose(q.q, [1, 0, 0, 0])


class TestDynamicsCoverage:
    """Additional dynamics tests for coverage."""

    def test_dynamics_gravity_only(self):
        """Test dynamics with only gravity."""
        mass = 100.0
        inertia = np.diag([1000, 1000, 1000])
        dynamics = AircraftDynamics(mass, inertia)

        state = State()
        state.altitude = 5000

        # No forces or moments
        def zero_forces(s):
            return (np.zeros(3), np.zeros(3))

        state_dot = dynamics.state_derivative(state, zero_forces)

        # Should have gravity acceleration in z
        assert state_dot.shape == (13,)
        # Vertical acceleration should be negative (gravity)
        assert state_dot[5] < 0  # w_dot

    def test_dynamics_with_forces(self):
        """Test dynamics with applied forces and moments."""
        mass = 100.0
        inertia = np.diag([1000, 1000, 1000])
        dynamics = AircraftDynamics(mass, inertia)

        state = State()
        state.velocity_body = np.array([100, 0, 0])

        # Apply thrust and moment
        def forces_moments(s):
            F = np.array([500, 0, 0])  # Thrust
            M = np.array([0, 100, 0])  # Pitch moment
            return (F, M)

        state_dot = dynamics.state_derivative(state, forces_moments)

        # Should have forward acceleration
        assert state_dot[3] > 0  # u_dot
        # Should have pitch rate change
        assert state_dot[11] != 0  # q_dot

    def test_dynamics_with_angular_rates(self):
        """Test dynamics with angular rates."""
        mass = 100.0
        inertia = np.diag([1000, 2000, 3000])
        dynamics = AircraftDynamics(mass, inertia)

        state = State()
        state.angular_rates = np.array([0.1, 0.2, 0.1])  # p, q, r

        def zero_forces(s):
            return (np.zeros(3), np.zeros(3))

        state_dot = dynamics.state_derivative(state, zero_forces)

        # Angular rates should cause quaternion to change
        assert not np.allclose(state_dot[6:10], 0)


class TestIntegratorCoverage:
    """Additional integrator tests for coverage."""

    def test_rk4_initialization(self):
        """Test RK4 integrator initialization."""
        integrator = RK4Integrator(dt=0.1)
        assert integrator.dt == 0.1

        integrator2 = RK4Integrator()  # Default
        assert integrator2.dt == 0.01

    def test_rk45_initialization(self):
        """Test RK45 integrator initialization."""
        integrator = RK45Integrator(rtol=1e-5, atol=1e-7)
        assert integrator.rtol == 1e-5
        assert integrator.atol == 1e-7

        integrator2 = RK45Integrator()  # Defaults
        assert integrator2.rtol == 1e-6
        assert integrator2.atol == 1e-8


class TestPropulsionCoverage:
    """Additional propulsion tests for coverage."""

    def test_constant_thrust_basic(self):
        """Test basic constant thrust model."""
        model = ConstantThrustModel(thrust=1000.0)

        state = State()
        state.altitude = 5000
        state.velocity_body = np.array([200, 0, 0])

        F, M = model.compute_thrust(state, throttle=1.0)

        assert F[0] == 1000.0  # Forward thrust
        assert np.allclose(M, 0)  # No moments

    def test_constant_thrust_with_offset(self):
        """Test constant thrust with moment arm offset."""
        offset = np.array([0, 0, 1.0])  # 1 ft above CG
        model = ConstantThrustModel(thrust=1000.0, thrust_offset=offset)

        state = State()
        F, M = model.compute_thrust(state, throttle=1.0)

        # Should have pitch moment
        assert M[1] != 0

    def test_constant_thrust_throttle(self):
        """Test throttle control."""
        model = ConstantThrustModel(thrust=1000.0)

        state = State()
        F1, _ = model.compute_thrust(state, throttle=0.5)
        F2, _ = model.compute_thrust(state, throttle=1.0)

        assert F1[0] == 500.0
        assert F2[0] == 1000.0

    def test_propeller_model_basic(self):
        """Test propeller model."""
        model = PropellerModel(power_max=100.0, prop_diameter=6.0, prop_efficiency=0.75)

        state = State()
        state.altitude = 5000
        state.velocity_body = np.array([200, 0, 0])

        F, M = model.compute_thrust(state, throttle=1.0)

        # Should produce thrust
        assert F[0] > 0
        assert np.allclose(M, 0)

    def test_propeller_at_zero_velocity(self):
        """Test propeller at zero velocity."""
        model = PropellerModel(power_max=100.0, prop_diameter=6.0)

        state = State()
        state.altitude = 0
        state.velocity_body = np.array([0, 0, 0])

        F, M = model.compute_thrust(state, throttle=1.0)

        # Should still produce thrust (static thrust)
        assert F[0] > 0

    def test_combined_force_model(self):
        """Test combined force model."""
        from src.core.aerodynamics import ConstantCoeffModel

        aero = ConstantCoeffModel(CL=0.5, CD=0.05)
        prop = ConstantThrustModel(thrust=500.0)

        combined = CombinedForceModel(aero, prop)

        state = State()
        state.altitude = 5000
        state.velocity_body = np.array([200, 0, 0])

        F, M = combined(state, throttle=1.0)

        # Should have both aero and propulsion forces
        assert F[0] != 0  # Drag + thrust
        assert F[2] != 0  # Lift


class TestAtmosphereCoverage:
    """Additional atmosphere tests for coverage."""

    def test_atmosphere_sea_level(self):
        """Test atmosphere at sea level."""
        atm = StandardAtmosphere(altitude=0)

        assert np.isclose(atm.temperature, 518.67, rtol=0.01)  # 59°F
        assert np.isclose(atm.pressure, 2116.2, rtol=0.01)  # psf
        assert np.isclose(atm.density, 0.002377, rtol=0.01)  # slugs/ft³

    def test_atmosphere_troposphere(self):
        """Test atmosphere in troposphere."""
        atm = StandardAtmosphere(altitude=20000)

        # Temperature should decrease with altitude
        assert atm.temperature < 518.67
        assert atm.pressure < 2116.2
        assert atm.density < 0.002377

    def test_atmosphere_stratosphere(self):
        """Test atmosphere in stratosphere."""
        atm = StandardAtmosphere(altitude=50000)

        # Should be in lower stratosphere
        assert atm.temperature > 0
        assert atm.pressure > 0
        assert atm.density > 0

    def test_atmosphere_upper_stratosphere(self):
        """Test atmosphere in upper stratosphere."""
        atm = StandardAtmosphere(altitude=75000)

        # Should still have valid properties
        assert atm.temperature > 0
        assert atm.pressure > 0
        assert atm.density > 0

    def test_atmosphere_mach_number(self):
        """Test Mach number calculation."""
        atm = StandardAtmosphere(altitude=20000)

        velocity = 250  # ft/s
        mach = atm.get_mach_number(velocity)

        assert mach > 0
        assert mach < 1  # Subsonic

    def test_atmosphere_dynamic_pressure(self):
        """Test dynamic pressure calculation."""
        atm = StandardAtmosphere(altitude=5000)

        velocity = 200  # ft/s
        q = atm.get_dynamic_pressure(velocity)

        expected_q = 0.5 * atm.density * velocity**2
        assert np.isclose(q, expected_q)

    def test_atmosphere_reynolds_number(self):
        """Test Reynolds number calculation."""
        atm = StandardAtmosphere(altitude=10000)

        velocity = 200  # ft/s
        length = 10  # ft
        Re = atm.get_reynolds_number(velocity, length)

        assert Re > 0

    def test_atmosphere_speed_of_sound(self):
        """Test speed of sound calculation."""
        atm = StandardAtmosphere(altitude=0)

        # Speed of sound at sea level ~1116 ft/s
        assert np.isclose(atm.speed_of_sound, 1116, rtol=0.01)

    def test_atmosphere_viscosity(self):
        """Test viscosity calculation."""
        atm = StandardAtmosphere(altitude=0)

        # Should have positive viscosity
        assert atm.dynamic_viscosity > 0
        assert atm.kinematic_viscosity > 0

    def test_atmosphere_string_representation(self):
        """Test string representation."""
        atm = StandardAtmosphere(altitude=5000)

        repr_str = repr(atm)
        str_str = str(atm)

        assert 'StandardAtmosphere' in repr_str
        assert 'altitude' in str_str.lower()


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
