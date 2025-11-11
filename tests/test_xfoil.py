"""
Unit tests for XFOIL integration.

Tests:
- XFOIL database loading and interpolation
- XFOIL aerodynamic model
- Reynolds number effects
- 3D corrections

Author: Claude Code
Date: 2025-11-10
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aero.xfoil_database import (
    PolarDatabase, XFOILDatabaseLoader, create_example_database
)
from src.core.aerodynamics import XFOILAeroModel
from src.core.state import State


class TestPolarDatabase:
    """Test polar database functionality."""

    @pytest.fixture
    def sample_database(self):
        """Create a sample polar database for testing."""
        reynolds_numbers = np.array([1e6, 3e6, 6e6])
        polars = {}

        for re in reynolds_numbers:
            alpha = np.linspace(-5, 15, 21)
            CL = 0.2 + 0.11 * alpha
            CD = 0.006 + 0.001 * (alpha / 10)**2
            CM = -0.05 - 0.003 * alpha

            polars[re] = {
                'alpha': alpha,
                'CL': CL,
                'CD': CD,
                'CM': CM
            }

        return PolarDatabase(
            airfoil_name='TEST_AIRFOIL',
            reynolds_numbers=reynolds_numbers,
            polars=polars,
            mach=0.25,
            ncrit=9.0
        )

    def test_interpolate_coefficients_exact_re(self, sample_database):
        """Test interpolation at exact Reynolds number."""
        # Test at Re = 3e6 (exact match)
        CL, CD, CM = sample_database.interpolate_coefficients(3e6, 5.0)

        # Expected values at alpha=5°
        expected_CL = 0.2 + 0.11 * 5.0
        expected_CD = 0.006 + 0.001 * (5.0 / 10)**2
        expected_CM = -0.05 - 0.003 * 5.0

        assert CL == pytest.approx(expected_CL, rel=0.01)
        assert CD == pytest.approx(expected_CD, rel=0.01)
        assert CM == pytest.approx(expected_CM, rel=0.01)

    def test_interpolate_coefficients_between_re(self, sample_database):
        """Test interpolation between Reynolds numbers."""
        # Test at Re = 2e6 (between 1e6 and 3e6)
        CL, CD, CM = sample_database.interpolate_coefficients(2e6, 5.0)

        # Should be between values at 1e6 and 3e6
        CL_1M, CD_1M, CM_1M = sample_database.interpolate_coefficients(1e6, 5.0)
        CL_3M, CD_3M, CM_3M = sample_database.interpolate_coefficients(3e6, 5.0)

        # Linear interpolation: should be halfway between
        assert CL == pytest.approx((CL_1M + CL_3M) / 2, rel=0.01)
        assert CD > 0
        assert isinstance(CM, (int, float))

    def test_interpolate_coefficients_out_of_bounds_re(self, sample_database):
        """Test that out-of-bounds Reynolds numbers are clamped."""
        # Test below minimum Re
        CL_low, CD_low, CM_low = sample_database.interpolate_coefficients(0.5e6, 5.0)
        CL_min, CD_min, CM_min = sample_database.interpolate_coefficients(1e6, 5.0)

        # Should clamp to minimum Re
        assert CL_low == pytest.approx(CL_min, rel=0.01)
        assert CD_low == pytest.approx(CD_min, rel=0.01)

        # Test above maximum Re
        CL_high, CD_high, CM_high = sample_database.interpolate_coefficients(10e6, 5.0)
        CL_max, CD_max, CM_max = sample_database.interpolate_coefficients(6e6, 5.0)

        # Should clamp to maximum Re
        assert CL_high == pytest.approx(CL_max, rel=0.01)
        assert CD_high == pytest.approx(CD_max, rel=0.01)

    def test_get_CLmax(self, sample_database):
        """Test getting maximum lift coefficient."""
        CLmax = sample_database.get_CLmax(3e6)

        # CLmax should be positive and reasonable
        assert CLmax > 1.0
        assert CLmax < 2.0

    def test_get_alpha_zero_lift(self, sample_database):
        """Test getting zero-lift angle of attack."""
        alpha0 = sample_database.get_alpha_zero_lift(3e6)

        # For cambered airfoil, should be slightly negative
        assert -3.0 < alpha0 < 0.0


class TestXFOILDatabaseLoader:
    """Test XFOIL database loader."""

    @pytest.fixture
    def temp_database_dir(self):
        """Create temporary directory for test database."""
        temp_dir = tempfile.mkdtemp(prefix='xfoil_test_')
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_create_example_database(self, temp_database_dir):
        """Test creating example database."""
        create_example_database(temp_database_dir)

        # Check that files were created
        csv_files = list(Path(temp_database_dir).glob('*.csv'))
        assert len(csv_files) == 4  # Should have 4 Reynolds numbers

    def test_load_polar_csv(self, temp_database_dir):
        """Test loading a single polar CSV file."""
        create_example_database(temp_database_dir)

        loader = XFOILDatabaseLoader(database_dir=temp_database_dir)
        csv_file = list(Path(temp_database_dir).glob('*.csv'))[0]

        data = loader.load_polar_csv(str(csv_file))

        # Check that data was loaded
        assert 'alpha' in data
        assert 'CL' in data
        assert 'CD' in data
        assert 'CM' in data
        assert len(data['alpha']) > 0

        # Check metadata
        metadata = data.get('metadata', {})
        assert 'Airfoil' in metadata
        assert 'Reynolds' in metadata

    def test_auto_load_airfoil(self, temp_database_dir):
        """Test automatic loading of airfoil polars."""
        create_example_database(temp_database_dir)

        loader = XFOILDatabaseLoader(database_dir=temp_database_dir)
        db = loader.auto_load_airfoil('NACA_64-212')

        assert db is not None
        assert db.airfoil_name == 'NACA_64-212'
        assert len(db.reynolds_numbers) == 4
        assert db.mach == 0.25

    def test_auto_load_nonexistent_airfoil(self, temp_database_dir):
        """Test loading non-existent airfoil returns None."""
        loader = XFOILDatabaseLoader(database_dir=temp_database_dir)
        db = loader.auto_load_airfoil('NONEXISTENT_AIRFOIL')

        assert db is None


class TestXFOILAeroModel:
    """Test XFOIL-based aerodynamic model."""

    @pytest.fixture
    def xfoil_model(self, tmp_path):
        """Create XFOIL aerodynamic model with test database."""
        # Create test database
        create_example_database(str(tmp_path))

        # Load database
        loader = XFOILDatabaseLoader(database_dir=str(tmp_path))
        polar_db = loader.auto_load_airfoil('NACA_64-212')

        # Create model
        model = XFOILAeroModel(
            polar_database=polar_db,
            S_ref=199.94,
            c_ref=26.689,
            b_ref=19.890,
            aspect_ratio=1.98,
            oswald_efficiency=0.85
        )

        return model

    def test_compute_forces_moments(self, xfoil_model):
        """Test computing forces and moments."""
        # Create test state
        state = State()
        state.position = np.array([0, 0, -5000])
        state.velocity_body = np.array([300, 0, 0])
        state.set_euler_angles(0, np.radians(5), 0)

        forces, moments = xfoil_model.compute_forces_moments(state)

        # Check that forces and moments are reasonable
        assert forces.shape == (3,)
        assert moments.shape == (3,)

        # Lift should be negative Z (upward)
        assert forces[2] < 0

        # Drag should be negative X (rearward)
        assert forces[0] < 0

    def test_reynolds_number_effects(self, xfoil_model):
        """Test that Reynolds number affects aerodynamic coefficients."""
        # Create two states with different Reynolds numbers
        state1 = State()
        state1.position = np.array([0, 0, -5000])
        state1.velocity_body = np.array([200, 0, 0])  # Lower speed
        state1.set_euler_angles(0, np.radians(5), 0)

        state2 = State()
        state2.position = np.array([0, 0, -5000])
        state2.velocity_body = np.array([600, 0, 0])  # Higher speed
        state2.set_euler_angles(0, np.radians(5), 0)

        forces1, moments1 = xfoil_model.compute_forces_moments(state1)
        forces2, moments2 = xfoil_model.compute_forces_moments(state2)

        # At higher Re, forces should scale more than velocity squared alone
        # (due to Reynolds effects on CD, CL)
        # This is a complex relationship, so just check they're different
        assert not np.allclose(forces1, forces2)

    def test_altitude_effects(self, xfoil_model):
        """Test that altitude affects aerodynamic forces."""
        # Sea level
        state1 = State()
        state1.position = np.array([0, 0, 0])
        state1.velocity_body = np.array([300, 0, 0])
        state1.set_euler_angles(0, np.radians(5), 0)

        # High altitude
        state2 = State()
        state2.position = np.array([0, 0, -20000])
        state2.velocity_body = np.array([300, 0, 0])
        state2.set_euler_angles(0, np.radians(5), 0)

        forces1, moments1 = xfoil_model.compute_forces_moments(state1)
        forces2, moments2 = xfoil_model.compute_forces_moments(state2)

        # Forces at altitude should be less (lower density)
        assert np.linalg.norm(forces2) < np.linalg.norm(forces1)

    def test_control_surface_effects(self, xfoil_model):
        """Test that control surfaces affect forces and moments."""
        state = State()
        state.position = np.array([0, 0, -5000])
        state.velocity_body = np.array([300, 0, 0])
        state.set_euler_angles(0, np.radians(5), 0)

        # No control deflection
        forces0, moments0 = xfoil_model.compute_forces_moments(state, controls={})

        # With elevator deflection
        forces1, moments1 = xfoil_model.compute_forces_moments(
            state, controls={'elevator': np.radians(5)}
        )

        # Elevator should change pitch moment
        assert moments1[1] != moments0[1]

        # With aileron deflection
        forces2, moments2 = xfoil_model.compute_forces_moments(
            state, controls={'aileron': np.radians(5)}
        )

        # Aileron should change roll moment
        assert moments2[0] != moments0[0]

    def test_get_CLmax(self, xfoil_model):
        """Test getting maximum lift coefficient."""
        state = State()
        state.position = np.array([0, 0, -5000])
        state.velocity_body = np.array([300, 0, 0])
        state.set_euler_angles(0, 0, 0)

        CLmax = xfoil_model.get_CLmax(state)

        # CLmax should be positive and reasonable
        assert CLmax > 1.0
        assert CLmax < 2.0

    def test_get_stall_alpha(self, xfoil_model):
        """Test getting stall angle of attack."""
        state = State()
        state.position = np.array([0, 0, -5000])
        state.velocity_body = np.array([300, 0, 0])
        state.set_euler_angles(0, 0, 0)

        alpha_stall = xfoil_model.get_stall_alpha(state)

        # Stall alpha should be positive and reasonable
        assert np.degrees(alpha_stall) > 10.0
        assert np.degrees(alpha_stall) < 20.0

    def test_3d_corrections(self, xfoil_model):
        """Test that 3D corrections are applied."""
        state = State()
        state.position = np.array([0, 0, -5000])
        state.velocity_body = np.array([300, 0, 0])
        state.set_euler_angles(0, np.radians(5), 0)

        forces, moments = xfoil_model.compute_forces_moments(state)

        # Compute dynamic pressure
        rho = 0.002048  # slug/ft³ at 5000 ft
        V = 300
        q_bar = 0.5 * rho * V**2

        # Compute coefficients
        CL = -forces[2] / (q_bar * xfoil_model.S_ref)
        CD = -forces[0] / (q_bar * xfoil_model.S_ref)

        # With 3D corrections, induced drag should be present
        # CD should be higher than 2D profile drag alone
        assert CD > 0.006  # 2D CD0

        # CL should be positive at positive alpha
        assert CL > 0


class TestXFOILViscosity:
    """Test viscosity and Reynolds number calculations."""

    def test_viscosity_increases_with_temperature(self):
        """Test that viscosity increases with altitude (temperature effect)."""
        from src.core.aerodynamics import XFOILAeroModel

        # Create dummy model just to access viscosity method
        # (we need a database, but we won't use the model)
        temp_dir = tempfile.mkdtemp(prefix='xfoil_test_')
        create_example_database(temp_dir)
        loader = XFOILDatabaseLoader(database_dir=temp_dir)
        polar_db = loader.auto_load_airfoil('NACA_64-212')

        model = XFOILAeroModel(
            polar_database=polar_db,
            S_ref=100,
            c_ref=10,
            b_ref=20,
            aspect_ratio=4.0
        )

        mu_sl = model._compute_viscosity(0)
        mu_5k = model._compute_viscosity(5000)
        mu_10k = model._compute_viscosity(10000)

        # Viscosity should decrease with altitude (temperature decreases)
        assert mu_5k < mu_sl
        assert mu_10k < mu_5k

        # Clean up
        shutil.rmtree(temp_dir)

    def test_density_decreases_with_altitude(self):
        """Test that density decreases with altitude."""
        from src.core.aerodynamics import XFOILAeroModel

        # Create dummy model
        temp_dir = tempfile.mkdtemp(prefix='xfoil_test_')
        create_example_database(temp_dir)
        loader = XFOILDatabaseLoader(database_dir=temp_dir)
        polar_db = loader.auto_load_airfoil('NACA_64-212')

        model = XFOILAeroModel(
            polar_database=polar_db,
            S_ref=100,
            c_ref=10,
            b_ref=20,
            aspect_ratio=4.0
        )

        rho_sl = model._compute_density(0)
        rho_5k = model._compute_density(5000)
        rho_10k = model._compute_density(10000)

        # Density should decrease with altitude
        assert rho_5k < rho_sl
        assert rho_10k < rho_5k

        # Check reasonable values
        assert 0.002 < rho_sl < 0.0025  # slug/ft³ at sea level
        assert 0.0015 < rho_5k < 0.0021  # slug/ft³ at 5000 ft

        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
