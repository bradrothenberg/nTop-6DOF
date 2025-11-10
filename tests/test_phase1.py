"""
Phase 1 Integration Tests

Tests for AVL geometry generation, mass properties, and basic analysis.
"""

import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.geometry import (
    read_csv_points,
    compute_wing_geometry,
    estimate_tail_geometry,
    WingGeometry
)
from src.io.mass_properties import read_mass_csv, MassProperties
from src.aero.avl_geometry import generate_avl_geometry_from_csv
from src.aero.avl_run_cases import atmosphere_us_standard


class TestGeometryParser:
    """Test geometry parsing from CSV files."""

    @pytest.fixture
    def data_path(self):
        return os.path.join(os.path.dirname(__file__), '..', 'Data')

    def test_read_le_points(self, data_path):
        """Test reading leading edge points."""
        le_file = os.path.join(data_path, 'LEpts.csv')
        le_points = read_csv_points(le_file, units='inches')

        assert le_points.shape[0] == 13  # 13 points
        assert le_points.shape[1] == 3   # x, y, z
        assert le_points.dtype == np.float64

        # Check conversion from inches to feet (should be ~1/12 of original)
        # Original data has values in ~100+ inches range
        assert np.max(np.abs(le_points)) < 20  # Should be in feet

    def test_read_te_points(self, data_path):
        """Test reading trailing edge points."""
        te_file = os.path.join(data_path, 'TEpts.csv')
        te_points = read_csv_points(te_file, units='inches')

        assert te_points.shape[0] == 13
        assert te_points.shape[1] == 3

    def test_wing_geometry_calculation(self, data_path):
        """Test wing geometry calculations."""
        le_file = os.path.join(data_path, 'LEpts.csv')
        te_file = os.path.join(data_path, 'TEpts.csv')

        le_points = read_csv_points(le_file, units='inches')
        te_points = read_csv_points(te_file, units='inches')

        wing = compute_wing_geometry(le_points, te_points)

        # Validate computed geometry
        assert isinstance(wing, WingGeometry)
        assert wing.span > 0
        assert wing.area > 0
        assert wing.mac > 0
        assert 0 < wing.aspect_ratio < 10  # Reasonable AR range
        assert 0 < wing.taper_ratio < 1    # Valid taper

        # Check specific values for this geometry
        assert abs(wing.span - 19.89) < 0.1      # ~19.89 ft span
        assert abs(wing.area - 199.94) < 1.0     # ~199.94 ft² area
        assert abs(wing.aspect_ratio - 1.98) < 0.1  # Low AR (delta wing)

    def test_tail_geometry_estimation(self, data_path):
        """Test tail surface estimation."""
        le_file = os.path.join(data_path, 'LEpts.csv')
        te_file = os.path.join(data_path, 'TEpts.csv')

        le_points = read_csv_points(le_file, units='inches')
        te_points = read_csv_points(te_file, units='inches')

        wing = compute_wing_geometry(le_points, te_points)
        h_tail, v_tail = estimate_tail_geometry(wing)

        # Horizontal tail
        assert h_tail['area'] > 0
        assert h_tail['span'] > 0
        assert h_tail['chord'] > 0
        assert h_tail['area'] < wing.area  # Tail should be smaller than wing

        # Vertical tail
        assert v_tail['area'] > 0
        assert v_tail['height'] > 0
        assert v_tail['chord'] > 0


class TestMassProperties:
    """Test mass properties conversion."""

    @pytest.fixture
    def data_path(self):
        return os.path.join(os.path.dirname(__file__), '..', 'Data')

    def test_read_mass_csv(self, data_path):
        """Test reading mass properties from CSV."""
        mass_file = os.path.join(data_path, 'mass.csv')
        mass_props = read_mass_csv(mass_file)

        assert isinstance(mass_props, MassProperties)
        assert mass_props.mass_lbm > 0
        assert len(mass_props.cg_inches) == 3
        assert len(mass_props.inertia_lbm_in2) == 6

    def test_unit_conversions(self, data_path):
        """Test unit conversions from US Customary."""
        mass_file = os.path.join(data_path, 'mass.csv')
        mass_props = read_mass_csv(mass_file)

        # Check mass conversions
        assert abs(mass_props.mass_lbm - 7555.639) < 0.01
        assert abs(mass_props.mass_slugs - 234.84) < 0.1  # lbm / 32.174
        assert abs(mass_props.mass_kg - 3427.18) < 1.0

        # Check CG conversions
        assert abs(mass_props.cg_ft[0] - 12.918) < 0.01  # inches / 12
        assert len(mass_props.cg_m) == 3

        # Check inertia conversions
        assert mass_props.inertia_slug_ft2[0] > 0  # Ixx
        assert mass_props.inertia_slug_ft2[1] > 0  # Iyy
        assert mass_props.inertia_slug_ft2[2] > 0  # Izz

    def test_inertia_matrix(self, data_path):
        """Test inertia tensor matrix generation."""
        mass_file = os.path.join(data_path, 'mass.csv')
        mass_props = read_mass_csv(mass_file)

        I_matrix = mass_props.get_inertia_matrix_slug_ft2()

        assert I_matrix.shape == (3, 3)
        assert I_matrix[0, 0] > 0  # Ixx diagonal
        assert I_matrix[1, 1] > 0  # Iyy diagonal
        assert I_matrix[2, 2] > 0  # Izz diagonal
        assert np.allclose(I_matrix, I_matrix.T)  # Should be symmetric


class TestAVLFileGeneration:
    """Test AVL file generation."""

    @pytest.fixture
    def data_path(self):
        return os.path.join(os.path.dirname(__file__), '..', 'Data')

    @pytest.fixture
    def output_path(self, tmp_path):
        return tmp_path

    def test_avl_file_generation(self, data_path, output_path):
        """Test complete AVL geometry file generation."""
        le_file = os.path.join(data_path, 'LEpts.csv')
        te_file = os.path.join(data_path, 'TEpts.csv')
        mass_file = os.path.join(data_path, 'mass.csv')
        avl_file = os.path.join(output_path, 'test.avl')

        wing, h_tail, v_tail, mass_props = generate_avl_geometry_from_csv(
            le_file, te_file, mass_file, avl_file, aircraft_name="TestUAV"
        )

        # Check file was created
        assert os.path.exists(avl_file)

        # Read and validate file contents
        with open(avl_file, 'r') as f:
            content = f.read()

        # Check header
        assert 'TestUAV' in content
        assert 'Sref' in content
        assert 'Cref' in content
        assert 'Bref' in content

        # Check surfaces are defined
        assert 'SURFACE' in content
        assert 'Wing' in content
        assert 'Horizontal Tail' in content
        assert 'Vertical Tail' in content

        # Check control surfaces
        assert 'flaperon' in content
        assert 'elevator' in content
        assert 'rudder' in content

        # Check panel discretization
        assert 'Nchordwise' in content
        assert 'Nspanwise' in content

        # Check airfoils
        assert 'NACA 2412' in content
        assert 'NACA 0012' in content


class TestAtmosphereModel:
    """Test US Standard Atmosphere model."""

    def test_sea_level_conditions(self):
        """Test atmosphere at sea level."""
        atm = atmosphere_us_standard(0.0)

        # Check sea level values
        assert abs(atm['temperature'] - 518.67) < 0.1  # 59°F in Rankine
        assert abs(atm['pressure'] - 2116.22) < 1.0    # psf
        assert abs(atm['density'] - 0.002377) < 0.0001 # slug/ft³
        assert atm['speed_of_sound'] > 1000  # ft/s

    def test_altitude_variations(self):
        """Test atmosphere at different altitudes."""
        altitudes = [0, 5000, 10000, 20000, 30000]

        for alt in altitudes:
            atm = atmosphere_us_standard(alt)

            # All values should be positive
            assert atm['temperature'] > 0
            assert atm['pressure'] > 0
            assert atm['density'] > 0
            assert atm['speed_of_sound'] > 0

        # Check that density decreases with altitude
        atm_0 = atmosphere_us_standard(0)
        atm_20k = atmosphere_us_standard(20000)
        assert atm_20k['density'] < atm_0['density']
        assert atm_20k['temperature'] < atm_0['temperature']


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self, tmp_path):
        """Test complete nTop to AVL workflow."""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'Data')
        le_file = os.path.join(data_path, 'LEpts.csv')
        te_file = os.path.join(data_path, 'TEpts.csv')
        mass_file = os.path.join(data_path, 'mass.csv')

        avl_file = os.path.join(tmp_path, 'workflow_test.avl')
        mass_output = os.path.join(tmp_path, 'workflow_test.mass')

        # Step 1: Parse geometry
        le_points = read_csv_points(le_file, units='inches')
        te_points = read_csv_points(te_file, units='inches')
        wing = compute_wing_geometry(le_points, te_points)

        # Step 2: Convert mass properties
        mass_props = read_mass_csv(mass_file)
        mass_props.write_avl_mass_file(mass_output, name="TestUAV")

        # Step 3: Generate AVL files
        wing, h_tail, v_tail, mass_props = generate_avl_geometry_from_csv(
            le_file, te_file, mass_file, avl_file
        )

        # Verify all outputs
        assert os.path.exists(avl_file)
        assert os.path.exists(mass_output)
        assert wing.area > 0
        assert mass_props.mass_slugs > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
