"""
Phase 5 Integration Tests

Tests for I/O and configuration:
- YAML configuration loading
- Aircraft configuration
- AVL file parsing
- Configuration builders
"""

import pytest
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.config import AircraftConfig, load_aircraft_config, save_aircraft_config, create_example_config
from src.io.avl_parser import (parse_avl_stability_file, parse_avl_forces_file,
                                 parse_avl_run_file, parse_avl_mass_file)
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import PropellerModel


class TestConfiguration:
    """Test configuration system."""

    def test_create_example_config(self):
        """Test creating example configuration."""
        config_dict = create_example_config()

        assert 'aircraft' in config_dict
        assert 'name' in config_dict['aircraft']
        assert 'mass' in config_dict['aircraft']
        assert 'inertia' in config_dict['aircraft']

    def test_aircraft_config_creation(self):
        """Test creating AircraftConfig from dictionary."""
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        assert config.name == 'nTop UAV'
        assert config.mass > 0
        assert config.inertia.shape == (3, 3)
        assert config.S_ref > 0

    def test_config_units(self):
        """Test configuration units."""
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        assert config.units in ['US', 'SI']

    def test_create_dynamics_from_config(self):
        """Test creating dynamics object from configuration."""
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        dynamics = config.create_dynamics()

        assert isinstance(dynamics, AircraftDynamics)
        assert dynamics.mass == config.mass
        assert np.allclose(dynamics.inertia, config.inertia)

    def test_create_aero_model_from_config(self):
        """Test creating aerodynamic model from configuration."""
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        aero = config.create_aero_model()

        assert isinstance(aero, LinearAeroModel)
        assert aero.S_ref == config.S_ref

    def test_create_propulsion_from_config(self):
        """Test creating propulsion model from configuration."""
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        prop = config.create_propulsion_model()

        assert isinstance(prop, PropellerModel)

    def test_initial_state_dict(self):
        """Test getting initial state dictionary."""
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        initial = config.get_initial_state_dict()

        assert 'altitude' in initial
        assert 'airspeed' in initial

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_aircraft_config(config, temp_path)

            # Load back
            loaded_config = load_aircraft_config(temp_path)

            assert loaded_config.name == config.name
            assert loaded_config.mass == config.mass

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestAVLParsers:
    """Test AVL file parsers."""

    def test_parse_stability_file_missing(self):
        """Test parsing non-existent stability file."""
        result = parse_avl_stability_file('nonexistent.st')

        # Should return empty dict with warning
        assert isinstance(result, dict)

    def test_parse_forces_file_missing(self):
        """Test parsing non-existent forces file."""
        result = parse_avl_forces_file('nonexistent.ft')

        assert isinstance(result, dict)

    def test_parse_run_file_missing(self):
        """Test parsing non-existent run file."""
        result = parse_avl_run_file('nonexistent.run')

        assert isinstance(result, dict)

    def test_parse_mass_file_missing(self):
        """Test parsing non-existent mass file."""
        result = parse_avl_mass_file('nonexistent.mass')

        assert isinstance(result, dict)

    def test_parse_stability_file_with_content(self):
        """Test parsing stability file with sample content."""
        # Create temporary file with sample AVL output
        content = """
        AVL Stability Derivatives

        CLa   =   5.234
        CYb   =  -0.312
        Clb   =  -0.105
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.st', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = parse_avl_stability_file(temp_path)

            assert 'CLa' in result
            assert np.isclose(result['CLa'], 5.234)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_parse_forces_file_with_content(self):
        """Test parsing forces file with sample content."""
        content = """
        Total Forces

        CLtot =   0.523
        CDtot =   0.045
        CYtot =   0.000
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ft', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = parse_avl_forces_file(temp_path)

            assert 'CL' in result
            assert np.isclose(result['CL'], 0.523)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_complete_aircraft_setup(self):
        """Test complete aircraft setup from configuration."""
        # Create configuration
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        # Create all components
        dynamics = config.create_dynamics()
        aero = config.create_aero_model()
        prop = config.create_propulsion_model()

        # Verify they work together
        assert dynamics.mass == config.mass
        assert aero.S_ref == config.S_ref
        assert prop is not None

    def test_config_with_different_aero_types(self):
        """Test configuration with different aerodynamic model types."""
        # Constant coefficient model
        config_dict = {
            'aircraft': {
                'name': 'Test',
                'mass': 100.0,
                'inertia': {'Ixx': 1000, 'Iyy': 1000, 'Izz': 1000},
                'reference': {'S': 10.0, 'b': 5.0, 'c': 2.0},
                'aerodynamics': {
                    'type': 'constant',
                    'CL': 0.5,
                    'CD': 0.05
                },
                'propulsion': {
                    'type': 'constant_thrust',
                    'max_thrust': 500.0
                }
            }
        }

        config = AircraftConfig(config_dict)
        aero = config.create_aero_model()

        assert aero is not None

    def test_config_representation(self):
        """Test configuration string representation."""
        config_dict = create_example_config()
        config = AircraftConfig(config_dict)

        repr_str = repr(config)

        assert 'nTop UAV' in repr_str
        assert 'mass' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
