"""
Aircraft Configuration System

Provides YAML-based configuration loading for aircraft parameters,
aerodynamics, propulsion, and simulation setup.
"""

import yaml
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel, ConstantCoeffModel
from src.core.propulsion import ConstantThrustModel, PropellerModel
from src.aero.avl_database import AVLDatabase


class AircraftConfig:
    """
    Aircraft configuration loaded from YAML file.

    Provides easy access to aircraft parameters, aerodynamics,
    propulsion, and initial conditions.

    Attributes
    ----------
    name : str
        Aircraft name
    mass : float
        Aircraft mass (slugs or kg depending on units)
    inertia : ndarray
        Inertia tensor (3x3)
    reference : dict
        Reference geometry (S_ref, c_ref, b_ref)
    aerodynamics : dict
        Aerodynamic model configuration
    propulsion : dict
        Propulsion model configuration
    initial_state : dict
        Initial state configuration
    units : str
        Unit system ('SI' or 'US')
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize aircraft configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary (typically from YAML)
        """
        self.raw_config = config_dict
        self._parse_config()

    def _parse_config(self):
        """Parse configuration dictionary."""
        aircraft = self.raw_config.get('aircraft', {})

        # Basic info
        self.name = aircraft.get('name', 'Unnamed Aircraft')
        self.units = aircraft.get('units', 'US')  # 'US' or 'SI'

        # Mass properties
        self.mass = aircraft.get('mass', 100.0)

        inertia_dict = aircraft.get('inertia', {})
        self.inertia = np.array([
            [inertia_dict.get('Ixx', 1000.0), 0, inertia_dict.get('Ixz', 0.0)],
            [0, inertia_dict.get('Iyy', 1000.0), 0],
            [inertia_dict.get('Ixz', 0.0), 0, inertia_dict.get('Izz', 1000.0)]
        ])

        # Reference geometry
        self.reference = aircraft.get('reference', {})
        self.S_ref = self.reference.get('S', 100.0)
        self.b_ref = self.reference.get('b', 10.0)
        self.c_ref = self.reference.get('c', 5.0)

        # Aerodynamics
        self.aerodynamics = aircraft.get('aerodynamics', {})

        # Propulsion
        self.propulsion = aircraft.get('propulsion', {})

        # Initial state
        self.initial_state = aircraft.get('initial_state', {})

    def create_dynamics(self) -> AircraftDynamics:
        """
        Create AircraftDynamics object from configuration.

        Returns
        -------
        AircraftDynamics
            Configured dynamics object
        """
        return AircraftDynamics(self.mass, self.inertia)

    def create_aero_model(self):
        """
        Create aerodynamic model from configuration.

        Returns
        -------
        AeroModel
            Configured aerodynamic model
        """
        aero_type = self.aerodynamics.get('type', 'linear')

        if aero_type == 'constant':
            # Constant coefficient model
            return ConstantCoeffModel(
                CL=self.aerodynamics.get('CL', 0.5),
                CD=self.aerodynamics.get('CD', 0.05),
                S_ref=self.S_ref
            )

        elif aero_type == 'linear':
            # Linear stability derivatives
            model = LinearAeroModel(
                S_ref=self.S_ref,
                c_ref=self.c_ref,
                b_ref=self.b_ref,
                rho=self.aerodynamics.get('rho', 0.002377)
            )

            # Set derivatives from config
            derivatives = self.aerodynamics.get('derivatives', {})
            for key, value in derivatives.items():
                if hasattr(model, key):
                    setattr(model, key, value)

            return model

        elif aero_type == 'avl_database':
            # AVL database
            db_file = self.aerodynamics.get('database_file')
            if db_file and Path(db_file).exists():
                return AVLDatabase.from_avl_sweep(
                    db_file,
                    self.S_ref,
                    self.c_ref,
                    self.b_ref
                )
            else:
                raise ValueError(f"AVL database file not found: {db_file}")

        else:
            raise ValueError(f"Unknown aerodynamic model type: {aero_type}")

    def create_propulsion_model(self):
        """
        Create propulsion model from configuration.

        Returns
        -------
        PropulsionModel
            Configured propulsion model
        """
        prop_type = self.propulsion.get('type', 'constant_thrust')

        if prop_type == 'constant_thrust':
            return ConstantThrustModel(
                thrust=self.propulsion.get('max_thrust', 500.0),
                thrust_offset=np.array(self.propulsion.get('offset', [0, 0, 0]))
            )

        elif prop_type == 'propeller':
            return PropellerModel(
                power_max=self.propulsion.get('power_max', 50.0),
                prop_diameter=self.propulsion.get('diameter', 6.0),
                prop_efficiency=self.propulsion.get('efficiency', 0.75)
            )

        else:
            raise ValueError(f"Unknown propulsion model type: {prop_type}")

    def get_initial_state_dict(self) -> Dict[str, Any]:
        """
        Get initial state as dictionary.

        Returns
        -------
        dict
            Initial state parameters
        """
        return self.initial_state.copy()

    def __repr__(self):
        """String representation."""
        return (f"AircraftConfig(name='{self.name}', "
                f"mass={self.mass}, "
                f"S_ref={self.S_ref}, "
                f"units='{self.units}')")


def load_aircraft_config(yaml_file: str) -> AircraftConfig:
    """
    Load aircraft configuration from YAML file.

    Parameters
    ----------
    yaml_file : str
        Path to YAML configuration file

    Returns
    -------
    AircraftConfig
        Loaded aircraft configuration

    Examples
    --------
    >>> config = load_aircraft_config('aircraft/cessna_172.yaml')
    >>> dynamics = config.create_dynamics()
    >>> aero = config.create_aero_model()
    """
    with open(yaml_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    return AircraftConfig(config_dict)


def save_aircraft_config(config: AircraftConfig, yaml_file: str):
    """
    Save aircraft configuration to YAML file.

    Parameters
    ----------
    config : AircraftConfig
        Aircraft configuration to save
    yaml_file : str
        Output YAML file path
    """
    with open(yaml_file, 'w') as f:
        yaml.dump(config.raw_config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {yaml_file}")


def create_example_config() -> Dict[str, Any]:
    """
    Create example aircraft configuration dictionary.

    Returns
    -------
    dict
        Example configuration
    """
    config = {
        'aircraft': {
            'name': 'nTop UAV',
            'units': 'US',  # US Customary units
            'mass': 234.8,  # slugs
            'inertia': {
                'Ixx': 14908.4,  # slug·ft²
                'Iyy': 2318.4,
                'Izz': 17226.9,
                'Ixz': 0.0
            },
            'reference': {
                'S': 199.94,  # ft²
                'b': 19.890,  # ft
                'c': 26.689   # ft
            },
            'aerodynamics': {
                'type': 'linear',
                'rho': 0.002377,  # slugs/ft³ (sea level)
                'derivatives': {
                    'CL_0': 0.2,
                    'CL_alpha': 4.5,
                    'CL_q': 3.5,
                    'CL_elevator': 0.4,
                    'CD_0': 0.03,
                    'CD_alpha2': 0.8,
                    'Cm_0': 0.0,
                    'Cm_alpha': -0.6,
                    'Cm_q': -8.0,
                    'Cm_elevator': -1.2,
                    'CY_beta': -0.3,
                    'Cl_beta': -0.1,
                    'Cl_p': -0.5,
                    'Cl_r': 0.1,
                    'Cl_aileron': 0.15,
                    'Cn_beta': 0.08,
                    'Cn_p': -0.05,
                    'Cn_r': -0.2,
                    'Cn_aileron': -0.02
                }
            },
            'propulsion': {
                'type': 'propeller',
                'power_max': 50.0,  # HP
                'diameter': 6.0,    # ft
                'efficiency': 0.75,
                'offset': [0.0, 0.0, 0.0]  # ft from CG
            },
            'initial_state': {
                'altitude': 5000.0,  # ft
                'airspeed': 200.0,   # ft/s
                'pitch': 2.0,        # deg
                'roll': 0.0,
                'yaw': 0.0
            }
        }
    }

    return config


def test_config():
    """Test configuration system."""
    print("=" * 60)
    print("Configuration System Test")
    print("=" * 60)
    print()

    # Create example config
    config_dict = create_example_config()
    config = AircraftConfig(config_dict)

    print(f"Aircraft: {config.name}")
    print(f"Mass: {config.mass} slugs")
    print(f"S_ref: {config.S_ref} ft²")
    print()

    # Create models
    dynamics = config.create_dynamics()
    print(f"Dynamics: {dynamics.mass} slugs")

    aero = config.create_aero_model()
    print(f"Aero model type: {type(aero).__name__}")

    prop = config.create_propulsion_model()
    print(f"Propulsion type: {type(prop).__name__}")
    print()


if __name__ == "__main__":
    test_config()
