"""
US Standard Atmosphere 1976 Model

Provides atmospheric properties as a function of altitude:
- Temperature
- Pressure
- Density
- Speed of sound
- Viscosity

Units: US Customary (feet, slugs, lbf, Rankine)
"""

import numpy as np


class StandardAtmosphere:
    """
    US Standard Atmosphere 1976 model.

    Provides atmospheric properties at any altitude from sea level to 80,000 ft.

    Parameters
    ----------
    altitude : float
        Geometric altitude in feet above MSL

    Attributes
    ----------
    temperature : float
        Static temperature (Rankine)
    pressure : float
        Static pressure (lbf/ft²)
    density : float
        Air density (slugs/ft³)
    speed_of_sound : float
        Speed of sound (ft/s)
    kinematic_viscosity : float
        Kinematic viscosity (ft²/s)
    dynamic_viscosity : float
        Dynamic viscosity (slug/(ft·s))

    Notes
    -----
    Model covers three atmospheric layers:
    - Troposphere: 0 - 36,089 ft (temperature decreases linearly)
    - Lower Stratosphere: 36,089 - 65,617 ft (isothermal)
    - Upper Stratosphere: 65,617 - 80,000 ft (temperature increases)

    Reference: U.S. Standard Atmosphere, 1976, NOAA/NASA/USAF
    """

    # Sea level conditions
    T0 = 518.67  # Rankine (59°F)
    P0 = 2116.22  # lbf/ft² (14.696 psi)
    rho0 = 0.002377  # slugs/ft³

    # Gas constant for air
    R = 1716.59  # ft·lbf/(slug·°R)

    # Ratio of specific heats
    gamma = 1.4

    # Sutherland's constants for viscosity
    S = 198.72  # Rankine
    T_ref = 518.67  # Rankine
    mu_ref = 3.7373e-7  # slug/(ft·s)

    # Layer boundaries (geometric altitude, ft)
    h_trop = 36089.0  # Tropopause
    h_strat1 = 65617.0  # Lower stratosphere boundary

    # Temperature lapse rates (°R/ft)
    lapse_trop = -0.00356616  # Troposphere: -3.566°R/1000ft
    lapse_strat1 = 0.0  # Lower stratosphere: isothermal
    lapse_strat2 = 0.00054864  # Upper stratosphere: +0.549°R/1000ft

    def __init__(self, altitude: float = 0.0):
        """
        Initialize atmosphere at specified altitude.

        Parameters
        ----------
        altitude : float, optional
            Geometric altitude in feet (default: 0.0, sea level)
        """
        self.altitude = altitude
        self._compute_properties()

    def _compute_properties(self):
        """Compute all atmospheric properties at current altitude."""
        h = self.altitude

        # Determine which atmospheric layer
        if h <= self.h_trop:
            # Troposphere
            self.temperature = self.T0 + self.lapse_trop * h

            # Pressure from hydrostatic equation with linear temperature
            theta = self.temperature / self.T0
            exponent = -32.174 / (self.lapse_trop * self.R)  # g / (L * R)
            self.pressure = self.P0 * theta**exponent

        elif h <= self.h_strat1:
            # Lower stratosphere (isothermal)
            self.temperature = self.T0 + self.lapse_trop * self.h_trop

            # Temperature at tropopause
            T_trop = self.T0 + self.lapse_trop * self.h_trop
            theta_trop = T_trop / self.T0
            exponent_trop = -32.174 / (self.lapse_trop * self.R)
            P_trop = self.P0 * theta_trop**exponent_trop

            # Pressure from isothermal layer
            exponent = -32.174 * (h - self.h_trop) / (self.R * self.temperature)
            self.pressure = P_trop * np.exp(exponent)

        else:
            # Upper stratosphere
            # Temperature at lower stratosphere boundary
            T_strat1 = self.T0 + self.lapse_trop * self.h_trop

            self.temperature = T_strat1 + self.lapse_strat2 * (h - self.h_strat1)

            # Pressure at lower stratosphere boundary
            theta_trop = T_strat1 / self.T0
            exponent_trop = -32.174 / (self.lapse_trop * self.R)
            P_trop = self.P0 * theta_trop**exponent_trop

            exponent_iso = -32.174 * (self.h_strat1 - self.h_trop) / (self.R * T_strat1)
            P_strat1 = P_trop * np.exp(exponent_iso)

            # Pressure in upper stratosphere
            theta = self.temperature / T_strat1
            exponent = -32.174 / (self.lapse_strat2 * self.R)
            self.pressure = P_strat1 * theta**exponent

        # Density from ideal gas law
        self.density = self.pressure / (self.R * self.temperature)

        # Speed of sound
        self.speed_of_sound = np.sqrt(self.gamma * self.R * self.temperature)

        # Viscosity (Sutherland's formula)
        self.dynamic_viscosity = (self.mu_ref * (self.temperature / self.T_ref)**1.5 *
                                   (self.T_ref + self.S) / (self.temperature + self.S))
        self.kinematic_viscosity = self.dynamic_viscosity / self.density

    def update(self, altitude: float):
        """
        Update atmospheric properties for new altitude.

        Parameters
        ----------
        altitude : float
            New geometric altitude in feet
        """
        self.altitude = altitude
        self._compute_properties()

    def get_properties(self) -> dict:
        """
        Get all atmospheric properties as dictionary.

        Returns
        -------
        dict
            Dictionary containing all atmospheric properties
        """
        return {
            'altitude': self.altitude,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'density': self.density,
            'speed_of_sound': self.speed_of_sound,
            'dynamic_viscosity': self.dynamic_viscosity,
            'kinematic_viscosity': self.kinematic_viscosity,
            'temperature_F': self.temperature - 459.67,
            'pressure_psi': self.pressure / 144.0,
            'mach_1': self.speed_of_sound
        }

    def get_mach_number(self, velocity: float) -> float:
        """
        Compute Mach number for given velocity.

        Parameters
        ----------
        velocity : float
            True airspeed in ft/s

        Returns
        -------
        float
            Mach number
        """
        return velocity / self.speed_of_sound

    def get_dynamic_pressure(self, velocity: float) -> float:
        """
        Compute dynamic pressure.

        Parameters
        ----------
        velocity : float
            True airspeed in ft/s

        Returns
        -------
        float
            Dynamic pressure q = 0.5 * rho * V²  (lbf/ft²)
        """
        return 0.5 * self.density * velocity**2

    def get_reynolds_number(self, velocity: float, length: float) -> float:
        """
        Compute Reynolds number.

        Parameters
        ----------
        velocity : float
            True airspeed in ft/s
        length : float
            Reference length (e.g., chord) in ft

        Returns
        -------
        float
            Reynolds number (dimensionless)
        """
        return velocity * length / self.kinematic_viscosity

    def __repr__(self):
        """String representation."""
        return (f"StandardAtmosphere(altitude={self.altitude:.0f} ft, "
                f"T={self.temperature-459.67:.1f}°F, "
                f"P={self.pressure/144:.2f} psi, "
                f"rho={self.density:.6f} slug/ft³)")

    @staticmethod
    def get_pressure_altitude(pressure: float) -> float:
        """
        Compute pressure altitude from static pressure.

        Parameters
        ----------
        pressure : float
            Static pressure (lbf/ft²)

        Returns
        -------
        float
            Pressure altitude (ft)

        Notes
        -----
        Valid only in troposphere (below 36,089 ft).
        Uses inverse of tropospheric pressure equation.
        """
        theta = (pressure / StandardAtmosphere.P0)**(
            -StandardAtmosphere.lapse_trop * StandardAtmosphere.R / 32.174)
        h = (theta * StandardAtmosphere.T0 - StandardAtmosphere.T0) / StandardAtmosphere.lapse_trop
        return h

    @staticmethod
    def get_density_altitude(density: float) -> float:
        """
        Compute density altitude from air density.

        Parameters
        ----------
        density : float
            Air density (slugs/ft³)

        Returns
        -------
        float
            Density altitude (ft)

        Notes
        -----
        Valid only in troposphere (below 36,089 ft).
        """
        sigma = density / StandardAtmosphere.rho0
        h = StandardAtmosphere.T0 / (-StandardAtmosphere.lapse_trop) * (1 - sigma**(
            -StandardAtmosphere.lapse_trop * StandardAtmosphere.R / 32.174))
        return h


def test_atmosphere():
    """Test atmosphere model with known values."""
    print("=" * 60)
    print("US Standard Atmosphere 1976 - Test")
    print("=" * 60)
    print()

    test_altitudes = [0, 5000, 10000, 20000, 36089, 50000]

    print(f"{'Alt (ft)':<10} {'T (F)':<10} {'P (psi)':<12} {'rho (slug/ft3)':<15} {'a (ft/s)':<10}")
    print("-" * 60)

    for alt in test_altitudes:
        atm = StandardAtmosphere(alt)
        props = atm.get_properties()

        print(f"{alt:<10.0f} {props['temperature_F']:<10.1f} {props['pressure_psi']:<12.3f} "
              f"{props['density']:<15.6f} {props['speed_of_sound']:<10.1f}")

    print()
    print("Expected values at sea level:")
    print("  T = 59.0 F, P = 14.696 psi, rho = 0.002377 slug/ft3, a = 1116.4 ft/s")
    print()

    # Test dynamic/kinematic properties
    print("Properties at 10,000 ft, V=200 ft/s, chord=10 ft:")
    atm = StandardAtmosphere(10000)
    print(f"  Mach number: {atm.get_mach_number(200):.3f}")
    print(f"  Dynamic pressure: {atm.get_dynamic_pressure(200):.2f} lbf/ft²")
    print(f"  Reynolds number: {atm.get_reynolds_number(200, 10):.2e}")
    print()


if __name__ == "__main__":
    test_atmosphere()
