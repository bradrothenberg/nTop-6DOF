"""
Phase 4 Integration Tests

Tests for analysis tools:
- Linearization
- Stability analysis
- Mode identification
- Frequency response
"""

import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.stability import LinearizedModel, StabilityAnalyzer, DynamicMode
from src.analysis.frequency import FrequencyAnalyzer
from src.core.state import State
from src.core.dynamics import AircraftDynamics
from src.core.aerodynamics import LinearAeroModel
from src.core.propulsion import ConstantThrustModel, CombinedForceModel
from src.control.trim import TrimSolver


class TestLinearization:
    """Test linearization about trim."""

    def test_linearized_model_creation(self):
        """Test creating a linearized model."""
        A = np.random.randn(13, 13)
        B = np.random.randn(13, 4)

        state = State()
        state.altitude = 5000
        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        model = LinearizedModel(A, B, state, controls)

        assert model.A.shape == (13, 13)
        assert model.B.shape == (13, 4)
        assert model.C.shape == (13, 13)
        assert model.D.shape == (13, 4)
        assert model.n_states == 13
        assert model.n_inputs == 4

    def test_eigenvalue_computation(self):
        """Test eigenvalue computation."""
        # Simple stable system
        A = np.array([[-1, 0], [0, -2]])
        B = np.array([[1], [0]])

        state = State()
        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        # Pad to 13x13
        A_full = np.eye(13) * -0.1
        A_full[0:2, 0:2] = A
        B_full = np.zeros((13, 4))
        B_full[0:2, 0] = B.flatten()

        model = LinearizedModel(A_full, B_full, state, controls)

        eigs = model.eigenvalues()
        assert eigs.shape == (13,)

        # Check if stable
        assert model.is_stable()

    def test_linearization_about_trim(self):
        """Test linearization about a trim point."""
        # Setup aircraft
        mass = 234.8
        inertia = np.diag([14908.4, 2318.4, 17226.9])
        dynamics = AircraftDynamics(mass, inertia)

        aero = LinearAeroModel(S_ref=199.94, c_ref=26.689, b_ref=19.890)
        aero.CL_0 = 0.2
        aero.CL_alpha = 4.5
        aero.CD_0 = 0.03
        aero.Cm_alpha = -0.6

        prop = ConstantThrustModel(thrust=600.0)
        force_model = CombinedForceModel(aero, prop)

        def dynamics_func(state, controls):
            throttle = controls.get('throttle', 0.5)
            return dynamics.state_derivative(state, lambda s: force_model(s, throttle))

        # Create analyzer
        analyzer = StabilityAnalyzer(dynamics_func)

        # Trim state
        trim_state = State()
        trim_state.position = np.array([0, 0, -5000])
        trim_state.velocity_body = np.array([200, 0, 0])
        trim_state.set_euler_angles(0, np.radians(2), 0)

        trim_controls = {'elevator': np.radians(2), 'aileron': 0.0,
                          'rudder': 0.0, 'throttle': 0.6}

        # Linearize
        linear_model = analyzer.linearize(trim_state, trim_controls, eps=1e-5)

        assert linear_model.A.shape == (13, 13)
        assert linear_model.B.shape == (13, 4)
        assert not np.all(linear_model.A == 0)  # Not all zeros
        assert not np.all(linear_model.B == 0)  # Not all zeros


class TestStabilityAnalysis:
    """Test stability analysis tools."""

    def test_mode_identification(self):
        """Test dynamic mode identification."""
        # Create simple oscillatory system (like short period)
        # x_dot = -0.5*x + 2*y
        # y_dot = -2*x - 0.5*y
        # Eigenvalues: -0.5 ± 2j

        A = np.eye(13) * -0.1
        A[3, 3] = -0.5  # u
        A[3, 5] = 2.0   # coupling to w
        A[5, 3] = -2.0
        A[5, 5] = -0.5  # w
        A[11, 5] = 1.0  # q coupled to w

        B = np.zeros((13, 4))

        state = State()
        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        model = LinearizedModel(A, B, state, controls)

        # Create analyzer
        def dummy_func(s, c):
            return np.zeros(13)

        analyzer = StabilityAnalyzer(dummy_func)

        # Identify modes
        modes = analyzer.identify_modes(model)

        assert len(modes) > 0
        assert all(isinstance(m, DynamicMode) for m in modes)

        # Check mode properties
        for mode in modes:
            assert hasattr(mode, 'name')
            assert hasattr(mode, 'eigenvalue')
            assert hasattr(mode, 'damping_ratio')
            assert hasattr(mode, 'natural_frequency')

    def test_stability_report(self, capsys):
        """Test stability report generation."""
        # Simple stable system
        A = np.eye(13) * -0.5
        B = np.zeros((13, 4))

        state = State()
        state.altitude = 5000
        state.velocity_body = np.array([200, 0, 0])

        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        model = LinearizedModel(A, B, state, controls)

        def dummy_func(s, c):
            return np.zeros(13)

        analyzer = StabilityAnalyzer(dummy_func)

        # Print report
        analyzer.print_stability_report(model)

        # Capture output
        captured = capsys.readouterr()
        assert "STABILITY ANALYSIS REPORT" in captured.out
        assert "STABLE" in captured.out


class TestFrequencyAnalysis:
    """Test frequency response analysis."""

    def test_frequency_analyzer_creation(self):
        """Test creating frequency analyzer."""
        A = np.eye(13) * -1.0
        B = np.zeros((13, 4))
        B[3, 0] = 1.0  # Elevator affects u

        state = State()
        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        model = LinearizedModel(A, B, state, controls)
        analyzer = FrequencyAnalyzer(model)

        assert analyzer.linear_model is model
        assert analyzer.sys is not None

    def test_bode_computation(self):
        """Test Bode plot data computation."""
        # Simple first-order system
        A = np.eye(13) * -1.0
        B = np.zeros((13, 4))
        B[3, 0] = 1.0

        state = State()
        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        model = LinearizedModel(A, B, state, controls)
        analyzer = FrequencyAnalyzer(model)

        omega, magnitude, phase = analyzer.bode(input_idx=0, output_idx=3)

        assert len(omega) > 0
        assert len(magnitude) == len(omega)
        assert len(phase) == len(omega)

        # Check magnitude decreases with frequency (first-order system)
        assert magnitude[0] > magnitude[-1]

    def test_step_response(self):
        """Test step response computation."""
        # Simple stable system
        A = np.eye(13) * -1.0
        B = np.zeros((13, 4))
        B[3, 0] = 1.0

        state = State()
        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        model = LinearizedModel(A, B, state, controls)
        analyzer = FrequencyAnalyzer(model)

        t, y = analyzer.step_response(input_idx=0, output_idx=3, t_final=10.0)

        assert len(t) > 0
        assert len(y) == len(t)

        # Response should settle for stable system
        assert np.abs(y[-1]) < 10  # Reasonable final value

    def test_impulse_response(self):
        """Test impulse response computation."""
        A = np.eye(13) * -1.0
        B = np.zeros((13, 4))
        B[3, 0] = 1.0

        state = State()
        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        model = LinearizedModel(A, B, state, controls)
        analyzer = FrequencyAnalyzer(model)

        t, y = analyzer.impulse_response(input_idx=0, output_idx=3, t_final=10.0)

        assert len(t) > 0
        assert len(y) == len(t)

        # Impulse response should decay for stable system
        assert np.abs(y[-1]) < np.abs(y[1])  # Decaying


class TestIntegration:
    """Integration tests combining multiple Phase 4 components."""

    def test_complete_stability_analysis(self):
        """Test complete stability analysis workflow."""
        # Setup aircraft
        mass = 234.8
        inertia = np.diag([14908.4, 2318.4, 17226.9])
        dynamics = AircraftDynamics(mass, inertia)

        aero = LinearAeroModel(S_ref=199.94, c_ref=26.689, b_ref=19.890)
        aero.CL_0 = 0.2
        aero.CL_alpha = 4.5
        aero.CD_0 = 0.03
        aero.CD_alpha2 = 0.5
        aero.Cm_alpha = -0.6
        aero.Cm_q = -8.0

        prop = ConstantThrustModel(thrust=600.0)
        force_model = CombinedForceModel(aero, prop)

        def dynamics_func(state, controls):
            throttle = controls.get('throttle', 0.5)
            return dynamics.state_derivative(state, lambda s: force_model(s, throttle))

        # Trim
        trim_state = State()
        trim_state.position = np.array([0, 0, -5000])
        trim_state.velocity_body = np.array([200, 0, 0])
        trim_state.set_euler_angles(0, np.radians(2), 0)

        trim_controls = {'elevator': np.radians(2), 'aileron': 0.0,
                          'rudder': 0.0, 'throttle': 0.6}

        # Stability analysis
        analyzer = StabilityAnalyzer(dynamics_func)
        linear_model = analyzer.linearize(trim_state, trim_controls)

        # Check stability
        assert isinstance(linear_model, LinearizedModel)
        assert linear_model.A.shape == (13, 13)

        # Identify modes
        modes = analyzer.identify_modes(linear_model)
        assert len(modes) > 0

    def test_frequency_analysis_workflow(self):
        """Test frequency analysis workflow."""
        # Simple system
        A = np.eye(13) * -1.0
        A[3, 3] = -2.0
        B = np.zeros((13, 4))
        B[3, 0] = 1.0

        state = State()
        controls = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

        model = LinearizedModel(A, B, state, controls)

        # Frequency analysis
        freq_analyzer = FrequencyAnalyzer(model)

        # Bode
        omega, mag, phase = freq_analyzer.bode(0, 3)
        assert len(omega) > 0

        # Step response
        t, y = freq_analyzer.step_response(0, 3)
        assert len(t) > 0

        # All checks passed
        assert True


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
