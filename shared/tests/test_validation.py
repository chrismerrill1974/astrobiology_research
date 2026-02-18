"""Tests for validation module."""

import numpy as np
import pytest

from dimensional_opening.validation import (
    ValidationResult, ValidationReport, run_validation, print_validation_table,
    generate_lorenz, generate_rossler, generate_henon,
    generate_uniform_cube, generate_uniform_line, generate_uniform_disk,
    generate_brusselator,
)
from dimensional_opening.correlation_dimension import CorrelationDimension, QualityFlag


class TestGenerators:
    """Tests for trajectory generators."""
    
    def test_lorenz_shape(self):
        traj = generate_lorenz(n_points=1000, transient=100)
        assert traj.shape == (1000, 3)
    
    def test_lorenz_bounded(self):
        traj = generate_lorenz(n_points=5000)
        assert np.all(np.abs(traj[:, :2]) < 50)
        assert np.all(traj[:, 2] > -10)
        assert np.all(traj[:, 2] < 60)
    
    def test_rossler_shape(self):
        traj = generate_rossler(n_points=1000, transient=100)
        assert traj.shape == (1000, 3)
    
    def test_henon_shape(self):
        traj = generate_henon(n_points=1000, transient=100)
        assert traj.shape == (1000, 2)
    
    def test_henon_bounded(self):
        traj = generate_henon(n_points=5000)
        assert np.all(np.abs(traj) < 2)
    
    def test_uniform_cube_shape(self):
        traj = generate_uniform_cube(n_points=1000, dim=3)
        assert traj.shape == (1000, 3)
    
    def test_uniform_cube_range(self):
        traj = generate_uniform_cube(n_points=1000, dim=3)
        assert np.all(traj >= 0)
        assert np.all(traj <= 1)
    
    def test_uniform_line_collinear(self):
        traj = generate_uniform_line(n_points=100)
        assert np.allclose(traj[:, 0], traj[:, 1])
        assert np.allclose(traj[:, 1], traj[:, 2])
    
    def test_uniform_disk_in_plane(self):
        traj = generate_uniform_disk(n_points=1000)
        assert np.allclose(traj[:, 2], 0)
    
    def test_uniform_disk_in_unit_circle(self):
        traj = generate_uniform_disk(n_points=1000)
        radii = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        assert np.all(radii <= 1 + 1e-10)
    
    def test_brusselator_shape(self):
        traj = generate_brusselator(n_points=1000, transient=500)
        assert traj.shape == (1000, 2)
    
    def test_brusselator_positive(self):
        traj = generate_brusselator(n_points=2000)
        assert np.all(traj > 0)
    
    def test_reproducibility(self):
        traj1 = generate_lorenz(n_points=100, seed=42)
        traj2 = generate_lorenz(n_points=100, seed=42)
        assert np.allclose(traj1, traj2)
        traj3 = generate_lorenz(n_points=100, seed=123)
        assert not np.allclose(traj1, traj3)


class TestD2Estimates:
    """Tests for D2 estimates on known systems."""
    
    def test_lorenz_d2(self):
        cd = CorrelationDimension()
        traj = generate_lorenz(n_points=10000, seed=42)
        result = cd.compute(traj, random_state=42)
        assert result.quality != QualityFlag.FAILED
        assert abs(result.D2 - 2.05) / 2.05 < 0.20
    
    def test_henon_d2(self):
        cd = CorrelationDimension()
        traj = generate_henon(n_points=10000, seed=42)
        result = cd.compute(traj, random_state=42)
        assert result.quality != QualityFlag.FAILED
        assert abs(result.D2 - 1.22) / 1.22 < 0.20
    
    def test_uniform_cube_d2(self):
        cd = CorrelationDimension()
        traj = generate_uniform_cube(n_points=5000, dim=3, seed=42)
        result = cd.compute(traj, theiler_window=1, random_state=42)
        assert result.quality != QualityFlag.FAILED
        assert abs(result.D2 - 3.0) / 3.0 < 0.15
    
    def test_uniform_line_d2(self):
        cd = CorrelationDimension()
        traj = generate_uniform_line(n_points=5000, seed=42)
        result = cd.compute(traj, theiler_window=1, random_state=42)
        assert result.quality != QualityFlag.FAILED
        assert abs(result.D2 - 1.0) / 1.0 < 0.15
    
    def test_brusselator_d2(self):
        cd = CorrelationDimension()
        traj = generate_brusselator(n_points=5000, seed=42)
        result = cd.compute(traj, random_state=42)
        assert result.quality != QualityFlag.FAILED
        assert abs(result.D2 - 1.0) / 1.0 < 0.20


class TestValidationReport:
    """Tests for validation report generation."""
    
    def test_run_validation(self):
        report = run_validation(tolerance=0.20, verbose=False, random_state=42)
        assert isinstance(report, ValidationReport)
        assert report.n_total == 7
        assert report.n_passed + report.n_failed == report.n_total
    
    def test_validation_result_fields(self):
        report = run_validation(tolerance=0.20, verbose=False, random_state=42)
        for result in report.results:
            assert isinstance(result, ValidationResult)
            assert result.name != ""
            assert result.expected_D2 > 0
    
    def test_print_validation_table(self):
        report = run_validation(tolerance=0.20, verbose=False, random_state=42)
        table = print_validation_table(report)
        assert "| System |" in table
        assert "Lorenz" in table
    
    def test_validation_to_dict(self):
        report = run_validation(tolerance=0.20, verbose=False, random_state=42)
        d = report.to_dict()
        assert 'n_passed' in d
        assert 'results' in d
        assert len(d['results']) == 7
