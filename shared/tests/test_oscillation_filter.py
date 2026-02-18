"""
Tests for the oscillation filter module (Paper 2, Phase 2).

Tests the locked oscillation-preservation criterion:
  1. Pure Brusselator passes
  2. Known fixed-point system fails
  3. Known runaway system fails
  4. Borderline CV cases (0.025 fails, 0.035 passes)
  5. Smoothing prevents noise false positives
  6. passes_oscillation_filter convenience function works
"""

import numpy as np
import pytest

from dimensional_opening.oscillation_filter import (
    OscillationResult,
    check_oscillation,
    passes_oscillation_filter,
    _count_sign_changes,
)
from dimensional_opening.network_generator import (
    NetworkGenerator,
    BRUSSELATOR,
    GeneratedNetwork,
)


class TestCountSignChanges:
    """Tests for the sign-change counting helper."""

    def test_no_changes(self):
        assert _count_sign_changes(np.array([1.0, 2.0, 3.0])) == 0

    def test_one_change(self):
        assert _count_sign_changes(np.array([1.0, -1.0])) == 1

    def test_multiple_changes(self):
        # Oscillating: +, -, +, -, +, - = 5 changes
        x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        assert _count_sign_changes(x) == 5

    def test_with_zeros(self):
        # Zeros are skipped, so [1, 0, -1] has 1 sign change
        x = np.array([1.0, 0.0, -1.0])
        assert _count_sign_changes(x) == 1

    def test_empty(self):
        assert _count_sign_changes(np.array([])) == 0

    def test_single_element(self):
        assert _count_sign_changes(np.array([1.0])) == 0


class TestCheckOscillation:
    """Tests for the check_oscillation function."""

    def _make_oscillating_trajectory(
        self, n_points=2000, freq=0.5, amplitude=1.0, mean=2.0
    ):
        """Create a synthetic oscillating trajectory."""
        t = np.linspace(0, 100, n_points)
        c = mean + amplitude * np.sin(2 * np.pi * freq * t)
        # Add a second non-oscillating species
        c2 = np.ones(n_points) * 1.5
        return t, np.column_stack([c, c2]), ["X", "Y"]

    def _make_fixed_point_trajectory(self, n_points=2000):
        """Create a trajectory that decays to a fixed point."""
        t = np.linspace(0, 100, n_points)
        # Exponential decay to steady state with tiny noise
        c = 1.0 + 0.5 * np.exp(-0.1 * t)
        noise = np.random.RandomState(42).normal(0, 0.001, n_points)
        c += noise
        c2 = 0.5 * np.ones(n_points)
        return t, np.column_stack([c, c2]), ["X", "Y"]

    def _make_runaway_trajectory(self, n_points=2000):
        """Create a trajectory with runaway growth."""
        t = np.linspace(0, 100, n_points)
        c = np.exp(0.05 * t)  # Exponential growth
        c2 = np.ones(n_points)
        return t, np.column_stack([c, c2]), ["X", "Y"]

    def test_oscillating_passes(self):
        t, c, names = self._make_oscillating_trajectory()
        result = check_oscillation(c, t, names)
        assert result.passes is True
        assert result.cv > 0.03
        assert result.sign_changes >= 5
        assert 0.2 < result.boundedness_ratio < 5.0

    def test_fixed_point_fails(self):
        t, c, names = self._make_fixed_point_trajectory()
        result = check_oscillation(c, t, names)
        assert result.passes is False

    def test_runaway_fails(self):
        t, c, names = self._make_runaway_trajectory()
        result = check_oscillation(c, t, names)
        # Should fail on boundedness (exponential growth means c(100)/c(50) >> 5)
        assert result.passes is False

    def test_food_species_excluded(self):
        """Food species should be excluded from evaluation."""
        t, c, names = self._make_oscillating_trajectory()
        # If the only oscillating species is labeled as food, filter should fail
        result = check_oscillation(c, t, names, food_species=["X"])
        # Y is non-oscillating, so should fail
        assert result.passes is False

    def test_borderline_cv_below_threshold(self):
        """CV = 0.025 should fail (below 0.03 threshold)."""
        t = np.linspace(0, 100, 2000)
        mean = 10.0
        # CV = std/mean, so std = 0.025 * 10 = 0.25
        # For sine wave, std = amplitude / sqrt(2)
        # So amplitude = 0.25 * sqrt(2) ~ 0.354
        amplitude = 0.025 * mean * np.sqrt(2)
        c = mean + amplitude * np.sin(2 * np.pi * 0.5 * t)
        c = np.column_stack([c])
        result = check_oscillation(c, t, ["X"])
        assert result.passes is False
        assert result.cv < 0.03

    def test_borderline_cv_above_threshold(self):
        """CV = 0.035 should pass (above 0.03 threshold)."""
        t = np.linspace(0, 100, 2000)
        mean = 10.0
        amplitude = 0.035 * mean * np.sqrt(2)
        c = mean + amplitude * np.sin(2 * np.pi * 0.5 * t)
        c = np.column_stack([c])
        result = check_oscillation(c, t, ["X"])
        assert result.passes is True
        assert result.cv > 0.03

    def test_smoothing_prevents_noise_false_positive(self):
        """
        High-frequency noise on a fixed-point signal should NOT pass.
        Without smoothing, random noise might generate many sign changes.
        """
        t = np.linspace(0, 100, 2000)
        # Steady state with noise that has small CV
        rng = np.random.RandomState(123)
        c = 5.0 + rng.normal(0, 0.01, 2000)  # CV ~ 0.002, well below 0.03
        c = np.column_stack([c])
        result = check_oscillation(c, t, ["X"])
        assert result.passes is False

    def test_waste_species_excluded(self):
        """Monotonically increasing (waste) species should be skipped."""
        t = np.linspace(0, 100, 2000)
        # X oscillates, W monotonically increases
        c_x = 2.0 + 1.0 * np.sin(2 * np.pi * 0.5 * t)
        c_w = np.linspace(0, 10, 2000)
        c = np.column_stack([c_x, c_w])
        result = check_oscillation(c, t, ["X", "W"])
        assert result.passes is True
        assert result.best_species_name == "X"

    def test_diagnostic_metrics_reported_on_failure(self):
        """Even when filter fails, diagnostic metrics should be populated."""
        t, c, names = self._make_fixed_point_trajectory()
        result = check_oscillation(c, t, names)
        assert result.passes is False
        assert result.best_species_idx >= 0
        # Should still have some metrics
        assert isinstance(result.cv, float)
        assert isinstance(result.sign_changes, int)


class TestPassesOscillationFilter:
    """Tests for the passes_oscillation_filter convenience function."""

    def test_pure_brusselator_passes(self):
        """The pure Brusselator template should pass the oscillation filter."""
        gen = NetworkGenerator(template='brusselator', seed=42)
        net = gen.generate_baseline()
        result = passes_oscillation_filter(net, dilution_rate=0.1)
        assert result.passes is True
        assert result.cv > 0.03
        assert result.sign_changes >= 5

    def test_result_is_oscillation_result(self):
        """Should return an OscillationResult dataclass."""
        gen = NetworkGenerator(template='brusselator', seed=42)
        net = gen.generate_baseline()
        result = passes_oscillation_filter(net)
        assert isinstance(result, OscillationResult)

    def test_brusselator_with_small_additions_often_passes(self):
        """
        Brusselator + 1 random autocatalytic addition should still sometimes
        pass (not every addition kills oscillation).
        """
        gen = NetworkGenerator(template='brusselator', seed=42)
        pass_count = 0
        for i in range(10):
            gen_i = NetworkGenerator(template='brusselator', seed=42 + i)
            net = gen_i.generate_test(n_autocatalytic=1, n_random=0)
            result = passes_oscillation_filter(net, dilution_rate=0.1)
            if result.passes:
                pass_count += 1
        # At least some should pass (not all additions kill oscillation)
        assert pass_count >= 1


class TestFilterEdgeCases:
    """Edge case tests."""

    def test_all_zero_concentrations(self):
        """Trajectory of all zeros should fail gracefully."""
        t = np.linspace(0, 100, 1000)
        c = np.zeros((1000, 2))
        result = check_oscillation(c, t, ["X", "Y"])
        assert result.passes is False

    def test_very_short_trajectory(self):
        """Very short trajectory should fail gracefully."""
        t = np.linspace(0, 100, 10)
        c = np.random.rand(10, 2)
        result = check_oscillation(c, t, ["X", "Y"])
        # Should not crash
        assert isinstance(result, OscillationResult)

    def test_single_species(self):
        """Single oscillating species should work."""
        t = np.linspace(0, 100, 2000)
        c = 2.0 + 1.0 * np.sin(2 * np.pi * 0.5 * t)
        c = c.reshape(-1, 1)
        result = check_oscillation(c, t, ["X"])
        assert result.passes is True
