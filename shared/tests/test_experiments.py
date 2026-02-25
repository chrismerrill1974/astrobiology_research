"""
Phase 3.1: Smoke tests for experiments.py module.

experiments.py is the largest module (~1200 lines) with zero unit tests.
It contains experiment runners and statistical analysis code (Mann-Whitney U,
KS test, Fisher exact, bootstrap CIs, permutation tests) that produce the
tables and p-values reported in the papers.

These smoke tests verify that each function runs without crashing and
returns results with correct structure and field types. They use minimal
parameters (n_networks=2, n_trajectories=2) to keep runtime manageable.
"""

import numpy as np
import pytest
import json

from dimensional_opening.experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_paper2,
    _compute_paper2_statistics,
    ExperimentResult,
    ProgressiveResult,
    DrivingResult,
    Paper2Result,
)


@pytest.mark.slow
class TestExperiment1Smoke:
    """Smoke test for run_experiment_1()."""

    def test_returns_correct_type_and_fields(self):
        """run_experiment_1 with minimal params should return ExperimentResult."""
        result = run_experiment_1(
            n_networks=2,
            n_added_control=1,
            n_autocatalytic_test=1,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, ExperimentResult)
        assert result.n_networks == 4  # 2 control + 2 test
        assert isinstance(result.group1_n, int)
        assert isinstance(result.group2_n, int)
        assert isinstance(result.group1_eta_median, float)
        assert isinstance(result.group2_eta_median, float)
        assert isinstance(result.delta_eta, float)
        assert isinstance(result.results, list)
        assert isinstance(result.timestamp, str)
        assert isinstance(result.parameters, dict)


@pytest.mark.slow
class TestExperiment2Smoke:
    """Smoke test for run_experiment_2()."""

    def test_returns_correct_type_and_fields(self):
        """run_experiment_2 with minimal params should return ProgressiveResult."""
        result = run_experiment_2(
            n_trajectories=2,
            n_steps=2,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, ProgressiveResult)
        assert result.n_trajectories == 2
        # n_steps=2 means baseline + 2 additions = 3 steps total
        assert len(result.n_autocatalytic_range) == 3
        assert len(result.eta_medians) == 3
        assert len(result.eta_lower) == 3
        assert len(result.eta_upper) == 3
        assert isinstance(result.eta_slope, float)


@pytest.mark.slow
class TestExperiment3Smoke:
    """Smoke test for run_experiment_3()."""

    def test_returns_correct_type_and_fields(self):
        """run_experiment_3 with minimal params should return DrivingResult."""
        result = run_experiment_3(
            dilution_rates=[0.1],
            n_networks=2,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, DrivingResult)
        assert result.dilution_rates == [0.1]
        assert len(result.eta_medians) == 1
        assert len(result.eta_iqrs) == 1
        assert len(result.n_successful) == 1
        assert isinstance(result.n_successful[0], int)


@pytest.mark.slow
class TestExperimentPaper2Smoke:
    """Smoke test for run_experiment_paper2()."""

    def test_returns_correct_type_and_fields(self):
        """run_experiment_paper2 with minimal params should return Paper2Result."""
        result = run_experiment_paper2(
            n_trajectories=2,
            n_steps=2,
            max_candidates=10,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, Paper2Result)
        assert result.n_trajectories == 2
        assert result.n_steps == 2
        assert result.max_candidates == 10
        # Matrices should have shape (n_trajectories, n_steps+1)
        assert result.group_a_eta_matrix.shape == (2, 3)
        assert result.group_b_eta_matrix.shape == (2, 3)
        # Statistical fields should be populated
        assert len(result.mann_whitney_p) == 3
        assert len(result.ks_p) == 3
        assert len(result.fisher_p) == 3
        assert isinstance(result.group_a_slope, float)
        assert isinstance(result.group_b_slope, float)
        assert isinstance(result.slope_difference, float)
        assert isinstance(result.early_termination_count, int)
        assert 0.0 <= result.early_termination_rate <= 1.0


class TestComputePaper2Statistics:
    """Tests for _compute_paper2_statistics() with known inputs."""

    def test_pvalues_in_valid_range(self):
        """p-values should be in [0, 1]."""
        rng = np.random.RandomState(42)
        n_traj, n_steps = 20, 3
        group_a = rng.uniform(0.1, 0.5, size=(n_traj, n_steps))
        group_b = rng.uniform(0.2, 0.6, size=(n_traj, n_steps))

        stats = _compute_paper2_statistics(
            group_a, group_b, n_bootstrap=200, seed=42,
        )

        for p in stats['mann_whitney_p']:
            if p is not None:
                assert 0.0 <= p <= 1.0, f"Mann-Whitney p={p} out of range"
        for p in stats['ks_p']:
            if p is not None:
                assert 0.0 <= p <= 1.0, f"KS p={p} out of range"
        for p in stats['fisher_p']:
            if p is not None:
                assert 0.0 <= p <= 1.0, f"Fisher p={p} out of range"

    def test_identical_groups_high_pvalues(self):
        """Identical groups should give p-values close to 1.0 (no difference)."""
        rng = np.random.RandomState(42)
        n_traj, n_steps = 30, 3
        data = rng.uniform(0.1, 0.5, size=(n_traj, n_steps))

        stats = _compute_paper2_statistics(
            data, data.copy(), n_bootstrap=200, seed=42,
        )

        for p in stats['mann_whitney_p']:
            if p is not None:
                assert p > 0.5, (
                    f"Identical groups should give high p, got {p:.4f}"
                )

    def test_empty_arrays_no_crash(self):
        """Empty eta arrays should not crash."""
        # All NaN — simulates the case where every simulation failed
        group_a = np.full((5, 3), np.nan)
        group_b = np.full((5, 3), np.nan)

        stats = _compute_paper2_statistics(
            group_a, group_b, n_bootstrap=100, seed=42,
        )

        # Should return NaN medians and None p-values, not crash
        assert all(np.isnan(m) for m in stats['group_a_medians'])
        assert all(np.isnan(m) for m in stats['group_b_medians'])
        assert all(p is None for p in stats['mann_whitney_p'])


class TestPaper2ResultSerialization:
    """Paper2Result.to_dict() should produce a JSON-serializable dict."""

    def test_to_dict_round_trip(self):
        """to_dict() should be JSON-serializable with all expected keys."""
        rng = np.random.RandomState(42)
        n_traj, n_steps = 5, 2
        total_steps = n_steps + 1

        result = Paper2Result(
            name="test",
            hypothesis="test hypothesis",
            n_trajectories=n_traj,
            n_steps=n_steps,
            max_candidates=10,
            group_a_eta_matrix=rng.uniform(0, 1, (n_traj, total_steps)),
            group_b_eta_matrix=rng.uniform(0, 1, (n_traj, total_steps)),
            group_a_valid_counts=[n_traj] * total_steps,
            group_b_valid_counts=[n_traj] * total_steps,
            group_a_medians=[0.3] * total_steps,
            group_a_iqr=[0.1] * total_steps,
            group_b_medians=[0.4] * total_steps,
            group_b_iqr=[0.1] * total_steps,
            group_a_slope=0.01,
            group_b_slope=0.02,
            group_a_slope_ci=(-0.01, 0.03),
            group_b_slope_ci=(0.0, 0.04),
            group_a_tail_prob=[0.1] * total_steps,
            group_b_tail_prob=[0.2] * total_steps,
            mann_whitney_p=[0.5] * total_steps,
            ks_p=[0.6] * total_steps,
            fisher_p=[0.7] * total_steps,
            acceptance_rates_per_step=[[0.5, 0.4]] * n_traj,
            acceptance_rate_medians=[0.5, 0.4],
            early_termination_count=1,
            early_termination_rate=0.2,
            cv_d2_correlation=0.1,
            cv_eta_correlation=-0.05,
            slope_difference=0.01,
            slope_difference_ci=(-0.02, 0.04),
            slope_difference_perm_p=0.3,
            group_b_filter_results=[],
            parameters={'seed': 42},
            timestamp="2026-02-25T00:00:00",
        )

        d = result.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Key fields should be present
        for key in [
            'name', 'hypothesis', 'n_trajectories', 'n_steps',
            'group_a_eta_matrix', 'group_b_eta_matrix',
            'mann_whitney_p', 'ks_p', 'fisher_p',
            'group_a_slope', 'group_b_slope',
            'slope_difference', 'slope_difference_perm_p',
            'parameters', 'timestamp',
        ]:
            assert key in d, f"Missing key: {key}"
