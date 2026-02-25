"""
Tests for correlation dimension module.

Step 4 test cases from the plan:
1. Lorenz attractor: D2 ≈ 2.05 (known chaotic attractor)
2. Hénon map: D2 ≈ 1.22 (known chaotic map)
3. iid point clouds: D2 = embedding dimension (geometry sanity check)
4. Quality flag behavior
5. Theiler window estimation
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dimensional_opening.correlation_dimension import (
    CorrelationDimension,
    CorrelationDimensionResult,
    EnsembleResult,
    QualityFlag,
    compute_activation_ratio,
    compute_D2_ensemble,
)


def generate_lorenz(n_points=5000, dt=0.01, discard=1000):
    """Generate Lorenz attractor trajectory."""
    # Lorenz parameters
    sigma, rho, beta = 10.0, 28.0, 8/3
    
    # Initial condition
    x, y, z = 1.0, 1.0, 1.0
    
    trajectory = []
    for i in range(n_points + discard):
        # Euler integration (simple but sufficient for testing)
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x, y, z = x + dx, y + dy, z + dz
        
        if i >= discard:
            trajectory.append([x, y, z])
    
    return np.array(trajectory)


def generate_henon(n_points=5000, discard=1000):
    """Generate Hénon map trajectory."""
    a, b = 1.4, 0.3
    
    x, y = 0.1, 0.1
    trajectory = []
    
    for i in range(n_points + discard):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        
        if i >= discard:
            trajectory.append([x, y])
    
    return np.array(trajectory)


def generate_uniform_cube(n_points=2000, dim=3):
    """Generate iid uniform points in a hypercube."""
    return np.random.uniform(0, 1, size=(n_points, dim))


def generate_limit_cycle(n_points=2000, dt=0.01):
    """Generate a simple limit cycle (circle)."""
    omega = 1.0
    t = np.arange(n_points) * dt
    x = np.cos(omega * t)
    y = np.sin(omega * t)
    return np.column_stack([x, y])


class TestCorrelationDimension:
    """Tests for CorrelationDimension class."""
    
    def setup_method(self):
        self.cd = CorrelationDimension()
    
    # -------------------------------------------------------------------------
    # Test Case 1: Lorenz attractor D2 ≈ 2.05
    # -------------------------------------------------------------------------
    
    def test_lorenz_dimension(self):
        """Lorenz attractor should have D2 ≈ 2.05."""
        np.random.seed(42)
        traj = generate_lorenz(n_points=8000, discard=2000)
        
        result = self.cd.compute(traj)
        
        # Lorenz D2 is approximately 2.05
        # Allow some tolerance due to finite data
        assert result.quality != QualityFlag.FAILED, f"Failed: {result.failure_reason}"
        assert 1.7 < result.D2 < 2.5, f"D2={result.D2}, expected ~2.05"
    
    # -------------------------------------------------------------------------
    # Test Case 2: Hénon map D2 ≈ 1.22
    # -------------------------------------------------------------------------
    
    def test_henon_dimension(self):
        """Hénon map should have D2 ≈ 1.22."""
        np.random.seed(42)
        traj = generate_henon(n_points=10000, discard=1000)
        
        result = self.cd.compute(traj)
        
        # Hénon D2 is approximately 1.22
        assert result.quality != QualityFlag.FAILED, f"Failed: {result.failure_reason}"
        assert 1.0 < result.D2 < 1.5, f"D2={result.D2}, expected ~1.22"
    
    # -------------------------------------------------------------------------
    # Test Case 3: iid point clouds (geometry sanity)
    # -------------------------------------------------------------------------
    
    def test_uniform_cube_3d(self):
        """Uniform 3D cube should have D2 ≈ 3."""
        np.random.seed(42)
        points = generate_uniform_cube(n_points=3000, dim=3)
        
        # For iid points, no temporal structure, so Theiler window = 1
        result = self.cd.compute(points, theiler_window=1)
        
        # Should get D2 close to 3
        assert result.quality != QualityFlag.FAILED, f"Failed: {result.failure_reason}"
        assert 2.5 < result.D2 < 3.5, f"D2={result.D2}, expected ~3.0"
    
    def test_uniform_cube_2d(self):
        """Uniform 2D square should have D2 ≈ 2."""
        np.random.seed(42)
        points = generate_uniform_cube(n_points=3000, dim=2)
        
        result = self.cd.compute(points, theiler_window=1)
        
        assert result.quality != QualityFlag.FAILED, f"Failed: {result.failure_reason}"
        assert 1.5 < result.D2 < 2.5, f"D2={result.D2}, expected ~2.0"
    
    # -------------------------------------------------------------------------
    # Test Case 4: Limit cycle D2 ≈ 1
    # -------------------------------------------------------------------------
    
    def test_limit_cycle(self):
        """Limit cycle should have D2 ≈ 1."""
        np.random.seed(42)
        traj = generate_limit_cycle(n_points=3000)
        
        result = self.cd.compute(traj)
        
        # Limit cycle is 1-dimensional
        assert result.quality != QualityFlag.FAILED, f"Failed: {result.failure_reason}"
        assert 0.8 < result.D2 < 1.5, f"D2={result.D2}, expected ~1.0"
    
    # -------------------------------------------------------------------------
    # Test Case 5: Quality flag behavior
    # -------------------------------------------------------------------------
    
    def test_constant_trajectory_fails(self):
        """Constant trajectory should fail (no scaling regime)."""
        traj = np.ones((1000, 3))
        
        result = self.cd.compute(traj)
        
        assert result.quality == QualityFlag.FAILED
    
    def test_too_short_trajectory_fails(self):
        """Very short trajectory should fail."""
        traj = np.random.randn(50, 3)
        
        result = self.cd.compute(traj)
        
        # May fail or be marginal due to insufficient data
        assert result.quality in [QualityFlag.FAILED, QualityFlag.MARGINAL]
    
    # -------------------------------------------------------------------------
    # Test Case 6: Theiler window estimation
    # -------------------------------------------------------------------------
    
    def test_theiler_window_auto(self):
        """Auto Theiler window should be reasonable."""
        traj = generate_lorenz(n_points=2000, discard=500)
        
        w = self.cd._estimate_theiler_window(traj)
        
        # Should be > 5 (minimum) and < N/5
        assert 5 <= w <= len(traj) // 5
    
    def test_theiler_window_constant(self):
        """Constant signal should give minimum Theiler window (5)."""
        traj = np.ones((1000, 2))
        
        w = self.cd._estimate_theiler_window(traj)
        
        assert w == 5  # Minimum for constant signal
    
    def test_theiler_window_too_small_warns(self):
        """Theiler window < 5 should warn and clamp to 5."""
        np.random.seed(42)
        traj = generate_lorenz(n_points=2000, discard=500)
        
        with pytest.warns(UserWarning, match="clamping to 5"):
            result = self.cd.compute(traj, theiler_window=2)
        
        assert result.theiler_window == 5


class TestResultObject:
    """Tests for CorrelationDimensionResult."""
    
    def test_result_repr(self):
        """Result should have readable repr."""
        cd = CorrelationDimension()
        traj = generate_lorenz(n_points=3000, discard=500)
        
        result = cd.compute(traj)
        
        # Should not raise
        repr_str = repr(result)
        assert "D2=" in repr_str
        assert "quality=" in repr_str
    
    def test_result_attributes(self):
        """Result should have all expected attributes."""
        cd = CorrelationDimension()
        traj = generate_lorenz(n_points=3000, discard=500)
        
        result = cd.compute(traj)
        
        assert hasattr(result, 'D2')
        assert hasattr(result, 'D2_uncertainty')
        assert hasattr(result, 'quality')
        assert hasattr(result, 'scaling_range')
        assert hasattr(result, 'r_values')
        assert hasattr(result, 'C_values')
        assert hasattr(result, 'local_slopes')
        assert hasattr(result, 'theiler_window')


class TestActivationRatio:
    """Tests for activation ratio computation."""
    
    def test_activation_ratio_normal(self):
        """Normal activation ratio calculation."""
        eta = compute_activation_ratio(D2=1.5, r_S=3)
        assert_allclose(eta, 0.5)
    
    def test_activation_ratio_small_rank(self):
        """r_S < 2 should return NaN."""
        eta = compute_activation_ratio(D2=1.0, r_S=1)
        assert np.isnan(eta)
    
    def test_activation_ratio_nan_d2(self):
        """NaN D2 should return NaN."""
        eta = compute_activation_ratio(D2=np.nan, r_S=5)
        assert np.isnan(eta)


class TestEnsembleComputation:
    """Tests for ensemble D2 computation."""
    
    def test_ensemble_basic(self):
        """Ensemble computation should work with multiple trajectories."""
        np.random.seed(42)
        
        # Generate multiple short trajectories
        trajectories = [generate_lorenz(n_points=2000, discard=500) for _ in range(3)]
        
        result = compute_D2_ensemble(trajectories, random_state=42)
        
        assert result.n_total == 3
        # At least some should succeed
        assert result.n_successful >= 1
        assert isinstance(result.D2_median, float)
        assert len(result.results) == 3
    
    def test_ensemble_all_fail(self):
        """If all trajectories fail, should return NaN with proper counts."""
        # Constant trajectories will all fail
        trajectories = [np.ones((100, 2)) for _ in range(3)]
        
        result = compute_D2_ensemble(trajectories)
        
        assert np.isnan(result.D2_median)
        assert np.isnan(result.D2_iqr)
        assert result.n_total == 3
        assert result.n_successful == 0
        assert result.n_failed == 3
        assert result.success_rate == 0.0
    
    def test_ensemble_empty(self):
        """Empty trajectory list should return NaN."""
        result = compute_D2_ensemble([])
        
        assert np.isnan(result.D2_median)
        assert result.n_total == 0
    
    def test_ensemble_quality_counts(self):
        """Quality counts should be tracked correctly."""
        # Mix of good trajectories (longer circles) and constant (will fail)
        t = np.linspace(0, 40*np.pi, 2000)
        trajectories = [
            np.column_stack([np.sin(t), np.cos(t)]),  # Should succeed
            np.ones((100, 2)),                         # Will fail
            np.column_stack([np.sin(t), np.cos(t)]),  # Should succeed
        ]
        
        result = compute_D2_ensemble(trajectories)
        
        assert result.n_total == 3
        assert result.n_failed >= 1  # At least the constant one fails
        assert result.n_good + result.n_marginal + result.n_failed == 3
    
    def test_ensemble_reproducibility(self):
        """Same random_state should give identical results."""
        # Use longer trajectories that reliably succeed
        t = np.linspace(0, 40*np.pi, 2000)
        trajectories = [
            np.column_stack([np.sin(t + i*0.1), np.cos(t + i*0.1)])
            for i in range(3)
        ]
        
        result1 = compute_D2_ensemble(trajectories, random_state=123)
        result2 = compute_D2_ensemble(trajectories, random_state=123)
        
        # Should have successful results
        assert result1.n_successful > 0, "Expected at least one successful result"
        assert result1.D2_median == result2.D2_median
        assert result1.random_state == 123
    
    def test_ensemble_qualities_property(self):
        """qualities property should return list of QualityFlags."""
        t = np.linspace(0, 40*np.pi, 2000)
        trajectories = [
            np.column_stack([np.sin(t), np.cos(t)]),  # Should succeed
            np.ones((100, 2)),                         # Will fail
        ]
        
        result = compute_D2_ensemble(trajectories)
        
        assert len(result.qualities) == 2
        assert all(isinstance(q, QualityFlag) for q in result.qualities)
        # At least one should be FAILED (the constant one)
        assert QualityFlag.FAILED in result.qualities


class TestReproducibility:
    """Tests for reproducible pair sampling.
    
    Uses minimal synthetic data to test the mechanism quickly.
    Statistical accuracy of D2 is tested elsewhere.
    """
    
    def test_random_state_stored_in_result(self):
        """random_state should be stored in result."""
        # Minimal 2D data - just needs to not crash
        traj = np.column_stack([np.sin(np.linspace(0, 4*np.pi, 200)),
                                np.cos(np.linspace(0, 4*np.pi, 200))])
        cd = CorrelationDimension()
        
        result = cd.compute(traj, random_state=42)
        assert result.random_state == 42
    
    def test_random_state_none_stored(self):
        """None random_state should be stored as None."""
        traj = np.column_stack([np.sin(np.linspace(0, 4*np.pi, 200)),
                                np.cos(np.linspace(0, 4*np.pi, 200))])
        cd = CorrelationDimension()
        
        result = cd.compute(traj, random_state=None)
        assert result.random_state is None
    
    def test_same_seed_same_pairs_sampled(self):
        """Same seed with max_pairs should sample identical pairs."""
        # Simple circle - deterministic, fast to create
        t = np.linspace(0, 10*np.pi, 300)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        cd = CorrelationDimension()
        
        # Use max_pairs to trigger sampling, explicit theiler_window to avoid auto-estimation
        result1 = cd.compute(traj, max_pairs=1000, random_state=123, theiler_window=5)
        result2 = cd.compute(traj, max_pairs=1000, random_state=123, theiler_window=5)
        
        # C(r) curves should be identical (same pairs sampled)
        assert np.allclose(result1.C_values, result2.C_values, equal_nan=True)
    
    def test_no_max_pairs_deterministic(self):
        """Without max_pairs, no sampling occurs - result independent of seed."""
        t = np.linspace(0, 4*np.pi, 150)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        cd = CorrelationDimension()
        
        # No max_pairs = all pairs used = deterministic
        result1 = cd.compute(traj, random_state=111)
        result2 = cd.compute(traj, random_state=999)
        
        # Should be identical - RNG not used
        assert np.allclose(result1.C_values, result2.C_values, equal_nan=True)


class TestDiagnosticPlot:
    """Tests for diagnostic plotting."""
    
    def test_plot_does_not_crash(self):
        """Diagnostic plot should not crash."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        cd = CorrelationDimension()
        traj = generate_lorenz(n_points=2000, discard=500)
        result = cd.compute(traj)
        
        # Should not raise
        cd.plot_diagnostics(result, show=False)
    
    def test_plot_saves_file(self, tmp_path):
        """Diagnostic plot should save to file."""
        import matplotlib
        matplotlib.use('Agg')
        
        cd = CorrelationDimension()
        traj = generate_lorenz(n_points=2000, discard=500)
        result = cd.compute(traj)
        
        save_path = tmp_path / "diagnostic.png"
        cd.plot_diagnostics(result, save_path=str(save_path), show=False)


# ===========================================================================
# Phase 1 additional tests — Guard against silent wrong answers
# ===========================================================================


class TestD2EmbeddingDimensionBound:
    """Phase 1.1: D2 should not exceed the embedding dimension.

    An estimated D2 greater than the number of trajectory dimensions is
    physically meaningless and would silently inflate eta = D2 / r_S.
    """

    def test_2d_system_d2_bounded(self):
        """D2 from a known 2D system should not exceed 2."""
        np.random.seed(42)
        traj = generate_henon(n_points=10000, discard=1000)
        assert traj.shape[1] == 2

        cd = CorrelationDimension()
        result = cd.compute(traj, random_state=42)

        if result.quality != QualityFlag.FAILED:
            assert result.D2 <= 2.5, (
                f"D2={result.D2:.2f} exceeds embedding dim=2 with tolerance"
            )

    def test_limit_cycle_2d_bounded(self):
        """A simple limit cycle in 2D should give D2 <= 2."""
        np.random.seed(42)
        traj = generate_limit_cycle(n_points=3000)
        assert traj.shape[1] == 2

        cd = CorrelationDimension()
        result = cd.compute(traj, random_state=42)

        if result.quality != QualityFlag.FAILED:
            assert result.D2 <= 2.5, (
                f"D2={result.D2:.2f} exceeds embedding dim=2 with tolerance"
            )

    def test_uniform_cube_d2_bounded_by_dimension(self):
        """D2 of uniform hypercube should not exceed its intrinsic dimension.

        Parameterized over several dimensions to guard against systematic bias.
        """
        cd = CorrelationDimension()
        for dim in [2, 3, 5]:
            np.random.seed(42)
            points = generate_uniform_cube(n_points=3000, dim=dim)
            result = cd.compute(points, theiler_window=1, random_state=42)

            if result.quality != QualityFlag.FAILED:
                assert result.D2 <= dim + 0.5, (
                    f"dim={dim}: D2={result.D2:.2f} exceeds dim + 0.5"
                )

    def test_embedding_dimension_stored(self):
        """Result should record the embedding dimension used."""
        cd = CorrelationDimension()
        traj = generate_lorenz(n_points=2000, discard=500)
        result = cd.compute(traj)

        assert result.embedding_dimension == 3


class TestComputeFromSimulation:
    """Phase 1.2: Direct tests for compute_from_simulation().

    This method bridges SimulationResult to D2 and is the core pipeline
    seam. It is currently only tested indirectly through ActivationTracker.
    """

    def setup_method(self):
        self.cd = CorrelationDimension()

    def test_brusselator_limit_cycle(self):
        """Brusselator via compute_from_simulation should give D2 ~ 1.

        Note: compute_from_simulation uses the raw concentration matrix.
        With chemostat mode, A and B are removed by the simulator, but
        waste products D and E are still present and monotonically
        increasing. We select only X, Y columns to get a clean trajectory
        — the ActivationTracker does this filtering automatically, but
        compute_from_simulation does not.
        """
        from dimensional_opening.simulator import (
            ReactionSimulator, SimulationResult, DrivingMode,
        )

        sim = ReactionSimulator()
        network = sim.build_network([
            "A -> X",
            "2X + Y -> 3X",
            "B + X -> Y + D",
            "X -> E",
        ])
        sim_result = sim.simulate(
            network,
            rate_constants=[1.0, 1.0, 1.0, 1.0],
            initial_concentrations={
                "A": 1.0, "B": 3.0, "X": 1.0, "Y": 1.0,
                "D": 0.0, "E": 0.0,
            },
            t_span=(0, 100),
            n_points=5000,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species={"A": 1.0, "B": 3.0},
        )

        # Extract only oscillating species (X, Y) — waste species D, E
        # are monotonically increasing and would corrupt the D2 estimate.
        x_idx = sim_result.species_names.index("X")
        y_idx = sim_result.species_names.index("Y")
        filtered = SimulationResult(
            time=sim_result.time,
            concentrations=sim_result.concentrations[:, [x_idx, y_idx]],
            species_names=["X", "Y"],
            n_species=2,
            driving_mode=sim_result.driving_mode,
            driving_params=sim_result.driving_params,
            rate_constants=sim_result.rate_constants,
            initial_concentrations=sim_result.initial_concentrations,
            solver_message=sim_result.solver_message,
            n_function_evals=sim_result.n_function_evals,
            success=sim_result.success,
        )

        result = self.cd.compute_from_simulation(
            filtered, remove_transient=0.5,
        )

        assert result.quality != QualityFlag.FAILED, (
            f"Failed: {result.failure_reason}"
        )
        # Limit cycle on chemostatted Brusselator — D2 should be ~1
        assert 0.7 < result.D2 < 1.5, f"D2={result.D2:.2f}, expected ~1.0"

    def test_transient_removal_fraction(self):
        """remove_transient=0.9 should use only the final 10% of the data."""
        from dimensional_opening.simulator import (
            ReactionSimulator, DrivingMode,
        )

        sim = ReactionSimulator()
        network = sim.build_network(["A -> X", "X -> Y"])
        sim_result = sim.simulate(
            network,
            rate_constants=[1.0, 1.0],
            initial_concentrations={"A": 1.0, "X": 0.0, "Y": 0.0},
            t_span=(0, 100),
            n_points=1000,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species={"A": 1.0},
        )

        total_points = len(sim_result.time)

        # Call with aggressive transient removal
        result = self.cd.compute_from_simulation(
            sim_result, remove_transient=0.9,
        )

        # The trajectory used should have only ~10% of the original points
        assert result.n_trajectory_points == total_points - int(
            total_points * 0.9
        )

    def test_monotonic_trajectory_handled(self):
        """Pure decay (no driving) should still return a result, not crash."""
        from dimensional_opening.simulator import (
            ReactionSimulator, DrivingMode,
        )

        sim = ReactionSimulator()
        network = sim.build_network(["A -> B", "B -> C"])
        sim_result = sim.simulate(
            network,
            rate_constants=[1.0, 0.5],
            initial_concentrations={"A": 2.0, "B": 0.0, "C": 0.0},
            t_span=(0, 20),
            n_points=500,
            driving_mode=DrivingMode.NONE,
        )

        # This should not crash — the trajectory is monotonic / decaying
        result = self.cd.compute_from_simulation(
            sim_result, remove_transient=0.5,
        )
        assert isinstance(result, CorrelationDimensionResult)
        # Quality is likely FAILED or MARGINAL for monotonic data
        # (no interesting attractor), but it should not crash
