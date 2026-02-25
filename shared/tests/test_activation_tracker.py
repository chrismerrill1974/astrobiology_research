"""
Tests for activation tracker and visualization modules.

Step 6 test cases:
1. Single network analysis
2. Batch processing
3. Skip conditions (r_S < 2, simulation failure)
4. Ensemble analysis
5. Result serialization
6. Visualization (smoke tests)
"""

import numpy as np
import pytest
import json
import tempfile
from pathlib import Path

from dimensional_opening.activation_tracker import (
    ActivationTracker,
    ActivationResult,
    BatchResult,
    save_results_csv,
    load_results_json,
)
from dimensional_opening.correlation_dimension import QualityFlag
from dimensional_opening import visualization as viz
from dimensional_opening.simulator import (
    ReactionSimulator,
    SimulationResult,
    DrivingMode,
)


class TestActivationTracker:
    """Tests for ActivationTracker."""
    
    def test_brusselator_analysis(self):
        """Brusselator should give valid η."""
        tracker = ActivationTracker(
            t_span=(0, 100),
            n_points=5000,
            remove_transient=0.5,
            random_state=42,
        )
        
        result = tracker.analyze_network(
            reactions=[
                "A -> X",
                "2X + Y -> 3X",
                "B + X -> Y + D",
                "X -> E",
            ],
            rate_constants=[1.0, 1.0, 1.0, 1.0],
            initial_concentrations={"A": 1.0, "B": 3.0, "X": 1.0, "Y": 1.0, "D": 0.0, "E": 0.0},
            chemostat_species={"A": 1.0, "B": 3.0},
            network_id="brusselator_test",
        )
        
        assert not result.skipped
        # Full network has r_S = 4 (6 species, 4 reactions, full rank)
        assert result.r_S >= 2
        assert result.quality in [QualityFlag.GOOD, QualityFlag.MARGINAL]
        # D2 ≈ 1 for limit cycle, η = D2/r_S
        assert 0.1 < result.eta < 0.8
        assert 0.8 < result.D2 < 1.5  # Limit cycle
    
    def test_skip_low_rank(self):
        """Networks with r_S < 2 should be skipped."""
        tracker = ActivationTracker()
        
        # Single species reaction: r_S = 1
        result = tracker.analyze_network(
            reactions=["A -> B"],
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            chemostat_species={"A": 1.0},
            network_id="low_rank",
        )
        
        assert result.skipped
        assert "r_S" in result.skip_reason
        assert result.r_S < 2
    
    def test_no_driving_warning(self):
        """Should warn when no driving is specified."""
        tracker = ActivationTracker(t_span=(0, 10), n_points=500)
        
        with pytest.warns(UserWarning, match="No driving"):
            result = tracker.analyze_network(
                reactions=["A + B -> C", "C -> A + B"],
                rate_constants=[1.0, 1.0],
                initial_concentrations={"A": 1.0, "B": 1.0, "C": 0.0},
                # No chemostat_species
            )
    
    def test_result_serialization(self):
        """ActivationResult should serialize to dict correctly."""
        result = ActivationResult(
            network_id="test",
            reactions=["A -> B"],
            species=["A", "B"],
            n_reactions=1,
            n_species=2,
            r_S=1,
            n_conservation_laws=1,
            D2=1.5,
            D2_uncertainty=0.1,
            quality=QualityFlag.GOOD,
            eta=0.75,
            driving_mode="chemostat",
            simulation_time=100,
            n_trajectory_points=5000,
            theiler_window=50,
        )
        
        d = result.to_dict()
        
        assert d['network_id'] == "test"
        assert d['D2'] == 1.5
        assert d['quality'] == "good"
        assert d['eta'] == 0.75
    
    def test_result_with_nan(self):
        """NaN values should serialize as None."""
        result = ActivationResult(
            network_id="failed",
            reactions=["A -> B"],
            species=["A", "B"],
            n_reactions=1,
            n_species=2,
            r_S=1,
            n_conservation_laws=1,
            D2=np.nan,
            D2_uncertainty=np.nan,
            quality=QualityFlag.FAILED,
            eta=np.nan,
            driving_mode="none",
            simulation_time=0,
            n_trajectory_points=0,
            theiler_window=0,
            skipped=True,
            skip_reason="Test",
        )
        
        d = result.to_dict()
        
        assert d['D2'] is None
        assert d['eta'] is None


class TestBatchAnalysis:
    """Tests for batch processing."""
    
    def test_batch_basic(self):
        """Batch analysis should process multiple networks."""
        tracker = ActivationTracker(
            t_span=(0, 100),
            n_points=5000,
            remove_transient=0.5,
            random_state=42,
        )
        
        networks = [
            {
                'reactions': ["A -> X", "2X + Y -> 3X", "B + X -> Y + D", "X -> E"],
                'rate_constants': [1.0, 1.0, 1.0, 1.0],
                'initial_concentrations': {"A": 1.0, "B": 3.0, "X": 1.0, "Y": 1.0, "D": 0.0, "E": 0.0},
                'chemostat_species': {"A": 1.0, "B": 3.0},
                'network_id': 'brusselator',
            },
            {
                # This simpler network will likely converge to fixed point
                'reactions': ["A -> X", "X -> Y", "Y -> Z"],
                'rate_constants': [1.0, 1.0, 1.0],
                'initial_concentrations': {"A": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0},
                'chemostat_species': {"A": 1.0},
                'network_id': 'cascade',
            },
        ]
        
        batch = tracker.analyze_batch(networks, verbose=False)
        
        assert batch.n_total == 2
        assert len(batch.results) == 2
    
    def test_batch_aggregation(self):
        """Batch should compute aggregate statistics correctly."""
        # Create mock results
        results = [
            ActivationResult(
                network_id="n1", reactions=[], species=[], n_reactions=2, n_species=3,
                r_S=2, n_conservation_laws=1, D2=1.0, D2_uncertainty=0.1,
                quality=QualityFlag.GOOD, eta=0.5, driving_mode="chemostat",
                simulation_time=100, n_trajectory_points=2500, theiler_window=50,
            ),
            ActivationResult(
                network_id="n2", reactions=[], species=[], n_reactions=3, n_species=4,
                r_S=3, n_conservation_laws=1, D2=1.5, D2_uncertainty=0.2,
                quality=QualityFlag.GOOD, eta=0.5, driving_mode="chemostat",
                simulation_time=100, n_trajectory_points=2500, theiler_window=50,
            ),
            ActivationResult(
                network_id="n3", reactions=[], species=[], n_reactions=2, n_species=2,
                r_S=1, n_conservation_laws=1, D2=np.nan, D2_uncertainty=np.nan,
                quality=QualityFlag.FAILED, eta=np.nan, driving_mode="chemostat",
                simulation_time=100, n_trajectory_points=0, theiler_window=0,
                skipped=True, skip_reason="r_S < 2",
            ),
        ]
        
        tracker = ActivationTracker()
        batch = tracker._aggregate_results(results)
        
        assert batch.n_total == 3
        assert batch.n_analyzed == 2  # One skipped
        assert batch.n_successful == 2
        assert batch.eta_median == 0.5


class TestCheckpointing:
    """Tests for result saving/loading."""
    
    def test_save_load_json(self):
        """Should round-trip through JSON."""
        tracker = ActivationTracker(
            t_span=(0, 50),
            n_points=2000,
            random_state=42,
        )
        
        result = tracker.analyze_network(
            reactions=["A -> X", "2X + Y -> 3X", "B + X -> Y"],
            rate_constants=[1.0, 1.0, 1.0],
            initial_concentrations={"A": 1.0, "B": 3.0, "X": 1.0, "Y": 1.0},
            chemostat_species={"A": 1.0, "B": 3.0},
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            # Save
            tracker._save_checkpoint([result], path)
            
            # Load
            loaded = load_results_json(path)
            
            assert len(loaded) == 1
            assert loaded[0].network_id == result.network_id
            assert loaded[0].r_S == result.r_S
            # D2 should be close (might have serialization precision)
            if not np.isnan(result.D2):
                assert abs(loaded[0].D2 - result.D2) < 1e-10
        finally:
            Path(path).unlink()
    
    def test_save_csv(self):
        """Should save to CSV correctly."""
        results = [
            ActivationResult(
                network_id="test", reactions=["A -> B"], species=["A", "B"],
                n_reactions=1, n_species=2, r_S=2, n_conservation_laws=0,
                D2=1.5, D2_uncertainty=0.1, quality=QualityFlag.GOOD, eta=0.75,
                driving_mode="chemostat", simulation_time=100,
                n_trajectory_points=2500, theiler_window=50,
            ),
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            save_results_csv(results, path)
            
            # Verify file exists and has content
            content = Path(path).read_text()
            assert 'network_id' in content
            assert 'test' in content
            assert '0.75' in content
        finally:
            Path(path).unlink()


class TestEnsembleAnalysis:
    """Tests for ensemble-based analysis."""
    
    def test_ensemble_analysis(self):
        """Ensemble analysis should run multiple trajectories."""
        tracker = ActivationTracker(
            t_span=(0, 100),
            n_points=5000,
            remove_transient=0.5,
            random_state=42,
        )
        
        # Use full Brusselator with waste products for oscillations
        result = tracker.analyze_network_ensemble(
            reactions=["A -> X", "2X + Y -> 3X", "B + X -> Y + D", "X -> E"],
            rate_constants=[1.0, 1.0, 1.0, 1.0],
            initial_concentrations={"A": 1.0, "B": 3.0, "X": 1.0, "Y": 1.0, "D": 0.0, "E": 0.0},
            chemostat_species={"A": 1.0, "B": 3.0},
            n_runs=3,
            ic_perturbation=0.1,
        )
        
        # Should have ensemble stats
        assert result.ensemble_D2_median is not None
        assert result.ensemble_success_rate is not None


class TestVisualization:
    """Smoke tests for visualization functions."""
    
    @pytest.fixture
    def sim_result(self):
        """Create a simulation result for plotting."""
        sim = ReactionSimulator()
        network = sim.build_network(["A -> X", "2X + Y -> 3X", "B + X -> Y"])
        return sim.simulate(
            network,
            rate_constants=[1.0, 1.0, 1.0],
            initial_concentrations={"A": 1.0, "B": 3.0, "X": 1.0, "Y": 1.0},
            t_span=(0, 100),
            n_points=5000,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species={"A": 1.0, "B": 3.0},
        )
    
    @pytest.fixture
    def d2_result(self, sim_result):
        """Create a D2 result for plotting."""
        from dimensional_opening.correlation_dimension import CorrelationDimension
        cd = CorrelationDimension()
        # Use more points post-transient for reliable D2
        traj = sim_result.concentrations[2500:, :2]  # X, Y post-transient
        return cd.compute(traj)
    
    @pytest.fixture
    def activation_results(self):
        """Create mock activation results for plotting."""
        results = []
        for i in range(10):
            results.append(ActivationResult(
                network_id=f"net_{i}",
                reactions=["A -> B"] * (i + 1),
                species=["A", "B"],
                n_reactions=i + 1,
                n_species=2,
                r_S=2 + i % 3,
                n_conservation_laws=1,
                D2=1.0 + 0.3 * np.random.randn(),
                D2_uncertainty=0.1,
                quality=QualityFlag.GOOD if np.random.rand() > 0.3 else QualityFlag.MARGINAL,
                eta=0.3 + 0.4 * np.random.rand(),
                driving_mode="chemostat",
                simulation_time=100,
                n_trajectory_points=2500,
                theiler_window=50,
            ))
        return results
    
    def test_plot_time_series(self, sim_result):
        """Time series plot should not crash."""
        import matplotlib
        matplotlib.use('Agg')
        
        ax = viz.plot_time_series(sim_result)
        assert ax is not None
    
    def test_plot_scaling_regime(self, d2_result):
        """Scaling plot should not crash."""
        import matplotlib
        matplotlib.use('Agg')
        
        ax = viz.plot_scaling_regime(d2_result)
        assert ax is not None
    
    def test_plot_eta_vs_complexity(self, activation_results):
        """η vs complexity plot should not crash."""
        import matplotlib
        matplotlib.use('Agg')
        
        ax = viz.plot_eta_vs_complexity(activation_results)
        assert ax is not None
    
    def test_plot_D2_vs_rS(self, activation_results):
        """D2 vs r_S plot should not crash."""
        import matplotlib
        matplotlib.use('Agg')
        
        ax = viz.plot_D2_vs_rS(activation_results)
        assert ax is not None
    
    def test_plot_eta_distribution(self, activation_results):
        """η distribution plot should not crash."""
        import matplotlib
        matplotlib.use('Agg')
        
        ax = viz.plot_eta_distribution(activation_results)
        assert ax is not None
    
    def test_plot_diagnostic_panel(self, sim_result, d2_result):
        """Diagnostic panel should not crash."""
        import matplotlib
        matplotlib.use('Agg')
        
        fig = viz.plot_diagnostic_panel(sim_result, d2_result)
        assert fig is not None
    
    def test_plot_saves_file(self, sim_result, d2_result):
        """Plots should save to file."""
        import matplotlib
        matplotlib.use('Agg')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        
        try:
            viz.plot_diagnostic_panel(sim_result, d2_result, save_path=path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink()


# ===========================================================================
# Phase 1 additional tests — Guard against silent wrong answers
# ===========================================================================


class TestTrajectoryExtraction:
    """Phase 1.3: Tests for _extract_trajectory species filtering.

    The tracker auto-excludes monotonically increasing/decreasing species
    before computing D2. This logic determines which dimensions of the
    trajectory are used, directly affecting D2 and eta.
    """

    def _make_sim_result(self, time, concentrations, species_names):
        """Helper to build a SimulationResult for testing extraction."""
        return SimulationResult(
            time=time,
            concentrations=concentrations,
            species_names=species_names,
            n_species=len(species_names),
            driving_mode=DrivingMode.NONE,
            driving_params={},
            rate_constants=np.array([1.0]),
            initial_concentrations=concentrations[0],
            solver_message="ok",
            n_function_evals=100,
            success=True,
        )

    def test_waste_species_excluded(self):
        """Monotonically increasing waste species should be excluded."""
        tracker = ActivationTracker(remove_transient=0.0)

        t = np.linspace(0, 100, 1000)
        # X oscillates, W monotonically increases (waste product)
        c_x = 2.0 + 1.0 * np.sin(2 * np.pi * 0.1 * t)
        c_y = 1.5 + 0.8 * np.cos(2 * np.pi * 0.1 * t)
        c_w = np.linspace(0, 10, 1000)

        sim_result = self._make_sim_result(
            t,
            np.column_stack([c_x, c_y, c_w]),
            ["X", "Y", "W"],
        )

        traj = tracker._extract_trajectory(sim_result, None, None)

        # Should include X and Y but not W
        assert traj is not None
        assert traj.shape[1] == 2, (
            f"Expected 2 species after filtering, got {traj.shape[1]}"
        )

    def test_constant_species_excluded(self):
        """Constant (e.g. chemostatted) species should be excluded."""
        tracker = ActivationTracker(remove_transient=0.0)

        t = np.linspace(0, 100, 1000)
        c_x = 2.0 + 1.0 * np.sin(2 * np.pi * 0.1 * t)
        c_const = np.ones(1000) * 5.0  # Constant — monotonic (non-decreasing)

        sim_result = self._make_sim_result(
            t,
            np.column_stack([c_x, c_const]),
            ["X", "CONST"],
        )

        traj = tracker._extract_trajectory(sim_result, None, None)

        assert traj is not None
        assert traj.shape[1] == 1, (
            f"Expected 1 species after filtering, got {traj.shape[1]}"
        )

    def test_all_monotonic_returns_none(self):
        """If ALL species are monotonic, extraction should return None."""
        tracker = ActivationTracker(remove_transient=0.0)

        t = np.linspace(0, 100, 1000)
        c_a = np.exp(-0.1 * t)        # Monotonically decreasing
        c_b = 1.0 - np.exp(-0.1 * t)  # Monotonically increasing

        sim_result = self._make_sim_result(
            t,
            np.column_stack([c_a, c_b]),
            ["A", "B"],
        )

        traj = tracker._extract_trajectory(sim_result, None, None)

        assert traj is None

    def test_single_non_monotonic_species_returns_1d(self):
        """If only 1 species survives filtering, it should still be returned."""
        tracker = ActivationTracker(remove_transient=0.0)

        t = np.linspace(0, 100, 1000)
        c_x = 2.0 + 1.0 * np.sin(2 * np.pi * 0.1 * t)
        c_sink = np.exp(-0.05 * t)  # Monotonically decreasing

        sim_result = self._make_sim_result(
            t,
            np.column_stack([c_x, c_sink]),
            ["X", "SINK"],
        )

        traj = tracker._extract_trajectory(sim_result, None, None)

        assert traj is not None
        assert traj.shape[1] == 1
        assert traj.shape[0] == 1000


class TestSimulationFailureSkip:
    """Phase 1.4: Tracker should skip networks when simulation fails.

    When solve_ivp returns success=False, the tracker should mark the
    network as skipped rather than computing D2 on garbage data.
    """

    def test_tracker_skips_when_sim_success_false(self):
        """Directly verify the sim_result.success check path."""
        tracker = ActivationTracker(
            t_span=(0, 10),
            n_points=500,
            remove_transient=0.0,
            random_state=42,
        )

        # Monkey-patch the simulator to return a failed result
        original_simulate = tracker.simulator.simulate

        def fake_simulate(network, **kwargs):
            result = original_simulate(network, **kwargs)
            # Force success=False to test the skip path
            object.__setattr__(result, 'success', False)
            return result

        tracker.simulator.simulate = fake_simulate

        try:
            result = tracker.analyze_network(
                reactions=["A -> X", "X -> B"],
                rate_constants=[1.0, 1.0],
                initial_concentrations={"A": 1.0, "X": 0.5, "B": 0.0},
                chemostat_species={"A": 1.0},
                network_id="forced_failure",
            )

            assert result.skipped is True
            assert "failed" in result.skip_reason.lower()
        finally:
            tracker.simulator.simulate = original_simulate


class TestSpeciesToTrack:
    """Phase 1.5: Tests for the species_to_track parameter.

    Callers can override automatic species selection by passing an explicit
    list of species names. This is used in paper scripts for reproducibility.
    """

    def _make_sim_result(self, time, concentrations, species_names):
        """Helper to build a SimulationResult for testing extraction."""
        return SimulationResult(
            time=time,
            concentrations=concentrations,
            species_names=species_names,
            n_species=len(species_names),
            driving_mode=DrivingMode.NONE,
            driving_params={},
            rate_constants=np.array([1.0]),
            initial_concentrations=concentrations[0],
            solver_message="ok",
            n_function_evals=100,
            success=True,
        )

    def test_explicit_species_selection(self):
        """species_to_track should select only the specified species."""
        tracker = ActivationTracker(remove_transient=0.0)

        t = np.linspace(0, 100, 1000)
        c_x = 2.0 + 1.0 * np.sin(2 * np.pi * 0.1 * t)
        c_y = 1.5 + 0.8 * np.cos(2 * np.pi * 0.1 * t)
        c_z = 3.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)

        sim_result = self._make_sim_result(
            t,
            np.column_stack([c_x, c_y, c_z]),
            ["X", "Y", "Z"],
        )

        traj = tracker._extract_trajectory(
            sim_result, species_to_track=["X", "Y"], chemostat_species=None
        )

        assert traj is not None
        assert traj.shape[1] == 2, (
            f"Expected 2 species, got {traj.shape[1]}"
        )
        # Verify the columns are X and Y (not Z)
        np.testing.assert_allclose(traj[:, 0], c_x, rtol=1e-10)
        np.testing.assert_allclose(traj[:, 1], c_y, rtol=1e-10)

    def test_nonexistent_species_silently_skipped(self):
        """species_to_track with a non-existent name should not crash.

        The current implementation uses a list comprehension with
        ``if sp in species_names``, so non-existent names are silently
        filtered out. This test documents that behavior.
        """
        tracker = ActivationTracker(remove_transient=0.0)

        t = np.linspace(0, 100, 1000)
        c_x = 2.0 + 1.0 * np.sin(2 * np.pi * 0.1 * t)

        sim_result = self._make_sim_result(
            t,
            c_x.reshape(-1, 1),
            ["X"],
        )

        traj = tracker._extract_trajectory(
            sim_result,
            species_to_track=["NONEXISTENT"],
            chemostat_species=None,
        )

        # No valid species matched — should return None
        assert traj is None

    def test_single_tracked_species(self):
        """Tracking a single species should return a 1-column trajectory."""
        tracker = ActivationTracker(remove_transient=0.0)

        t = np.linspace(0, 100, 1000)
        c_x = 2.0 + 1.0 * np.sin(2 * np.pi * 0.1 * t)
        c_y = 1.5 + 0.8 * np.cos(2 * np.pi * 0.1 * t)

        sim_result = self._make_sim_result(
            t,
            np.column_stack([c_x, c_y]),
            ["X", "Y"],
        )

        traj = tracker._extract_trajectory(
            sim_result, species_to_track=["X"], chemostat_species=None,
        )

        assert traj is not None
        assert traj.shape == (1000, 1)
        np.testing.assert_allclose(traj[:, 0], c_x, rtol=1e-10)
