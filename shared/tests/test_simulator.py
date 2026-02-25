"""
Tests for simulator module.

Step 2 test cases from the plan:
1. First-order decay: A -> B with analytical solution [A](t) = [A]_0 * exp(-kt)
2. Chemostatted equilibrium: A <-> B with [A] fixed; verify [B] reaches steady state
3. Brusselator with CSTR: Known limit cycle behavior
4. Lotka-Volterra with inflow: Sustained oscillations (vs damped without driving)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dimensional_opening.simulator import (
    ReactionSimulator,
    ReactionNetwork,
    SimulationResult,
    DrivingMode,
    simulate_reactions,
)


class TestReactionSimulator:
    """Tests for ReactionSimulator class."""
    
    def setup_method(self):
        self.sim = ReactionSimulator()
    
    # -------------------------------------------------------------------------
    # Test Case 1: First-order decay A -> B
    # -------------------------------------------------------------------------
    
    def test_first_order_decay_analytical(self):
        """A -> B with known analytical solution [A](t) = [A]_0 * exp(-kt)."""
        network = self.sim.build_network(["A -> B"])
        
        k = 0.5
        A0 = 1.0
        t_span = (0, 10)
        
        result = self.sim.simulate(
            network,
            rate_constants=[k],
            initial_concentrations={"A": A0, "B": 0.0},
            t_span=t_span,
            n_points=100,
        )
        
        # Analytical solution
        t = result.time
        A_analytical = A0 * np.exp(-k * t)
        B_analytical = A0 * (1 - np.exp(-k * t))
        
        # Compare
        A_numerical = result.get_species("A")
        B_numerical = result.get_species("B")
        
        assert_allclose(A_numerical, A_analytical, rtol=1e-4)
        assert_allclose(B_numerical, B_analytical, rtol=1e-4)
    
    def test_first_order_decay_mass_conservation(self):
        """Total mass should be conserved in A -> B (closed system, no driving).
        
        Note: Mass conservation only applies to DrivingMode.NONE (closed system).
        CSTR and CHEMOSTAT modes exchange mass with external reservoirs.
        """
        network = self.sim.build_network(["A -> B"])
        
        result = self.sim.simulate(
            network,
            rate_constants=[1.0],
            initial_concentrations={"A": 2.5, "B": 0.5},
            t_span=(0, 10),
        )
        
        total = result.get_species("A") + result.get_species("B")
        assert_allclose(total, 3.0, rtol=1e-6)
    
    # -------------------------------------------------------------------------
    # Test Case 2: Chemostatted equilibrium
    # -------------------------------------------------------------------------
    
    def test_chemostat_steady_state(self):
        """A <-> B with [A] fixed should drive [B] to steady state."""
        network = self.sim.build_network(["A -> B", "B -> A"])
        
        k_forward = 1.0
        k_reverse = 0.5
        A_fixed = 2.0
        
        result = self.sim.simulate(
            network,
            rate_constants=[k_forward, k_reverse],
            initial_concentrations={"A": A_fixed, "B": 0.0},
            t_span=(0, 20),
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species={"A": A_fixed},
        )
        
        # At steady state: k_f * A = k_r * B  =>  B = (k_f/k_r) * A
        B_expected = (k_forward / k_reverse) * A_fixed
        B_final = result.get_species("B")[-1]
        
        assert_allclose(B_final, B_expected, rtol=1e-3)
    
    def test_chemostat_removes_species_from_ode(self):
        """Chemostat mode should only integrate dynamic species."""
        network = self.sim.build_network(["A -> B", "B -> C"])
        
        result = self.sim.simulate(
            network,
            rate_constants=[1.0, 0.5],
            initial_concentrations={"A": 1.0, "B": 0.0, "C": 0.0},
            t_span=(0, 10),
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species={"A": 1.0},
        )
        
        # A should not be in output (it's chemostatted)
        assert "A" not in result.species_names
        assert "B" in result.species_names
        assert "C" in result.species_names
        assert result.n_species == 2
    
    # -------------------------------------------------------------------------
    # Test Case 3: CSTR mode
    # -------------------------------------------------------------------------
    
    def test_cstr_steady_state(self):
        """CSTR with inflow of A should reach non-trivial steady state."""
        # A -> B with inflow of A and outflow of both
        network = self.sim.build_network(["A -> B"])
        
        k = 1.0
        D = 0.5  # Dilution rate
        A_feed = 2.0
        
        result = self.sim.simulate(
            network,
            rate_constants=[k],
            initial_concentrations={"A": 0.0, "B": 0.0},
            t_span=(0, 30),
            driving_mode=DrivingMode.CSTR,
            cstr_dilution_rate=D,
            cstr_feed_concentrations={"A": A_feed},
        )
        
        # At steady state:
        # dA/dt = 0 = -k*A + D*(A_feed - A)  =>  A = D*A_feed / (k + D)
        # dB/dt = 0 = k*A - D*B  =>  B = k*A / D
        A_ss = D * A_feed / (k + D)
        B_ss = k * A_ss / D
        
        A_final = result.get_species("A")[-1]
        B_final = result.get_species("B")[-1]
        
        assert_allclose(A_final, A_ss, rtol=1e-2)
        assert_allclose(B_final, B_ss, rtol=1e-2)
    
    def test_cstr_vs_no_driving(self):
        """Same network with CSTR should reach different state than without driving."""
        network = self.sim.build_network(["A -> B"])
        
        # Without driving: goes to B = A0, A = 0
        result_no_drive = self.sim.simulate(
            network,
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 20),
            driving_mode=DrivingMode.NONE,
        )
        
        # With CSTR: maintains non-zero A
        result_cstr = self.sim.simulate(
            network,
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 20),
            driving_mode=DrivingMode.CSTR,
            cstr_dilution_rate=0.5,
            cstr_feed_concentrations={"A": 1.0},
        )
        
        A_final_no_drive = result_no_drive.get_species("A")[-1]
        A_final_cstr = result_cstr.get_species("A")[-1]
        
        # Without driving, A decays to ~0
        assert A_final_no_drive < 0.01
        # With CSTR, A is maintained
        assert A_final_cstr > 0.3
    
    # -------------------------------------------------------------------------
    # Test Case 4: Brusselator (oscillations)
    # -------------------------------------------------------------------------
    
    def test_brusselator_oscillates(self):
        """Brusselator should exhibit limit cycle oscillations with CSTR driving."""
        # Brusselator: A -> X, 2X + Y -> 3X, B + X -> Y + D, X -> E
        # With A, B as food species (chemostatted)
        # Standard parameters: A=1, B=3 gives oscillations
        
        network = self.sim.build_network([
            "A -> X",
            "2X + Y -> 3X",
            "B + X -> Y + D", 
            "X -> E",
        ])
        
        A, B = 1.0, 3.0  # Parameters that give oscillations
        
        result = self.sim.simulate(
            network,
            rate_constants=[1.0, 1.0, 1.0, 1.0],
            initial_concentrations={"A": A, "B": B, "X": 1.0, "Y": 1.0, "D": 0.0, "E": 0.0},
            t_span=(0, 50),
            n_points=2000,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species={"A": A, "B": B},
        )
        
        X = result.get_species("X")
        
        # Remove transient (first half)
        X_ss = X[len(X)//2:]
        
        # Check for oscillations: variance should be significant
        X_mean = np.mean(X_ss)
        X_std = np.std(X_ss)
        
        # Oscillations should give std/mean > 0.1 (coefficient of variation)
        assert X_std / X_mean > 0.1, "Brusselator should oscillate"
    
    # -------------------------------------------------------------------------
    # Test Case 5: Lotka-Volterra
    # -------------------------------------------------------------------------
    
    def test_lotka_volterra_oscillates(self):
        """Lotka-Volterra predator-prey should oscillate."""
        # X + Y -> 2Y (predation), Y -> Z (predator death), A -> X (prey birth)
        # Simplified: prey X, predator Y
        # X -> 2X (prey reproduction), X + Y -> 2Y (predation), Y -> (death)
        
        network = self.sim.build_network([
            "X -> 2X",       # Prey reproduction (need food source)
            "X + Y -> 2Y",   # Predation
            "Y -> D",        # Predator death
        ])
        
        result = self.sim.simulate(
            network,
            rate_constants=[1.0, 0.5, 0.5],
            initial_concentrations={"X": 2.0, "Y": 1.0, "D": 0.0},
            t_span=(0, 30),
            n_points=1000,
            driving_mode=DrivingMode.CSTR,
            cstr_dilution_rate=0.1,
            cstr_feed_concentrations={"X": 1.0},  # Prey inflow
        )
        
        X = result.get_species("X")
        Y = result.get_species("Y")
        
        # Both populations should oscillate (check non-trivial dynamics)
        # Skip transient
        X_ss = X[len(X)//2:]
        Y_ss = Y[len(Y)//2:]
        
        # Check that we have variation (not collapsed to fixed point)
        assert np.std(X_ss) > 0.01 or np.std(Y_ss) > 0.01, \
            "Lotka-Volterra should show non-trivial dynamics"


class TestReactionNetwork:
    """Tests for ReactionNetwork building."""
    
    def setup_method(self):
        self.sim = ReactionSimulator()
    
    def test_network_from_strings(self):
        """Build network from reaction strings."""
        network = self.sim.build_network(["A + B -> C", "C -> A + B"])
        
        assert network.n_species == 3
        assert network.n_reactions == 2
        assert network.species_names == ["A", "B", "C"]
    
    def test_stoichiometric_matrix(self):
        """Verify stoichiometric matrix is correct."""
        network = self.sim.build_network(["A -> B"])
        
        # S should be [[-1], [1]]
        expected_S = np.array([[-1], [1]], dtype=float)
        assert_allclose(network.stoichiometric_matrix, expected_S)
    
    def test_reactant_matrix(self):
        """Verify reactant matrix is correct."""
        network = self.sim.build_network(["2A + B -> C"])
        
        # R should be [[2], [1], [0]] (A appears twice, B once, C not a reactant)
        expected_R = np.array([[2], [1], [0]], dtype=float)
        assert_allclose(network.reactant_matrix, expected_R)
    
    def test_network_from_chempy(self):
        """Build network from chempy ReactionSystem."""
        pytest.importorskip("chempy")
        from chempy import ReactionSystem, Reaction
        
        rxns = [
            Reaction({'A': 1}, {'B': 1}),
            Reaction({'B': 1}, {'A': 1}),
        ]
        rsys = ReactionSystem(rxns, substances='A B'.split())
        
        network = self.sim.build_network_from_chempy(rsys)
        
        assert network.n_species == 2
        assert network.n_reactions == 2


class TestSimulationResult:
    """Tests for SimulationResult."""
    
    def test_get_species(self):
        """Test species retrieval by name."""
        result = simulate_reactions(
            ["A -> B"],
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 1),
        )
        
        A = result.get_species("A")
        assert len(A) == len(result.time)
    
    def test_get_species_unknown(self):
        """Unknown species should raise KeyError."""
        result = simulate_reactions(
            ["A -> B"],
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 1),
        )
        
        with pytest.raises(KeyError):
            result.get_species("C")
    
    def test_result_metadata(self):
        """Result should contain metadata."""
        result = simulate_reactions(
            ["A -> B"],
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 1),
            driving_mode=DrivingMode.CSTR,
            cstr_dilution_rate=0.5,
            cstr_feed_concentrations={"A": 1.0},
        )
        
        assert result.driving_mode == DrivingMode.CSTR
        assert result.driving_params["dilution_rate"] == 0.5
        assert result.success is True


class TestConvenienceFunction:
    """Tests for simulate_reactions convenience function."""
    
    def test_simulate_reactions(self):
        """Basic usage of convenience function."""
        result = simulate_reactions(
            ["A -> B", "B -> A"],
            rate_constants=[1.0, 0.5],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 10),
        )
        
        assert result.success
        assert len(result.time) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def setup_method(self):
        self.sim = ReactionSimulator()
    
    def test_wrong_number_of_rate_constants(self):
        """Should raise error for wrong number of rate constants."""
        network = self.sim.build_network(["A -> B", "B -> A"])
        
        with pytest.raises(ValueError, match="rate constants"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],  # Should be 2
                initial_concentrations={"A": 1.0, "B": 0.0},
                t_span=(0, 1),
            )
    
    def test_unknown_species_in_initial(self):
        """Should raise error for unknown species in initial conditions."""
        network = self.sim.build_network(["A -> B"])
        
        with pytest.raises(ValueError, match="Unknown species"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],
                initial_concentrations={"A": 1.0, "C": 0.0},
                t_span=(0, 1),
            )
    
    def test_unknown_chemostat_species(self):
        """Should raise error for unknown chemostat species."""
        network = self.sim.build_network(["A -> B"])
        
        with pytest.raises(ValueError, match="Unknown chemostat"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],
                initial_concentrations={"A": 1.0, "B": 0.0},
                t_span=(0, 1),
                driving_mode=DrivingMode.CHEMOSTAT,
                chemostat_species={"C": 1.0},
            )
    
    def test_chemostat_mode_requires_species(self):
        """CHEMOSTAT mode should require chemostat_species."""
        network = self.sim.build_network(["A -> B"])
        
        with pytest.raises(ValueError, match="chemostat_species required"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],
                initial_concentrations={"A": 1.0, "B": 0.0},
                t_span=(0, 1),
                driving_mode=DrivingMode.CHEMOSTAT,
            )
    
    def test_chemostat_override_warning(self):
        """Should warn if initial concentration disagrees with chemostat value."""
        network = self.sim.build_network(["A -> B"])
        
        with pytest.warns(UserWarning, match="overridden by fixed value"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],
                initial_concentrations={"A": 0.5, "B": 0.0},  # A=0.5
                t_span=(0, 1),
                driving_mode=DrivingMode.CHEMOSTAT,
                chemostat_species={"A": 1.0},  # but chemostat wants A=1.0
            )
    
    def test_negative_dilution_rate_error(self):
        """Negative dilution rate should raise error."""
        network = self.sim.build_network(["A -> B"])
        
        with pytest.raises(ValueError, match="Dilution rate must be >= 0"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],
                initial_concentrations={"A": 1.0, "B": 0.0},
                t_span=(0, 1),
                driving_mode=DrivingMode.CSTR,
                cstr_dilution_rate=-0.1,
            )
    
    def test_zero_dilution_rate_warning(self):
        """Zero dilution rate in CSTR mode should warn."""
        network = self.sim.build_network(["A -> B"])
        
        with pytest.warns(UserWarning, match="effectively DrivingMode.NONE"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],
                initial_concentrations={"A": 1.0, "B": 0.0},
                t_span=(0, 1),
                driving_mode=DrivingMode.CSTR,
                cstr_dilution_rate=0.0,
            )
    
    def test_negative_chemostat_value_error(self):
        """Negative chemostat concentration should raise error."""
        network = self.sim.build_network(["A -> B"])
        
        with pytest.raises(ValueError, match="Chemostat concentration must be >= 0"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],
                initial_concentrations={"A": 1.0, "B": 0.0},
                t_span=(0, 1),
                driving_mode=DrivingMode.CHEMOSTAT,
                chemostat_species={"A": -1.0},
            )
    
    def test_negative_feed_concentration_error(self):
        """Negative feed concentration should raise error."""
        network = self.sim.build_network(["A -> B"])
        
        with pytest.raises(ValueError, match="Feed concentration must be >= 0"):
            self.sim.simulate(
                network,
                rate_constants=[1.0],
                initial_concentrations={"A": 1.0, "B": 0.0},
                t_span=(0, 1),
                driving_mode=DrivingMode.CSTR,
                cstr_dilution_rate=0.5,
                cstr_feed_concentrations={"A": -1.0},
            )
    
    def test_remove_transient(self):
        """Test transient removal."""
        result = simulate_reactions(
            ["A -> B"],
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 10),
            n_points=100,
            remove_transient=0.5,
        )
        
        # Should have removed first 50% of points
        assert len(result.time) == 50
        assert result.time[0] > 4.0  # Should start around t=5


# ===========================================================================
# Phase 1 additional tests — Guard against silent wrong answers
# ===========================================================================


class TestSolverFailureHandling:
    """Phase 1.4: ODE solver failure should be surfaced, not hidden.

    When solve_ivp returns success=False, the result should clearly
    indicate failure so downstream code can skip it.
    """

    def setup_method(self):
        self.sim = ReactionSimulator()

    def test_result_has_success_field(self):
        """Every simulation result should have a success boolean."""
        network = self.sim.build_network(["A -> B"])
        result = self.sim.simulate(
            network,
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 10),
        )

        assert hasattr(result, 'success')
        assert isinstance(result.success, bool)

    def test_normal_simulation_succeeds(self):
        """A well-behaved system should have success=True."""
        network = self.sim.build_network(["A -> B"])
        result = self.sim.simulate(
            network,
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 10),
        )

        assert result.success is True

    def test_result_has_solver_message(self):
        """Simulation result should include the solver's message for debugging."""
        network = self.sim.build_network(["A -> B"])
        result = self.sim.simulate(
            network,
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 10),
        )

        assert hasattr(result, 'solver_message')
        assert isinstance(result.solver_message, str)
        assert len(result.solver_message) > 0


# ===========================================================================
# Phase 2 additional tests — Edge cases for Paper 5
# ===========================================================================


class TestStiffSystemAccuracy:
    """Phase 2.2: Stiff systems with widely separated rate constants.

    Paper 5 may involve larger networks with timescales spanning several
    orders of magnitude. The default solver (LSODA) should handle this,
    but these tests verify correctness on known stiff problems.
    """

    def setup_method(self):
        self.sim = ReactionSimulator()

    def test_two_timescale_mass_conservation(self):
        """Fast equilibrium A<->B coupled with slow B->C.

        With k_fwd=k_rev=100 and k_slow=0.01 the fast sub-system
        equilibrates rapidly. Total mass A+B+C should be conserved.
        """
        network = self.sim.build_network(["A -> B", "B -> A", "B -> C"])

        result = self.sim.simulate(
            network,
            rate_constants=[100.0, 100.0, 0.01],
            initial_concentrations={"A": 1.0, "B": 0.0, "C": 0.0},
            t_span=(0, 50),
            n_points=500,
        )

        total = (
            result.get_species("A")
            + result.get_species("B")
            + result.get_species("C")
        )
        assert_allclose(total, 1.0, rtol=1e-4), (
            "Mass conservation violated in stiff system"
        )

    def test_two_timescale_slow_product(self):
        """Slow product C should grow at roughly k_slow * [B]_eq.

        With fast equilibrium A<->B (k=100 each) and slow B->C (k=0.01),
        A and B rapidly equilibrate to [A]=[B]=0.5, then C grows at
        rate ~ 0.01 * 0.5 = 0.005 per unit time.
        """
        network = self.sim.build_network(["A -> B", "B -> A", "B -> C"])

        result = self.sim.simulate(
            network,
            rate_constants=[100.0, 100.0, 0.01],
            initial_concentrations={"A": 1.0, "B": 0.0, "C": 0.0},
            t_span=(0, 50),
            n_points=500,
        )

        C = result.get_species("C")
        # After initial transient (~0.1 time units), C should grow
        # At t=50, C ~ 0.005 * 50 = 0.25 (approximate, A+B are depleting)
        assert C[-1] > 0.1, f"C_final={C[-1]:.4f}, expected > 0.1"
        assert C[-1] < 0.5, f"C_final={C[-1]:.4f}, expected < 0.5"


class TestRemoveTransientBoundary:
    """Phase 2.3: Boundary values for remove_transient parameter.

    Only tested at 0.5 in the existing suite. Extreme values could
    produce empty or near-empty trajectories.
    """

    def setup_method(self):
        self.sim = ReactionSimulator()

    def test_remove_transient_zero(self):
        """remove_transient=0.0 should keep the full trajectory."""
        result = simulate_reactions(
            ["A -> B"],
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 10),
            n_points=100,
            remove_transient=0.0,
        )

        assert len(result.time) == 100
        assert_allclose(result.time[0], 0.0, atol=0.2)

    def test_remove_transient_099(self):
        """remove_transient=0.99 with 100 points.

        The simulator guards against removing too many points
        (requires at least 10 remaining). With n_points=100 and
        remove_transient=0.99, n_remove=99, which exceeds len-10=90,
        so the transient removal is skipped.
        """
        result = simulate_reactions(
            ["A -> B"],
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 10),
            n_points=100,
            remove_transient=0.99,
        )

        # The guard `n_remove < len(t) - 10` prevents removing 99 of 100.
        # So either all 100 points remain, or only the safe amount is removed.
        assert len(result.time) >= 10

    def test_remove_transient_1_0(self):
        """remove_transient=1.0 should not produce an empty trajectory.

        With n_points=100, n_remove=100, which fails the guard
        n_remove < len(t) - 10, so no points are removed.
        """
        result = simulate_reactions(
            ["A -> B"],
            rate_constants=[1.0],
            initial_concentrations={"A": 1.0, "B": 0.0},
            t_span=(0, 10),
            n_points=100,
            remove_transient=1.0,
        )

        # Should NOT be empty — guard prevents it
        assert len(result.time) >= 10
