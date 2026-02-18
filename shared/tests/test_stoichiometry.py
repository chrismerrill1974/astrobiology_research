"""
Tests for stoichiometry module.

Step 1 test cases from the plan:
1. Simple reversible reaction: A <-> B (rank 1, 1 conservation law)
2. Linear chain: A -> B -> C (rank 2, 1 conservation law)
3. Autocatalytic cycle: A + B -> 2B, B -> A (rank 1, stoichiometrically equivalent to A <-> B)
4. Manually constructed matrices with known rank
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from dimensional_opening.stoichiometry import (
    StoichiometricAnalyzer,
    StoichiometricAnalysis,
    compute_rank,
    get_conservation_laws,
)


class TestStoichiometricAnalyzer:
    """Tests for StoichiometricAnalyzer class."""
    
    def setup_method(self):
        self.analyzer = StoichiometricAnalyzer()
    
    # -------------------------------------------------------------------------
    # Test Case 1: Simple reversible reaction A <-> B
    # -------------------------------------------------------------------------
    
    def test_reversible_reaction_rank(self):
        """A <-> B should have rank 1."""
        # Two reactions: A -> B and B -> A
        # S = [[-1, 1], [1, -1]]  (rows: A, B; cols: rxn1, rxn2)
        reactions = ["A -> B", "B -> A"]
        result = self.analyzer.from_reaction_strings(reactions)
        
        assert result.rank == 1
        assert result.n_species == 2
        assert result.n_reactions == 2
    
    def test_reversible_reaction_conservation(self):
        """A <-> B should have 1 conservation law: [A] + [B] = const."""
        reactions = ["A -> B", "B -> A"]
        result = self.analyzer.from_reaction_strings(reactions)
        
        assert result.n_conservation_laws == 1
        
        # The null space vector should be proportional to [1, 1]
        # (meaning A + B is conserved)
        null_vec = result.null_space_basis[:, 0]
        # Check that both components have the same sign and magnitude
        assert np.allclose(np.abs(null_vec[0]), np.abs(null_vec[1]))
        assert np.sign(null_vec[0]) == np.sign(null_vec[1])
    
    def test_reversible_reaction_from_matrix(self):
        """Test from_matrix gives same result."""
        S = np.array([[-1, 1], [1, -1]], dtype=float)
        result = self.analyzer.from_matrix(S)
        
        assert result.rank == 1
        assert result.n_conservation_laws == 1
    
    # -------------------------------------------------------------------------
    # Test Case 2: Linear chain A -> B -> C
    # -------------------------------------------------------------------------
    
    def test_linear_chain_rank(self):
        """A -> B -> C should have rank 2."""
        reactions = ["A -> B", "B -> C"]
        result = self.analyzer.from_reaction_strings(reactions)
        
        assert result.rank == 2
        assert result.n_species == 3
        assert result.n_reactions == 2
    
    def test_linear_chain_conservation(self):
        """A -> B -> C should have 1 conservation law: [A] + [B] + [C] = const."""
        reactions = ["A -> B", "B -> C"]
        result = self.analyzer.from_reaction_strings(reactions)
        
        assert result.n_conservation_laws == 1
        
        # The null space vector should be proportional to [1, 1, 1]
        null_vec = result.null_space_basis[:, 0]
        # All components should have same sign and magnitude
        assert np.allclose(np.abs(null_vec), np.abs(null_vec[0]))
    
    def test_linear_chain_matrix(self):
        """Verify the stoichiometric matrix for A -> B -> C."""
        reactions = ["A -> B", "B -> C"]
        result = self.analyzer.from_reaction_strings(reactions)
        
        # Expected S (rows: A, B, C; cols: rxn1, rxn2):
        # rxn1 (A->B): A decreases, B increases
        # rxn2 (B->C): B decreases, C increases
        expected = np.array([
            [-1,  0],  # A
            [ 1, -1],  # B
            [ 0,  1],  # C
        ], dtype=float)
        
        assert_array_almost_equal(result.stoichiometric_matrix, expected)
    
    # -------------------------------------------------------------------------
    # Test Case 3: Autocatalytic cycle A + B -> 2B, B -> A
    # -------------------------------------------------------------------------
    
    def test_autocatalytic_rank(self):
        """A + B -> 2B, B -> A has rank 1 (reactions are stoichiometrically inverse).
        
        Note: The original plan stated rank=2, but this is incorrect.
        The net stoichiometry for both reactions gives parallel columns:
        - A + B -> 2B: delta = [-1, +1]
        - B -> A:      delta = [+1, -1]
        These are linearly dependent, so rank = 1.
        """
        reactions = ["A + B -> 2B", "B -> A"]
        result = self.analyzer.from_reaction_strings(reactions)
        
        assert result.rank == 1  # Corrected from plan's "rank 2"
        assert result.n_species == 2
        assert result.n_reactions == 2
    
    def test_autocatalytic_matrix(self):
        """Verify the stoichiometric matrix for autocatalytic cycle."""
        reactions = ["A + B -> 2B", "B -> A"]
        result = self.analyzer.from_reaction_strings(reactions)
        
        # rxn1 (A + B -> 2B): A decreases by 1, B increases by 1 (net)
        # rxn2 (B -> A): B decreases by 1, A increases by 1
        expected = np.array([
            [-1,  1],  # A
            [ 1, -1],  # B
        ], dtype=float)
        
        assert_array_almost_equal(result.stoichiometric_matrix, expected)
    
    def test_autocatalytic_conservation(self):
        """Autocatalytic cycle A + B -> 2B, B -> A has 1 conservation law.
        
        This is stoichiometrically equivalent to A <-> B, so [A] + [B] = const.
        """
        reactions = ["A + B -> 2B", "B -> A"]
        result = self.analyzer.from_reaction_strings(reactions)
        
        assert result.n_conservation_laws == 1
        
        # Conservation law should be [1, 1] (up to scaling)
        null_vec = result.null_space_basis[:, 0]
        assert np.allclose(np.abs(null_vec[0]), np.abs(null_vec[1]))
    
    def test_true_rank2_system(self):
        """A proper rank-2 system with 2 species needs different reactions."""
        # To get rank 2 with 2 species, need 2 linearly independent reactions
        # Example: A -> B, A -> 2B (production rates differ)
        # S = [[-1, -1], [1, 2]]
        S = np.array([[-1, -1], [1, 2]], dtype=float)
        result = self.analyzer.from_matrix(S)
        
        assert result.rank == 2
        assert result.n_conservation_laws == 0
    
    # -------------------------------------------------------------------------
    # Test Case 4: Manually constructed matrices with known rank
    # -------------------------------------------------------------------------
    
    def test_rank_zero_matrix(self):
        """Zero matrix has rank 0."""
        S = np.zeros((3, 2))
        result = self.analyzer.from_matrix(S)
        
        assert result.rank == 0
        assert result.n_conservation_laws == 3
    
    def test_full_rank_square(self):
        """Full rank square matrix."""
        S = np.array([[1, 0], [0, 1]], dtype=float)
        result = self.analyzer.from_matrix(S)
        
        assert result.rank == 2
        assert result.n_conservation_laws == 0
    
    def test_rank_deficient_3x3(self):
        """3x3 matrix with rank 2."""
        # Third row is sum of first two
        S = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 2],
        ], dtype=float)
        result = self.analyzer.from_matrix(S)
        
        assert result.rank == 2
        assert result.n_conservation_laws == 1
    
    def test_tall_matrix(self):
        """More species than reactions."""
        S = np.array([
            [-1],
            [ 1],
            [ 0],
        ], dtype=float)  # A -> B, C is inert
        result = self.analyzer.from_matrix(S)
        
        assert result.rank == 1
        assert result.n_conservation_laws == 2  # A+B conserved, C conserved
    
    def test_wide_matrix(self):
        """More reactions than species."""
        S = np.array([
            [-1, 1, -1],
            [ 1, -1, 1],
        ], dtype=float)  # A <-> B via three reactions
        result = self.analyzer.from_matrix(S)
        
        assert result.rank == 1  # All columns are linearly dependent
        assert result.n_conservation_laws == 1
    
    # -------------------------------------------------------------------------
    # Test numerical properties
    # -------------------------------------------------------------------------
    
    def test_singular_values_logged(self):
        """Verify singular values are captured."""
        S = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.analyzer.from_matrix(S)
        
        assert len(result.singular_values) == 2
        assert result.singular_values[0] >= result.singular_values[1]
        assert result.svd_tolerance > 0
    
    def test_integer_coefficients_stable(self):
        """Typical stoichiometric matrices with small integers are stable."""
        # Brusselator-like system
        S = np.array([
            [-1,  1,  0, -1],  # X
            [ 1, -1, -1,  1],  # Y
            [ 0,  0,  1,  0],  # A (if included as species)
        ], dtype=float)
        result = self.analyzer.from_matrix(S)
        
        # Should get clean integer rank
        assert isinstance(result.rank, int)
        assert result.rank + result.n_conservation_laws == result.n_species


class TestReactionStringParsing:
    """Tests for reaction string parsing."""
    
    def setup_method(self):
        self.analyzer = StoichiometricAnalyzer()
    
    def test_simple_reaction(self):
        """Parse A -> B."""
        result = self.analyzer.from_reaction_strings(["A -> B"])
        expected = np.array([[-1], [1]], dtype=float)
        assert_array_almost_equal(result.stoichiometric_matrix, expected)
    
    def test_coefficients(self):
        """Parse 2A -> B."""
        result = self.analyzer.from_reaction_strings(["2A -> B"])
        expected = np.array([[-2], [1]], dtype=float)
        assert_array_almost_equal(result.stoichiometric_matrix, expected)
    
    def test_multiple_reactants(self):
        """Parse A + B -> C."""
        result = self.analyzer.from_reaction_strings(["A + B -> C"])
        expected = np.array([[-1], [-1], [1]], dtype=float)
        assert_array_almost_equal(result.stoichiometric_matrix, expected)
    
    def test_multiple_products(self):
        """Parse A -> B + C."""
        result = self.analyzer.from_reaction_strings(["A -> B + C"])
        expected = np.array([[-1], [1], [1]], dtype=float)
        assert_array_almost_equal(result.stoichiometric_matrix, expected)
    
    def test_complex_reaction(self):
        """Parse 2A + B -> 3C + D."""
        result = self.analyzer.from_reaction_strings(["2A + B -> 3C + D"])
        expected = np.array([[-2], [-1], [3], [1]], dtype=float)
        assert_array_almost_equal(result.stoichiometric_matrix, expected)
    
    def test_species_order_preserved(self):
        """Species order matches first appearance."""
        result = self.analyzer.from_reaction_strings(["X -> Y", "Z -> X"])
        assert result.species_names == ["X", "Y", "Z"]
    
    def test_explicit_species_order(self):
        """Explicit species order is respected."""
        result = self.analyzer.from_reaction_strings(
            ["A -> B"],
            species=["B", "A"]
        )
        expected = np.array([[1], [-1]], dtype=float)  # B first, then A
        assert_array_almost_equal(result.stoichiometric_matrix, expected)
    
    def test_invalid_reaction_no_arrow(self):
        """Reaction without -> raises error."""
        with pytest.raises(ValueError, match="must contain"):
            self.analyzer.from_reaction_strings(["A + B"])
    
    def test_unknown_species(self):
        """Unknown species with explicit list raises error."""
        with pytest.raises(ValueError, match="Unknown species"):
            self.analyzer.from_reaction_strings(["A -> B"], species=["A"])
    
    def test_non_integer_coefficient_rejected(self):
        """Non-integer coefficients like 0.5A should raise error."""
        with pytest.raises(ValueError, match="Non-integer coefficient"):
            self.analyzer.from_reaction_strings(["0.5A -> B"])
    
    def test_scientific_notation_rejected(self):
        """Scientific notation like 1e3A should raise error."""
        with pytest.raises(ValueError, match="Scientific notation not supported"):
            self.analyzer.from_reaction_strings(["1e3A -> B"])


class TestChempyIntegration:
    """Tests for chempy integration."""
    
    def setup_method(self):
        self.analyzer = StoichiometricAnalyzer()
    
    def test_chempy_simple(self):
        """Test with a simple chempy ReactionSystem."""
        pytest.importorskip("chempy")
        from chempy import ReactionSystem, Reaction
        
        # A <-> B
        rxns = [
            Reaction({'A': 1}, {'B': 1}),
            Reaction({'B': 1}, {'A': 1}),
        ]
        rsys = ReactionSystem(rxns, substances='A B'.split())
        
        result = self.analyzer.from_chempy(rsys)
        
        assert result.rank == 1
        assert result.n_species == 2
        assert result.n_reactions == 2
        assert result.n_conservation_laws == 1
    
    def test_chempy_autocatalytic(self):
        """Test autocatalytic reaction with chempy."""
        pytest.importorskip("chempy")
        from chempy import ReactionSystem, Reaction
        
        # A + B -> 2B (autocatalysis)
        # B -> A (decay)
        rxns = [
            Reaction({'A': 1, 'B': 1}, {'B': 2}),
            Reaction({'B': 1}, {'A': 1}),
        ]
        rsys = ReactionSystem(rxns, substances='A B'.split())
        
        result = self.analyzer.from_chempy(rsys)
        
        # Same as reversible: rank 1
        assert result.rank == 1
    
    def test_chempy_brusselator(self):
        """Test Brusselator system."""
        pytest.importorskip("chempy")
        from chempy import ReactionSystem, Reaction
        
        # Brusselator:
        # A -> X
        # 2X + Y -> 3X
        # B + X -> Y + D
        # X -> E
        # (A, B held constant as food; D, E are waste)
        rxns = [
            Reaction({'A': 1}, {'X': 1}),
            Reaction({'X': 2, 'Y': 1}, {'X': 3}),
            Reaction({'B': 1, 'X': 1}, {'Y': 1, 'D': 1}),
            Reaction({'X': 1}, {'E': 1}),
        ]
        rsys = ReactionSystem(rxns, substances='A B X Y D E'.split())
        
        result = self.analyzer.from_chempy(rsys)
        
        # 6 species, 4 reactions
        assert result.n_species == 6
        assert result.n_reactions == 4


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_compute_rank(self):
        """Test compute_rank convenience function."""
        S = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        assert compute_rank(S) == 2
    
    def test_get_conservation_laws(self):
        """Test get_conservation_laws convenience function."""
        # A -> B: conservation law is [1, 1] (A + B = const)
        S = np.array([[-1], [1]], dtype=float)
        laws = get_conservation_laws(S)
        
        assert laws.shape == (2, 1)
        # Both entries should have same magnitude
        assert np.allclose(np.abs(laws[0]), np.abs(laws[1]))


class TestEdgeCases:
    """Tests for edge cases."""
    
    def setup_method(self):
        self.analyzer = StoichiometricAnalyzer()
    
    def test_single_species_single_reaction(self):
        """Single species, single reaction: A -> A (null reaction)."""
        S = np.array([[0]], dtype=float)
        result = self.analyzer.from_matrix(S)
        
        assert result.rank == 0
        assert result.n_conservation_laws == 1
    
    def test_1d_matrix(self):
        """1D input should raise error."""
        with pytest.raises(ValueError, match="must be 2D"):
            self.analyzer.from_matrix(np.array([1, 2, 3]))
    
    def test_empty_reactions_list(self):
        """Empty reaction list."""
        with pytest.raises((ValueError, IndexError)):
            self.analyzer.from_reaction_strings([])
