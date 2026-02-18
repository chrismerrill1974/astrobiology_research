"""
Stoichiometric rank calculator for reaction networks.

Step 1 of the Dimensional Activation pipeline.

Computes:
- Stoichiometric rank r_S (structural upper bound on dynamical dimensionality)
- Conservation law space dimension (m - r_S)
- Null space basis (conservation law vectors)
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import null_space
import scipy
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class StoichiometricAnalysis:
    """Results of stoichiometric analysis."""
    
    # Core results
    rank: int                           # r_S
    n_species: int                      # m
    n_reactions: int                    # n
    n_conservation_laws: int            # m - r_S
    
    # The matrix and its decomposition
    stoichiometric_matrix: np.ndarray   # S (m x n)
    null_space_basis: np.ndarray        # Basis for ker(S^T), shape (m, m-r_S)
    singular_values: np.ndarray         # Full singular spectrum
    
    # Numerical details for reproducibility
    svd_tolerance: float
    numpy_version: str = ""
    scipy_version: str = ""
    
    # Optional metadata
    species_names: Optional[List[str]] = None
    reaction_names: Optional[List[str]] = None
    
    def __repr__(self) -> str:
        return (
            f"StoichiometricAnalysis(\n"
            f"  rank={self.rank}, n_species={self.n_species}, "
            f"n_reactions={self.n_reactions},\n"
            f"  n_conservation_laws={self.n_conservation_laws},\n"
            f"  svd_tolerance={self.svd_tolerance:.2e},\n"
            f"  singular_values={self.singular_values},\n"
            f"  numpy={self.numpy_version}, scipy={self.scipy_version}\n"
            f")"
        )


class StoichiometricAnalyzer:
    """
    Analyzer for stoichiometric properties of reaction networks.
    
    Computes rank, conservation laws, and related quantities from:
    - A stoichiometric matrix (NumPy array)
    - A list of reaction strings
    - A chempy.ReactionSystem object
    
    Examples
    --------
    >>> analyzer = StoichiometricAnalyzer()
    
    # From matrix
    >>> S = np.array([[-1, 1], [1, -1]])  # A <-> B
    >>> result = analyzer.from_matrix(S)
    >>> result.rank
    1
    
    # From reaction strings
    >>> reactions = ["A -> B", "B -> A"]
    >>> result = analyzer.from_reaction_strings(reactions)
    >>> result.rank
    1
    """
    
    def from_matrix(
        self,
        S: ArrayLike,
        species_names: Optional[List[str]] = None,
        reaction_names: Optional[List[str]] = None,
    ) -> StoichiometricAnalysis:
        """
        Analyze a stoichiometric matrix directly.
        
        Parameters
        ----------
        S : array_like
            Stoichiometric matrix of shape (n_species, n_reactions).
            Entry S[i,j] is the net change in species i for reaction j.
        species_names : list of str, optional
            Names for each species (rows).
        reaction_names : list of str, optional
            Names for each reaction (columns).
            
        Returns
        -------
        StoichiometricAnalysis
            Complete analysis results.
        """
        S = np.asarray(S, dtype=float)
        
        if S.ndim != 2:
            raise ValueError(f"S must be 2D, got shape {S.shape}")
        
        m, n = S.shape  # m species, n reactions
        
        # Compute SVD for rank determination
        singular_values = np.linalg.svd(S, compute_uv=False)
        
        # Compute tolerance (NumPy default convention)
        eps = np.finfo(S.dtype).eps
        tol = max(m, n) * eps * singular_values[0] if singular_values[0] > 0 else eps
        
        # Rank = number of singular values above tolerance
        rank = int(np.sum(singular_values > tol))
        
        # Conservation law space = null space of S^T
        # dim(ker(S^T)) = m - rank(S)
        null_basis = null_space(S.T)
        
        # Validate: null space dimension should equal m - rank
        expected_null_dim = m - rank
        actual_null_dim = null_basis.shape[1] if null_basis.size > 0 else 0
        
        if actual_null_dim != expected_null_dim:
            # This shouldn't happen with consistent tolerances, but check anyway
            import warnings
            warnings.warn(
                f"Null space dimension mismatch: expected {expected_null_dim}, "
                f"got {actual_null_dim}. May indicate numerical issues."
            )
        
        return StoichiometricAnalysis(
            rank=rank,
            n_species=m,
            n_reactions=n,
            n_conservation_laws=expected_null_dim,
            stoichiometric_matrix=S,
            null_space_basis=null_basis,
            singular_values=singular_values,
            svd_tolerance=tol,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            species_names=species_names,
            reaction_names=reaction_names,
        )
    
    def from_reaction_strings(
        self,
        reactions: List[str],
        species: Optional[List[str]] = None,
    ) -> StoichiometricAnalysis:
        """
        Analyze a reaction network from reaction strings.
        
        Parameters
        ----------
        reactions : list of str
            Reaction strings like "A + B -> C" or "2A -> B + C".
            Supports "->" for irreversible reactions.
            For reversible, use two separate reactions.
        species : list of str, optional
            Ordered list of species names. If not provided, extracted
            from reactions in order of first appearance.
            
        Returns
        -------
        StoichiometricAnalysis
            Complete analysis results.
            
        Examples
        --------
        >>> analyzer = StoichiometricAnalyzer()
        >>> result = analyzer.from_reaction_strings(["A + B -> 2C", "C -> A"])
        >>> result.rank
        2
        """
        S, species_names, reaction_names = self._parse_reactions(reactions, species)
        return self.from_matrix(S, species_names=species_names, reaction_names=reaction_names)
    
    def from_chempy(self, reaction_system) -> StoichiometricAnalysis:
        """
        Analyze a chempy.ReactionSystem.
        
        Parameters
        ----------
        reaction_system : chempy.ReactionSystem
            A chempy reaction system object.
            
        Returns
        -------
        StoichiometricAnalysis
            Complete analysis results.
        """
        # Get species and reactions
        species_names = list(reaction_system.substances.keys())
        species_index = {name: i for i, name in enumerate(species_names)}  # O(1) lookup
        n_species = len(species_names)
        n_reactions = len(reaction_system.rxns)
        
        # Build stoichiometric matrix
        # S[i,j] = net change in species i for reaction j
        S = np.zeros((n_species, n_reactions))
        
        for j, rxn in enumerate(reaction_system.rxns):
            # Net stoichiometry = products - reactants
            for species, coeff in rxn.prod.items():
                S[species_index[species], j] += coeff
            for species, coeff in rxn.reac.items():
                S[species_index[species], j] -= coeff
        
        reaction_names = [str(rxn) for rxn in reaction_system.rxns]
        
        return self.from_matrix(S, species_names=species_names, reaction_names=reaction_names)
    
    def _parse_reactions(
        self,
        reactions: List[str],
        species: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Parse reaction strings into stoichiometric matrix.
        
        Handles formats like:
        - "A -> B"
        - "A + B -> C"
        - "2A -> B + C"
        - "A + 2B -> 3C"
        """
        # First pass: collect all species if not provided
        if species is None:
            species_set = []
            for rxn in reactions:
                for side in rxn.replace("->", "+").split("+"):
                    s = self._parse_species_term(side.strip())
                    if s and s[1] not in species_set:
                        species_set.append(s[1])
            species_names = species_set
        else:
            species_names = list(species)
        
        n_species = len(species_names)
        n_reactions = len(reactions)
        S = np.zeros((n_species, n_reactions))
        
        for j, rxn in enumerate(reactions):
            if "->" not in rxn:
                raise ValueError(f"Reaction must contain '->': {rxn}")
            
            left, right = rxn.split("->")
            
            # Parse reactants (negative contribution)
            for term in left.split("+"):
                parsed = self._parse_species_term(term.strip())
                if parsed:
                    coeff, name = parsed
                    if name not in species_names:
                        raise ValueError(f"Unknown species '{name}' in reaction: {rxn}")
                    i = species_names.index(name)
                    S[i, j] -= coeff
            
            # Parse products (positive contribution)
            for term in right.split("+"):
                parsed = self._parse_species_term(term.strip())
                if parsed:
                    coeff, name = parsed
                    if name not in species_names:
                        raise ValueError(f"Unknown species '{name}' in reaction: {rxn}")
                    i = species_names.index(name)
                    S[i, j] += coeff
        
        return S, species_names, reactions
    
    def _parse_species_term(self, term: str) -> Optional[Tuple[int, str]]:
        """
        Parse a term like "2A" or "B" into (coefficient, species_name).
        
        Returns None for empty terms.
        
        Raises
        ------
        ValueError
            If coefficient is not a positive integer, or species name is malformed.
        """
        term = term.strip()
        if not term:
            return None
        
        # Check for decimal point anywhere (non-integer coefficient)
        if '.' in term:
            # Check if it's in a coefficient position (before species name)
            first_alpha = next((i for i, c in enumerate(term) if c.isalpha()), len(term))
            if '.' in term[:first_alpha]:
                raise ValueError(
                    f"Non-integer coefficient not supported: '{term}'. "
                    f"Use integer stoichiometric coefficients only."
                )
        
        # Find where the coefficient ends and species name begins
        i = 0
        while i < len(term) and term[i].isdigit():
            i += 1
        
        if i == 0:
            # No coefficient, default to 1
            # But first character must be alphabetic (valid species name start)
            if not term[0].isalpha():
                raise ValueError(
                    f"Species name must start with a letter: '{term}'"
                )
            return (1, term)
        elif i == len(term):
            raise ValueError(f"Invalid term (no species name): {term}")
        else:
            coeff = int(term[:i])
            if coeff <= 0:
                raise ValueError(f"Coefficient must be positive: '{term}'")
            name = term[i:].strip()
            
            # Check for scientific notation attempt: coefficient followed by e/E and digit
            # e.g., "1e3A" would have name="e3A" which looks like sci notation
            if name and name[0].lower() == 'e' and len(name) > 1 and name[1].isdigit():
                raise ValueError(
                    f"Scientific notation not supported in coefficients: '{term}'. "
                    f"Use integer stoichiometric coefficients only."
                )
            
            # Species name must start with a letter
            if not name or not name[0].isalpha():
                raise ValueError(
                    f"Species name must start with a letter: '{term}'"
                )
            return (coeff, name)


def compute_rank(S: ArrayLike) -> int:
    """
    Convenience function to compute stoichiometric rank.
    
    Parameters
    ----------
    S : array_like
        Stoichiometric matrix.
        
    Returns
    -------
    int
        Rank of the matrix.
    """
    analyzer = StoichiometricAnalyzer()
    return analyzer.from_matrix(S).rank


def get_conservation_laws(S: ArrayLike) -> np.ndarray:
    """
    Convenience function to get conservation law vectors.
    
    Parameters
    ----------
    S : array_like
        Stoichiometric matrix.
        
    Returns
    -------
    np.ndarray
        Null space basis of S^T. Each column is a conservation law.
        For example, if column is [1, 1, 0], then species 0 + species 1 is conserved.
    """
    analyzer = StoichiometricAnalyzer()
    return analyzer.from_matrix(S).null_space_basis
