"""
Reaction network simulator with non-equilibrium driving.

Step 2 of the Dimensional Activation pipeline.

Integrates mass-action kinetics ODEs with support for:
- Chemostatted species (fixed concentrations)
- CSTR-style inflow/outflow
- Energy carrier species

Critical design choice: Without explicit driving, most mass-action networks
relax to fixed points (D2 ≈ 0) or at best limit cycles (D2 ≈ 1), which is
uninformative for our purposes.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple, Union
from enum import Enum
import warnings


class DrivingMode(Enum):
    """Non-equilibrium driving modes."""
    NONE = "none"                    # No driving (will likely go to fixed point)
    CHEMOSTAT = "chemostat"          # Fixed concentrations for food species
    CSTR = "cstr"                    # Continuous stirred tank reactor (inflow/outflow)
    # ENERGY_CARRIER = "energy"      # Future: explicit ATP-like species


@dataclass
class SimulationResult:
    """Results of a reaction network simulation."""
    
    # Time and concentration data
    time: np.ndarray                          # Shape (n_times,)
    concentrations: np.ndarray                # Shape (n_times, n_dynamic_species)
    
    # Species information
    species_names: List[str]                  # Names of dynamic species
    n_species: int                            # Number of dynamic species
    
    # Simulation metadata
    driving_mode: DrivingMode
    driving_params: Dict
    rate_constants: np.ndarray
    initial_concentrations: np.ndarray
    
    # Integration metadata
    solver_message: str
    n_function_evals: int
    success: bool
    
    # Version info for reproducibility
    numpy_version: str = ""
    scipy_version: str = ""
    
    def __repr__(self) -> str:
        return (
            f"SimulationResult(\n"
            f"  n_times={len(self.time)}, n_species={self.n_species},\n"
            f"  t=[{self.time[0]:.2f}, {self.time[-1]:.2f}],\n"
            f"  driving={self.driving_mode.value},\n"
            f"  success={self.success}\n"
            f")"
        )
    
    def get_species(self, name: str) -> np.ndarray:
        """Get concentration time series for a specific species."""
        if name not in self.species_names:
            raise KeyError(f"Species '{name}' not found. Available: {self.species_names}")
        idx = self.species_names.index(name)
        return self.concentrations[:, idx]


@dataclass 
class ReactionNetwork:
    """
    A reaction network ready for simulation.
    
    Separates network topology from simulation parameters.
    """
    # Network structure
    species_names: List[str]
    n_species: int
    n_reactions: int
    
    # Stoichiometry: S[i,j] = net change in species i for reaction j
    stoichiometric_matrix: np.ndarray
    
    # Reactant stoichiometry for rate laws: R[i,j] = stoich coeff of species i as reactant in rxn j
    reactant_matrix: np.ndarray
    
    # Optional: reaction strings for documentation
    reaction_strings: Optional[List[str]] = None


class ReactionSimulator:
    """
    Simulator for reaction networks with mass-action kinetics.
    
    Supports non-equilibrium driving via:
    - Chemostatted species (fixed concentrations)
    - CSTR inflow/outflow terms
    
    Examples
    --------
    >>> sim = ReactionSimulator()
    >>> 
    >>> # Build network from reaction strings
    >>> network = sim.build_network(["A -> B", "B -> A"])
    >>> 
    >>> # Simulate with chemostat driving
    >>> result = sim.simulate(
    ...     network,
    ...     rate_constants=[1.0, 0.5],
    ...     initial_concentrations={"A": 1.0, "B": 0.0},
    ...     t_span=(0, 10),
    ...     driving_mode=DrivingMode.CHEMOSTAT,
    ...     chemostat_species={"A": 1.0},  # Hold A fixed
    ... )
    """
    
    def build_network(
        self,
        reactions: List[str],
        species: Optional[List[str]] = None,
    ) -> ReactionNetwork:
        """
        Build a reaction network from reaction strings.
        
        Parameters
        ----------
        reactions : list of str
            Reaction strings like "A + B -> C" or "2A -> B".
        species : list of str, optional
            Ordered species list. If not provided, extracted from reactions.
            
        Returns
        -------
        ReactionNetwork
            Network ready for simulation.
        """
        # Parse to get stoichiometric matrix and reactant matrix
        S, R, species_names = self._parse_reactions(reactions, species)
        
        return ReactionNetwork(
            species_names=species_names,
            n_species=len(species_names),
            n_reactions=len(reactions),
            stoichiometric_matrix=S,
            reactant_matrix=R,
            reaction_strings=reactions,
        )
    
    def build_network_from_chempy(self, reaction_system) -> ReactionNetwork:
        """
        Build a reaction network from a chempy.ReactionSystem.
        
        Parameters
        ----------
        reaction_system : chempy.ReactionSystem
            A chempy reaction system.
            
        Returns
        -------
        ReactionNetwork
            Network ready for simulation.
        """
        species_names = list(reaction_system.substances.keys())
        species_index = {name: i for i, name in enumerate(species_names)}
        n_species = len(species_names)
        n_reactions = len(reaction_system.rxns)
        
        S = np.zeros((n_species, n_reactions))
        R = np.zeros((n_species, n_reactions))
        
        for j, rxn in enumerate(reaction_system.rxns):
            for species, coeff in rxn.prod.items():
                S[species_index[species], j] += coeff
            for species, coeff in rxn.reac.items():
                S[species_index[species], j] -= coeff
                R[species_index[species], j] = coeff
        
        return ReactionNetwork(
            species_names=species_names,
            n_species=n_species,
            n_reactions=n_reactions,
            stoichiometric_matrix=S,
            reactant_matrix=R,
            reaction_strings=[str(rxn) for rxn in reaction_system.rxns],
        )
    
    def simulate(
        self,
        network: ReactionNetwork,
        rate_constants: ArrayLike,
        initial_concentrations: Union[ArrayLike, Dict[str, float]],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        n_points: int = 1000,
        driving_mode: DrivingMode = DrivingMode.NONE,
        chemostat_species: Optional[Dict[str, float]] = None,
        cstr_dilution_rate: float = 0.0,
        cstr_feed_concentrations: Optional[Dict[str, float]] = None,
        solver: str = "LSODA",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: float = np.inf,
        remove_transient: float = 0.0,
    ) -> SimulationResult:
        """
        Simulate the reaction network.
        
        Parameters
        ----------
        network : ReactionNetwork
            The reaction network to simulate.
        rate_constants : array_like
            Rate constants for each reaction, shape (n_reactions,).
        initial_concentrations : array_like or dict
            Initial concentrations. Either array (same order as species_names)
            or dict mapping species names to concentrations.
        t_span : tuple
            (t_start, t_end) for integration.
        t_eval : array_like, optional
            Times at which to store solution. If None, uses n_points equally spaced.
        n_points : int
            Number of output points if t_eval not specified.
        driving_mode : DrivingMode
            Type of non-equilibrium driving.
        chemostat_species : dict, optional
            For CHEMOSTAT mode: {species_name: fixed_concentration}.
        cstr_dilution_rate : float
            For CSTR mode: dilution rate D.
        cstr_feed_concentrations : dict, optional
            For CSTR mode: {species_name: feed_concentration}. Species not
            listed have feed concentration 0.
        solver : str
            ODE solver. "LSODA" recommended (handles stiff/non-stiff).
        rtol, atol : float
            Relative and absolute tolerances.
        max_step : float
            Maximum step size.
        remove_transient : float
            Fraction of trajectory to remove from start (0 to 1).
            
        Returns
        -------
        SimulationResult
            Simulation results including time series and metadata.
        """
        import scipy
        
        k = np.asarray(rate_constants, dtype=float)
        if len(k) != network.n_reactions:
            raise ValueError(
                f"Expected {network.n_reactions} rate constants, got {len(k)}"
            )
        
        # Convert initial concentrations to array
        if isinstance(initial_concentrations, dict):
            c0 = np.zeros(network.n_species)
            for name, conc in initial_concentrations.items():
                if name not in network.species_names:
                    raise ValueError(f"Unknown species: {name}")
                idx = network.species_names.index(name)
                c0[idx] = conc
        else:
            c0 = np.asarray(initial_concentrations, dtype=float)
            if len(c0) != network.n_species:
                raise ValueError(
                    f"Expected {network.n_species} initial concentrations, got {len(c0)}"
                )
        
        # Validate driving parameters
        if driving_mode == DrivingMode.CHEMOSTAT:
            if chemostat_species is None:
                raise ValueError("chemostat_species required for CHEMOSTAT mode")
            for name, val in chemostat_species.items():
                if val < 0:
                    raise ValueError(f"Chemostat concentration must be >= 0: {name}={val}")
                # Warn if initial concentration disagrees with fixed value
                if name in network.species_names:
                    idx = network.species_names.index(name)
                    if abs(c0[idx] - val) > 1e-10:
                        warnings.warn(
                            f"Initial concentration for chemostatted species '{name}' "
                            f"({c0[idx]}) overridden by fixed value ({val})."
                        )
        
        if driving_mode == DrivingMode.CSTR:
            if cstr_dilution_rate is None:
                cstr_dilution_rate = 0.0
            if cstr_dilution_rate < 0:
                raise ValueError(f"Dilution rate must be >= 0: {cstr_dilution_rate}")
            if cstr_dilution_rate == 0:
                warnings.warn(
                    "CSTR dilution rate is 0, which is effectively DrivingMode.NONE. "
                    "Consider using DrivingMode.NONE explicitly."
                )
            if cstr_feed_concentrations:
                for name, val in cstr_feed_concentrations.items():
                    if val < 0:
                        raise ValueError(f"Feed concentration must be >= 0: {name}={val}")
        
        # Set up time evaluation points
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], n_points)
        else:
            t_eval = np.asarray(t_eval)
        
        # Build the RHS function based on driving mode
        if driving_mode == DrivingMode.NONE:
            rhs = self._build_rhs_basic(network, k)
            driving_params = {}
            
        elif driving_mode == DrivingMode.CHEMOSTAT:
            if chemostat_species is None:
                raise ValueError("chemostat_species required for CHEMOSTAT mode")
            rhs, dynamic_mask = self._build_rhs_chemostat(
                network, k, chemostat_species
            )
            driving_params = {"chemostat_species": chemostat_species}
            # Adjust c0 and species list for dynamic species only
            c0_dynamic = c0[dynamic_mask]
            dynamic_names = [n for n, m in zip(network.species_names, dynamic_mask) if m]
            
        elif driving_mode == DrivingMode.CSTR:
            if cstr_feed_concentrations is None:
                cstr_feed_concentrations = {}
            rhs = self._build_rhs_cstr(
                network, k, cstr_dilution_rate, cstr_feed_concentrations
            )
            driving_params = {
                "dilution_rate": cstr_dilution_rate,
                "feed_concentrations": cstr_feed_concentrations,
            }
        else:
            raise ValueError(f"Unknown driving mode: {driving_mode}")
        
        # Select initial conditions based on mode
        if driving_mode == DrivingMode.CHEMOSTAT:
            y0 = c0_dynamic
            output_species = dynamic_names
        else:
            y0 = c0
            output_species = network.species_names
        
        # Integrate
        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method=solver,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        
        if not sol.success:
            warnings.warn(f"Integration warning: {sol.message}")
        
        # Extract results
        t = sol.t
        c = sol.y.T  # Shape (n_times, n_species)
        
        # Remove transient if requested
        if remove_transient > 0:
            n_remove = int(len(t) * remove_transient)
            if n_remove > 0 and n_remove < len(t) - 10:
                t = t[n_remove:]
                c = c[n_remove:]
        
        return SimulationResult(
            time=t,
            concentrations=c,
            species_names=output_species,
            n_species=len(output_species),
            driving_mode=driving_mode,
            driving_params=driving_params,
            rate_constants=k,
            initial_concentrations=c0,
            solver_message=sol.message,
            n_function_evals=sol.nfev,
            success=sol.success,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
        )
    
    def _build_rhs_basic(
        self,
        network: ReactionNetwork,
        k: np.ndarray,
    ) -> Callable:
        """Build RHS function for basic mass-action kinetics (no driving)."""
        S = network.stoichiometric_matrix
        R = network.reactant_matrix
        
        def rhs(t, c):
            # Ensure non-negative concentrations for numerical stability
            c = np.maximum(c, 0)
            # Reaction rates: r_j = k_j * prod_i c_i^R_{ij}
            rates = k * np.prod(np.power(c[:, np.newaxis], R), axis=0)
            # dc/dt = S @ rates
            return S @ rates
        
        return rhs
    
    def _build_rhs_chemostat(
        self,
        network: ReactionNetwork,
        k: np.ndarray,
        chemostat_species: Dict[str, float],
    ) -> Tuple[Callable, np.ndarray]:
        """
        Build RHS function for chemostat mode.
        
        Chemostatted species have fixed concentrations and are removed from
        the ODE system. They appear as constants in the rate expressions.
        
        Returns
        -------
        rhs : callable
            RHS function for dynamic species only.
        dynamic_mask : np.ndarray
            Boolean mask indicating which species are dynamic.
        """
        S = network.stoichiometric_matrix
        R = network.reactant_matrix
        n_species = network.n_species
        
        # Identify chemostatted vs dynamic species
        chemostat_indices = []
        chemostat_values = []
        for name, conc in chemostat_species.items():
            if name not in network.species_names:
                raise ValueError(f"Unknown chemostat species: {name}")
            idx = network.species_names.index(name)
            chemostat_indices.append(idx)
            chemostat_values.append(conc)
        
        chemostat_indices = np.array(chemostat_indices)
        chemostat_values = np.array(chemostat_values)
        
        dynamic_mask = np.ones(n_species, dtype=bool)
        dynamic_mask[chemostat_indices] = False
        dynamic_indices = np.where(dynamic_mask)[0]
        
        # Reduced matrices for dynamic species
        S_dynamic = S[dynamic_mask, :]
        
        def rhs(t, c_dynamic):
            # Reconstruct full concentration vector
            c_full = np.zeros(n_species)
            c_full[dynamic_indices] = np.maximum(c_dynamic, 0)
            c_full[chemostat_indices] = chemostat_values
            
            # Reaction rates
            rates = k * np.prod(np.power(c_full[:, np.newaxis], R), axis=0)
            
            # Return only dynamic species derivatives
            return S_dynamic @ rates
        
        return rhs, dynamic_mask
    
    def _build_rhs_cstr(
        self,
        network: ReactionNetwork,
        k: np.ndarray,
        D: float,
        feed_concentrations: Dict[str, float],
    ) -> Callable:
        """
        Build RHS function for CSTR mode.
        
        dc_i/dt = (reaction terms) + D * (c_i_feed - c_i)
        
        where D is the dilution rate.
        """
        S = network.stoichiometric_matrix
        R = network.reactant_matrix
        
        # Build feed concentration vector
        c_feed = np.zeros(network.n_species)
        for name, conc in feed_concentrations.items():
            if name not in network.species_names:
                raise ValueError(f"Unknown feed species: {name}")
            idx = network.species_names.index(name)
            c_feed[idx] = conc
        
        def rhs(t, c):
            c = np.maximum(c, 0)
            # Reaction rates
            rates = k * np.prod(np.power(c[:, np.newaxis], R), axis=0)
            # dc/dt = S @ rates + D * (c_feed - c)
            return S @ rates + D * (c_feed - c)
        
        return rhs
    
    def _parse_reactions(
        self,
        reactions: List[str],
        species: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Parse reaction strings into stoichiometric and reactant matrices.
        
        Returns
        -------
        S : np.ndarray
            Stoichiometric matrix (net change), shape (n_species, n_reactions).
        R : np.ndarray
            Reactant matrix (reactant coefficients), shape (n_species, n_reactions).
        species_names : list of str
            Species names in order.
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
        R = np.zeros((n_species, n_reactions))
        
        # O(1) lookup for species index
        species_index = {name: i for i, name in enumerate(species_names)}
        
        for j, rxn in enumerate(reactions):
            if "->" not in rxn:
                raise ValueError(f"Reaction must contain '->': {rxn}")
            
            left, right = rxn.split("->")
            
            # Parse reactants
            for term in left.split("+"):
                parsed = self._parse_species_term(term.strip())
                if parsed:
                    coeff, name = parsed
                    if name not in species_index:
                        raise ValueError(f"Unknown species '{name}' in reaction: {rxn}")
                    i = species_index[name]
                    S[i, j] -= coeff      # Net: reactants consumed
                    R[i, j] += coeff      # Reactant stoichiometry for rate law
            
            # Parse products
            for term in right.split("+"):
                parsed = self._parse_species_term(term.strip())
                if parsed:
                    coeff, name = parsed
                    if name not in species_index:
                        raise ValueError(f"Unknown species '{name}' in reaction: {rxn}")
                    i = species_index[name]
                    S[i, j] += coeff      # Net: products produced
        
        return S, R, species_names
    
    def _parse_species_term(self, term: str) -> Optional[Tuple[int, str]]:
        """Parse a term like "2A" or "B" into (coefficient, species_name)."""
        term = term.strip()
        if not term:
            return None
        
        # Check for decimal point (non-integer coefficient)
        if '.' in term:
            first_alpha = next((i for i, c in enumerate(term) if c.isalpha()), len(term))
            if '.' in term[:first_alpha]:
                raise ValueError(
                    f"Non-integer coefficient not supported: '{term}'. "
                    f"Use integer stoichiometric coefficients only."
                )
        
        i = 0
        while i < len(term) and term[i].isdigit():
            i += 1
        
        if i == 0:
            if not term[0].isalpha():
                raise ValueError(f"Species name must start with a letter: '{term}'")
            return (1, term)
        elif i == len(term):
            raise ValueError(f"Invalid term (no species name): {term}")
        else:
            coeff = int(term[:i])
            if coeff <= 0:
                raise ValueError(f"Coefficient must be positive: '{term}'")
            name = term[i:].strip()
            if name and name[0].lower() == 'e' and len(name) > 1 and name[1].isdigit():
                raise ValueError(
                    f"Scientific notation not supported: '{term}'. "
                    f"Use integer coefficients only."
                )
            if not name or not name[0].isalpha():
                raise ValueError(f"Species name must start with a letter: '{term}'")
            return (coeff, name)


# Convenience functions

def simulate_reactions(
    reactions: List[str],
    rate_constants: ArrayLike,
    initial_concentrations: Dict[str, float],
    t_span: Tuple[float, float],
    **kwargs
) -> SimulationResult:
    """
    Convenience function to simulate a reaction network.
    
    Parameters
    ----------
    reactions : list of str
        Reaction strings.
    rate_constants : array_like
        Rate constants for each reaction.
    initial_concentrations : dict
        {species_name: initial_concentration}.
    t_span : tuple
        (t_start, t_end).
    **kwargs
        Additional arguments passed to ReactionSimulator.simulate().
        
    Returns
    -------
    SimulationResult
        Simulation results.
    """
    sim = ReactionSimulator()
    network = sim.build_network(reactions)
    return sim.simulate(network, rate_constants, initial_concentrations, t_span, **kwargs)
