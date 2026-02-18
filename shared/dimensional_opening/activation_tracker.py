"""
Activation Tracker: Integration layer for dimensional activation analysis.

Step 6 of the Dimensional Activation pipeline.

Combines stoichiometry, simulation, and correlation dimension analysis
to track the activation ratio η = D2 / r_S across reaction networks.

Design principles (from plan):
- Sequential execution (parallelization is premature optimization)
- Checkpointing: save results after each network
- Descriptive statistics with uncertainty, not p-values
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import json
import csv
from datetime import datetime
import warnings

from .stoichiometry import StoichiometricAnalyzer, StoichiometricAnalysis
from .simulator import ReactionSimulator, ReactionNetwork, SimulationResult, DrivingMode
from .correlation_dimension import (
    CorrelationDimension,
    CorrelationDimensionResult,
    EnsembleResult,
    QualityFlag,
    compute_activation_ratio,
    compute_D2_ensemble,
)


@dataclass
class ActivationResult:
    """Results from activation analysis of a single network."""
    
    # Network info
    network_id: str
    reactions: List[str]
    species: List[str]
    n_reactions: int
    n_species: int
    
    # Stoichiometric analysis
    r_S: int                            # Stoichiometric rank
    n_conservation_laws: int
    
    # Correlation dimension
    D2: float
    D2_uncertainty: float
    quality: QualityFlag
    
    # Activation ratio
    eta: float                          # η = D2 / r_S
    
    # Metadata
    driving_mode: str
    simulation_time: float
    n_trajectory_points: int
    theiler_window: int
    random_state: Optional[int] = None
    
    # Optional ensemble stats (if multiple runs)
    ensemble_D2_median: Optional[float] = None
    ensemble_D2_iqr: Optional[float] = None
    ensemble_success_rate: Optional[float] = None
    
    # Flags
    skipped: bool = False
    skip_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'network_id': self.network_id,
            'reactions': self.reactions,
            'species': self.species,
            'n_reactions': self.n_reactions,
            'n_species': self.n_species,
            'r_S': self.r_S,
            'n_conservation_laws': self.n_conservation_laws,
            'D2': self.D2 if not np.isnan(self.D2) else None,
            'D2_uncertainty': self.D2_uncertainty if not np.isnan(self.D2_uncertainty) else None,
            'quality': self.quality.value,
            'eta': self.eta if not np.isnan(self.eta) else None,
            'driving_mode': self.driving_mode,
            'simulation_time': self.simulation_time,
            'n_trajectory_points': self.n_trajectory_points,
            'theiler_window': self.theiler_window,
            'random_state': self.random_state,
            'ensemble_D2_median': self.ensemble_D2_median,
            'ensemble_D2_iqr': self.ensemble_D2_iqr,
            'ensemble_success_rate': self.ensemble_success_rate,
            'skipped': self.skipped,
            'skip_reason': self.skip_reason,
        }
    
    def __repr__(self) -> str:
        if self.skipped:
            return f"ActivationResult({self.network_id}: SKIPPED - {self.skip_reason})"
        return (
            f"ActivationResult(\n"
            f"  network_id={self.network_id},\n"
            f"  r_S={self.r_S}, D2={self.D2:.3f}, η={self.eta:.3f},\n"
            f"  quality={self.quality.value}\n"
            f")"
        )


@dataclass 
class BatchResult:
    """Results from batch activation analysis."""
    
    results: List[ActivationResult]
    
    # Summary statistics
    n_total: int
    n_analyzed: int                     # Not skipped
    n_successful: int                   # GOOD or MARGINAL quality
    
    # Aggregate stats (computed on successful only)
    eta_median: float
    eta_iqr: float
    eta_mean: float
    eta_std: float
    
    D2_median: float
    D2_iqr: float
    
    r_S_median: float
    r_S_range: Tuple[int, int]
    
    # Metadata
    timestamp: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'n_total': self.n_total,
            'n_analyzed': self.n_analyzed,
            'n_successful': self.n_successful,
            'eta_median': self.eta_median if not np.isnan(self.eta_median) else None,
            'eta_iqr': self.eta_iqr if not np.isnan(self.eta_iqr) else None,
            'eta_mean': self.eta_mean if not np.isnan(self.eta_mean) else None,
            'eta_std': self.eta_std if not np.isnan(self.eta_std) else None,
            'D2_median': self.D2_median if not np.isnan(self.D2_median) else None,
            'D2_iqr': self.D2_iqr if not np.isnan(self.D2_iqr) else None,
            'r_S_median': self.r_S_median if not np.isnan(self.r_S_median) else None,
            'r_S_range': list(self.r_S_range),
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'results': [r.to_dict() for r in self.results],
        }
    
    def __repr__(self) -> str:
        return (
            f"BatchResult(\n"
            f"  n_successful={self.n_successful}/{self.n_total},\n"
            f"  η = {self.eta_median:.3f} (IQR: {self.eta_iqr:.3f}),\n"
            f"  D2 = {self.D2_median:.3f}, r_S = {self.r_S_median:.1f}\n"
            f")"
        )


class ActivationTracker:
    """
    Track activation ratio across reaction networks.
    
    Integrates stoichiometry, simulation, and correlation dimension
    analysis to compute η = D2 / r_S.
    
    Examples
    --------
    >>> tracker = ActivationTracker()
    >>> 
    >>> # Single network analysis
    >>> result = tracker.analyze_network(
    ...     reactions=["A -> X", "2X + Y -> 3X", "B + X -> Y"],
    ...     rate_constants=[1.0, 1.0, 1.0],
    ...     initial_concentrations={"A": 1.0, "B": 3.0, "X": 1.0, "Y": 1.0},
    ...     chemostat_species={"A": 1.0, "B": 3.0},
    ... )
    >>> print(f"η = {result.eta:.3f}")
    >>>
    >>> # Batch analysis
    >>> networks = [...]  # List of network specifications
    >>> batch = tracker.analyze_batch(networks)
    >>> print(f"Median η = {batch.eta_median:.3f}")
    """
    
    def __init__(
        self,
        t_span: Tuple[float, float] = (0, 200),
        n_points: int = 10000,
        remove_transient: float = 0.5,
        theiler_window: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize tracker with default parameters.
        
        Parameters
        ----------
        t_span : tuple
            Simulation time span (t_start, t_end).
        n_points : int
            Number of time points in simulation.
        remove_transient : float
            Fraction of trajectory to discard (0 to 1).
        theiler_window : int, optional
            Theiler window for D2 (None for auto).
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.t_span = t_span
        self.n_points = n_points
        self.remove_transient = remove_transient
        self.theiler_window = theiler_window
        self.random_state = random_state
        
        # Initialize components
        self.stoich_analyzer = StoichiometricAnalyzer()
        self.simulator = ReactionSimulator()
        self.cd = CorrelationDimension()
    
    def analyze_network(
        self,
        reactions: List[str],
        rate_constants: List[float],
        initial_concentrations: Dict[str, float],
        chemostat_species: Optional[Dict[str, float]] = None,
        cstr_dilution_rate: Optional[float] = None,
        cstr_feed_concentrations: Optional[Dict[str, float]] = None,
        network_id: Optional[str] = None,
        species_to_track: Optional[List[str]] = None,
    ) -> ActivationResult:
        """
        Analyze activation for a single reaction network.
        
        Parameters
        ----------
        reactions : list of str
            Reaction strings (e.g., ["A -> B", "B + C -> D"]).
        rate_constants : list of float
            Rate constant for each reaction.
        initial_concentrations : dict
            Initial concentration for each species.
        chemostat_species : dict, optional
            Species held at fixed concentration (chemostat mode).
        cstr_dilution_rate : float, optional
            Dilution rate for CSTR mode.
        cstr_feed_concentrations : dict, optional
            Feed concentrations for CSTR mode.
        network_id : str, optional
            Identifier for this network (auto-generated if None).
        species_to_track : list of str, optional
            Species to include in D2 calculation. If None, uses all
            dynamic (non-chemostatted) species except monotonic accumulators.
            
        Returns
        -------
        ActivationResult
            Complete analysis results.
        """
        if network_id is None:
            network_id = f"net_{datetime.now().strftime('%H%M%S%f')}"
        
        # Determine driving mode
        if chemostat_species:
            driving_mode = DrivingMode.CHEMOSTAT
        elif cstr_dilution_rate is not None:
            driving_mode = DrivingMode.CSTR
        else:
            driving_mode = DrivingMode.NONE
            warnings.warn(
                "No driving specified. System will likely collapse to fixed point."
            )
        
        # Step 1: Stoichiometric analysis
        try:
            stoich = self.stoich_analyzer.from_reaction_strings(reactions)
        except Exception as e:
            return self._skipped_result(
                network_id, reactions, f"Stoichiometry error: {e}"
            )
        
        r_S = stoich.rank
        
        # Filter: r_S < 2 is excluded
        if r_S < 2:
            return self._skipped_result(
                network_id, reactions, f"r_S={r_S} < 2: η ill-defined",
                stoich=stoich
            )
        
        # Step 2: Simulate dynamics
        try:
            network = self.simulator.build_network(reactions)
            
            sim_kwargs = {
                'rate_constants': rate_constants,
                'initial_concentrations': initial_concentrations,
                't_span': self.t_span,
                'n_points': self.n_points,
                'driving_mode': driving_mode,
            }
            
            if chemostat_species:
                sim_kwargs['chemostat_species'] = chemostat_species
            if cstr_dilution_rate is not None:
                sim_kwargs['cstr_dilution_rate'] = cstr_dilution_rate
            if cstr_feed_concentrations is not None:
                sim_kwargs['cstr_feed_concentrations'] = cstr_feed_concentrations
            
            sim_result = self.simulator.simulate(network, **sim_kwargs)
            
            if not sim_result.success:
                return self._skipped_result(
                    network_id, reactions, "Simulation failed",
                    stoich=stoich
                )
                
        except Exception as e:
            return self._skipped_result(
                network_id, reactions, f"Simulation error: {e}",
                stoich=stoich
            )
        
        # Step 3: Extract trajectory for D2
        trajectory = self._extract_trajectory(
            sim_result, species_to_track, chemostat_species
        )
        
        if trajectory is None or trajectory.shape[1] < 1:
            return self._skipped_result(
                network_id, reactions, "No valid species for D2",
                stoich=stoich
            )
        
        # Step 4: Compute correlation dimension
        try:
            d2_result = self.cd.compute(
                trajectory,
                theiler_window=self.theiler_window,
                random_state=self.random_state,
            )
        except Exception as e:
            return self._skipped_result(
                network_id, reactions, f"D2 error: {e}",
                stoich=stoich
            )
        
        # Step 5: Compute activation ratio
        D2 = d2_result.D2
        eta = compute_activation_ratio(D2, r_S)
        
        return ActivationResult(
            network_id=network_id,
            reactions=reactions,
            species=stoich.species_names,
            n_reactions=len(reactions),
            n_species=len(stoich.species_names),
            r_S=r_S,
            n_conservation_laws=stoich.n_conservation_laws,
            D2=D2,
            D2_uncertainty=d2_result.D2_uncertainty,
            quality=d2_result.quality,
            eta=eta,
            driving_mode=driving_mode.value,
            simulation_time=self.t_span[1] - self.t_span[0],
            n_trajectory_points=d2_result.n_trajectory_points,
            theiler_window=d2_result.theiler_window,
            random_state=self.random_state,
        )
    
    def analyze_network_ensemble(
        self,
        reactions: List[str],
        rate_constants: List[float],
        initial_concentrations: Dict[str, float],
        n_runs: int = 10,
        ic_perturbation: float = 0.1,
        **kwargs,
    ) -> ActivationResult:
        """
        Analyze with ensemble of trajectories for uncertainty.
        
        Runs multiple simulations with perturbed initial conditions
        and aggregates D2 estimates.
        
        Parameters
        ----------
        reactions : list of str
            Reaction strings.
        rate_constants : list of float
            Rate constants.
        initial_concentrations : dict
            Base initial concentrations.
        n_runs : int
            Number of ensemble members.
        ic_perturbation : float
            Relative perturbation to initial conditions (0.1 = 10%).
        **kwargs
            Additional arguments to analyze_network().
            
        Returns
        -------
        ActivationResult
            Results with ensemble statistics.
        """
        network_id = kwargs.pop('network_id', None)
        if network_id is None:
            network_id = f"ens_{datetime.now().strftime('%H%M%S%f')}"
        
        # Get base result for stoichiometry
        base_result = self.analyze_network(
            reactions, rate_constants, initial_concentrations,
            network_id=f"{network_id}_base", **kwargs
        )
        
        if base_result.skipped:
            return base_result
        
        # Generate ensemble of trajectories
        trajectories = []
        rng = np.random.default_rng(self.random_state)
        
        chemostat_species = kwargs.get('chemostat_species', {})
        species_to_track = kwargs.get('species_to_track', None)
        
        for i in range(n_runs):
            # Perturb non-chemostatted species
            perturbed_ic = {}
            for sp, conc in initial_concentrations.items():
                if sp in chemostat_species:
                    perturbed_ic[sp] = conc
                else:
                    perturbed_ic[sp] = conc * (1 + ic_perturbation * rng.standard_normal())
                    perturbed_ic[sp] = max(0, perturbed_ic[sp])  # Non-negative
            
            try:
                network = self.simulator.build_network(reactions)
                
                sim_kwargs = {
                    'rate_constants': rate_constants,
                    'initial_concentrations': perturbed_ic,
                    't_span': self.t_span,
                    'n_points': self.n_points,
                    'driving_mode': DrivingMode.CHEMOSTAT if chemostat_species else DrivingMode.NONE,
                }
                if chemostat_species:
                    sim_kwargs['chemostat_species'] = chemostat_species
                
                sim_result = self.simulator.simulate(network, **sim_kwargs)
                
                if sim_result.success:
                    traj = self._extract_trajectory(
                        sim_result, species_to_track, chemostat_species
                    )
                    if traj is not None:
                        trajectories.append(traj)
            except:
                continue
        
        if len(trajectories) < 2:
            # Fall back to single-run result
            base_result.network_id = network_id
            return base_result
        
        # Compute ensemble D2
        ensemble = compute_D2_ensemble(
            trajectories,
            theiler_window=self.theiler_window,
            random_state=self.random_state,
        )
        
        # Use ensemble median for η
        eta = compute_activation_ratio(ensemble.D2_median, base_result.r_S)
        
        return ActivationResult(
            network_id=network_id,
            reactions=reactions,
            species=base_result.species,
            n_reactions=base_result.n_reactions,
            n_species=base_result.n_species,
            r_S=base_result.r_S,
            n_conservation_laws=base_result.n_conservation_laws,
            D2=ensemble.D2_median,
            D2_uncertainty=ensemble.D2_std,
            quality=QualityFlag.GOOD if ensemble.n_good > ensemble.n_total // 2 else QualityFlag.MARGINAL,
            eta=eta,
            driving_mode=base_result.driving_mode,
            simulation_time=base_result.simulation_time,
            n_trajectory_points=base_result.n_trajectory_points,
            theiler_window=ensemble.theiler_window,
            random_state=self.random_state,
            ensemble_D2_median=ensemble.D2_median,
            ensemble_D2_iqr=ensemble.D2_iqr,
            ensemble_success_rate=ensemble.success_rate,
        )
    
    def analyze_batch(
        self,
        networks: List[Dict[str, Any]],
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
    ) -> BatchResult:
        """
        Analyze multiple networks.
        
        Parameters
        ----------
        networks : list of dict
            Each dict contains 'reactions', 'rate_constants', 
            'initial_concentrations', and optionally 'chemostat_species',
            'network_id', etc.
        checkpoint_path : str, optional
            Path to save results after each network.
        verbose : bool
            Print progress.
            
        Returns
        -------
        BatchResult
            Aggregated results.
        """
        results = []
        
        for i, net_spec in enumerate(networks):
            if verbose:
                print(f"Analyzing network {i+1}/{len(networks)}...", end=" ")
            
            result = self.analyze_network(**net_spec)
            results.append(result)
            
            if verbose:
                if result.skipped:
                    print(f"SKIPPED: {result.skip_reason}")
                else:
                    print(f"η={result.eta:.3f}, quality={result.quality.value}")
            
            # Checkpoint
            if checkpoint_path:
                self._save_checkpoint(results, checkpoint_path)
        
        return self._aggregate_results(results)
    
    def _extract_trajectory(
        self,
        sim_result: SimulationResult,
        species_to_track: Optional[List[str]],
        chemostat_species: Optional[Dict[str, float]],
    ) -> Optional[np.ndarray]:
        """Extract trajectory for D2 calculation.
        
        Excludes monotonically increasing/decreasing species (accumulators/sinks)
        as these inflate dimensionality artificially.
        """
        c = sim_result.concentrations
        species_names = sim_result.species_names
        
        # Remove transient
        n_remove = int(len(c) * self.remove_transient)
        c = c[n_remove:]
        
        if species_to_track:
            # Use specified species
            indices = [species_names.index(sp) for sp in species_to_track 
                      if sp in species_names]
        else:
            # Auto-select: exclude monotonic accumulators/sinks
            indices = []
            for i, sp in enumerate(species_names):
                col = c[:, i]
                diffs = np.diff(col)
                
                # Check if strictly monotonic (accumulator or sink)
                is_monotonic_increasing = np.all(diffs >= -1e-10)
                is_monotonic_decreasing = np.all(diffs <= 1e-10)
                
                if is_monotonic_increasing or is_monotonic_decreasing:
                    # Exclude monotonic species - they're accumulators/sinks
                    continue
                
                # Include non-monotonic species
                indices.append(i)
        
        if len(indices) == 0:
            return None
        
        return c[:, indices]
    
    def _skipped_result(
        self,
        network_id: str,
        reactions: List[str],
        reason: str,
        stoich: Optional[StoichiometricAnalysis] = None,
    ) -> ActivationResult:
        """Create a skipped result."""
        return ActivationResult(
            network_id=network_id,
            reactions=reactions,
            species=stoich.species_names if stoich else [],
            n_reactions=len(reactions),
            n_species=len(stoich.species_names) if stoich else 0,
            r_S=stoich.rank if stoich else 0,
            n_conservation_laws=stoich.n_conservation_laws if stoich else 0,
            D2=np.nan,
            D2_uncertainty=np.nan,
            quality=QualityFlag.FAILED,
            eta=np.nan,
            driving_mode="none",
            simulation_time=0,
            n_trajectory_points=0,
            theiler_window=0,
            skipped=True,
            skip_reason=reason,
        )
    
    def _aggregate_results(self, results: List[ActivationResult]) -> BatchResult:
        """Compute aggregate statistics."""
        n_total = len(results)
        
        # Filter to non-skipped, successful
        analyzed = [r for r in results if not r.skipped]
        successful = [r for r in analyzed if r.quality != QualityFlag.FAILED]
        
        n_analyzed = len(analyzed)
        n_successful = len(successful)
        
        if n_successful == 0:
            return BatchResult(
                results=results,
                n_total=n_total,
                n_analyzed=n_analyzed,
                n_successful=0,
                eta_median=np.nan, eta_iqr=np.nan, eta_mean=np.nan, eta_std=np.nan,
                D2_median=np.nan, D2_iqr=np.nan,
                r_S_median=np.nan, r_S_range=(0, 0),
                timestamp=datetime.now().isoformat(),
                parameters=self._get_parameters(),
            )
        
        etas = np.array([r.eta for r in successful])
        D2s = np.array([r.D2 for r in successful])
        r_Ss = np.array([r.r_S for r in successful])
        
        return BatchResult(
            results=results,
            n_total=n_total,
            n_analyzed=n_analyzed,
            n_successful=n_successful,
            eta_median=np.median(etas),
            eta_iqr=np.percentile(etas, 75) - np.percentile(etas, 25),
            eta_mean=np.mean(etas),
            eta_std=np.std(etas),
            D2_median=np.median(D2s),
            D2_iqr=np.percentile(D2s, 75) - np.percentile(D2s, 25),
            r_S_median=np.median(r_Ss),
            r_S_range=(int(np.min(r_Ss)), int(np.max(r_Ss))),
            timestamp=datetime.now().isoformat(),
            parameters=self._get_parameters(),
        )
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get tracker parameters for logging."""
        return {
            't_span': self.t_span,
            'n_points': self.n_points,
            'remove_transient': self.remove_transient,
            'theiler_window': self.theiler_window,
            'random_state': self.random_state,
        }
    
    def _save_checkpoint(self, results: List[ActivationResult], path: str):
        """Save results to checkpoint file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'n_results': len(results),
            'results': [r.to_dict() for r in results],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def save_results_csv(results: List[ActivationResult], path: str):
    """Save results to CSV file."""
    fieldnames = [
        'network_id', 'n_reactions', 'n_species', 'r_S', 
        'D2', 'D2_uncertainty', 'eta', 'quality', 
        'skipped', 'skip_reason'
    ]
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'network_id': r.network_id,
                'n_reactions': r.n_reactions,
                'n_species': r.n_species,
                'r_S': r.r_S,
                'D2': r.D2 if not np.isnan(r.D2) else '',
                'D2_uncertainty': r.D2_uncertainty if not np.isnan(r.D2_uncertainty) else '',
                'eta': r.eta if not np.isnan(r.eta) else '',
                'quality': r.quality.value,
                'skipped': r.skipped,
                'skip_reason': r.skip_reason or '',
            })


def load_results_json(path: str) -> List[ActivationResult]:
    """Load results from JSON checkpoint."""
    with open(path) as f:
        data = json.load(f)
    
    results = []
    for r in data['results']:
        results.append(ActivationResult(
            network_id=r['network_id'],
            reactions=r['reactions'],
            species=r['species'],
            n_reactions=r['n_reactions'],
            n_species=r['n_species'],
            r_S=r['r_S'],
            n_conservation_laws=r['n_conservation_laws'],
            D2=r['D2'] if r['D2'] is not None else np.nan,
            D2_uncertainty=r['D2_uncertainty'] if r['D2_uncertainty'] is not None else np.nan,
            quality=QualityFlag(r['quality']),
            eta=r['eta'] if r['eta'] is not None else np.nan,
            driving_mode=r['driving_mode'],
            simulation_time=r['simulation_time'],
            n_trajectory_points=r['n_trajectory_points'],
            theiler_window=r['theiler_window'],
            random_state=r.get('random_state'),
            ensemble_D2_median=r.get('ensemble_D2_median'),
            ensemble_D2_iqr=r.get('ensemble_D2_iqr'),
            ensemble_success_rate=r.get('ensemble_success_rate'),
            skipped=r['skipped'],
            skip_reason=r.get('skip_reason'),
        ))
    
    return results
