"""
Scientific Experiments for dimensional activation hypothesis.

Step 7 Part 3: Run experiments from the plan.

Experiments:
1. Random vs Autocatalytic Networks
2. Progressive Autocatalysis  
3. Driving Strength Dependence
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from datetime import datetime
import time

from .activation_tracker import ActivationTracker, ActivationResult
from .network_generator import NetworkGenerator, GeneratedNetwork
from .correlation_dimension import QualityFlag

# Try to import tqdm for progress bars, fall back to simple version
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ProgressTracker:
    """Simple progress tracker with ETA."""
    
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.times = []
    
    def update(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        
        if self.current > 0:
            avg_time = elapsed / self.current
            remaining = (self.total - self.current) * avg_time
            eta_str = self._format_time(remaining)
            elapsed_str = self._format_time(elapsed)
            
            pct = 100 * self.current / self.total
            print(f"\r  {self.desc}: {self.current}/{self.total} ({pct:.0f}%) "
                  f"[{elapsed_str} elapsed, ETA: {eta_str}]", end="", flush=True)
    
    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def close(self):
        elapsed = time.time() - self.start_time
        print(f"\r  {self.desc}: {self.total}/{self.total} (100%) "
              f"[{self._format_time(elapsed)} total]" + " " * 20)


def get_progress_bar(iterable, total: int, desc: str, verbose: bool):
    """Get appropriate progress indicator."""
    if not verbose:
        return iterable
    
    if HAS_TQDM:
        return tqdm(iterable, total=total, desc=f"  {desc}", 
                    bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        # Return iterable with manual tracking
        return iterable


@dataclass
class ExperimentResult:
    """Results from Experiment 1: two-group comparison."""
    name: str
    hypothesis: str
    n_networks: int
    n_successful: int
    group1_label: str
    group1_eta_median: float
    group1_eta_iqr: float
    group1_n: int
    group2_label: str
    group2_eta_median: float
    group2_eta_iqr: float
    group2_n: int
    delta_eta: float
    results: List[ActivationResult]
    parameters: Dict
    timestamp: str
    
    def __repr__(self) -> str:
        return (
            f"ExperimentResult({self.name})\n"
            f"  Hypothesis: {self.hypothesis}\n"
            f"  {self.group1_label}: eta = {self.group1_eta_median:.3f} (IQR: {self.group1_eta_iqr:.3f}, n={self.group1_n})\n"
            f"  {self.group2_label}: eta = {self.group2_eta_median:.3f} (IQR: {self.group2_eta_iqr:.3f}, n={self.group2_n})\n"
            f"  Delta_eta = {self.delta_eta:+.3f}\n"
        )
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name, 'hypothesis': self.hypothesis,
            'n_networks': self.n_networks, 'n_successful': self.n_successful,
            'group1': {'label': self.group1_label, 'eta_median': self.group1_eta_median,
                       'eta_iqr': self.group1_eta_iqr, 'n': self.group1_n},
            'group2': {'label': self.group2_label, 'eta_median': self.group2_eta_median,
                       'eta_iqr': self.group2_eta_iqr, 'n': self.group2_n},
            'delta_eta': self.delta_eta, 'parameters': self.parameters, 'timestamp': self.timestamp,
        }


@dataclass
class ProgressiveResult:
    """Results from Experiment 2: progressive autocatalysis."""
    name: str
    hypothesis: str
    n_trajectories: int
    n_autocatalytic_range: List[int]
    eta_medians: List[float]
    eta_lower: List[float]
    eta_upper: List[float]
    eta_slope: float
    parameters: Dict
    timestamp: str
    
    def __repr__(self) -> str:
        return (
            f"ProgressiveResult({self.name})\n"
            f"  Hypothesis: {self.hypothesis}\n"
            f"  Trajectories: {self.n_trajectories}\n"
            f"  eta at 0 autocatalytic: {self.eta_medians[0]:.3f}\n"
            f"  eta at {self.n_autocatalytic_range[-1]} autocatalytic: {self.eta_medians[-1]:.3f}\n"
            f"  Slope: {self.eta_slope:+.4f} per autocatalytic reaction\n"
        )


@dataclass
class DrivingResult:
    """Results from Experiment 3: driving strength."""
    name: str
    hypothesis: str
    dilution_rates: List[float]
    eta_medians: List[float]
    eta_iqrs: List[float]
    n_successful: List[int]
    parameters: Dict
    timestamp: str
    
    def __repr__(self) -> str:
        lines = [f"DrivingResult({self.name})", f"  Hypothesis: {self.hypothesis}"]
        for D, eta, iqr, n in zip(self.dilution_rates, self.eta_medians, 
                                   self.eta_iqrs, self.n_successful):
            lines.append(f"  D={D:.3f}: eta = {eta:.3f} +/- {iqr:.3f} (n={n})")
        return "\n".join(lines)


def run_experiment_1(
    n_networks: int = 50, n_species: int = 10, n_reactions: int = 15,
    n_autocatalytic: int = 3, seed: int = 42, verbose: bool = True,
) -> ExperimentResult:
    """Experiment 1: Random vs Autocatalytic Networks."""
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 1: Random vs Autocatalytic Networks")
        print("=" * 60)
    
    gen = NetworkGenerator(n_species=n_species, n_food=3, seed=seed)
    tracker = ActivationTracker(t_span=(0, 200), n_points=10000, 
                                 remove_transient=0.5, random_state=seed)
    all_results = []
    
    # Analyze random networks
    if verbose:
        print(f"\nAnalyzing {n_networks} random networks...")
    random_nets = gen.generate_batch_random(n_networks, n_reactions)
    random_results = []
    
    if HAS_TQDM and verbose:
        iterator = tqdm(enumerate(random_nets), total=n_networks, 
                        desc="  Random", 
                        bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        iterator = enumerate(random_nets)
        if verbose and not HAS_TQDM:
            progress = ProgressTracker(n_networks, "Random")
    
    for i, net in iterator:
        result = tracker.analyze_network(**net.to_tracker_input())
        result.network_id = f"random_{i}"
        random_results.append(result)
        all_results.append(result)
        
        if verbose and not HAS_TQDM:
            progress.update()
    
    if verbose and not HAS_TQDM:
        progress.close()
    
    # Analyze autocatalytic networks
    if verbose:
        print(f"\nAnalyzing {n_networks} autocatalytic networks...")
    autocat_nets = gen.generate_batch_autocatalytic(n_networks, n_reactions, n_autocatalytic)
    autocat_results = []
    
    if HAS_TQDM and verbose:
        iterator = tqdm(enumerate(autocat_nets), total=n_networks,
                        desc="  Autocatalytic",
                        bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        iterator = enumerate(autocat_nets)
        if verbose and not HAS_TQDM:
            progress = ProgressTracker(n_networks, "Autocatalytic")
    
    for i, net in iterator:
        result = tracker.analyze_network(**net.to_tracker_input())
        result.network_id = f"autocat_{i}"
        autocat_results.append(result)
        all_results.append(result)
        
        if verbose and not HAS_TQDM:
            progress.update()
    
    if verbose and not HAS_TQDM:
        progress.close()
    
    # Compute statistics
    random_etas = [r.eta for r in random_results 
                   if not r.skipped and r.quality != QualityFlag.FAILED and not np.isnan(r.eta)]
    autocat_etas = [r.eta for r in autocat_results 
                    if not r.skipped and r.quality != QualityFlag.FAILED and not np.isnan(r.eta)]
    
    if len(random_etas) == 0:
        random_median, random_iqr = np.nan, np.nan
    else:
        random_median = np.median(random_etas)
        random_iqr = np.percentile(random_etas, 75) - np.percentile(random_etas, 25)
    
    if len(autocat_etas) == 0:
        autocat_median, autocat_iqr = np.nan, np.nan
    else:
        autocat_median = np.median(autocat_etas)
        autocat_iqr = np.percentile(autocat_etas, 75) - np.percentile(autocat_etas, 25)
    
    delta = autocat_median - random_median if not (np.isnan(autocat_median) or np.isnan(random_median)) else np.nan
    
    result = ExperimentResult(
        name="Random vs Autocatalytic",
        hypothesis="Autocatalytic networks show higher eta than random networks",
        n_networks=n_networks * 2, n_successful=len(random_etas) + len(autocat_etas),
        group1_label="Random", group1_eta_median=random_median, 
        group1_eta_iqr=random_iqr, group1_n=len(random_etas),
        group2_label="Autocatalytic", group2_eta_median=autocat_median,
        group2_eta_iqr=autocat_iqr, group2_n=len(autocat_etas),
        delta_eta=delta, results=all_results,
        parameters={'n_networks': n_networks, 'n_species': n_species,
                   'n_reactions': n_reactions, 'n_autocatalytic': n_autocatalytic, 'seed': seed},
        timestamp=datetime.now().isoformat(),
    )
    
    if verbose:
        print(f"\n{result}")
    return result


def run_experiment_2(
    n_trajectories: int = 20, n_base_reactions: int = 10, n_autocatalytic_to_add: int = 10,
    n_species: int = 10, seed: int = 42, verbose: bool = True,
) -> ProgressiveResult:
    """Experiment 2: Progressive Autocatalysis."""
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 2: Progressive Autocatalysis")
        print("=" * 60)
    
    tracker = ActivationTracker(t_span=(0, 200), n_points=10000,
                                 remove_transient=0.5, random_state=seed)
    
    n_steps = n_autocatalytic_to_add + 1
    total_analyses = n_trajectories * n_steps
    eta_matrix = np.full((n_trajectories, n_steps), np.nan)
    
    if verbose:
        print(f"\nRunning {n_trajectories} trajectories x {n_steps} steps = {total_analyses} analyses")
    
    analysis_count = 0
    start_time = time.time()
    
    for traj_idx in range(n_trajectories):
        gen = NetworkGenerator(n_species=n_species, n_food=3, seed=seed + traj_idx)
        networks = gen.generate_progressive(n_base_reactions, n_autocatalytic_to_add)
        
        for step_idx, net in enumerate(networks):
            result = tracker.analyze_network(**net.to_tracker_input())
            if not result.skipped and result.quality != QualityFlag.FAILED:
                eta_matrix[traj_idx, step_idx] = result.eta
            
            analysis_count += 1
            
            if verbose:
                elapsed = time.time() - start_time
                avg_time = elapsed / analysis_count
                remaining = (total_analyses - analysis_count) * avg_time
                
                if remaining < 60:
                    eta_str = f"{remaining:.0f}s"
                elif remaining < 3600:
                    eta_str = f"{remaining/60:.1f}m"
                else:
                    eta_str = f"{remaining/3600:.1f}h"
                
                pct = 100 * analysis_count / total_analyses
                print(f"\r  Progress: {analysis_count}/{total_analyses} ({pct:.0f}%) "
                      f"[Traj {traj_idx+1}/{n_trajectories}, Step {step_idx}/{n_steps-1}] "
                      f"ETA: {eta_str}    ", end="", flush=True)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\r  Completed {total_analyses} analyses in {elapsed:.1f}s" + " " * 40)
    
    # Compute aggregates
    n_autocatalytic_range = list(range(n_steps))
    eta_medians, eta_lower, eta_upper = [], [], []
    
    for col in range(n_steps):
        vals = eta_matrix[:, col]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            eta_medians.append(np.median(valid))
            eta_lower.append(np.percentile(valid, 25))
            eta_upper.append(np.percentile(valid, 75))
        else:
            eta_medians.append(np.nan)
            eta_lower.append(np.nan)
            eta_upper.append(np.nan)
    
    valid_x = [x for x, y in zip(n_autocatalytic_range, eta_medians) if not np.isnan(y)]
    valid_y = [y for y in eta_medians if not np.isnan(y)]
    slope = np.polyfit(valid_x, valid_y, 1)[0] if len(valid_x) >= 2 else np.nan
    
    result = ProgressiveResult(
        name="Progressive Autocatalysis",
        hypothesis="Adding autocatalytic reactions increases eta",
        n_trajectories=n_trajectories, n_autocatalytic_range=n_autocatalytic_range,
        eta_medians=eta_medians, eta_lower=eta_lower, eta_upper=eta_upper, eta_slope=slope,
        parameters={'n_trajectories': n_trajectories, 'n_base_reactions': n_base_reactions,
                   'n_autocatalytic_to_add': n_autocatalytic_to_add, 'n_species': n_species, 'seed': seed},
        timestamp=datetime.now().isoformat(),
    )
    
    if verbose:
        print(f"\n{result}")
    return result


def run_experiment_3(
    dilution_rates: List[float] = [0.01, 0.03, 0.1, 0.3, 1.0],
    n_networks: int = 10, n_species: int = 10, n_reactions: int = 20,
    n_autocatalytic: int = 5, seed: int = 42, verbose: bool = True,
) -> DrivingResult:
    """Experiment 3: Driving Strength Dependence."""
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 3: Driving Strength Dependence")
        print("=" * 60)
    
    gen = NetworkGenerator(n_species=n_species, n_food=3, seed=seed)
    networks = gen.generate_batch_autocatalytic(n_networks, n_reactions, n_autocatalytic)
    
    total_analyses = len(dilution_rates) * n_networks
    analysis_count = 0
    start_time = time.time()
    
    if verbose:
        print(f"\nTesting {len(dilution_rates)} dilution rates x {n_networks} networks = {total_analyses} analyses")
    
    eta_medians, eta_iqrs, n_successful = [], [], []
    
    for D in dilution_rates:
        tracker = ActivationTracker(t_span=(0, 200), n_points=10000,
                                     remove_transient=0.5, random_state=seed)
        etas = []
        
        for i, net in enumerate(networks):
            input_dict = net.to_tracker_input()
            feed_conc = input_dict.pop('chemostat_species')
            input_dict['cstr_dilution_rate'] = D
            input_dict['cstr_feed_concentrations'] = feed_conc
            
            result = tracker.analyze_network(**input_dict)
            if not result.skipped and result.quality != QualityFlag.FAILED:
                etas.append(result.eta)
            
            analysis_count += 1
            
            if verbose:
                elapsed = time.time() - start_time
                avg_time = elapsed / analysis_count
                remaining = (total_analyses - analysis_count) * avg_time
                
                if remaining < 60:
                    eta_str = f"{remaining:.0f}s"
                elif remaining < 3600:
                    eta_str = f"{remaining/60:.1f}m"
                else:
                    eta_str = f"{remaining/3600:.1f}h"
                
                pct = 100 * analysis_count / total_analyses
                print(f"\r  Progress: {analysis_count}/{total_analyses} ({pct:.0f}%) "
                      f"[D={D:.2f}, Net {i+1}/{n_networks}] ETA: {eta_str}    ", 
                      end="", flush=True)
        
        if len(etas) > 0:
            eta_medians.append(np.median(etas))
            eta_iqrs.append(np.percentile(etas, 75) - np.percentile(etas, 25))
        else:
            eta_medians.append(np.nan)
            eta_iqrs.append(np.nan)
        n_successful.append(len(etas))
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\r  Completed {total_analyses} analyses in {elapsed:.1f}s" + " " * 40)
    
    result = DrivingResult(
        name="Driving Strength Dependence",
        hypothesis="Stronger driving enables higher activation",
        dilution_rates=dilution_rates, eta_medians=eta_medians,
        eta_iqrs=eta_iqrs, n_successful=n_successful,
        parameters={'dilution_rates': dilution_rates, 'n_networks': n_networks,
                   'n_species': n_species, 'n_reactions': n_reactions,
                   'n_autocatalytic': n_autocatalytic, 'seed': seed},
        timestamp=datetime.now().isoformat(),
    )
    
    if verbose:
        print(f"\n{result}")
    return result


def run_all_experiments(seed: int = 42, verbose: bool = True, save_path: Optional[str] = None) -> Dict:
    """Run all three experiments."""
    total_start = time.time()
    
    results = {
        'experiment_1': run_experiment_1(n_networks=50, seed=seed, verbose=verbose),
        'experiment_2': run_experiment_2(n_trajectories=20, seed=seed, verbose=verbose),
        'experiment_3': run_experiment_3(n_networks=10, seed=seed, verbose=verbose),
    }
    
    if verbose:
        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"All experiments completed in {total_elapsed/60:.1f} minutes")
        print(f"{'='*60}")
    
    if save_path:
        save_dict = {
            'experiment_1': results['experiment_1'].to_dict(),
            'experiment_2': {
                'name': results['experiment_2'].name,
                'hypothesis': results['experiment_2'].hypothesis,
                'n_trajectories': results['experiment_2'].n_trajectories,
                'eta_medians': results['experiment_2'].eta_medians,
                'eta_slope': results['experiment_2'].eta_slope,
                'parameters': results['experiment_2'].parameters,
            },
            'experiment_3': {
                'name': results['experiment_3'].name,
                'hypothesis': results['experiment_3'].hypothesis,
                'dilution_rates': results['experiment_3'].dilution_rates,
                'eta_medians': results['experiment_3'].eta_medians,
                'n_successful': results['experiment_3'].n_successful,
                'parameters': results['experiment_3'].parameters,
            },
            'timestamp': datetime.now().isoformat(),
        }
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
    
    return results
