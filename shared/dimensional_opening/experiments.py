"""
Scientific Experiments for dimensional activation hypothesis.

Step 7 Part 3 (v4): Template-based experiments.

Key change: Uses oscillator templates (Brusselator) instead of random networks.
This ensures we get oscillating systems that can actually be analyzed.

Experiments:
1. Control vs Test: Brusselator + random vs Brusselator + autocatalytic
2. Progressive Autocatalysis: Baseline → increasing autocatalysis
3. Driving Strength Dependence: Vary CSTR dilution rate
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

# Try to import tqdm for progress bars
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
    
    def update(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        
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
        g1_eta = f"{self.group1_eta_median:.3f}" if not np.isnan(self.group1_eta_median) else "nan"
        g2_eta = f"{self.group2_eta_median:.3f}" if not np.isnan(self.group2_eta_median) else "nan"
        g1_iqr = f"{self.group1_eta_iqr:.3f}" if not np.isnan(self.group1_eta_iqr) else "nan"
        g2_iqr = f"{self.group2_eta_iqr:.3f}" if not np.isnan(self.group2_eta_iqr) else "nan"
        delta = f"{self.delta_eta:+.3f}" if not np.isnan(self.delta_eta) else "nan"
        
        return (
            f"ExperimentResult({self.name})\n"
            f"  Hypothesis: {self.hypothesis}\n"
            f"  {self.group1_label}: eta = {g1_eta} (IQR: {g1_iqr}, n={self.group1_n})\n"
            f"  {self.group2_label}: eta = {g2_eta} (IQR: {g2_iqr}, n={self.group2_n})\n"
            f"  Delta_eta = {delta}\n"
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
        first_valid = next((x for x in self.eta_medians if not np.isnan(x)), np.nan)
        last_valid = next((x for x in reversed(self.eta_medians) if not np.isnan(x)), np.nan)
        first_str = f"{first_valid:.3f}" if not np.isnan(first_valid) else "nan"
        last_str = f"{last_valid:.3f}" if not np.isnan(last_valid) else "nan"
        slope_str = f"{self.eta_slope:+.4f}" if not np.isnan(self.eta_slope) else "nan"
        
        return (
            f"ProgressiveResult({self.name})\n"
            f"  Hypothesis: {self.hypothesis}\n"
            f"  Trajectories: {self.n_trajectories}\n"
            f"  eta at baseline: {first_str}\n"
            f"  eta at +{self.n_autocatalytic_range[-1]} autocatalytic: {last_str}\n"
            f"  Slope: {slope_str} per autocatalytic reaction\n"
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
            eta_str = f"{eta:.3f}" if not np.isnan(eta) else "nan"
            iqr_str = f"{iqr:.3f}" if not np.isnan(iqr) else "nan"
            lines.append(f"  D={D:.3f}: eta = {eta_str} +/- {iqr_str} (n={n})")
        return "\n".join(lines)


def _analyze_network(
    tracker: ActivationTracker,
    net: GeneratedNetwork,
    dilution_rate: float = 0.1,
) -> ActivationResult:
    """Analyze a network using CHEMOSTAT mode (food species clamped)."""
    input_dict = net.to_tracker_input()
    return tracker.analyze_network(**input_dict)


def run_experiment_1(
    n_networks: int = 20,
    n_added_control: int = 3,
    n_autocatalytic_test: int = 2,
    n_random_test: int = 1,
    template: str = 'brusselator',
    dilution_rate: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Experiment 1: Control vs Test (additional autocatalysis).
    
    Control: Template + random non-autocatalytic reactions
    Test: Template + autocatalytic reactions (+ some random)
    
    Hypothesis: Test networks show higher η than control networks.
    """
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 1: Control vs Test (Additional Autocatalysis)")
        print("=" * 60)
        print(f"  Template: {template}")
        print(f"  Control: +{n_added_control} random reactions")
        print(f"  Test: +{n_autocatalytic_test} autocatalytic, +{n_random_test} random")
    
    gen = NetworkGenerator(template=template, seed=seed)
    tracker = ActivationTracker(t_span=(0, 200), n_points=10000, 
                                remove_transient=0.5, random_state=seed)
    all_results = []
    
    # Generate and analyze control networks
    if verbose:
        print(f"\nAnalyzing {n_networks} control networks...")
    
    control_nets = gen.generate_batch_control(n_networks, n_added_control, verbose=verbose)
    control_results = []
    
    if HAS_TQDM and verbose:
        iterator = tqdm(enumerate(control_nets), total=n_networks, 
                        desc="  Control", 
                        bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        iterator = enumerate(control_nets)
        if verbose:
            progress = ProgressTracker(n_networks, "Control")
    
    for i, net in iterator:
        result = _analyze_network(tracker, net, dilution_rate)
        result.network_id = f"control_{i}"
        control_results.append(result)
        all_results.append(result)
        
        if verbose and not HAS_TQDM:
            progress.update()
    
    if verbose and not HAS_TQDM:
        progress.close()
    
    # Generate and analyze test networks
    if verbose:
        print(f"\nAnalyzing {n_networks} test networks...")
    
    test_nets = gen.generate_batch_test(n_networks, n_autocatalytic_test, n_random_test, verbose=verbose)
    test_results = []
    
    if HAS_TQDM and verbose:
        iterator = tqdm(enumerate(test_nets), total=n_networks,
                        desc="  Test",
                        bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        iterator = enumerate(test_nets)
        if verbose:
            progress = ProgressTracker(n_networks, "Test")
    
    for i, net in iterator:
        result = _analyze_network(tracker, net, dilution_rate)
        result.network_id = f"test_{i}"
        test_results.append(result)
        all_results.append(result)
        
        if verbose and not HAS_TQDM:
            progress.update()
    
    if verbose and not HAS_TQDM:
        progress.close()
    
    # Compute statistics
    control_etas = [r.eta for r in control_results 
                    if not r.skipped and r.quality != QualityFlag.FAILED and not np.isnan(r.eta)]
    test_etas = [r.eta for r in test_results 
                 if not r.skipped and r.quality != QualityFlag.FAILED and not np.isnan(r.eta)]
    
    if verbose:
        print(f"\n  Control: {len(control_etas)}/{n_networks} successful")
        print(f"  Test: {len(test_etas)}/{n_networks} successful")
    
    if len(control_etas) == 0:
        control_median, control_iqr = np.nan, np.nan
    else:
        control_median = np.median(control_etas)
        control_iqr = np.percentile(control_etas, 75) - np.percentile(control_etas, 25)
    
    if len(test_etas) == 0:
        test_median, test_iqr = np.nan, np.nan
    else:
        test_median = np.median(test_etas)
        test_iqr = np.percentile(test_etas, 75) - np.percentile(test_etas, 25)
    
    delta = test_median - control_median if not (np.isnan(test_median) or np.isnan(control_median)) else np.nan
    
    result = ExperimentResult(
        name="Control vs Test (Additional Autocatalysis)",
        hypothesis="Additional autocatalytic reactions increase eta",
        n_networks=n_networks * 2, n_successful=len(control_etas) + len(test_etas),
        group1_label="Control", group1_eta_median=control_median, 
        group1_eta_iqr=control_iqr, group1_n=len(control_etas),
        group2_label="Test", group2_eta_median=test_median,
        group2_eta_iqr=test_iqr, group2_n=len(test_etas),
        delta_eta=delta, results=all_results,
        parameters={
            'n_networks': n_networks, 'template': template,
            'n_added_control': n_added_control,
            'n_autocatalytic_test': n_autocatalytic_test,
            'n_random_test': n_random_test,
            'dilution_rate': dilution_rate, 'seed': seed,
        },
        timestamp=datetime.now().isoformat(),
    )
    
    if verbose:
        print(f"\n{result}")
    return result


def run_experiment_2(
    n_trajectories: int = 15,
    n_steps: int = 5,
    template: str = 'brusselator',
    dilution_rate: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> ProgressiveResult:
    """
    Experiment 2: Progressive Autocatalysis.
    
    Start with template, progressively add autocatalytic reactions.
    
    Hypothesis: Adding autocatalytic reactions increases η.
    """
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 2: Progressive Autocatalysis")
        print("=" * 60)
        print(f"  Template: {template}")
        print(f"  Steps: baseline + {n_steps} autocatalytic additions")
    
    tracker = ActivationTracker(t_span=(0, 200), n_points=10000,
                                remove_transient=0.5, random_state=seed)
    
    total_steps = n_steps + 1  # baseline + n_steps
    total_analyses = n_trajectories * total_steps
    eta_matrix = np.full((n_trajectories, total_steps), np.nan)
    
    if verbose:
        print(f"\nRunning {n_trajectories} trajectories x {total_steps} steps = {total_analyses} analyses")
    
    analysis_count = 0
    start_time = time.time()
    
    for traj_idx in range(n_trajectories):
        gen = NetworkGenerator(template=template, seed=seed + traj_idx)
        networks = gen.generate_progressive(n_steps)
        
        for step_idx, net in enumerate(networks):
            result = _analyze_network(tracker, net, dilution_rate)
            
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
                      f"[Traj {traj_idx+1}/{n_trajectories}, Step {step_idx}/{n_steps}] "
                      f"ETA: {eta_str}    ", end="", flush=True)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\r  Completed {total_analyses} analyses in {elapsed:.1f}s" + " " * 40)
    
    # Compute aggregates
    n_autocatalytic_range = list(range(total_steps))
    eta_medians, eta_lower, eta_upper = [], [], []
    
    for col in range(total_steps):
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
        parameters={
            'n_trajectories': n_trajectories, 'n_steps': n_steps,
            'template': template, 'dilution_rate': dilution_rate, 'seed': seed,
        },
        timestamp=datetime.now().isoformat(),
    )
    
    if verbose:
        print(f"\n{result}")
    return result


def run_experiment_3(
    dilution_rates: List[float] = [0.01, 0.03, 0.1, 0.3, 1.0],
    n_networks: int = 10,
    n_autocatalytic: int = 2,
    template: str = 'brusselator',
    seed: int = 42,
    verbose: bool = True,
) -> DrivingResult:
    """
    Experiment 3: Driving Strength Dependence.
    
    Vary CSTR dilution rate, measure effect on η.
    
    Hypothesis: Stronger driving enables higher activation.
    """
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 3: Driving Strength Dependence")
        print("=" * 60)
        print(f"  Template: {template}")
        print(f"  Dilution rates: {dilution_rates}")
    
    gen = NetworkGenerator(template=template, seed=seed)
    
    if verbose:
        print(f"\nGenerating {n_networks} test networks...")
    networks = gen.generate_batch_test(n_networks, n_autocatalytic, verbose=verbose)
    
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
            result = _analyze_network(tracker, net, D)
            
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
        parameters={
            'dilution_rates': dilution_rates, 'n_networks': n_networks,
            'n_autocatalytic': n_autocatalytic, 'template': template, 'seed': seed,
        },
        timestamp=datetime.now().isoformat(),
    )
    
    if verbose:
        print(f"\n{result}")
    return result


def run_all_experiments(
    seed: int = 42,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> Dict:
    """Run all three experiments."""
    total_start = time.time()
    
    results = {
        'experiment_1': run_experiment_1(n_networks=20, seed=seed, verbose=verbose),
        'experiment_2': run_experiment_2(n_trajectories=15, seed=seed, verbose=verbose),
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
                'n_autocatalytic_range': results['experiment_2'].n_autocatalytic_range,
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


# =====================================================================
# Paper 2: Feedback-Aligned vs Random Autocatalysis
# =====================================================================

@dataclass
class Paper2Result:
    """Results from Paper 2: Random vs Feedback-Aligned progressive autocatalysis."""
    name: str
    hypothesis: str
    n_trajectories: int
    n_steps: int
    max_candidates: int

    # Raw η matrices: shape (n_trajectories, n_steps+1), NaN for invalid/missing
    group_a_eta_matrix: np.ndarray  # Random
    group_b_eta_matrix: np.ndarray  # Aligned

    # Valid counts at each step
    group_a_valid_counts: List[int]
    group_b_valid_counts: List[int]

    # Aggregates per step (length n_steps+1)
    group_a_medians: List[float]
    group_a_iqr: List[float]
    group_b_medians: List[float]
    group_b_iqr: List[float]

    # Slopes
    group_a_slope: float
    group_b_slope: float
    group_a_slope_ci: tuple  # (low, high) 95% bootstrap CI
    group_b_slope_ci: tuple

    # Tail probabilities P(η > 0.25) at each step
    group_a_tail_prob: List[float]
    group_b_tail_prob: List[float]

    # Statistical tests per step (length n_steps+1)
    mann_whitney_p: List[Optional[float]]
    ks_p: List[Optional[float]]
    fisher_p: List[Optional[float]]  # Fisher exact on tail probabilities

    # Group B metadata
    acceptance_rates_per_step: List[List[float]]  # list of lists (per trajectory per step)
    acceptance_rate_medians: List[float]           # median acceptance rate at each step
    early_termination_count: int
    early_termination_rate: float

    # Selection bias (reportable artifacts for referee)
    cv_d2_correlation: Optional[float]    # Pearson r(CV, D₂) in accepted set
    cv_eta_correlation: Optional[float]   # Pearson r(CV, η) in accepted set

    # Slope comparison
    slope_difference: float               # Group B slope - Group A slope
    slope_difference_ci: tuple            # 95% bootstrap CI on slope difference
    slope_difference_perm_p: Optional[float]  # Permutation test p-value

    # Diagnostics
    group_b_filter_results: list  # nested: per trajectory, per step

    parameters: Dict
    timestamp: str

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        def _safe(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, float) and np.isnan(v):
                return None
            return v

        d = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            if isinstance(val, np.ndarray):
                d[key] = [[None if np.isnan(x) else float(x) for x in row]
                          for row in val]
            elif isinstance(val, list):
                d[key] = [_safe(x) for x in val]
            elif isinstance(val, tuple):
                d[key] = [_safe(x) for x in val]
            else:
                d[key] = _safe(val)
        return d


def _compute_paper2_statistics(
    group_a_eta: np.ndarray,
    group_b_eta: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Compute all statistical comparisons between Group A and Group B.

    Parameters
    ----------
    group_a_eta, group_b_eta : ndarray, shape (n_traj, n_steps+1)
    n_bootstrap : int
    seed : int

    Returns
    -------
    dict with keys: mann_whitney_p, ks_p, fisher_p,
                    group_a_slope, group_b_slope,
                    group_a_slope_ci, group_b_slope_ci,
                    group_a_tail_prob, group_b_tail_prob,
                    group_a_medians, group_a_iqr,
                    group_b_medians, group_b_iqr,
                    group_a_valid_counts, group_b_valid_counts
    """
    from scipy.stats import mannwhitneyu, ks_2samp, fisher_exact

    rng = np.random.RandomState(seed)
    n_steps_plus_1 = group_a_eta.shape[1]
    steps = np.arange(n_steps_plus_1)

    # Per-step aggregates
    a_medians, a_iqr, a_valid, a_tail = [], [], [], []
    b_medians, b_iqr, b_valid, b_tail = [], [], [], []
    mw_p, ks_p_vals, fish_p = [], [], []

    eta_threshold = 0.25

    for k in range(n_steps_plus_1):
        a_vals = group_a_eta[:, k]
        b_vals = group_b_eta[:, k]
        a_ok = a_vals[~np.isnan(a_vals)]
        b_ok = b_vals[~np.isnan(b_vals)]

        a_valid.append(len(a_ok))
        b_valid.append(len(b_ok))

        if len(a_ok) > 0:
            a_medians.append(float(np.median(a_ok)))
            a_iqr.append(float(np.percentile(a_ok, 75) - np.percentile(a_ok, 25)))
            a_tail.append(float(np.mean(a_ok > eta_threshold)))
        else:
            a_medians.append(np.nan)
            a_iqr.append(np.nan)
            a_tail.append(np.nan)

        if len(b_ok) > 0:
            b_medians.append(float(np.median(b_ok)))
            b_iqr.append(float(np.percentile(b_ok, 75) - np.percentile(b_ok, 25)))
            b_tail.append(float(np.mean(b_ok > eta_threshold)))
        else:
            b_medians.append(np.nan)
            b_iqr.append(np.nan)
            b_tail.append(np.nan)

        # Mann-Whitney U
        if len(a_ok) >= 3 and len(b_ok) >= 3:
            try:
                _, p = mannwhitneyu(a_ok, b_ok, alternative='two-sided')
                mw_p.append(float(p))
            except ValueError:
                mw_p.append(None)
        else:
            mw_p.append(None)

        # KS test
        if len(a_ok) >= 3 and len(b_ok) >= 3:
            try:
                _, p = ks_2samp(a_ok, b_ok)
                ks_p_vals.append(float(p))
            except ValueError:
                ks_p_vals.append(None)
        else:
            ks_p_vals.append(None)

        # Fisher exact on tail probabilities
        if len(a_ok) >= 3 and len(b_ok) >= 3:
            a_above = int(np.sum(a_ok > eta_threshold))
            a_below = len(a_ok) - a_above
            b_above = int(np.sum(b_ok > eta_threshold))
            b_below = len(b_ok) - b_above
            try:
                _, p = fisher_exact([[a_above, a_below], [b_above, b_below]])
                fish_p.append(float(p))
            except ValueError:
                fish_p.append(None)
        else:
            fish_p.append(None)

    # Slopes: linear fit of median η vs step
    def _fit_slope(medians):
        valid_mask = ~np.isnan(medians)
        if np.sum(valid_mask) < 2:
            return np.nan
        return float(np.polyfit(steps[valid_mask], np.array(medians)[valid_mask], 1)[0])

    a_slope = _fit_slope(np.array(a_medians))
    b_slope = _fit_slope(np.array(b_medians))

    # Bootstrap CI for slopes
    def _bootstrap_slope(eta_matrix, n_boot, rng):
        n_traj = eta_matrix.shape[0]
        slopes = []
        for _ in range(n_boot):
            idx = rng.randint(0, n_traj, size=n_traj)
            boot_matrix = eta_matrix[idx, :]
            meds = []
            for k in range(eta_matrix.shape[1]):
                col = boot_matrix[:, k]
                ok = col[~np.isnan(col)]
                meds.append(float(np.median(ok)) if len(ok) > 0 else np.nan)
            meds = np.array(meds)
            valid_mask = ~np.isnan(meds)
            if np.sum(valid_mask) >= 2:
                slopes.append(float(np.polyfit(steps[valid_mask], meds[valid_mask], 1)[0]))
        if len(slopes) < 100:
            return (np.nan, np.nan)
        slopes = np.array(slopes)
        return (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))

    a_slope_ci = _bootstrap_slope(group_a_eta, n_bootstrap, rng)
    b_slope_ci = _bootstrap_slope(group_b_eta, n_bootstrap, rng)

    # Bootstrap CI on slope DIFFERENCE (Δs = slope_B - slope_A)
    slope_diff = b_slope - a_slope
    n_traj_a = group_a_eta.shape[0]
    n_traj_b = group_b_eta.shape[0]
    diff_slopes = []
    for _ in range(n_bootstrap):
        # Resample each group independently
        idx_a = rng.randint(0, n_traj_a, size=n_traj_a)
        idx_b = rng.randint(0, n_traj_b, size=n_traj_b)
        boot_a = group_a_eta[idx_a, :]
        boot_b = group_b_eta[idx_b, :]
        meds_a, meds_b = [], []
        for k in range(group_a_eta.shape[1]):
            ok_a = boot_a[:, k]; ok_a = ok_a[~np.isnan(ok_a)]
            ok_b = boot_b[:, k]; ok_b = ok_b[~np.isnan(ok_b)]
            meds_a.append(float(np.median(ok_a)) if len(ok_a) > 0 else np.nan)
            meds_b.append(float(np.median(ok_b)) if len(ok_b) > 0 else np.nan)
        meds_a, meds_b = np.array(meds_a), np.array(meds_b)
        vm_a, vm_b = ~np.isnan(meds_a), ~np.isnan(meds_b)
        if np.sum(vm_a) >= 2 and np.sum(vm_b) >= 2:
            s_a = float(np.polyfit(steps[vm_a], meds_a[vm_a], 1)[0])
            s_b = float(np.polyfit(steps[vm_b], meds_b[vm_b], 1)[0])
            diff_slopes.append(s_b - s_a)
    if len(diff_slopes) >= 100:
        diff_slopes = np.array(diff_slopes)
        slope_diff_ci = (float(np.percentile(diff_slopes, 2.5)),
                         float(np.percentile(diff_slopes, 97.5)))
    else:
        slope_diff_ci = (np.nan, np.nan)

    # Permutation test on slope difference
    # Pool all trajectories, randomly assign to A/B, compute slope diff
    combined = np.vstack([group_a_eta, group_b_eta])
    n_total = combined.shape[0]
    n_perm = min(n_bootstrap, 5000)
    perm_diffs = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(n_total)
        perm_a = combined[perm_idx[:n_traj_a], :]
        perm_b = combined[perm_idx[n_traj_a:], :]
        meds_a, meds_b = [], []
        for k in range(combined.shape[1]):
            ok_a = perm_a[:, k]; ok_a = ok_a[~np.isnan(ok_a)]
            ok_b = perm_b[:, k]; ok_b = ok_b[~np.isnan(ok_b)]
            meds_a.append(float(np.median(ok_a)) if len(ok_a) > 0 else np.nan)
            meds_b.append(float(np.median(ok_b)) if len(ok_b) > 0 else np.nan)
        meds_a, meds_b = np.array(meds_a), np.array(meds_b)
        vm_a, vm_b = ~np.isnan(meds_a), ~np.isnan(meds_b)
        if np.sum(vm_a) >= 2 and np.sum(vm_b) >= 2:
            s_a = float(np.polyfit(steps[vm_a], meds_a[vm_a], 1)[0])
            s_b = float(np.polyfit(steps[vm_b], meds_b[vm_b], 1)[0])
            perm_diffs.append(s_b - s_a)
    if len(perm_diffs) >= 100:
        perm_diffs = np.array(perm_diffs)
        # Two-sided p: fraction of permutations with |diff| >= |observed|
        perm_p = float(np.mean(np.abs(perm_diffs) >= abs(slope_diff)))
    else:
        perm_p = None

    return {
        'group_a_medians': a_medians, 'group_a_iqr': a_iqr,
        'group_b_medians': b_medians, 'group_b_iqr': b_iqr,
        'group_a_valid_counts': a_valid, 'group_b_valid_counts': b_valid,
        'group_a_tail_prob': a_tail, 'group_b_tail_prob': b_tail,
        'group_a_slope': a_slope, 'group_b_slope': b_slope,
        'group_a_slope_ci': a_slope_ci, 'group_b_slope_ci': b_slope_ci,
        'slope_difference': slope_diff, 'slope_difference_ci': slope_diff_ci,
        'slope_difference_perm_p': perm_p,
        'mann_whitney_p': mw_p, 'ks_p': ks_p_vals, 'fisher_p': fish_p,
    }


def run_experiment_paper2(
    n_trajectories: int = 200,
    n_steps: int = 5,
    max_candidates: int = 50,
    template: str = 'brusselator',
    seed: int = 42,
    checkpoint_every: int = 50,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> Paper2Result:
    """
    Paper 2 main experiment: Random (Group A) vs Aligned (Group B).

    Group A: Progressive autocatalytic additions with no filtering.
    Group B: Progressive autocatalytic additions that preserve oscillation.

    Both groups start from the same Brusselator baseline and add up to
    n_steps autocatalytic reactions. η = D₂/r_S is measured at each step.

    Parameters
    ----------
    n_trajectories : int
        Number of independent trajectories per group.
    n_steps : int
        Number of autocatalytic additions (0..n_steps).
    max_candidates : int
        Max candidates per step for Group B aligned additions.
    template : str
        Oscillator template name.
    seed : int
        Random seed for reproducibility.
    checkpoint_every : int
        Save intermediate results every N trajectories.
    checkpoint_dir : str, optional
        Directory for checkpoint files. If None, no checkpoints saved.
    verbose : bool
        Print progress.

    Returns
    -------
    Paper2Result
    """
    if verbose:
        print("=" * 70)
        print("PAPER 2: Random vs Feedback-Aligned Autocatalysis")
        print("=" * 70)
        print(f"  Template: {template}")
        print(f"  Trajectories per group: {n_trajectories}")
        print(f"  Steps: baseline + {n_steps} additions")
        print(f"  Max candidates (Group B): {max_candidates}")
        print(f"  Seed: {seed}")
        print()

    total_steps = n_steps + 1  # baseline + n_steps additions
    tracker = ActivationTracker(
        t_span=(0, 200), n_points=10000,
        remove_transient=0.5, random_state=seed,
    )

    # Allocate storage
    a_eta = np.full((n_trajectories, total_steps), np.nan)
    b_eta = np.full((n_trajectories, total_steps), np.nan)

    # Group B metadata
    all_acceptance_rates = []     # per trajectory: list of rates per step
    all_filter_results = []       # per trajectory: list of OscillationResults
    early_term_count = 0

    # CV, D2, η for selection bias check (Group B accepted networks)
    bias_cvs = []
    bias_d2s = []
    bias_etas = []

    total_work = n_trajectories * 2  # A + B
    work_done = 0
    start_time = time.time()

    # --- GROUP A: Random progressive ---
    if verbose:
        print(">>> GROUP A: Random Progressive Autocatalysis")

    for traj in range(n_trajectories):
        gen = NetworkGenerator(template=template, seed=seed + traj)
        networks = gen.generate_progressive(n_steps)

        for step, net in enumerate(networks):
            result = _analyze_network(tracker, net)
            if not result.skipped and result.quality != QualityFlag.FAILED:
                a_eta[traj, step] = result.eta

        work_done += 1
        if verbose:
            elapsed = time.time() - start_time
            avg = elapsed / work_done
            remaining = (total_work - work_done) * avg
            pct = 100 * work_done / total_work
            eta_str = f"{remaining/60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"
            print(f"\r  Group A: {traj+1}/{n_trajectories} "
                  f"[{pct:.0f}% overall, ETA: {eta_str}]    ",
                  end="", flush=True)

        # Checkpoint
        if checkpoint_dir and (traj + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_dir, 'group_a', traj + 1, a_eta)

    if verbose:
        print()

    # --- GROUP B: Feedback-aligned progressive ---
    if verbose:
        print(">>> GROUP B: Feedback-Aligned Progressive Autocatalysis")

    for traj in range(n_trajectories):
        gen = NetworkGenerator(template=template, seed=seed + n_trajectories + traj)
        aligned_result = gen.generate_progressive_aligned(
            n_steps=n_steps, max_candidates=max_candidates,
        )

        traj_acceptance = []
        traj_filter = []

        if aligned_result.terminated_early:
            early_term_count += 1

        for step, net in enumerate(aligned_result.networks):
            result = _analyze_network(tracker, net)
            if not result.skipped and result.quality != QualityFlag.FAILED:
                b_eta[traj, step] = result.eta

                # Collect selection bias data for non-baseline steps
                if step > 0 and step - 1 < len(aligned_result.filter_results):
                    fr = aligned_result.filter_results[step - 1]
                    bias_cvs.append(fr.cv)
                    bias_d2s.append(result.D2)
                    bias_etas.append(result.eta)

        traj_acceptance = list(aligned_result.acceptance_rates)
        traj_filter = list(aligned_result.filter_results)
        all_acceptance_rates.append(traj_acceptance)
        all_filter_results.append(traj_filter)

        work_done += 1
        if verbose:
            elapsed = time.time() - start_time
            avg = elapsed / work_done
            remaining = (total_work - work_done) * avg
            pct = 100 * work_done / total_work
            eta_str = f"{remaining/60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"
            term_str = f" [{early_term_count} early term]" if early_term_count > 0 else ""
            print(f"\r  Group B: {traj+1}/{n_trajectories} "
                  f"[{pct:.0f}% overall, ETA: {eta_str}]{term_str}    ",
                  end="", flush=True)

        # Checkpoint
        if checkpoint_dir and (traj + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_dir, 'group_b', traj + 1, b_eta)

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # --- Compute statistics ---
    if verbose:
        print("\n>>> Computing statistics...")

    stats = _compute_paper2_statistics(a_eta, b_eta, n_bootstrap=10000, seed=seed)

    # Selection bias: Pearson correlations within accepted set
    cv_d2_corr = None
    cv_eta_corr = None
    if len(bias_cvs) >= 10:
        from scipy.stats import pearsonr
        # CV vs D₂
        valid_d2 = [(c, d) for c, d in zip(bias_cvs, bias_d2s) if not np.isnan(d)]
        if len(valid_d2) >= 10:
            cvs, d2s = zip(*valid_d2)
            cv_d2_corr, _ = pearsonr(cvs, d2s)
            cv_d2_corr = float(cv_d2_corr)
        # CV vs η
        valid_eta = [(c, e) for c, e in zip(bias_cvs, bias_etas) if not np.isnan(e)]
        if len(valid_eta) >= 10:
            cvs, etas = zip(*valid_eta)
            cv_eta_corr, _ = pearsonr(cvs, etas)
            cv_eta_corr = float(cv_eta_corr)

    # Acceptance rate medians per step
    acc_rate_medians = []
    for step_idx in range(n_steps):
        step_rates = [
            rates[step_idx]
            for rates in all_acceptance_rates
            if step_idx < len(rates)
        ]
        if step_rates:
            acc_rate_medians.append(float(np.median(step_rates)))
        else:
            acc_rate_medians.append(np.nan)

    result = Paper2Result(
        name="Paper 2: Random vs Feedback-Aligned Autocatalysis",
        hypothesis="Feedback-aligned additions show less dynamical dilution than random additions",
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        max_candidates=max_candidates,
        group_a_eta_matrix=a_eta,
        group_b_eta_matrix=b_eta,
        group_a_valid_counts=stats['group_a_valid_counts'],
        group_b_valid_counts=stats['group_b_valid_counts'],
        group_a_medians=stats['group_a_medians'],
        group_a_iqr=stats['group_a_iqr'],
        group_b_medians=stats['group_b_medians'],
        group_b_iqr=stats['group_b_iqr'],
        group_a_slope=stats['group_a_slope'],
        group_b_slope=stats['group_b_slope'],
        group_a_slope_ci=stats['group_a_slope_ci'],
        group_b_slope_ci=stats['group_b_slope_ci'],
        group_a_tail_prob=stats['group_a_tail_prob'],
        group_b_tail_prob=stats['group_b_tail_prob'],
        mann_whitney_p=stats['mann_whitney_p'],
        ks_p=stats['ks_p'],
        fisher_p=stats['fisher_p'],
        acceptance_rates_per_step=all_acceptance_rates,
        acceptance_rate_medians=acc_rate_medians,
        early_termination_count=early_term_count,
        early_termination_rate=early_term_count / n_trajectories,
        cv_d2_correlation=cv_d2_corr,
        cv_eta_correlation=cv_eta_corr,
        slope_difference=stats['slope_difference'],
        slope_difference_ci=stats['slope_difference_ci'],
        slope_difference_perm_p=stats['slope_difference_perm_p'],
        group_b_filter_results=all_filter_results,
        parameters={
            'n_trajectories': n_trajectories,
            'n_steps': n_steps,
            'max_candidates': max_candidates,
            'template': template,
            'seed': seed,
        },
        timestamp=datetime.now().isoformat(),
    )

    if verbose:
        _print_paper2_summary(result)

    return result


def _save_checkpoint(checkpoint_dir: str, group_name: str, n_done: int,
                     eta_matrix: np.ndarray):
    """Save intermediate checkpoint."""
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'{group_name}_checkpoint_{n_done}.json')
    data = {
        'group': group_name,
        'n_done': n_done,
        'eta_matrix': [[None if np.isnan(x) else float(x) for x in row]
                        for row in eta_matrix[:n_done]],
        'timestamp': datetime.now().isoformat(),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _print_paper2_summary(result: Paper2Result):
    """Print summary table for Paper 2 results."""
    print("\n" + "=" * 70)
    print("PAPER 2 RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTrajectories: {result.n_trajectories} per group")
    print(f"Early termination (Group B): {result.early_termination_count} "
          f"({result.early_termination_rate:.1%})")

    print(f"\n{'Step':>4}  {'A med':>7} {'A IQR':>7} {'A n':>4}  "
          f"{'B med':>7} {'B IQR':>7} {'B n':>4}  {'MW p':>8} {'KS p':>8}")
    print("-" * 70)

    for k in range(result.n_steps + 1):
        a_m = f"{result.group_a_medians[k]:.4f}" if not np.isnan(result.group_a_medians[k]) else "  N/A"
        a_i = f"{result.group_a_iqr[k]:.4f}" if not np.isnan(result.group_a_iqr[k]) else "  N/A"
        b_m = f"{result.group_b_medians[k]:.4f}" if not np.isnan(result.group_b_medians[k]) else "  N/A"
        b_i = f"{result.group_b_iqr[k]:.4f}" if not np.isnan(result.group_b_iqr[k]) else "  N/A"
        mw = f"{result.mann_whitney_p[k]:.4f}" if result.mann_whitney_p[k] is not None else "    N/A"
        ks = f"{result.ks_p[k]:.4f}" if result.ks_p[k] is not None else "    N/A"
        print(f"  +{k}  {a_m:>7} {a_i:>7} {result.group_a_valid_counts[k]:>4}  "
              f"{b_m:>7} {b_i:>7} {result.group_b_valid_counts[k]:>4}  {mw:>8} {ks:>8}")

    print(f"\nSlopes:")
    a_ci = result.group_a_slope_ci
    b_ci = result.group_b_slope_ci
    print(f"  Group A: {result.group_a_slope:+.4f}/rxn "
          f"[95% CI: {a_ci[0]:+.4f}, {a_ci[1]:+.4f}]")
    print(f"  Group B: {result.group_b_slope:+.4f}/rxn "
          f"[95% CI: {b_ci[0]:+.4f}, {b_ci[1]:+.4f}]")
    d_ci = result.slope_difference_ci
    print(f"  Δslope (B−A): {result.slope_difference:+.4f}/rxn "
          f"[95% CI: {d_ci[0]:+.4f}, {d_ci[1]:+.4f}]")
    if result.slope_difference_perm_p is not None:
        print(f"  Permutation test p = {result.slope_difference_perm_p:.4f}")

    print(f"\nTail probability P(η > 0.25) at k={result.n_steps}:")
    a_t = result.group_a_tail_prob[-1]
    b_t = result.group_b_tail_prob[-1]
    f_p = result.fisher_p[-1]
    print(f"  Group A: {a_t:.3f}" if not np.isnan(a_t) else "  Group A: N/A")
    print(f"  Group B: {b_t:.3f}" if not np.isnan(b_t) else "  Group B: N/A")
    print(f"  Fisher p: {f_p:.4f}" if f_p is not None else "  Fisher p: N/A")

    print(f"\nSelection bias checks:")
    if result.cv_d2_correlation is not None:
        print(f"  r(CV, D₂) = {result.cv_d2_correlation:.3f}")
    else:
        print(f"  r(CV, D₂) = insufficient data")
    if result.cv_eta_correlation is not None:
        print(f"  r(CV, η)  = {result.cv_eta_correlation:.3f}")
    else:
        print(f"  r(CV, η)  = insufficient data")

    if result.acceptance_rate_medians:
        print(f"\nAcceptance rates (Group B medians):")
        for i, rate in enumerate(result.acceptance_rate_medians):
            r_str = f"{rate:.3f}" if not np.isnan(rate) else "N/A"
            print(f"  Step {i+1}: {r_str}")

    print("=" * 70)
