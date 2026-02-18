"""
Pilot 1: Drive window sweep.

Sweep drive strength J to find the regime where:
- Oscillations survive (not blown up or stuck at fixed point)
- There's a chance of D₂ > 1 (higher-dimensional dynamics)

Also runs a small growth experiment at promising J values.
"""

import sys
import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add astrobiology2 to path
_astro2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from dimensional_opening.activation_tracker import ActivationTracker, ActivationResult
from dimensional_opening.oscillation_filter import passes_oscillation_filter
from dimensional_opening.correlation_dimension import QualityFlag

from .driven_brusselator import make_driven_brusselator, add_random_autocatalytic_reaction


@dataclass
class RegimeClassification:
    """Classification of a single run."""
    J: float
    blowup: bool
    fixed_point: bool
    oscillating: bool
    D2: float
    eta: float
    r_S: int
    quality: str


@dataclass
class JLevelSummary:
    """Summary statistics at one J level."""
    J: float
    n_runs: int
    n_blowup: int
    n_fixed_point: int
    n_oscillating: int
    n_D2_above_1: int
    n_D2_above_1p2: int
    median_D2: float
    iqr_D2: float
    median_eta: float


def classify_run(
    net,
    tracker: ActivationTracker,
) -> RegimeClassification:
    """
    Simulate and classify a single network run.

    Returns classification: blowup, fixed_point, or oscillating,
    plus D₂ and η if measurable.
    """
    J = float(net.network_id.split("J")[1].split("_")[0]) if "J" in net.network_id else 0.0

    # Check oscillation
    osc = passes_oscillation_filter(net)

    if not osc.passes:
        # Distinguish blowup from fixed point
        # A blowup typically has boundedness_ratio far from 1
        if osc.boundedness_ratio > 10 or osc.boundedness_ratio < 0.01:
            return RegimeClassification(
                J=J, blowup=True, fixed_point=False, oscillating=False,
                D2=np.nan, eta=np.nan, r_S=0, quality="FAILED",
            )
        else:
            return RegimeClassification(
                J=J, blowup=False, fixed_point=True, oscillating=False,
                D2=np.nan, eta=np.nan, r_S=0, quality="FAILED",
            )

    # Oscillating — now measure D₂
    input_dict = net.to_tracker_input()
    result = tracker.analyze_network(**input_dict)

    if result.skipped or result.quality == QualityFlag.FAILED:
        return RegimeClassification(
            J=J, blowup=False, fixed_point=False, oscillating=True,
            D2=np.nan, eta=np.nan, r_S=result.r_S, quality="FAILED",
        )

    return RegimeClassification(
        J=J, blowup=False, fixed_point=False, oscillating=True,
        D2=result.D2, eta=result.eta, r_S=result.r_S,
        quality=result.quality.value,
    )


def run_j_sweep(
    J_values: List[float] = None,
    n_runs: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> List[JLevelSummary]:
    """
    Sweep drive strength J and classify regime at each level.

    Parameters
    ----------
    J_values : list of float
        Drive strengths to test. Defaults to [0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0].
    n_runs : int
        Number of runs per J level.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    list of JLevelSummary
    """
    if J_values is None:
        J_values = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

    tracker = ActivationTracker(
        t_span=(0, 200), n_points=10000,
        remove_transient=0.5, random_state=seed,
    )

    summaries = []
    total = len(J_values) * n_runs
    done = 0
    start_time = time.time()

    for J in J_values:
        classifications = []

        for i in range(n_runs):
            net = make_driven_brusselator(J)
            # Give each run a unique ID
            net.network_id = f"driven_J{J:.2f}_run{i}"

            cls = classify_run(net, tracker)
            classifications.append(cls)

            done += 1
            if verbose:
                elapsed = time.time() - start_time
                eta_sec = (total - done) * elapsed / done if done > 0 else 0
                print(f"\r  J-sweep: {done}/{total} "
                      f"(J={J:.1f}, run {i+1}/{n_runs}) "
                      f"ETA: {eta_sec:.0f}s    ", end="", flush=True)

        # Summarize
        n_blowup = sum(1 for c in classifications if c.blowup)
        n_fp = sum(1 for c in classifications if c.fixed_point)
        n_osc = sum(1 for c in classifications if c.oscillating)
        d2_vals = [c.D2 for c in classifications if not np.isnan(c.D2)]
        eta_vals = [c.eta for c in classifications if not np.isnan(c.eta)]

        summaries.append(JLevelSummary(
            J=J,
            n_runs=n_runs,
            n_blowup=n_blowup,
            n_fixed_point=n_fp,
            n_oscillating=n_osc,
            n_D2_above_1=sum(1 for d in d2_vals if d > 1.0),
            n_D2_above_1p2=sum(1 for d in d2_vals if d > 1.2),
            median_D2=float(np.median(d2_vals)) if d2_vals else np.nan,
            iqr_D2=float(np.percentile(d2_vals, 75) - np.percentile(d2_vals, 25)) if len(d2_vals) >= 2 else np.nan,
            median_eta=float(np.median(eta_vals)) if eta_vals else np.nan,
        ))

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  J-sweep complete in {elapsed:.1f}s")

    return summaries


@dataclass
class GrowthCellResult:
    """Result for one cell of the growth x drive grid."""
    J: float
    k: int
    group: str  # "random" or "aligned"
    n_runs: int
    n_surviving: int
    n_D2_above_1: int
    median_D2: float
    median_eta: float


def run_growth_at_J(
    J_values: List[float],
    k_max: int = 3,
    n_runs: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> List[GrowthCellResult]:
    """
    Run growth protocol at specific J levels.

    For each J, grow the network k=0..k_max with random additions
    and measure survival + D₂.

    Parameters
    ----------
    J_values : list of float
        Drive strengths to test.
    k_max : int
        Max additions.
    n_runs : int
        Runs per cell.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    list of GrowthCellResult
    """
    tracker = ActivationTracker(
        t_span=(0, 200), n_points=10000,
        remove_transient=0.5, random_state=seed,
    )
    rng = np.random.default_rng(seed)

    results = []
    total = len(J_values) * (k_max + 1) * n_runs
    done = 0
    start_time = time.time()

    for J in J_values:
        for k in range(k_max + 1):
            d2_vals = []
            eta_vals = []
            n_surviving = 0

            for run_idx in range(n_runs):
                # Build base driven network
                net = make_driven_brusselator(J)

                # Add k random autocatalytic reactions
                for add_idx in range(k):
                    try:
                        net = add_random_autocatalytic_reaction(
                            net, rng=np.random.default_rng(seed + run_idx * 100 + add_idx)
                        )
                    except RuntimeError:
                        break

                net.network_id = f"growth_J{J:.2f}_k{k}_run{run_idx}"

                cls = classify_run(net, tracker)
                if cls.oscillating:
                    n_surviving += 1
                    if not np.isnan(cls.D2):
                        d2_vals.append(cls.D2)
                    if not np.isnan(cls.eta):
                        eta_vals.append(cls.eta)

                done += 1
                if verbose:
                    elapsed = time.time() - start_time
                    eta_sec = (total - done) * elapsed / done if done > 0 else 0
                    print(f"\r  Growth: {done}/{total} "
                          f"(J={J:.1f}, k={k}, run {run_idx+1}/{n_runs}) "
                          f"ETA: {eta_sec:.0f}s    ", end="", flush=True)

            results.append(GrowthCellResult(
                J=J, k=k, group="random",
                n_runs=n_runs,
                n_surviving=n_surviving,
                n_D2_above_1=sum(1 for d in d2_vals if d > 1.0),
                median_D2=float(np.median(d2_vals)) if d2_vals else np.nan,
                median_eta=float(np.median(eta_vals)) if eta_vals else np.nan,
            ))

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  Growth experiment complete in {elapsed:.1f}s")

    return results


def print_j_sweep_table(summaries: List[JLevelSummary]):
    """Print a formatted table of J-sweep results."""
    print("\n" + "=" * 80)
    print("PILOT 1: Drive Window Sweep")
    print("=" * 80)
    print(f"{'J':>5}  {'N':>3}  {'Blowup':>7}  {'FixPt':>6}  {'Osc':>5}  "
          f"{'D2>1':>5}  {'D2>1.2':>6}  {'med D2':>7}  {'IQR D2':>7}  {'med eta':>8}")
    print("-" * 80)

    for s in summaries:
        d2_str = f"{s.median_D2:.3f}" if not np.isnan(s.median_D2) else "  N/A"
        iqr_str = f"{s.iqr_D2:.3f}" if not np.isnan(s.iqr_D2) else "  N/A"
        eta_str = f"{s.median_eta:.4f}" if not np.isnan(s.median_eta) else "   N/A"
        print(f"{s.J:>5.1f}  {s.n_runs:>3}  "
              f"{s.n_blowup:>7}  {s.n_fixed_point:>6}  {s.n_oscillating:>5}  "
              f"{s.n_D2_above_1:>5}  {s.n_D2_above_1p2:>6}  "
              f"{d2_str:>7}  {iqr_str:>7}  {eta_str:>8}")

    print("=" * 80)


def print_growth_table(results: List[GrowthCellResult]):
    """Print a formatted table of growth experiment results."""
    print("\n" + "=" * 70)
    print("PILOT 1: Growth at Promising J Values")
    print("=" * 70)
    print(f"{'J':>5}  {'k':>2}  {'Surv':>5}  {'N':>3}  {'D2>1':>5}  {'med D2':>7}  {'med eta':>8}")
    print("-" * 70)

    for r in results:
        d2_str = f"{r.median_D2:.3f}" if not np.isnan(r.median_D2) else "  N/A"
        eta_str = f"{r.median_eta:.4f}" if not np.isnan(r.median_eta) else "   N/A"
        print(f"{r.J:>5.1f}  {r.k:>2}  {r.n_surviving:>5}  {r.n_runs:>3}  "
              f"{r.n_D2_above_1:>5}  {d2_str:>7}  {eta_str:>8}")

    print("=" * 70)


def run_pilot1(
    seed: int = 42,
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Run complete Pilot 1: J sweep + growth at promising values.

    Returns dict with 'j_sweep' and 'growth' results.
    """
    if verbose:
        print("\n" + "#" * 80)
        print("# PILOT 1: Drive Window Identification")
        print("#" * 80)

    # Phase 1: J sweep
    if verbose:
        print("\n>>> Phase 1: J-level sweep")
    summaries = run_j_sweep(seed=seed, verbose=verbose)
    print_j_sweep_table(summaries)

    # Phase 2: Growth at promising J values
    # Pick J values where oscillation survives and there's any sign of D₂ > 1
    promising = [s.J for s in summaries
                 if s.n_oscillating >= 5 and s.J > 0]
    if not promising:
        # Fall back to any J > 0 with oscillations
        promising = [s.J for s in summaries if s.n_oscillating >= 1 and s.J > 0]
    # Take up to 3
    promising = promising[:3]

    growth_results = []
    if promising:
        if verbose:
            print(f"\n>>> Phase 2: Growth experiment at J = {promising}")
        growth_results = run_growth_at_J(
            J_values=promising, k_max=3, n_runs=10, seed=seed, verbose=verbose,
        )
        print_growth_table(growth_results)
    else:
        if verbose:
            print("\n>>> Phase 2: Skipped (no promising J values found)")

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        data = {
            'j_sweep': [asdict(s) for s in summaries],
            'growth': [asdict(r) for r in growth_results],
            'promising_J': promising,
        }
        path = os.path.join(save_dir, 'pilot1_results.json')
        with open(path, 'w') as f:
            # Handle NaN for JSON
            json.dump(data, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
        if verbose:
            print(f"\n  Results saved to {path}")

    return {'j_sweep': summaries, 'growth': growth_results, 'promising_J': promising}
