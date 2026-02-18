"""
Pilot 2: D₂ estimator validation on known systems.

Confirms that the Grassberger-Procaccia D₂ estimator can reliably
distinguish:
- Limit cycles (D₂ ≈ 1) from torus/chaotic behavior (D₂ ≈ 2)

Tests on:
1. Brusselator limit cycle (D₂ ≈ 1.0) — existing
2. Single Rössler attractor (D₂ ≈ 2.0) — existing
3. Coupled Rössler pair on torus (D₂ ≈ 2.0–2.5) — new
4. Lorenz attractor (D₂ ≈ 2.05) — existing

Each system is tested with multiple seeds to get error bars.
"""

import sys
import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional

# Add astrobiology2 to path
_astro2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from dimensional_opening.correlation_dimension import CorrelationDimension
from dimensional_opening.validation import (
    generate_rossler,
    generate_brusselator,
    generate_lorenz,
)


def generate_coupled_rossler(
    n_points: int = 10000,
    dt: float = 0.02,
    transient: int = 10000,
    coupling: float = 0.03,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a coupled Rössler system (two oscillators with weak coupling).

    With weak coupling and incommensurate frequencies, this produces
    quasi-periodic (torus) dynamics with D₂ ≈ 2.

    Parameters
    ----------
    n_points : int
        Post-transient points.
    dt : float
        Integration time step.
    transient : int
        Transient steps to discard.
    coupling : float
        Coupling strength (small → torus, large → sync).
    seed : int
        Random seed for initial conditions.

    Returns
    -------
    np.ndarray, shape (n_points, 6)
        Trajectory of [x1, y1, z1, x2, y2, z2].
    """
    # Two Rössler oscillators with slightly different frequencies
    a1, b1, c1 = 0.15, 0.2, 10.0  # Oscillator 1 (slightly different a, c)
    a2, b2, c2 = 0.2, 0.2, 5.7    # Oscillator 2 (standard Rössler)

    rng = np.random.default_rng(seed)
    state = rng.uniform(-1, 1, 6) + np.array([1.0, 1.0, 0.5, -1.0, 1.0, 0.5])

    trajectory = []
    for i in range(n_points + transient):
        x1, y1, z1, x2, y2, z2 = state

        # Oscillator 1
        dx1 = -y1 - z1 + coupling * (x2 - x1)
        dy1 = x1 + a1 * y1
        dz1 = b1 + z1 * (x1 - c1)

        # Oscillator 2
        dx2 = -y2 - z2 + coupling * (x1 - x2)
        dy2 = x2 + a2 * y2
        dz2 = b2 + z2 * (x2 - c2)

        state = state + dt * np.array([dx1, dy1, dz1, dx2, dy2, dz2])

        if i >= transient:
            trajectory.append(state.copy())

    return np.array(trajectory)


@dataclass
class D2Estimate:
    """D₂ estimate for one system at one seed."""
    system: str
    expected_D2: float
    seed: int
    measured_D2: float
    uncertainty: float
    quality: str


@dataclass
class SystemSummary:
    """Summary of D₂ estimates for one system across seeds."""
    system: str
    expected_D2: float
    n_seeds: int
    n_good: int
    mean_D2: float
    std_D2: float
    min_D2: float
    max_D2: float
    separable_from_1: bool  # Is the 95% CI entirely above 1.3?


def run_pilot2(
    n_seeds: int = 5,
    base_seed: int = 42,
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> List[SystemSummary]:
    """
    Run D₂ validation on known systems.

    Parameters
    ----------
    n_seeds : int
        Number of independent seeds per system.
    base_seed : int
        Base random seed.
    verbose : bool
        Print progress.
    save_dir : str, optional
        Directory to save results JSON.

    Returns
    -------
    list of SystemSummary
    """
    if verbose:
        print("\n" + "#" * 80)
        print("# PILOT 2: D₂ Estimator Validation")
        print("#" * 80)

    cd = CorrelationDimension()

    systems = [
        ("Brusselator limit cycle", 1.0,
         lambda seed: generate_brusselator(n_points=5000, seed=seed)),
        ("Rössler attractor", 1.99,
         lambda seed: generate_rossler(n_points=10000, seed=seed)),
        ("Lorenz attractor", 2.05,
         lambda seed: generate_lorenz(n_points=10000, seed=seed)),
        ("Coupled Rössler (torus)", 2.0,
         lambda seed: generate_coupled_rossler(n_points=10000, seed=seed)),
    ]

    all_estimates = []
    summaries = []

    total = len(systems) * n_seeds
    done = 0
    start_time = time.time()

    for sys_name, expected, generator in systems:
        estimates = []

        for i in range(n_seeds):
            seed = base_seed + i * 7

            if verbose:
                done += 1
                elapsed = time.time() - start_time
                eta_sec = (total - done) * elapsed / done if done > 0 else 0
                print(f"\r  Validating: {done}/{total} "
                      f"({sys_name}, seed {i+1}/{n_seeds}) "
                      f"ETA: {eta_sec:.0f}s    ", end="", flush=True)

            try:
                traj = generator(seed)
                result = cd.compute(traj, random_state=seed)
                est = D2Estimate(
                    system=sys_name, expected_D2=expected, seed=seed,
                    measured_D2=float(result.D2),
                    uncertainty=float(result.D2_uncertainty),
                    quality=result.quality.value,
                )
            except Exception as e:
                est = D2Estimate(
                    system=sys_name, expected_D2=expected, seed=seed,
                    measured_D2=np.nan, uncertainty=np.nan, quality="FAILED",
                )

            estimates.append(est)
            all_estimates.append(est)

        # Summarize
        good_d2 = [e.measured_D2 for e in estimates
                    if not np.isnan(e.measured_D2) and e.quality != "FAILED"]

        if len(good_d2) >= 2:
            mean_d2 = float(np.mean(good_d2))
            std_d2 = float(np.std(good_d2, ddof=1))
            # 95% CI lower bound
            ci_lower = mean_d2 - 2 * std_d2
            separable = ci_lower > 1.3  # clearly above limit cycle
        elif len(good_d2) == 1:
            mean_d2 = good_d2[0]
            std_d2 = 0.0
            separable = mean_d2 > 1.3
        else:
            mean_d2 = np.nan
            std_d2 = np.nan
            separable = False

        summaries.append(SystemSummary(
            system=sys_name,
            expected_D2=expected,
            n_seeds=n_seeds,
            n_good=len(good_d2),
            mean_D2=mean_d2,
            std_D2=std_d2,
            min_D2=float(np.min(good_d2)) if good_d2 else np.nan,
            max_D2=float(np.max(good_d2)) if good_d2 else np.nan,
            separable_from_1=separable if expected > 1.5 else False,
        ))

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  Validation complete in {elapsed:.1f}s")

    # Print table
    _print_validation_table(summaries)

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        data = {
            'estimates': [asdict(e) for e in all_estimates],
            'summaries': [asdict(s) for s in summaries],
        }
        path = os.path.join(save_dir, 'pilot2_results.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2,
                      default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
        if verbose:
            print(f"\n  Results saved to {path}")

    return summaries


def _print_validation_table(summaries: List[SystemSummary]):
    """Print formatted validation results."""
    print("\n" + "=" * 80)
    print("PILOT 2: D₂ Estimator Validation Results")
    print("=" * 80)
    print(f"{'System':<28}  {'Expected':>8}  {'Mean D2':>8}  {'Std':>6}  "
          f"{'Range':>13}  {'Good':>4}  {'Sep?':>4}")
    print("-" * 80)

    for s in summaries:
        mean_str = f"{s.mean_D2:.3f}" if not np.isnan(s.mean_D2) else "  N/A"
        std_str = f"{s.std_D2:.3f}" if not np.isnan(s.std_D2) else " N/A"
        if not np.isnan(s.min_D2):
            range_str = f"[{s.min_D2:.2f}, {s.max_D2:.2f}]"
        else:
            range_str = "     N/A"
        sep_str = "YES" if s.separable_from_1 else "no"

        print(f"{s.system:<28}  {s.expected_D2:>8.2f}  {mean_str:>8}  {std_str:>6}  "
              f"{range_str:>13}  {s.n_good:>4}  {sep_str:>4}")

    # Key question
    limit_cycle = next((s for s in summaries if s.expected_D2 <= 1.1), None)
    higher_dim = [s for s in summaries if s.expected_D2 > 1.5]

    print("\n" + "-" * 80)
    if limit_cycle and higher_dim:
        lc_upper = limit_cycle.mean_D2 + 2 * limit_cycle.std_D2 if not np.isnan(limit_cycle.std_D2) else np.nan
        hd_lowers = [s.mean_D2 - 2 * s.std_D2 for s in higher_dim
                     if not np.isnan(s.std_D2)]
        if hd_lowers and not np.isnan(lc_upper):
            min_hd_lower = min(hd_lowers)
            gap = min_hd_lower - lc_upper
            print(f"  Limit cycle 95% upper: {lc_upper:.3f}")
            print(f"  Higher-dim 95% lower:  {min_hd_lower:.3f}")
            print(f"  Gap: {gap:.3f}")
            if gap > 0:
                print("  VERDICT: D₂ estimator CAN separate limit cycle from torus/chaos")
            else:
                print("  VERDICT: D₂ estimator CANNOT reliably separate — CIs overlap")
        else:
            print("  VERDICT: Insufficient data to assess separation")
    else:
        print("  VERDICT: Missing reference systems")

    print("=" * 80)
