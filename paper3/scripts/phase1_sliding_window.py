"""
Sliding-window D₂ analysis: resolve whether D₂ > 1.2 is transient or stable.

The convergence test (T-doubling) showed D₂ dropping from ~1.6 at T=10000
to ~1.0 at T=20000. But this compares DIFFERENT time windows:
  T=10000: analyzes t=[5000, 10000]  (50% discard)
  T=20000: analyzes t=[10000, 20000] (50% discard)

If the system phase-locks between t=10000 and t=20000, the T=20000
window would average complex + simple dynamics, pulling D₂ down.

This script runs ONE long integration (T=20000) and computes D₂ on
non-overlapping windows to see exactly when/if D₂ drops.

Windows: [2500,5000], [5000,7500], [7500,10000], [10000,12500],
         [12500,15000], [15000,17500], [17500,20000]
Each window = 2500 time units ≈ 250 oscillator periods.
"""

import sys
import os
import json
import time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_astro2_path = os.path.join(_this_dir, '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from pilot.pilot5b_enzyme_complex import (
    EnzymeComplexParams, make_enzyme_complex_network,
)
from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension


def simulate_long(p: EnzymeComplexParams, seed: int, t_end: float = 20000,
                  n_points: int = 20000) -> dict | None:
    """Run long simulation, return FULL trajectory (no discard)."""
    net = make_enzyme_complex_network(p, seed=seed)
    sim = ReactionSimulator()
    try:
        network_obj = sim.build_network(net.reactions)
        result = sim.simulate(
            network_obj,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=(0, t_end),
            n_points=n_points,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=net.chemostat_species,
        )
    except Exception as e:
        print(f"  Simulation failed: {e}")
        return None

    if not result.success:
        print(f"  Solver failed: {result.solver_message}")
        return None

    t = result.time
    c = result.concentrations
    species = result.species_names

    def idx(name):
        return species.index(name)

    return {
        't': t,
        'X1': c[:, idx('X1')],
        'Y1': c[:, idx('Y1')],
        'X2': c[:, idx('X2')],
        'Y2': c[:, idx('Y2')],
        'E':  c[:, idx('E')],
    }


def compute_d2_window(traj: dict, t_start: float, t_end: float) -> dict:
    """Compute D₂ on a specific time window of the trajectory."""
    t = traj['t']
    mask = (t >= t_start) & (t < t_end)
    n_pts = np.sum(mask)

    if n_pts < 500:
        return {'t_start': t_start, 't_end': t_end, 'D2': float('nan'),
                'n_points': int(n_pts), 'quality': 'TOO_FEW_POINTS'}

    trajectory = np.column_stack([
        traj['X1'][mask], traj['Y1'][mask],
        traj['X2'][mask], traj['Y2'][mask],
        traj['E'][mask],
    ])

    cd = CorrelationDimension()
    try:
        result = cd.compute(trajectory)
        d2 = float(result.D2) if result.D2 is not None else float('nan')
        unc = float(result.D2_uncertainty) if result.D2_uncertainty else float('nan')
        quality = result.quality.name if result.quality else 'UNKNOWN'
    except Exception as e:
        d2 = float('nan')
        unc = float('nan')
        quality = f'FAILED: {e}'

    # Also compute phase correlation in this window
    X1w = traj['X1'][mask]
    X2w = traj['X2'][mask]
    r = float(np.corrcoef(X1w, X2w)[0, 1])

    return {
        't_start': t_start, 't_end': t_end,
        'D2': d2, 'D2_unc': unc, 'quality': quality,
        'n_points': int(n_pts), 'r_X1X2': r,
    }


def run_sliding_window(save_dir: str | None = None) -> dict:
    """Run sliding-window D₂ analysis on both exemplar points."""

    print("\n" + "#" * 80)
    print("# SLIDING-WINDOW D₂ ANALYSIS")
    print("# Purpose: determine if D₂ > 1.2 is transient or stable")
    print("#" * 80)

    exemplars = [
        EnzymeComplexParams(
            J=5.0, gamma=0.002, k_on=10.0, k_off=10.0,
            k_cat=0.3, G_total=1.0,
            label="Regime2_J5_g0.002_kc0.3",
        ),
        EnzymeComplexParams(
            J=7.0, gamma=0.001, k_on=10.0, k_off=10.0,
            k_cat=0.5, G_total=1.0,
            label="Regime1_J7_g0.001_kc0.5",
        ),
    ]

    # Windows: skip first 2500 as initial transient, then 2500-unit windows
    window_size = 2500
    windows = [(i * window_size, (i + 1) * window_size)
               for i in range(1, 8)]  # [2500,5000] to [17500,20000]

    seeds = [42, 179]
    all_results = {}
    start_time = time.time()

    for p in exemplars:
        print(f"\n{'=' * 70}")
        print(f"  {p.label}  (J={p.J}, γ={p.gamma}, k_cat={p.k_cat})")
        print(f"{'=' * 70}")

        exemplar_results = []

        for seed in seeds:
            print(f"\n  Seed {seed}: simulating T=20000...", end='', flush=True)
            t0 = time.time()
            traj = simulate_long(p, seed=seed, t_end=20000, n_points=40000)
            if traj is None:
                print(" FAILED")
                continue
            sim_time = time.time() - t0
            print(f" done ({sim_time:.1f}s)")

            print(f"  {'Window':>20}  {'D₂':>7}  {'±':>6}  {'r(X1,X2)':>9}  {'Quality':>10}  {'N':>5}")
            print(f"  {'-'*65}")

            seed_results = {'seed': seed, 'sim_time': sim_time, 'windows': []}

            for t_start, t_end in windows:
                w = compute_d2_window(traj, t_start, t_end)
                seed_results['windows'].append(w)

                d2_str = f"{w['D2']:.3f}" if not np.isnan(w['D2']) else "N/A"
                unc_str = f"{w['D2_unc']:.3f}" if not np.isnan(w.get('D2_unc', float('nan'))) else ""
                r_str = f"{w['r_X1X2']:.3f}" if not np.isnan(w.get('r_X1X2', float('nan'))) else ""
                marker = " ***" if w['D2'] > 1.2 and not np.isnan(w['D2']) else ""

                print(f"  [{t_start:>6},{t_end:>6}]  {d2_str:>7}  {unc_str:>6}  {r_str:>9}  "
                      f"{w['quality']:>10}  {w['n_points']:>5}{marker}")

            # Also compute D₂ on the full post-2500 window [2500, 20000]
            print(f"\n  Full window [2500, 20000]:")
            w_full = compute_d2_window(traj, 2500, 20000)
            seed_results['full_window'] = w_full
            d2_str = f"{w_full['D2']:.3f}" if not np.isnan(w_full['D2']) else "N/A"
            r_str = f"{w_full['r_X1X2']:.3f}" if not np.isnan(w_full.get('r_X1X2', float('nan'))) else ""
            print(f"  [{2500:>6},{20000:>6}]  D₂={d2_str}  r={r_str}")

            # And [5000, 10000] to match the standard sweep
            print(f"  Standard sweep window [5000, 10000]:")
            w_std = compute_d2_window(traj, 5000, 10000)
            seed_results['standard_window'] = w_std
            d2_str = f"{w_std['D2']:.3f}" if not np.isnan(w_std['D2']) else "N/A"
            print(f"  [{5000:>6},{10000:>6}]  D₂={d2_str}")

            exemplar_results.append(seed_results)

        all_results[p.label] = exemplar_results

    total_time = time.time() - start_time

    # ── Verdict ──────────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print(f"  SLIDING-WINDOW VERDICT  ({total_time:.0f}s / {total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    for label, results in all_results.items():
        print(f"\n  {label}:")
        for sr in results:
            d2_by_window = [(w['t_start'], w['D2']) for w in sr['windows']
                            if not np.isnan(w['D2'])]
            if d2_by_window:
                # Check if D₂ stays > 1.2 throughout
                all_above = all(d2 > 1.2 for _, d2 in d2_by_window)
                late_above = all(d2 > 1.2 for t, d2 in d2_by_window if t >= 10000)
                early_d2 = [d2 for t, d2 in d2_by_window if t < 10000]
                late_d2 = [d2 for t, d2 in d2_by_window if t >= 10000]

                early_med = np.median(early_d2) if early_d2 else float('nan')
                late_med = np.median(late_d2) if late_d2 else float('nan')

                status = "STABLE ✓" if all_above else ("LATE OK" if late_above else "TRANSIENT ✗")
                print(f"    Seed {sr['seed']}: early D₂={early_med:.3f}, "
                      f"late D₂={late_med:.3f}, {status}")
                print(f"      Trajectory: {' → '.join(f'{d2:.2f}' for _, d2 in d2_by_window)}")

    print(f"{'=' * 70}")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        'test': 'sliding_window_d2',
        'description': 'D₂ on non-overlapping 2500-unit windows from T=20000 integration',
        'runtime_seconds': total_time,
        'window_size': window_size,
        'windows': [(t0, t1) for t0, t1 in windows],
        'exemplars': {},
    }
    for label, results in all_results.items():
        output['exemplars'][label] = results

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'phase1_sliding_window_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_sliding_window(save_dir=save_dir)
