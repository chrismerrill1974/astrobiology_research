"""
Phase 0: Baseline Characterisation for Paper 4.

Batch-mode: processes one parameter set at a time (sequentially),
writes each result to a JSONL file incrementally, and supports
resume from partial runs.

Usage:
  python3 phase0_baseline.py                # run all 200
  python3 phase0_baseline.py --start 50     # resume from param 50
  python3 phase0_baseline.py --end 10       # just first 10 (test)
  python3 phase0_baseline.py --analyse      # analyse existing results
"""

import sys
import os
import json
import time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_shared_path = os.path.join(_this_dir, '..', '..', 'shared')
_paper3_scripts = os.path.join(_this_dir, '..', '..', 'paper3', 'scripts')
if os.path.abspath(_shared_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_shared_path))
if os.path.abspath(_paper3_scripts) not in sys.path:
    sys.path.insert(0, os.path.abspath(_paper3_scripts))

from pilot5b_enzyme_complex import (
    EnzymeComplexParams, make_enzyme_complex_network,
)
from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension


# ── Configuration ─────────────────────────────────────────────────────

N_SAMPLES = 200
N_SEEDS = 5
T_END = 20000
N_POINTS = 40000
WINDOW_SIZE = 2500
D2_THRESHOLD = 1.2
TLOCK_THRESHOLD = 1.1

GAMMA_MIN = 0.0005
GAMMA_MAX = 0.01
J_MIN = 2.0
J_MAX = 12.0
KCAT_MIN = 0.05
KCAT_MAX = 0.8

MAX_CONCENTRATION = 1e6
SEEDS = [42 + s * 137 for s in range(N_SEEDS)]


# ── Core computation (single-threaded) ────────────────────────────────

def simulate_long(p: EnzymeComplexParams, seed: int) -> dict | None:
    """Run long simulation, return full trajectory or None on failure."""
    net = make_enzyme_complex_network(p, seed=seed)
    sim = ReactionSimulator()
    try:
        network_obj = sim.build_network(net.reactions)
        result = sim.simulate(
            network_obj,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=(0, T_END),
            n_points=N_POINTS,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=net.chemostat_species,
        )
    except Exception:
        return None

    if not result.success:
        return None

    t = result.time
    c = result.concentrations
    species = result.species_names

    def idx(name):
        return species.index(name)

    for sp_name in ['X1', 'Y1', 'X2', 'Y2', 'E']:
        vals = c[:, idx(sp_name)]
        if np.any(vals > MAX_CONCENTRATION) or np.any(vals < 0):
            return None

    return {
        't': t,
        'X1': c[:, idx('X1')],
        'Y1': c[:, idx('Y1')],
        'X2': c[:, idx('X2')],
        'Y2': c[:, idx('Y2')],
        'E':  c[:, idx('E')],
    }


def compute_d2_window(traj: dict, t_start: float, t_end: float) -> float:
    """Compute D2 on a specific time window. Returns D2 or NaN."""
    t = traj['t']
    mask = (t >= t_start) & (t < t_end)
    if np.sum(mask) < 500:
        return float('nan')

    trajectory = np.column_stack([
        traj['X1'][mask], traj['Y1'][mask],
        traj['X2'][mask], traj['Y2'][mask],
        traj['E'][mask],
    ])

    cd = CorrelationDimension()
    try:
        result = cd.compute(trajectory)
        return float(result.D2) if result.D2 is not None else float('nan')
    except Exception:
        return float('nan')


def evaluate_one_param(param_idx: int, gamma: float, J: float,
                       k_cat: float) -> dict:
    """
    Evaluate one parameter set across all seeds. SEQUENTIAL.
    Returns a summary dict.
    """
    taus = []
    t_locks = []
    n_failed = 0

    for seed in SEEDS:
        p = EnzymeComplexParams(
            A=1.0, B=3.0, k_on=10.0, k_off=10.0,
            k_cat=k_cat, G_total=1.0, J=J, gamma=gamma,
            label=f"p{param_idx}_s{seed}",
        )
        traj = simulate_long(p, seed=seed)
        if traj is None:
            taus.append(0)
            t_locks.append(None)
            n_failed += 1
            continue

        # Sliding-window D2
        tau = 0
        t_lock = None
        d2_vals = []
        for win_idx in range(1, 8):
            t_start = win_idx * WINDOW_SIZE
            t_end = (win_idx + 1) * WINDOW_SIZE
            d2 = compute_d2_window(traj, t_start, t_end)
            d2_vals.append(d2)
            if not np.isnan(d2) and d2 > D2_THRESHOLD:
                tau += 1

        # T_lock
        for i in range(len(d2_vals)):
            all_below = True
            for j in range(i, len(d2_vals)):
                if np.isnan(d2_vals[j]) or d2_vals[j] >= TLOCK_THRESHOLD:
                    all_below = False
                    break
            if all_below:
                t_lock = (i + 1) * WINDOW_SIZE
                break

        taus.append(tau)
        t_locks.append(t_lock)

    return {
        'param_idx': param_idx,
        'gamma': gamma,
        'J': J,
        'k_cat': k_cat,
        'taus': taus,
        'mean_tau': float(np.mean(taus)),
        'max_tau': int(max(taus)),
        'n_failed': n_failed,
        't_locks': t_locks,
    }


# ── Parameter generation (deterministic) ─────────────────────────────

def generate_params():
    """Generate the 200 parameter vectors (deterministic seed)."""
    rng = np.random.RandomState(2024)
    log_gamma_min = np.log10(GAMMA_MIN)
    log_gamma_max = np.log10(GAMMA_MAX)
    gammas = 10 ** rng.uniform(log_gamma_min, log_gamma_max, N_SAMPLES)
    Js = rng.uniform(J_MIN, J_MAX, N_SAMPLES)
    kcats = rng.uniform(KCAT_MIN, KCAT_MAX, N_SAMPLES)
    return gammas, Js, kcats


# ── Batch runner with incremental writes ──────────────────────────────

def run_batch(start: int = 0, end: int = None, save_dir: str = None):
    """Run Phase 0 in batch mode with incremental JSONL output."""
    if save_dir is None:
        save_dir = os.path.join(_this_dir, '..', 'data')
    os.makedirs(save_dir, exist_ok=True)

    jsonl_path = os.path.join(save_dir, 'phase0_incremental.jsonl')
    gammas, Js, kcats = generate_params()

    if end is None:
        end = N_SAMPLES

    # Check what's already done
    done_indices = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_indices.add(rec['param_idx'])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Found {len(done_indices)} already completed in {jsonl_path}")

    todo = [i for i in range(start, end) if i not in done_indices]

    print(f"\n  Phase 0: params {start}-{end-1}, {len(todo)} remaining "
          f"({len(done_indices)} already done)")
    print(f"  Output: {jsonl_path}")
    print(f"  Each param set = {N_SEEDS} seeds x ~5-30s = ~25-150s\n")

    batch_start = time.time()

    for count, idx in enumerate(todo):
        t0 = time.time()
        result = evaluate_one_param(
            idx, float(gammas[idx]), float(Js[idx]), float(kcats[idx]))
        elapsed = time.time() - t0

        # Append to JSONL immediately
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')

        total_done = count + 1
        total_elapsed = time.time() - batch_start
        rate = total_done / total_elapsed if total_elapsed > 0 else 0
        remaining = len(todo) - total_done
        eta = remaining / rate if rate > 0 else 0

        print(f"  [{total_done}/{len(todo)}] param {idx:3d} | "
              f"gamma={result['gamma']:.5f} J={result['J']:.2f} "
              f"kcat={result['k_cat']:.3f} | "
              f"mean_tau={result['mean_tau']:.1f} taus={result['taus']} "
              f"failed={result['n_failed']} | "
              f"{elapsed:.1f}s | ETA {eta:.0f}s", flush=True)

    total_time = time.time() - batch_start
    print(f"\n  Batch done in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Results in {jsonl_path}")


# ── Analysis of completed results ─────────────────────────────────────

def analyse(save_dir: str = None):
    """Analyse results from incremental JSONL file."""
    if save_dir is None:
        save_dir = os.path.join(_this_dir, '..', 'data')

    jsonl_path = os.path.join(save_dir, 'phase0_incremental.jsonl')
    if not os.path.exists(jsonl_path):
        print(f"  No results file found at {jsonl_path}")
        return

    summaries = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                summaries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"\n{'=' * 70}")
    print(f"PHASE 0 RESULTS ({len(summaries)} parameter sets)")
    print(f"{'=' * 70}")

    if not summaries:
        print("  No results to analyse.")
        return

    mean_taus = [s['mean_tau'] for s in summaries]
    n_with_tau_gt0 = sum(1 for s in summaries if s['max_tau'] > 0)
    n_with_mean_tau_gt1 = sum(1 for s in summaries if s['mean_tau'] >= 1.0)
    n_total_failed = sum(s['n_failed'] for s in summaries)
    total_runs = len(summaries) * N_SEEDS

    print(f"\n  Parameter draws: {len(summaries)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Failed integrations: {n_total_failed} "
          f"({100*n_total_failed/total_runs:.1f}%)")
    print(f"\n  Draws with any tau > 0: {n_with_tau_gt0} "
          f"({100*n_with_tau_gt0/len(summaries):.1f}%)")
    print(f"  Draws with mean tau >= 1: {n_with_mean_tau_gt1} "
          f"({100*n_with_mean_tau_gt1/len(summaries):.1f}%)")
    print(f"\n  mean(tau) distribution:")
    print(f"    min={min(mean_taus):.2f}, median={np.median(mean_taus):.2f}, "
          f"mean={np.mean(mean_taus):.2f}, max={max(mean_taus):.2f}")
    print(f"    std={np.std(mean_taus):.2f}")

    # Histogram
    tau_counts = [0] * 8
    for s in summaries:
        for t in s['taus']:
            tau_counts[min(t, 7)] += 1
    print(f"\n  tau distribution (all runs):")
    for i, c in enumerate(tau_counts):
        bar = '#' * (c // 5)
        print(f"    tau={i}: {c:>5} {bar}")

    # Go/no-go
    pct_zero = 100 * sum(1 for s in summaries if s['max_tau'] == 0) / len(summaries)
    print(f"\n  Draws with ALL tau=0: {pct_zero:.1f}%")

    if pct_zero > 90:
        print("\n  *** GO/NO-GO: FAIL — >90% of draws have tau=0 ***")
    else:
        print("\n  *** GO/NO-GO: PASS — sufficient tau variance ***")

    # Candidate baseline
    candidates = sorted(
        [s for s in summaries if 0.5 <= s['mean_tau'] <= 3.0],
        key=lambda s: abs(s['mean_tau'] - 1.5))
    if candidates:
        best = candidates[0]
        print(f"\n  Recommended Phase 1 baseline:")
        print(f"    param_idx={best['param_idx']}, "
              f"gamma={best['gamma']:.5f}, J={best['J']:.3f}, "
              f"k_cat={best['k_cat']:.3f}")
        print(f"    mean_tau={best['mean_tau']:.2f}, taus={best['taus']}")
    else:
        print("\n  No candidate with mean tau in [0.5, 3.0] found.")
        top5 = sorted(summaries, key=lambda s: s['mean_tau'], reverse=True)[:5]
        print("  Top 5 by mean_tau:")
        for s in top5:
            print(f"    gamma={s['gamma']:.5f}, J={s['J']:.3f}, "
                  f"k_cat={s['k_cat']:.3f}, mean_tau={s['mean_tau']:.2f}")

    # Top 20
    print(f"\n  Top 20 by mean_tau:")
    top20 = sorted(summaries, key=lambda s: s['mean_tau'], reverse=True)[:20]
    for s in top20:
        print(f"    gamma={s['gamma']:.5f}  J={s['J']:.2f}  "
              f"kcat={s['k_cat']:.3f}  mean_tau={s['mean_tau']:.2f}  "
              f"taus={s['taus']}")

    # Write consolidated JSON for Phase 1 to read
    consolidated_path = os.path.join(save_dir, 'phase0_results.json')
    output = {
        'phase': 'phase0_baseline',
        'n_samples': len(summaries),
        'n_seeds': N_SEEDS,
        'go': pct_zero <= 90,
        'pct_zero_tau': pct_zero,
        'param_summaries': summaries,
    }
    with open(consolidated_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Consolidated results saved to {consolidated_path}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Phase 0: Baseline Characterisation (batch mode)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start param index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='End param index exclusive (default: all 200)')
    parser.add_argument('--analyse', action='store_true',
                        help='Analyse existing results instead of running')
    args = parser.parse_args()

    save_dir = os.path.join(_this_dir, '..', 'data')

    if args.analyse:
        analyse(save_dir)
    else:
        run_batch(start=args.start, end=args.end, save_dir=save_dir)
