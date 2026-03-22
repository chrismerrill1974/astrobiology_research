"""
Paper 5 Phase 2: Drift-Buffering Test.

For each promoted topology, start from the best parameter set found in Phase 1,
apply controlled multiplicative drift at increasing magnitudes, and measure
survival probability S(sigma_d) = Pr[tau > 0 | sigma_d].

Single-process, batch-mode, incremental JSONL output, resumable.

Usage:
  python3 paper5_phase2_drift.py --topology T0 --results-dir results_phase2
  python3 paper5_phase2_drift.py --topology rw16 --results-dir results_phase2
  python3 paper5_phase2_drift.py --analyse --results-dir results_phase2
"""

import sys
import os
import json
import time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))

# Reuse Phase 1's evaluate_single (same integration + D2 protocol)
sys.path.insert(0, _this_dir)
from paper5_phase1_screen import evaluate_single


# ── Configuration ────────────────────────────────────────────────────

# Drift magnitudes (sigma_d): multiplicative noise in log-space
# theta' = theta * 10^N(0, sigma_d^2)
SIGMA_D_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

N_DRAWS_PER_SIGMA = 100   # parameter draws per drift magnitude
N_SEEDS = 2
SEEDS = [42, 179]

# Phase 2 candidates and their best Phase 1 parameters
# These are the centroids from which drift is applied.
PHASE2_CANDIDATES = {
    'T0': {
        'gamma': 0.00050432,
        'J': 7.1114,
        'k_cat': 0.4574,
        'motif_rates': None,
        'rewiring_seed': None,
        'best_tau': 4,
    },
    'T6': {
        'gamma': 0.0059866,
        'J': 3.3386,
        'k_cat': 0.0642,
        'motif_rates': [0.3828, 0.7504, 9.6278, 0.1879, 3.4234, 1.2274],
        'rewiring_seed': None,
        'best_tau': 3,
    },
    'rw16': {
        'gamma': 0.00060372,
        'J': 5.4204,
        'k_cat': 0.2732,
        'motif_rates': None,
        'rewiring_seed': 6552,
        'best_tau': 3,
    },
    'rw4': {
        'gamma': 0.00056628,
        'J': 9.3882,
        'k_cat': 0.7918,
        'motif_rates': None,
        'rewiring_seed': 5388,
        'best_tau': 4,
    },
}

# Parameter bounds (same as Phase 1)
GAMMA_BOUNDS = (0.0005, 0.01)
J_BOUNDS = (2.0, 12.0)
KCAT_BOUNDS = (0.05, 0.8)


# ── Drift application ───────────────────────────────────────────────

def apply_drift(center: dict, sigma_d: float, rng: np.random.RandomState) -> dict:
    """
    Apply multiplicative Gaussian drift to (gamma, J, k_cat).
    theta' = theta * 10^N(0, sigma_d^2), clamped to bounds.
    """
    if sigma_d == 0.0:
        return {
            'gamma': center['gamma'],
            'J': center['J'],
            'k_cat': center['k_cat'],
        }

    def drift_param(val, bounds):
        log_noise = rng.normal(0, sigma_d)
        new_val = val * (10 ** log_noise)
        return float(np.clip(new_val, bounds[0], bounds[1]))

    return {
        'gamma': drift_param(center['gamma'], GAMMA_BOUNDS),
        'J': drift_param(center['J'], J_BOUNDS),
        'k_cat': drift_param(center['k_cat'], KCAT_BOUNDS),
    }


# ── Main drift loop ─────────────────────────────────────────────────

def run_drift_test(
    topology_name: str,
    results_dir: str = 'results_phase2',
    verbose: bool = True,
):
    """
    Run drift-buffering test for one topology.
    Writes incremental JSONL output (resumable).
    """
    os.makedirs(results_dir, exist_ok=True)

    if topology_name not in PHASE2_CANDIDATES:
        print(f"ERROR: Unknown Phase 2 candidate '{topology_name}'. "
              f"Known: {sorted(PHASE2_CANDIDATES.keys())}")
        sys.exit(1)

    center = PHASE2_CANDIDATES[topology_name]
    jsonl_path = os.path.join(results_dir, f'drift_{topology_name}.jsonl')

    # Resume: count completed
    completed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (rec['sigma_d'], rec['draw'], rec['seed'])
                    completed.add(key)
                except (json.JSONDecodeError, KeyError):
                    pass
        if verbose and completed:
            print(f"Resuming {topology_name}: {len(completed)} evaluations already done")

    # sigma_d=0 is deterministic (1 draw), all others get N_DRAWS_PER_SIGMA
    n_nonzero = sum(1 for s in SIGMA_D_VALUES if s > 0)
    n_zero = sum(1 for s in SIGMA_D_VALUES if s == 0.0)
    total = (n_zero * 1 + n_nonzero * N_DRAWS_PER_SIGMA) * N_SEEDS
    done = len(completed)

    # Map topology name to the right evaluate_single args
    if topology_name.startswith('rw'):
        eval_topo_name = f'T1_{topology_name}'   # e.g. 'T1_rw16'
        eval_rewiring_seed = center['rewiring_seed']
    else:
        eval_topo_name = topology_name
        eval_rewiring_seed = None

    with open(jsonl_path, 'a') as out:
        for sigma_d in SIGMA_D_VALUES:
            # sigma_d=0 is deterministic — only 1 draw needed
            n_draws = 1 if sigma_d == 0.0 else N_DRAWS_PER_SIGMA
            for draw in range(n_draws):
                # Deterministic RNG per (sigma_d, draw)
                drift_rng = np.random.RandomState(
                    int(sigma_d * 10000) * 100003 + draw * 7 + 31
                )
                params = apply_drift(center, sigma_d, drift_rng)

                for seed in SEEDS:
                    key = (sigma_d, draw, seed)
                    if key in completed:
                        continue

                    result = evaluate_single(
                        eval_topo_name,
                        gamma=params['gamma'],
                        J=params['J'],
                        k_cat=params['k_cat'],
                        seed=seed,
                        motif_rates=center['motif_rates'],
                        n_rewirings_seed=eval_rewiring_seed,
                    )

                    record = {
                        'topology': topology_name,
                        'sigma_d': sigma_d,
                        'draw': draw,
                        'seed': seed,
                        'gamma': round(params['gamma'], 8),
                        'J': round(params['J'], 4),
                        'k_cat': round(params['k_cat'], 4),
                        'center_gamma': center['gamma'],
                        'center_J': center['J'],
                        'center_k_cat': center['k_cat'],
                        **result,
                    }
                    out.write(json.dumps(record) + '\n')
                    out.flush()

                    done += 1
                    if verbose and done % 50 == 0:
                        pct = 100 * done / total
                        print(f"  {topology_name}: {done}/{total} ({pct:.0f}%) "
                              f"sigma_d={sigma_d} tau={result['tau']} "
                              f"status={result['status']} {result['elapsed_s']:.1f}s")

    if verbose:
        print(f"  {topology_name}: COMPLETE ({done} evaluations)")


# ── Analysis ─────────────────────────────────────────────────────────

def analyse(results_dir: str = 'results_phase2'):
    """Analyse drift-buffering results."""
    print(f"\nAnalysing drift results in {results_dir}/\n")

    topology_data = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.startswith('drift_') or not fname.endswith('.jsonl'):
            continue
        topo = fname.replace('drift_', '').replace('.jsonl', '')
        records = []
        with open(os.path.join(results_dir, fname)) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        topology_data[topo] = records

    if not topology_data:
        print("No drift results found.")
        return

    # Header
    print(f"{'Topo':<8}", end='')
    for sd in SIGMA_D_VALUES:
        print(f"  σ={sd:<5}", end='')
    print("  σ_d*")
    print("-" * (8 + len(SIGMA_D_VALUES) * 9 + 6))

    summary = {}
    for topo in sorted(topology_data.keys()):
        records = topology_data[topo]
        curve = {}
        for sd in SIGMA_D_VALUES:
            sd_recs = [r for r in records if abs(r['sigma_d'] - sd) < 1e-6
                       and r['status'] == 'ok']
            if not sd_recs:
                curve[sd] = {'S': None, 'mean_tau': None, 'n': 0}
                continue
            taus = [r['tau'] for r in sd_recs]
            s = sum(1 for t in taus if t > 0) / len(taus)
            curve[sd] = {
                'S': round(s, 4),
                'mean_tau': round(float(np.mean(taus)), 4),
                'mean_tau_positive': round(float(np.mean([t for t in taus if t > 0])), 4) if any(t > 0 for t in taus) else 0,
                'n': len(sd_recs),
            }

        # Find critical drift sigma_d* where S < 0.5
        sigma_d_star = None
        for sd in SIGMA_D_VALUES:
            if curve[sd]['S'] is not None and curve[sd]['S'] < 0.5:
                sigma_d_star = sd
                break

        print(f"{topo:<8}", end='')
        for sd in SIGMA_D_VALUES:
            s = curve[sd]['S']
            if s is None:
                print(f"  {'N/A':<5}", end='')
            else:
                print(f"  {s:<5.3f}", end='')
        print(f"  {sigma_d_star if sigma_d_star is not None else '>0.5'}")

        summary[topo] = {
            'survival_curve': {str(sd): curve[sd] for sd in SIGMA_D_VALUES},
            'sigma_d_star': sigma_d_star,
        }

    # Save
    output_path = os.path.join(results_dir, 'phase2_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")

    # Check success criteria
    print("\n=== Phase 2 Success Criteria ===")
    if 'rw16' in summary and 'T0' in summary:
        rw16_star = summary['rw16']['sigma_d_star']
        t0_star = summary['T0']['sigma_d_star']
        if rw16_star is not None and t0_star is not None:
            print(f"rw16 σ_d* = {rw16_star}, T0 σ_d* = {t0_star}")
            if rw16_star > t0_star:
                print("  → rw16 is MORE robust to drift than T0")
            elif rw16_star == t0_star:
                print("  → rw16 and T0 collapse at same drift magnitude")
            else:
                print("  → rw16 is LESS robust to drift than T0")
        else:
            print(f"rw16 σ_d* = {rw16_star}, T0 σ_d* = {t0_star}")
            print("  → One or both survive all tested drift magnitudes")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Paper 5 Phase 2: Drift-Buffering Test')
    parser.add_argument('--topology', type=str,
                        help=f'Topology to test ({", ".join(sorted(PHASE2_CANDIDATES.keys()))})')
    parser.add_argument('--results-dir', type=str, default='results_phase2')
    parser.add_argument('--analyse', action='store_true')
    args = parser.parse_args()

    if args.analyse:
        analyse(args.results_dir)
        return

    if not args.topology:
        parser.error(f"--topology required (one of {sorted(PHASE2_CANDIDATES.keys())}), or use --analyse")

    print(f"Paper 5 Phase 2: Drift-buffering test for {args.topology}")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Center params: {PHASE2_CANDIDATES[args.topology]}")
    print(f"  Drift magnitudes: {SIGMA_D_VALUES}")
    print(f"  Draws per sigma: {N_DRAWS_PER_SIGMA}")
    print()

    run_drift_test(
        args.topology,
        results_dir=args.results_dir,
        verbose=True,
    )


if __name__ == '__main__':
    main()
