"""
Paper 5 Phase 1: Ridge-Width Screening.

For each topology, sample random parameter draws and measure tau_{>1.2}.
Compute ridge width R_w = fraction of draws with tau > 0.

Single-process, batch-mode, incremental JSONL output, resumable.

Usage:
  python3 paper5_phase1_screen.py --topology T0                       # run one topology
  python3 paper5_phase1_screen.py --topology T0 --results-dir results_phase1
  python3 paper5_phase1_screen.py --topology T1_random --n-rewirings 20  # extended random ensemble
  python3 paper5_phase1_screen.py --analyse --results-dir results_phase1  # analyse all
"""

import sys
import os
import json
import time
import signal
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_research_dir = os.path.join(_this_dir, '..', 'astrobiology_research')
_shared_path = os.path.join(_research_dir, 'shared')
_paper3_scripts = os.path.join(_research_dir, 'paper3', 'scripts')

for p in [_shared_path, _paper3_scripts]:
    if os.path.abspath(p) not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

from pilot5b_enzyme_complex import EnzymeComplexParams
from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension

# Import topology library
sys.path.insert(0, _this_dir)
from paper5_topology_library import (
    TOPOLOGY_BUILDERS, TOPOLOGY_GROUPS, TOPOLOGY_DESCRIPTIONS,
    get_d2_species, make_t1_random_control,
    BASELINE_GAMMA, BASELINE_J, BASELINE_KCAT,
)


# ── Configuration ────────────────────────────────────────────────────

N_PARAM_DRAWS = 300         # random parameter draws per motif-rate instantiation
N_MOTIF_RATE_DRAWS = 3      # independent motif-rate draws per topology
N_SEEDS = 2                 # seeds per parameter draw
SEEDS = [42, 179]           # matching Paper 4

# Parameter ranges (matching Paper 4 Phase 0)
GAMMA_RANGE = (0.0005, 0.01)   # log-uniform
J_RANGE = (2.0, 12.0)          # uniform
KCAT_RANGE = (0.05, 0.8)       # uniform

# Motif rate range (new for Paper 5)
MOTIF_RATE_RANGE = (0.1, 10.0)  # log-uniform

# Integration parameters (matching Paper 4)
T_END = 20000
N_POINTS = 40000
WINDOW_SIZE = 2500          # time units per D2 window
D2_THRESHOLD = 1.2
MAX_STEP = 10.0

# Numerical safeguards
MAX_CONCENTRATION = 1e6
EVAL_TIMEOUT_SECONDS = 120  # 2 min (increased from Paper 4's 60s for larger networks)


# ── Timeout mechanism ────────────────────────────────────────────────

class EvalTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise EvalTimeout(f"Integration exceeded {EVAL_TIMEOUT_SECONDS}s")

# Only set signal handler on Unix (not on Windows, not in threads)
_HAS_SIGALRM = hasattr(signal, 'SIGALRM')


# ── Core evaluation ──────────────────────────────────────────────────

def evaluate_single(
    topology_name: str,
    gamma: float, J: float, k_cat: float,
    seed: int,
    motif_rates: list = None,
    motif_rng_seed: int = 0,
    n_rewirings_seed: int = None,
) -> dict:
    """
    Run one integration for a given topology + parameters.
    Returns dict with tau, d2_windows, status, elapsed_s.
    """
    t0 = time.time()

    p = EnzymeComplexParams(
        A=1.0, B=3.0,
        k_on=10.0, k_off=10.0,
        k_cat=k_cat, G_total=1.0,
        J=J, gamma=gamma,
        label=f"screen_{topology_name}",
    )

    # Build the network for this topology
    try:
        if n_rewirings_seed is not None:
            # Extended random ensemble: use specific rewiring seed
            net = make_t1_random_control(p, seed=seed, rng_seed=n_rewirings_seed)
        elif topology_name in TOPOLOGY_BUILDERS:
            builder = TOPOLOGY_BUILDERS[topology_name]
            # Motif builders accept motif_rates for Group B
            if topology_name.startswith('T') and not topology_name.startswith('T0') and not topology_name.startswith('T1'):
                net = builder(p, seed=seed, motif_rates=motif_rates)
            else:
                net = builder(p, seed=seed)
        else:
            return {'tau': 0, 'status': f'unknown_topology:{topology_name}',
                    'elapsed_s': time.time() - t0}
    except Exception as e:
        return {'tau': 0, 'status': f'build_fail:{str(e)[:60]}',
                'elapsed_s': time.time() - t0}

    d2_species = get_d2_species(topology_name, net)

    # Set timeout
    if _HAS_SIGALRM:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EVAL_TIMEOUT_SECONDS)

    status = 'ok'
    tau = 0
    d2_windows = []

    try:
        sim = ReactionSimulator()
        network_obj = sim.build_network(net.reactions)
        sim_result = sim.simulate(
            network_obj,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=(0, T_END),
            n_points=N_POINTS,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=net.chemostat_species,
            max_step=MAX_STEP,
        )

        if not sim_result.success:
            status = 'solver_fail'
        else:
            c = sim_result.concentrations
            species_names = sim_result.species_names

            # NaN/Inf guard
            if not np.all(np.isfinite(c)):
                status = 'nan_inf'
            elif np.any(c > MAX_CONCENTRATION):
                status = 'pathological'
            else:
                # Compute D2 in sliding windows (Paper 4 protocol)
                sp_indices = [species_names.index(sp) for sp in d2_species
                              if sp in species_names]

                pts_per_unit = N_POINTS / T_END  # 2 pts/unit
                t_start_idx = int(2500 * pts_per_unit)   # skip first 2500 time units
                window_pts = int(WINDOW_SIZE * pts_per_unit)
                n_windows = 7

                cd = CorrelationDimension()
                for w in range(n_windows):
                    w_start = t_start_idx + w * window_pts
                    w_end = w_start + window_pts
                    if w_end > len(c):
                        break
                    traj = c[w_start:w_end, :][:, sp_indices]

                    try:
                        d2_result = cd.compute(traj)
                        d2_val = float(d2_result.D2) if d2_result.D2 is not None else 0.0
                    except Exception:
                        d2_val = 0.0

                    d2_windows.append(round(d2_val, 4))
                    if d2_val > D2_THRESHOLD:
                        tau += 1

    except EvalTimeout:
        status = 'timeout'
    except Exception as e:
        status = f'exception:{str(e)[:60]}'
    finally:
        if _HAS_SIGALRM:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.time() - t0
    return {
        'tau': tau,
        'd2_windows': d2_windows,
        'status': status,
        'elapsed_s': round(elapsed, 1),
    }


# ── Main screening loop ─────────────────────────────────────────────

def run_topology_screen(
    topology_name: str,
    results_dir: str = 'results_phase1',
    n_param_draws: int = N_PARAM_DRAWS,
    n_motif_rate_draws: int = N_MOTIF_RATE_DRAWS,
    n_rewirings: int = 0,
    rewiring_offset: int = 0,
    verbose: bool = True,
):
    """
    Screen one topology: sample random parameter draws, measure tau.
    Writes incremental JSONL output (resumable).

    For the extended random ensemble (n_rewirings > 0), runs multiple
    independent random rewirings with fewer draws each.
    """
    os.makedirs(results_dir, exist_ok=True)

    if n_rewirings > 0:
        # Extended random ensemble mode
        _run_random_ensemble(topology_name, results_dir, n_param_draws,
                             n_rewirings, verbose,
                             rewiring_offset=rewiring_offset)
        return

    jsonl_path = os.path.join(results_dir, f'screen_{topology_name}.jsonl')

    # Resume: count completed draws
    completed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (rec['motif_rate_draw'], rec['param_draw'], rec['seed'])
                    completed.add(key)
                except (json.JSONDecodeError, KeyError):
                    pass
        if verbose:
            print(f"Resuming {topology_name}: {len(completed)} evaluations already done")

    total = n_param_draws * n_motif_rate_draws * N_SEEDS
    done = len(completed)

    # Determine number of motif rates needed
    is_group_b = topology_name in ('T2', 'T3', 'T4', 'T5', 'T6')

    with open(jsonl_path, 'a') as out:
        for mrd in range(n_motif_rate_draws):
            # Sample motif rates for this draw
            motif_rng = np.random.RandomState(mrd * 7919 + 42)
            if is_group_b:
                # Number of motif reactions depends on topology
                # T2: 4, T3: 6, T4: 6, T5: 5, T6: 6
                n_motif_rxns = {'T2': 4, 'T3': 6, 'T4': 6, 'T5': 5, 'T6': 6}
                n_rates = n_motif_rxns.get(topology_name, 6)
                motif_rates = list(10 ** motif_rng.uniform(
                    np.log10(MOTIF_RATE_RANGE[0]),
                    np.log10(MOTIF_RATE_RANGE[1]),
                    size=n_rates
                ))
            else:
                motif_rates = None

            # Sample parameter draws
            param_rng = np.random.RandomState(mrd * 10007 + 1)
            gammas = 10 ** param_rng.uniform(
                np.log10(GAMMA_RANGE[0]), np.log10(GAMMA_RANGE[1]),
                size=n_param_draws)
            Js = param_rng.uniform(J_RANGE[0], J_RANGE[1], size=n_param_draws)
            kcats = param_rng.uniform(KCAT_RANGE[0], KCAT_RANGE[1], size=n_param_draws)

            for pd in range(n_param_draws):
                for seed in SEEDS:
                    key = (mrd, pd, seed)
                    if key in completed:
                        continue

                    result = evaluate_single(
                        topology_name,
                        gamma=float(gammas[pd]),
                        J=float(Js[pd]),
                        k_cat=float(kcats[pd]),
                        seed=seed,
                        motif_rates=motif_rates,
                    )

                    record = {
                        'topology': topology_name,
                        'motif_rate_draw': mrd,
                        'param_draw': pd,
                        'seed': seed,
                        'gamma': round(float(gammas[pd]), 8),
                        'J': round(float(Js[pd]), 4),
                        'k_cat': round(float(kcats[pd]), 4),
                        'motif_rates': [round(r, 4) for r in motif_rates] if motif_rates else None,
                        **result,
                    }
                    out.write(json.dumps(record) + '\n')
                    out.flush()

                    done += 1
                    if verbose and done % 50 == 0:
                        pct = 100 * done / total
                        print(f"  {topology_name}: {done}/{total} ({pct:.0f}%) "
                              f"tau={result['tau']} status={result['status']} "
                              f"{result['elapsed_s']:.1f}s")

    if verbose:
        print(f"  {topology_name}: COMPLETE ({done} evaluations)")


def _run_random_ensemble(
    topology_name: str,
    results_dir: str,
    n_param_draws: int,
    n_rewirings: int,
    verbose: bool,
    rewiring_offset: int = 0,
):
    """Run extended random-wiring ensemble: multiple rewirings, fewer draws each."""
    # 100 draws × 2 seeds = 200 evals per rewiring (matches EXPECTED_EVALS_PER_REWIRING)
    draws_per_rewiring = 100

    for rw in range(n_rewirings):
        rw_idx = rewiring_offset + rw
        rw_seed = 5000 + rw_idx * 97
        jsonl_path = os.path.join(results_dir, f'screen_T1_rw{rw_idx}.jsonl')

        # Resume
        completed = set()
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        key = (rec['param_draw'], rec['seed'])
                        completed.add(key)
                    except (json.JSONDecodeError, KeyError):
                        pass

        param_rng = np.random.RandomState(rw_idx * 10007 + 1)
        gammas = 10 ** param_rng.uniform(
            np.log10(GAMMA_RANGE[0]), np.log10(GAMMA_RANGE[1]),
            size=draws_per_rewiring)
        Js = param_rng.uniform(J_RANGE[0], J_RANGE[1], size=draws_per_rewiring)
        kcats = param_rng.uniform(KCAT_RANGE[0], KCAT_RANGE[1], size=draws_per_rewiring)

        done = len(completed)
        total = draws_per_rewiring * N_SEEDS

        with open(jsonl_path, 'a') as out:
            for pd in range(draws_per_rewiring):
                for seed in SEEDS:
                    key = (pd, seed)
                    if key in completed:
                        continue

                    result = evaluate_single(
                        f'T1_rw{rw_idx}',
                        gamma=float(gammas[pd]),
                        J=float(Js[pd]),
                        k_cat=float(kcats[pd]),
                        seed=seed,
                        n_rewirings_seed=rw_seed,
                    )

                    record = {
                        'topology': f'T1_rw{rw_idx}',
                        'rewiring_seed': rw_seed,
                        'param_draw': pd,
                        'seed': seed,
                        'gamma': round(float(gammas[pd]), 8),
                        'J': round(float(Js[pd]), 4),
                        'k_cat': round(float(kcats[pd]), 4),
                        **result,
                    }
                    out.write(json.dumps(record) + '\n')
                    out.flush()
                    done += 1

                    if verbose and done % 50 == 0:
                        print(f"  T1_rw{rw_idx}: {done}/{total} ({100*done/total:.0f}%)")

        if verbose:
            print(f"  T1_rw{rw_idx} (seed={rw_seed}): COMPLETE ({done} evaluations)")


# ── Analysis ─────────────────────────────────────────────────────────

def analyse(results_dir: str = 'results_phase1'):
    """Analyse all completed topology screens."""
    print(f"\nAnalysing results in {results_dir}/\n")

    # Load all screen files
    topology_data = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.startswith('screen_') or not fname.endswith('.jsonl'):
            continue

        records = []
        with open(os.path.join(results_dir, fname)) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        topo = fname.replace('screen_', '').replace('.jsonl', '')
        topology_data[topo] = records

    if not topology_data:
        print("No screen results found.")
        return

    # Compute ridge width per topology
    print(f"{'Topology':<12} {'N draws':>8} {'N ok':>6} {'R_w':>8} "
          f"{'R_w(2)':>8} {'Mean τ':>8} {'Mean τ|>0':>10} "
          f"{'Timeout%':>9}")
    print("-" * 90)

    summary = {}
    for topo in sorted(topology_data.keys()):
        records = topology_data[topo]
        n_total = len(records)
        n_ok = sum(1 for r in records if r['status'] == 'ok')
        taus = [r['tau'] for r in records if r['status'] == 'ok']
        n_timeout = sum(1 for r in records if r['status'] == 'timeout')

        if not taus:
            print(f"{topo:<12} {n_total:>8} {n_ok:>6} {'N/A':>8} "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>10} "
                  f"{100*n_timeout/max(n_total,1):>8.1f}%")
            continue

        r_w = sum(1 for t in taus if t > 0) / len(taus)
        r_w2 = sum(1 for t in taus if t > 2) / len(taus)
        mean_tau = np.mean(taus)
        positive_taus = [t for t in taus if t > 0]
        mean_tau_pos = np.mean(positive_taus) if positive_taus else 0

        print(f"{topo:<12} {n_total:>8} {n_ok:>6} {r_w:>8.4f} "
              f"{r_w2:>8.4f} {mean_tau:>8.3f} {mean_tau_pos:>10.3f} "
              f"{100*n_timeout/max(n_total,1):>8.1f}%")

        summary[topo] = {
            'n_total': n_total,
            'n_ok': n_ok,
            'r_w': round(r_w, 6),
            'r_w2': round(r_w2, 6),
            'mean_tau': round(mean_tau, 4),
            'mean_tau_positive': round(mean_tau_pos, 4),
            'timeout_pct': round(100 * n_timeout / max(n_total, 1), 2),
        }

    print("-" * 90)

    # Compare Group B vs random controls
    group_b = {k: v for k, v in summary.items()
               if k in ('T2', 'T3', 'T4', 'T5', 'T6')}
    controls = {k: v for k, v in summary.items()
                if k.startswith('T1')}

    if group_b and controls:
        control_rws = [v['r_w'] for v in controls.values()]
        mean_control_rw = np.mean(control_rws)

        print(f"\nMean control R_w: {mean_control_rw:.4f}")
        print("\nGroup B vs controls:")
        for topo, data in sorted(group_b.items()):
            ratio = data['r_w'] / mean_control_rw if mean_control_rw > 0 else float('inf')
            verdict = "PASS (>2×)" if ratio > 2.0 else "FAIL (<2×)" if ratio < 1.5 else "MARGINAL"
            print(f"  {topo}: R_w={data['r_w']:.4f}, ratio={ratio:.2f}× control → {verdict}")

    # Save summary
    output_path = os.path.join(results_dir, 'phase1_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Paper 5 Phase 1: Ridge-Width Screen')
    parser.add_argument('--topology', type=str,
                        help='Topology to screen (T0, T1a, T1b, T2-T6, or T1_random)')
    parser.add_argument('--results-dir', type=str, default='results_phase1')
    parser.add_argument('--n-draws', type=int, default=N_PARAM_DRAWS)
    parser.add_argument('--n-motif-draws', type=int, default=N_MOTIF_RATE_DRAWS)
    parser.add_argument('--n-rewirings', type=int, default=0,
                        help='Extended random ensemble: number of independent rewirings')
    parser.add_argument('--rewiring-offset', type=int, default=0,
                        help='Starting rewiring index (so node 9 can do rw10-19)')
    parser.add_argument('--analyse', action='store_true')
    args = parser.parse_args()

    if args.analyse:
        analyse(args.results_dir)
        return

    if not args.topology:
        parser.error("--topology required (or use --analyse)")

    # Fail fast: T1_random MUST be run with --n-rewirings
    if args.topology == 'T1_random' and args.n_rewirings <= 0:
        parser.error("T1_random requires --n-rewirings N (e.g. --n-rewirings 10). "
                      "Without it, the script falls through to the regular path "
                      "which cannot build topology 'T1_random'.")

    # Fail fast: named topology must exist in TOPOLOGY_BUILDERS
    if args.topology != 'T1_random' and args.topology not in TOPOLOGY_BUILDERS:
        parser.error(f"Unknown topology '{args.topology}'. "
                      f"Known: {sorted(TOPOLOGY_BUILDERS.keys())} or T1_random with --n-rewirings")

    print(f"Paper 5 Phase 1: Screening topology {args.topology}")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Param draws: {args.n_draws}")
    print(f"  Motif-rate draws: {args.n_motif_draws}")
    if args.n_rewirings > 0:
        print(f"  Extended random ensemble: {args.n_rewirings} rewirings "
              f"(offset={args.rewiring_offset}, rw{args.rewiring_offset}-rw{args.rewiring_offset + args.n_rewirings - 1})")
    print()

    run_topology_screen(
        args.topology,
        results_dir=args.results_dir,
        n_param_draws=args.n_draws,
        n_motif_rate_draws=args.n_motif_draws,
        n_rewirings=args.n_rewirings,
        rewiring_offset=args.rewiring_offset,
        verbose=True,
    )


if __name__ == '__main__':
    main()
