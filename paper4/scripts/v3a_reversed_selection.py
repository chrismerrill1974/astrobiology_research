"""
V3a: Reversed Selection — Minimize tau_{>1.2} (gamma only).

Tests bidirectionality: if selection can push tau DOWN as well as up,
it confirms the evolutionary algorithm genuinely acts on the fitness
landscape rather than exploiting an artifact or regression to the mean.

Identical to phase1_evolution_v2.py except:
  - Tournament selection picks the LEAST fit individual (minimize tau)
  - File naming uses 'v3a' prefix
  - Same baseline, same params (N=20, G=40, 2 seeds, 10+10 replicates)

Expected outcome: gamma drifts UPWARD (faster energy leak -> shorter
transient -> lower tau). If V2 found gamma converged to ~0.0004,
reversed selection should push gamma toward 0.01+ or GAMMA_MAX.

Usage:
  python3 v3a_reversed_selection.py --replicate 0              # reversed selection rep 0
  python3 v3a_reversed_selection.py --replicate 0 --neutral    # neutral rep 0
  python3 v3a_reversed_selection.py --analyse                  # analyse all results
  python3 v3a_reversed_selection.py --analyse --results-dir ../data/v3a_reversed
"""

import sys
import os
import json
import time
import signal
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


# ── Configuration ────────────────────────────────────────────────────

POP_SIZE = 20
G_MAX = 40
N_SELECTION_REPS = 10
N_NEUTRAL_REPS = 10

# Mutation
SIGMA_M = 0.1  # std in log10-space
GAMMA_MIN = 1e-4
GAMMA_MAX = 0.1

# Fitness evaluation
N_FITNESS_SEEDS = 2
FITNESS_SEEDS = [42, 179]

# Selection
TOURNAMENT_K = 3

# Integration
T_END = 20000
N_POINTS = 40000
WINDOW_SIZE = 2500
D2_THRESHOLD = 1.2

# Numerical safeguards
MAX_CONCENTRATION = 1e6
EVAL_TIMEOUT_SECONDS = 60
MAX_STEP = 10.0

# V3a-specific
FILE_PREFIX = 'v3a'  # output files: v3a_rep0_reversed.jsonl, etc.


# ── Timeout mechanism ────────────────────────────────────────────────

class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout(f"Integration exceeded {EVAL_TIMEOUT_SECONDS}s wall-clock limit")


# ── Fitness evaluation (identical to V2) ─────────────────────────────

def simulate_and_get_tau(gamma: float, J: float, k_cat: float,
                         seed: int) -> dict:
    """Run one integration and return tau_{>1.2} plus diagnostics."""
    t0 = time.time()

    p = EnzymeComplexParams(
        A=1.0, B=3.0,
        k_on=10.0, k_off=10.0,
        k_cat=k_cat, G_total=1.0,
        J=J, gamma=gamma,
        label=f"v3a_g{gamma:.6f}_s{seed}",
    )

    net = make_enzyme_complex_network(p, seed=seed)
    sim = ReactionSimulator()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(EVAL_TIMEOUT_SECONDS)

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
            max_step=MAX_STEP,
        )
    except EvalTimeout:
        return {'tau': 0, 'status': 'timeout', 'elapsed_s': time.time() - t0}
    except Exception as e:
        return {'tau': 0, 'status': 'exception', 'elapsed_s': time.time() - t0,
                'error': str(e)[:200]}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not result.success:
        return {'tau': 0, 'status': 'solver_fail', 'elapsed_s': time.time() - t0,
                'message': getattr(result, 'message', '')[:200]}

    t = result.time
    c = result.concentrations
    species = result.species_names

    if not np.isfinite(c).all():
        return {'tau': 0, 'status': 'nan_inf', 'elapsed_s': time.time() - t0}

    def idx(name):
        return species.index(name)

    for sp_name in ['X1', 'Y1', 'X2', 'Y2', 'E']:
        vals = c[:, idx(sp_name)]
        if np.any(vals > MAX_CONCENTRATION) or np.any(vals < 0):
            return {'tau': 0, 'status': 'pathological',
                    'elapsed_s': time.time() - t0}

    tau = 0
    for win_idx in range(1, 8):
        t_start = win_idx * WINDOW_SIZE
        t_end_w = (win_idx + 1) * WINDOW_SIZE
        mask = (t >= t_start) & (t < t_end_w)
        n_pts = np.sum(mask)

        if n_pts < 500:
            continue

        trajectory = np.column_stack([
            c[:, idx('X1')][mask], c[:, idx('Y1')][mask],
            c[:, idx('X2')][mask], c[:, idx('Y2')][mask],
            c[:, idx('E')][mask],
        ])

        if not np.isfinite(trajectory).all():
            continue

        cd = CorrelationDimension()
        try:
            d2_result = cd.compute(trajectory)
            d2 = float(d2_result.D2) if d2_result.D2 is not None else 0.0
        except Exception:
            d2 = 0.0

        if d2 > D2_THRESHOLD:
            tau += 1

    return {'tau': tau, 'status': 'ok', 'elapsed_s': time.time() - t0}


def evaluate_individual(gamma: float, J: float, k_cat: float) -> dict:
    """Evaluate one individual across all fitness seeds. SEQUENTIAL."""
    taus = []
    statuses = []
    total_elapsed = 0.0
    for seed in FITNESS_SEEDS:
        result = simulate_and_get_tau(gamma, J, k_cat, seed)
        taus.append(result['tau'])
        statuses.append(result['status'])
        total_elapsed += result.get('elapsed_s', 0)
    fitness = float(np.mean(taus))
    return {
        'gamma': float(gamma),
        'fitness': fitness,
        'taus': taus,
        'statuses': statuses,
        'eval_seconds': round(total_elapsed, 1),
    }


# ── Mutation (identical to V2) ───────────────────────────────────────

def mutate_gamma(gamma: float, rng: np.random.RandomState) -> float:
    """Gaussian mutation in log10-space with reflection at bounds."""
    log_gamma = np.log10(gamma)
    log_gamma_new = log_gamma + rng.normal(0, SIGMA_M)

    log_min = np.log10(GAMMA_MIN)
    log_max = np.log10(GAMMA_MAX)

    if log_gamma_new < log_min:
        log_gamma_new = log_min + abs(log_gamma_new - log_min)
    elif log_gamma_new > log_max:
        log_gamma_new = log_max - abs(log_gamma_new - log_max)

    log_gamma_new = np.clip(log_gamma_new, log_min, log_max)
    return float(10 ** log_gamma_new)


# ── Selection ────────────────────────────────────────────────────────

def tournament_select_reversed(individuals: list, k: int,
                               rng: np.random.RandomState) -> dict:
    """
    REVERSED tournament selection: pick k random individuals,
    return the LEAST fit (minimum fitness).
    This is the key V3a change — selects for minimum tau.
    """
    contestants = rng.choice(len(individuals), size=k, replace=False)
    worst = min(contestants, key=lambda i: individuals[i]['fitness'])
    return individuals[worst]


def tournament_select_normal(individuals: list, k: int,
                             rng: np.random.RandomState) -> dict:
    """Normal tournament selection (for RNG compatibility with neutral)."""
    contestants = rng.choice(len(individuals), size=k, replace=False)
    best = max(contestants, key=lambda i: individuals[i]['fitness'])
    return individuals[best]


# ── Single replicate (batch, incremental, resumable) ─────────────────

def run_replicate(replicate_id: int, rep_type: str,
                  gamma_0: float, J: float, k_cat: float,
                  save_dir: str):
    """
    Run one replicate generation-by-generation.
    rep_type: 'reversed' or 'neutral'
    """
    jsonl_path = os.path.join(save_dir, f'{FILE_PREFIX}_rep{replicate_id}_{rep_type}.jsonl')

    # Check what generations are already done
    done_gens = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_gens[rec['generation']] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Found {len(done_gens)} completed generations in {jsonl_path}")

    last_done = max(done_gens.keys()) if done_gens else -1

    if last_done >= G_MAX:
        print(f"  Replicate {replicate_id} ({rep_type}) already complete ({last_done + 1} gens)")
        return

    # Use different RNG seed range from V2 to avoid any correlation
    # V2 used replicate_id * 1000 + 7; V3a uses replicate_id * 1000 + 31
    rng = np.random.RandomState(replicate_id * 1000 + 31)

    # Reconstruct population state from saved generations
    if last_done >= 0:
        last_rec = done_gens[last_done]
        gammas = np.array(last_rec['gammas'])

        for g in range(last_done):
            for _ in range(POP_SIZE):
                rng.choice(POP_SIZE, size=TOURNAMENT_K, replace=False)
                if rep_type == 'neutral':
                    rng.randint(0, POP_SIZE)
                rng.normal(0, SIGMA_M)

        if last_done < G_MAX:
            stored_individuals = []
            for i in range(POP_SIZE):
                stored_individuals.append({
                    'gamma': last_rec['gammas'][i],
                    'fitness': last_rec['fitnesses'][i],
                    'taus': last_rec['individual_taus'][i],
                })
            new_gammas = np.zeros(POP_SIZE)
            for i in range(POP_SIZE):
                if rep_type == 'reversed':
                    parent = tournament_select_reversed(stored_individuals, TOURNAMENT_K, rng)
                else:
                    rng.choice(POP_SIZE, size=TOURNAMENT_K, replace=False)
                    parent = stored_individuals[rng.randint(0, POP_SIZE)]
                new_gammas[i] = mutate_gamma(parent['gamma'], rng)
            gammas = new_gammas

        print(f"  Resuming from generation {last_done + 1}")
    else:
        gammas = np.full(POP_SIZE, gamma_0)

    start_gen = last_done + 1
    batch_start = time.time()

    print(f"\n  {'=' * 60}")
    print(f"  V3a {rep_type.upper()} REPLICATE {replicate_id}")
    print(f"  gamma_0={gamma_0:.5f}, J={J:.3f}, k_cat={k_cat:.3f}")
    print(f"  Generations {start_gen} to {G_MAX}")
    if rep_type == 'reversed':
        print(f"  *** REVERSED SELECTION: minimizing tau ***")
    print(f"  Safeguards: max_step={MAX_STEP}, timeout={EVAL_TIMEOUT_SECONDS}s, "
          f"NaN/Inf check=ON")
    print(f"  {'=' * 60}\n")

    total_timeouts = 0
    total_nan_inf = 0
    total_solver_fail = 0
    total_pathological = 0

    for gen in range(start_gen, G_MAX + 1):
        gen_start = time.time()

        individuals = []
        gen_timeouts = 0
        gen_nan_inf = 0
        gen_solver_fail = 0
        gen_pathological = 0

        for i in range(POP_SIZE):
            ind = evaluate_individual(float(gammas[i]), J, k_cat)
            individuals.append(ind)

            for st in ind.get('statuses', []):
                if st == 'timeout':
                    gen_timeouts += 1
                elif st == 'nan_inf':
                    gen_nan_inf += 1
                elif st == 'solver_fail':
                    gen_solver_fail += 1
                elif st == 'pathological':
                    gen_pathological += 1

        total_timeouts += gen_timeouts
        total_nan_inf += gen_nan_inf
        total_solver_fail += gen_solver_fail
        total_pathological += gen_pathological

        fitnesses = [ind['fitness'] for ind in individuals]
        gamma_vals = [ind['gamma'] for ind in individuals]
        n_lethal = sum(1 for ind in individuals
                       if all(t == 0 for t in ind['taus']))

        gen_record = {
            'generation': gen,
            'replicate_id': replicate_id,
            'type': rep_type,
            'experiment': 'v3a_reversed',
            'mean_fitness': float(np.mean(fitnesses)),
            'median_fitness': float(np.median(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'mean_gamma': float(np.mean(gamma_vals)),
            'std_gamma': float(np.std(gamma_vals)),
            'min_gamma': float(np.min(gamma_vals)),
            'max_gamma': float(np.max(gamma_vals)),
            'n_lethal': n_lethal,
            'gammas': [float(g) for g in gammas],
            'fitnesses': fitnesses,
            'individual_taus': [ind['taus'] for ind in individuals],
            'diagnostics': {
                'timeouts': gen_timeouts,
                'nan_inf': gen_nan_inf,
                'solver_fail': gen_solver_fail,
                'pathological': gen_pathological,
                'eval_seconds': [ind.get('eval_seconds', 0) for ind in individuals],
            },
        }

        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(gen_record) + '\n')

        gen_elapsed = time.time() - gen_start
        total_elapsed = time.time() - batch_start
        gens_done = gen - start_gen + 1
        gens_remaining = G_MAX - gen
        rate = gens_done / total_elapsed if total_elapsed > 0 else 0
        eta = gens_remaining / rate if rate > 0 else 0

        diag_parts = []
        if gen_timeouts:
            diag_parts.append(f"TO={gen_timeouts}")
        if gen_nan_inf:
            diag_parts.append(f"NaN={gen_nan_inf}")
        if gen_solver_fail:
            diag_parts.append(f"SF={gen_solver_fail}")
        if gen_pathological:
            diag_parts.append(f"PATH={gen_pathological}")
        diag_str = f" | {'  '.join(diag_parts)}" if diag_parts else ""

        print(f"  [{rep_type[0].upper()}{replicate_id}] Gen {gen:3d}/{G_MAX} | "
              f"mean_tau={gen_record['mean_fitness']:.2f} "
              f"min_tau={gen_record['min_fitness']:.1f} | "
              f"mean_gamma={gen_record['mean_gamma']:.5f} "
              f"std={gen_record['std_gamma']:.5f} | "
              f"lethal={n_lethal} | "
              f"{gen_elapsed:.0f}s | ETA {eta/60:.0f}min{diag_str}", flush=True)

        # Selection + reproduction (skip on last generation)
        if gen < G_MAX:
            new_gammas = np.zeros(POP_SIZE)
            for i in range(POP_SIZE):
                if rep_type == 'reversed':
                    parent = tournament_select_reversed(individuals, TOURNAMENT_K, rng)
                else:
                    # Neutral: random parent (no fitness pressure)
                    rng.choice(POP_SIZE, size=TOURNAMENT_K,
                               replace=False)  # consume same RNG
                    parent = individuals[rng.randint(0, POP_SIZE)]
                new_gammas[i] = mutate_gamma(parent['gamma'], rng)
            gammas = new_gammas

    total_time = time.time() - batch_start
    print(f"\n  Replicate {replicate_id} ({rep_type}) done in "
          f"{total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Cumulative diagnostics: timeouts={total_timeouts}, "
          f"nan_inf={total_nan_inf}, solver_fail={total_solver_fail}, "
          f"pathological={total_pathological}")


# ── Analysis ─────────────────────────────────────────────────────────

def analyse(save_dir: str):
    """Analyse all completed V3a results."""
    from scipy import stats

    reversed_reps = []
    neutral_reps = []

    for rep_id in range(N_SELECTION_REPS):
        path = os.path.join(save_dir, f'{FILE_PREFIX}_rep{rep_id}_reversed.jsonl')
        if os.path.exists(path):
            gens = []
            with open(path) as f:
                for line in f:
                    try:
                        gens.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if gens:
                reversed_reps.append({
                    'replicate_id': rep_id,
                    'generations': sorted(gens, key=lambda g: g['generation']),
                })

    for rep_id in range(N_NEUTRAL_REPS):
        path = os.path.join(save_dir, f'{FILE_PREFIX}_rep{rep_id}_neutral.jsonl')
        if os.path.exists(path):
            gens = []
            with open(path) as f:
                for line in f:
                    try:
                        gens.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if gens:
                neutral_reps.append({
                    'replicate_id': rep_id,
                    'generations': sorted(gens, key=lambda g: g['generation']),
                })

    print(f"\n{'=' * 70}")
    print(f"V3a REVERSED SELECTION ANALYSIS (n={N_SELECTION_REPS})")
    print(f"{'=' * 70}")
    print(f"\n  Reversed replicates found: {len(reversed_reps)}")
    print(f"  Neutral replicates found: {len(neutral_reps)}")

    if not reversed_reps:
        print("  No reversed selection results to analyse.")
        return

    print(f"\n  {'─' * 50}")
    print(f"  PER-REPLICATE SUMMARY")
    print(f"  {'─' * 50}")

    gen0_fitness = []
    genmax_fitness = []
    final_gammas = []

    for rep in reversed_reps:
        gens = rep['generations']
        g0 = gens[0]
        gmax = gens[-1]
        gen0_fitness.append(g0['mean_fitness'])
        genmax_fitness.append(gmax['mean_fitness'])
        final_gammas.append(gmax['mean_gamma'])

        print(f"\n  Reversed rep {rep['replicate_id']}:")
        print(f"    Gen 0:  mean_tau={g0['mean_fitness']:.3f}, "
              f"mean_gamma={g0['mean_gamma']:.5f}")
        print(f"    Gen {gmax['generation']}: mean_tau={gmax['mean_fitness']:.3f}, "
              f"mean_gamma={gmax['mean_gamma']:.5f}")
        print(f"    Delta tau: {gmax['mean_fitness'] - g0['mean_fitness']:+.3f}")
        print(f"    Delta gamma: {gmax['mean_gamma'] - g0['mean_gamma']:+.5f}")

    gen0_fitness = np.array(gen0_fitness)
    genmax_fitness = np.array(genmax_fitness)
    delta = genmax_fitness - gen0_fitness

    print(f"\n  {'─' * 50}")
    print(f"  AGGREGATE")
    print(f"  {'─' * 50}")
    print(f"  Gen 0 mean tau: {gen0_fitness.mean():.3f} +/- {gen0_fitness.std():.3f}")
    print(f"  Gen {G_MAX} mean tau: {genmax_fitness.mean():.3f} +/- {genmax_fitness.std():.3f}")
    print(f"  Delta: {delta.mean():.3f} +/- {delta.std():.3f}")
    print(f"  Final mean gamma: {np.mean(final_gammas):.5f} +/- {np.std(final_gammas):.5f}")

    # Wilcoxon: test if tau DECREASED (gen 40 < gen 0)
    if len(gen0_fitness) >= 3 and not np.all(delta == 0):
        stat, p_val = stats.wilcoxon(gen0_fitness, genmax_fitness,
                                      alternative='greater')
        print(f"\n  Wilcoxon (gen 0 > gen {G_MAX}, i.e. tau decreased): W={stat}, p={p_val:.4f}")
    else:
        p_val = 1.0
        print(f"\n  Wilcoxon: insufficient data or zero variation (p=1.0)")

    # Compare reversed to neutral
    p_neutral = 1.0
    d = 0.0
    if neutral_reps:
        neutral_genmax = []
        for rep in neutral_reps:
            gmax = rep['generations'][-1]
            neutral_genmax.append(gmax['mean_fitness'])
        neutral_genmax = np.array(neutral_genmax)

        print(f"\n  Neutral gen {G_MAX} mean tau: {neutral_genmax.mean():.3f} "
              f"+/- {neutral_genmax.std():.3f}")

        # Test: reversed should have LOWER tau than neutral
        if len(genmax_fitness) >= 3 and len(neutral_genmax) >= 3:
            u_stat, p_neutral = stats.mannwhitneyu(
                neutral_genmax, genmax_fitness, alternative='greater')
            print(f"  Mann-Whitney (neutral > reversed): U={u_stat}, p={p_neutral:.4f}")

            pooled_std = np.sqrt((np.var(genmax_fitness) + np.var(neutral_genmax)) / 2)
            if pooled_std > 0:
                d = (neutral_genmax.mean() - genmax_fitness.mean()) / pooled_std
                print(f"  Cohen's d (neutral - reversed): {d:.3f}")

    # V3a success criteria
    print(f"\n  {'─' * 50}")
    print(f"  V3a BIDIRECTIONALITY ASSESSMENT")
    print(f"  {'─' * 50}")

    gamma_increased = np.mean(final_gammas) > 0.00223  # baseline gamma_0
    tau_decreased = delta.mean() < 0

    if tau_decreased and p_val < 0.05:
        if gamma_increased:
            verdict = "BIDIRECTIONAL CONFIRMED"
            print(f"  -> BIDIRECTIONAL CONFIRMED: tau decreased (p={p_val:.4f}), "
                  f"gamma increased to {np.mean(final_gammas):.5f}")
        else:
            verdict = "TAU DECREASED BUT GAMMA UNCHANGED"
            print(f"  -> TAU DECREASED (p={p_val:.4f}) but gamma did not increase")
    elif tau_decreased:
        verdict = "TREND (underpowered)"
        print(f"  -> TREND: tau decreased but p={p_val:.4f} > 0.05")
    else:
        verdict = "NO EFFECT"
        print(f"  -> NO EFFECT: reversed selection did not decrease tau")

    # Save consolidated
    output = {
        'experiment': 'v3a_reversed_selection',
        'verdict': verdict,
        'config': {
            'pop_size': POP_SIZE,
            'g_max': G_MAX,
            'n_reversed_reps': N_SELECTION_REPS,
            'n_neutral_reps': N_NEUTRAL_REPS,
            'n_fitness_seeds': N_FITNESS_SEEDS,
            'sigma_m': SIGMA_M,
            'gamma_min': GAMMA_MIN,
            'gamma_max': GAMMA_MAX,
            'tournament_k': TOURNAMENT_K,
        },
        'reversed_reps': [{
            'replicate_id': r['replicate_id'],
            'gen0_fitness': r['generations'][0]['mean_fitness'],
            'genmax_fitness': r['generations'][-1]['mean_fitness'],
            'final_mean_gamma': r['generations'][-1]['mean_gamma'],
        } for r in reversed_reps],
        'neutral_reps': [{
            'replicate_id': r['replicate_id'],
            'genmax_fitness': r['generations'][-1]['mean_fitness'],
            'final_mean_gamma': r['generations'][-1]['mean_gamma'],
        } for r in neutral_reps],
        'statistics': {
            'p_wilcoxon_decreased': float(p_val),
            'p_mannwhitney_neutral_gt_reversed': float(p_neutral),
            'cohens_d_neutral_minus_reversed': float(d),
        },
    }

    path = os.path.join(save_dir, f'{FILE_PREFIX}_results.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Consolidated results saved to {path}")


# ── Load Phase 0 baseline ───────────────────────────────────────────

def load_phase0_baseline(phase0_path: str = None) -> dict:
    """Load Phase 0 results and extract recommended baseline."""
    if phase0_path is None:
        phase0_path = os.path.join(_this_dir, '..', 'data', 'phase0_results.json')

    if os.path.exists(phase0_path):
        with open(phase0_path) as f:
            data = json.load(f)

        summaries = data['param_summaries']
        candidates = [s for s in summaries if 0.5 <= s['mean_tau'] <= 3.0]
        if candidates:
            best = min(candidates, key=lambda s: abs(s['mean_tau'] - 1.5))
            return {
                'gamma_0': best['gamma'],
                'J': best['J'],
                'k_cat': best['k_cat'],
                'mean_tau': best['mean_tau'],
                'source': 'phase0',
            }

    print("  WARNING: No Phase 0 baseline found, using Paper 3 exemplar")
    return {
        'gamma_0': 0.002,
        'J': 5.0,
        'k_cat': 0.3,
        'mean_tau': 1.5,
        'source': 'paper3_exemplar',
    }


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='V3a: Reversed Selection (minimize tau, gamma only)')
    parser.add_argument('--replicate', type=int, default=None,
                        help='Replicate index to run (0, 1, 2, ...)')
    parser.add_argument('--neutral', action='store_true',
                        help='Run as neutral drift control')
    parser.add_argument('--analyse', action='store_true',
                        help='Analyse existing results')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory (default: ../data/v3a_reversed/)')
    args = parser.parse_args()

    save_dir = args.results_dir if args.results_dir else os.path.join(
        _this_dir, '..', 'data', 'v3a_reversed')
    os.makedirs(save_dir, exist_ok=True)

    if args.analyse:
        analyse(save_dir)
    elif args.replicate is not None:
        baseline = load_phase0_baseline()
        rep_type = 'neutral' if args.neutral else 'reversed'

        print(f"\n  V3a Reversed Selection — {rep_type} replicate {args.replicate}")
        print(f"  Baseline: gamma_0={baseline['gamma_0']:.5f}, "
              f"J={baseline['J']:.3f}, k_cat={baseline['k_cat']:.3f} "
              f"(source: {baseline['source']})")
        print(f"  Pop={POP_SIZE}, Gen={G_MAX}, Seeds={N_FITNESS_SEEDS}")
        if rep_type == 'reversed':
            print(f"  *** SELECTING FOR MINIMUM TAU ***")
        print(f"  Output: {save_dir}/{FILE_PREFIX}_rep{args.replicate}_{rep_type}.jsonl")

        run_replicate(
            replicate_id=args.replicate,
            rep_type=rep_type,
            gamma_0=baseline['gamma_0'],
            J=baseline['J'],
            k_cat=baseline['k_cat'],
            save_dir=save_dir,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 v3a_reversed_selection.py --replicate 0              # reversed rep 0")
        print("  python3 v3a_reversed_selection.py --replicate 0 --neutral    # neutral rep 0")
        print("  python3 v3a_reversed_selection.py --analyse                  # analyse results")
