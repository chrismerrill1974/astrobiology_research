"""
Phase 1: Evolutionary Selection on tau_{>1.2} (gamma only).

Single-process, batch-mode, incremental JSONL output, resumable.
v2: Bulletproofed per Gemini review — NaN/Inf guards, per-eval timeout,
    stiff-regime max_step cap, and enhanced diagnostics.

Usage:
  python3 phase1_evolution_v2.py --replicate 0              # run selection replicate 0
  python3 phase1_evolution_v2.py --replicate 0 --neutral    # run neutral replicate 0
  python3 phase1_evolution_v2.py --analyse                  # analyse all completed results
  python3 phase1_evolution_v2.py --analyse --results-dir ../data/v2_confirmed
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


# ── Configuration (Pilot) ────────────────────────────────────────────

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
EVAL_TIMEOUT_SECONDS = 60   # 1 min hard wall-clock timeout per integration
MAX_STEP = 10.0             # prevent LSODA from taking huge steps in stiff regime


# ── Timeout mechanism ────────────────────────────────────────────────

class EvalTimeout(Exception):
    """Raised when a single evaluation exceeds the wall-clock timeout."""
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout(f"Integration exceeded {EVAL_TIMEOUT_SECONDS}s wall-clock limit")


# ── Fitness evaluation (single-threaded) ─────────────────────────────

def simulate_and_get_tau(gamma: float, J: float, k_cat: float,
                         seed: int) -> dict:
    """
    Run one integration and return tau_{>1.2} plus diagnostics.
    Returns dict with 'tau', 'status', 'elapsed_s'.
    Status: 'ok', 'solver_fail', 'pathological', 'nan_inf', 'timeout', 'exception'
    """
    t0 = time.time()

    p = EnzymeComplexParams(
        A=1.0, B=3.0,
        k_on=10.0, k_off=10.0,
        k_cat=k_cat, G_total=1.0,
        J=J, gamma=gamma,
        label=f"evo_g{gamma:.6f}_s{seed}",
    )

    net = make_enzyme_complex_network(p, seed=seed)
    sim = ReactionSimulator()

    # Set alarm-based timeout (Unix only; safe for single-threaded)
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
        signal.alarm(0)  # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)  # restore handler

    if not result.success:
        return {'tau': 0, 'status': 'solver_fail', 'elapsed_s': time.time() - t0,
                'message': getattr(result, 'message', '')[:200]}

    t = result.time
    c = result.concentrations
    species = result.species_names

    # ── NaN / Inf check ───────────────────────────────────────────────
    if not np.isfinite(c).all():
        return {'tau': 0, 'status': 'nan_inf', 'elapsed_s': time.time() - t0}

    def idx(name):
        return species.index(name)

    # ── Pathological regime check ─────────────────────────────────────
    for sp_name in ['X1', 'Y1', 'X2', 'Y2', 'E']:
        vals = c[:, idx(sp_name)]
        if np.any(vals > MAX_CONCENTRATION) or np.any(vals < 0):
            return {'tau': 0, 'status': 'pathological',
                    'elapsed_s': time.time() - t0}

    # ── Sliding-window D2 ────────────────────────────────────────────
    tau = 0
    for win_idx in range(1, 8):  # windows [2500,5000] to [17500,20000]
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

        # Guard: skip window if trajectory has NaN/Inf
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


# ── Mutation ─────────────────────────────────────────────────────────

def mutate_gamma(gamma: float, rng: np.random.RandomState) -> float:
    """Gaussian mutation in log10-space with reflection at bounds."""
    log_gamma = np.log10(gamma)
    log_gamma_new = log_gamma + rng.normal(0, SIGMA_M)

    log_min = np.log10(GAMMA_MIN)
    log_max = np.log10(GAMMA_MAX)

    # Reflection: if optimum is below GAMMA_MIN, population clusters
    # just above the floor (boundary collapse criterion).
    if log_gamma_new < log_min:
        log_gamma_new = log_min + abs(log_gamma_new - log_min)
    elif log_gamma_new > log_max:
        log_gamma_new = log_max - abs(log_gamma_new - log_max)

    log_gamma_new = np.clip(log_gamma_new, log_min, log_max)
    return float(10 ** log_gamma_new)


# ── Selection ────────────────────────────────────────────────────────

def tournament_select(individuals: list, k: int,
                      rng: np.random.RandomState) -> dict:
    """Tournament selection: pick k random individuals, return the fittest."""
    contestants = rng.choice(len(individuals), size=k, replace=False)
    best = max(contestants, key=lambda i: individuals[i]['fitness'])
    return individuals[best]


# ── Single replicate (batch, incremental, resumable) ─────────────────

def run_replicate(replicate_id: int, rep_type: str,
                  gamma_0: float, J: float, k_cat: float,
                  save_dir: str):
    """
    Run one replicate generation-by-generation.
    Writes one JSONL line per generation. Resumes from last completed gen.

    rep_type: 'selection' or 'neutral'
    """
    jsonl_path = os.path.join(save_dir, f'phase1_rep{replicate_id}_{rep_type}.jsonl')

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

    # If fully done, skip
    if last_done >= G_MAX:
        print(f"  Replicate {replicate_id} ({rep_type}) already complete ({last_done + 1} gens)")
        return

    rng = np.random.RandomState(replicate_id * 1000 + 7)

    # Reconstruct population state from saved generations
    if last_done >= 0:
        last_rec = done_gens[last_done]
        gammas = np.array(last_rec['gammas'])

        # Advance RNG to match state after last_done generations.
        for g in range(last_done):
            for _ in range(POP_SIZE):
                rng.choice(POP_SIZE, size=TOURNAMENT_K, replace=False)
                if rep_type == 'neutral':
                    rng.randint(0, POP_SIZE)
                rng.normal(0, SIGMA_M)

        # Apply the final mutation (gen last_done → last_done+1) using stored
        # fitness data, so gammas are ready for the next generation's evaluation.
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
                if rep_type == 'selection':
                    parent = tournament_select(stored_individuals, TOURNAMENT_K, rng)
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
    print(f"  {rep_type.upper()} REPLICATE {replicate_id}")
    print(f"  gamma_0={gamma_0:.5f}, J={J:.3f}, k_cat={k_cat:.3f}")
    print(f"  Generations {start_gen} to {G_MAX}")
    print(f"  Safeguards: max_step={MAX_STEP}, timeout={EVAL_TIMEOUT_SECONDS}s, "
          f"NaN/Inf check=ON")
    print(f"  {'=' * 60}\n")

    # Track cumulative diagnostics
    total_timeouts = 0
    total_nan_inf = 0
    total_solver_fail = 0
    total_pathological = 0

    for gen in range(start_gen, G_MAX + 1):
        gen_start = time.time()

        # Evaluate all individuals sequentially
        individuals = []
        gen_timeouts = 0
        gen_nan_inf = 0
        gen_solver_fail = 0
        gen_pathological = 0

        for i in range(POP_SIZE):
            ind = evaluate_individual(float(gammas[i]), J, k_cat)
            individuals.append(ind)

            # Count diagnostic statuses
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

        # Compute generation stats
        fitnesses = [ind['fitness'] for ind in individuals]
        gamma_vals = [ind['gamma'] for ind in individuals]
        n_lethal = sum(1 for ind in individuals
                       if all(t == 0 for t in ind['taus']))

        gen_record = {
            'generation': gen,
            'replicate_id': replicate_id,
            'type': rep_type,
            'mean_fitness': float(np.mean(fitnesses)),
            'median_fitness': float(np.median(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'mean_gamma': float(np.mean(gamma_vals)),
            'std_gamma': float(np.std(gamma_vals)),
            'min_gamma': float(np.min(gamma_vals)),
            'max_gamma': float(np.max(gamma_vals)),
            'n_lethal': n_lethal,
            'gammas': [float(g) for g in gammas],
            'fitnesses': fitnesses,
            'individual_taus': [ind['taus'] for ind in individuals],
            # v2 diagnostics
            'diagnostics': {
                'timeouts': gen_timeouts,
                'nan_inf': gen_nan_inf,
                'solver_fail': gen_solver_fail,
                'pathological': gen_pathological,
                'eval_seconds': [ind.get('eval_seconds', 0) for ind in individuals],
            },
        }

        # Write immediately
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(gen_record) + '\n')

        gen_elapsed = time.time() - gen_start
        total_elapsed = time.time() - batch_start
        gens_done = gen - start_gen + 1
        gens_remaining = G_MAX - gen
        rate = gens_done / total_elapsed if total_elapsed > 0 else 0
        eta = gens_remaining / rate if rate > 0 else 0

        # Build diagnostic suffix
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
              f"max_tau={gen_record['max_fitness']:.1f} | "
              f"mean_gamma={gen_record['mean_gamma']:.5f} "
              f"std={gen_record['std_gamma']:.5f} | "
              f"lethal={n_lethal} | "
              f"{gen_elapsed:.0f}s | ETA {eta/60:.0f}min{diag_str}", flush=True)

        # Selection + reproduction (skip on last generation)
        if gen < G_MAX:
            new_gammas = np.zeros(POP_SIZE)
            for i in range(POP_SIZE):
                if rep_type == 'selection':
                    parent = tournament_select(individuals, TOURNAMENT_K, rng)
                else:
                    # Neutral: random parent (no fitness pressure)
                    parent_idx = rng.choice(POP_SIZE, size=TOURNAMENT_K,
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
    """Analyse all completed Phase 1 results."""
    from scipy import stats

    # Load all replicate files
    selection_reps = []
    neutral_reps = []

    for rep_id in range(N_SELECTION_REPS):
        path = os.path.join(save_dir, f'phase1_rep{rep_id}_selection.jsonl')
        if os.path.exists(path):
            gens = []
            with open(path) as f:
                for line in f:
                    try:
                        gens.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if gens:
                selection_reps.append({
                    'replicate_id': rep_id,
                    'generations': sorted(gens, key=lambda g: g['generation']),
                })

    for rep_id in range(N_NEUTRAL_REPS):
        path = os.path.join(save_dir, f'phase1_rep{rep_id}_neutral.jsonl')
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
    print(f"PHASE 1 ANALYSIS (V2, n={N_SELECTION_REPS})")
    print(f"{'=' * 70}")
    print(f"\n  Selection replicates found: {len(selection_reps)}")
    print(f"  Neutral replicates found: {len(neutral_reps)}")

    if not selection_reps:
        print("  No selection results to analyse.")
        return

    # Per-replicate summary
    print(f"\n  {'─' * 50}")
    print(f"  PER-REPLICATE SUMMARY")
    print(f"  {'─' * 50}")

    gen0_fitness = []
    genmax_fitness = []
    final_gammas = []
    final_lethals = []

    for rep in selection_reps:
        gens = rep['generations']
        g0 = gens[0]
        gmax = gens[-1]
        gen0_fitness.append(g0['mean_fitness'])
        genmax_fitness.append(gmax['mean_fitness'])
        final_gammas.append(gmax['mean_gamma'])
        final_lethals.append(gmax['n_lethal'])

        print(f"\n  Selection rep {rep['replicate_id']}:")
        print(f"    Gen 0:  mean_tau={g0['mean_fitness']:.3f}, "
              f"mean_gamma={g0['mean_gamma']:.5f}")
        print(f"    Gen {gmax['generation']}: mean_tau={gmax['mean_fitness']:.3f}, "
              f"mean_gamma={gmax['mean_gamma']:.5f}")
        print(f"    Delta tau: {gmax['mean_fitness'] - g0['mean_fitness']:+.3f}")

        # v2: aggregate diagnostics
        diag_totals = {'timeouts': 0, 'nan_inf': 0, 'solver_fail': 0,
                       'pathological': 0}
        for g in gens:
            d = g.get('diagnostics', {})
            for k in diag_totals:
                diag_totals[k] += d.get(k, 0)
        if any(v > 0 for v in diag_totals.values()):
            print(f"    Diagnostics: {diag_totals}")

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

    # Wilcoxon signed-rank: gen 0 vs gen G_max
    if len(gen0_fitness) >= 3 and not np.all(delta == 0):
        stat, p_val = stats.wilcoxon(genmax_fitness, gen0_fitness,
                                      alternative='greater')
        print(f"\n  Wilcoxon (gen {G_MAX} > gen 0): W={stat}, p={p_val:.4f}")
    else:
        p_val = 1.0
        print(f"\n  Wilcoxon: insufficient data or zero variation (p=1.0)")

    # Compare to neutral
    p_neutral = 1.0
    d = 0.0
    if neutral_reps:
        neutral_genmax = []
        for rep in neutral_reps:
            gmax = rep['generations'][-1]
            neutral_genmax.append(gmax['mean_fitness'])
            print(f"\n  Neutral rep {rep['replicate_id']}:")
            print(f"    Gen {gmax['generation']}: mean_tau={gmax['mean_fitness']:.3f}, "
                  f"mean_gamma={gmax['mean_gamma']:.5f}")
        neutral_genmax = np.array(neutral_genmax)

        print(f"\n  Neutral gen {G_MAX} mean tau: {neutral_genmax.mean():.3f} "
              f"+/- {neutral_genmax.std():.3f}")

        if len(genmax_fitness) >= 3 and len(neutral_genmax) >= 3:
            u_stat, p_neutral = stats.mannwhitneyu(
                genmax_fitness, neutral_genmax, alternative='greater')
            print(f"  Mann-Whitney (selection > neutral): U={u_stat}, p={p_neutral:.4f}")

            pooled_std = np.sqrt((np.var(genmax_fitness) + np.var(neutral_genmax)) / 2)
            if pooled_std > 0:
                d = (genmax_fitness.mean() - neutral_genmax.mean()) / pooled_std
                print(f"  Cohen's d: {d:.3f}")
            else:
                print(f"  Cohen's d: 0.000 (zero variance)")

    # Success criteria
    print(f"\n  {'─' * 50}")
    print(f"  SUCCESS CRITERIA EVALUATION")
    print(f"  {'─' * 50}")

    pct_at_floor = sum(1 for g in final_gammas if g < GAMMA_MIN * 2) / len(final_gammas)
    mean_lethal_pct = np.mean([l / POP_SIZE for l in final_lethals])

    if pct_at_floor >= 0.8 and mean_lethal_pct > 0.2:
        verdict = "BOUNDARY COLLAPSE"
        print(f"  -> BOUNDARY COLLAPSE: {pct_at_floor*100:.0f}% at gamma floor, "
              f"{mean_lethal_pct*100:.0f}% mean lethal rate")
    elif neutral_reps and p_val < 0.05 and p_neutral < 0.05 and d > 0.8:
        verdict = "STRONG POSITIVE"
        print(f"  -> STRONG POSITIVE: p_gen0={p_val:.4f}, p_neutral={p_neutral:.4f}, d={d:.2f}")
    elif neutral_reps and p_val < 0.05 and p_neutral < 0.05:
        if d < 0.5:
            verdict = "AMBIGUOUS"
            print(f"  -> AMBIGUOUS: p={p_val:.4f} but d={d:.2f} < 0.5")
        else:
            verdict = "POSITIVE"
            print(f"  -> POSITIVE: p_gen0={p_val:.4f}, p_neutral={p_neutral:.4f}, d={d:.2f}")
    elif p_val < 0.05:
        verdict = "PRELIMINARY POSITIVE (need neutral)"
        print(f"  -> PRELIMINARY POSITIVE: p_gen0={p_val:.4f} (neutral control needed)")
    else:
        verdict = "NEGATIVE"
        if neutral_reps:
            print(f"  -> NEGATIVE: p_gen0={p_val:.4f}, p_neutral={p_neutral:.4f}, d={d:.2f}")
        else:
            print(f"  -> NEGATIVE: p_gen0={p_val:.4f}")

    if len(selection_reps) < N_SELECTION_REPS or len(neutral_reps) < N_NEUTRAL_REPS:
        print(f"\n  NOTE: Only {len(selection_reps)}/{N_SELECTION_REPS} selection and "
              f"{len(neutral_reps)}/{N_NEUTRAL_REPS} neutral replicates found.")
        print(f"  Statistics may change when all replicates complete.")
    print(f"  {'─' * 50}")

    # Save consolidated
    output = {
        'phase': 'phase1_v2',
        'verdict': verdict,
        'script_version': 'v2',
        'config': {
            'pop_size': POP_SIZE,
            'g_max': G_MAX,
            'n_selection_reps': N_SELECTION_REPS,
            'n_neutral_reps': N_NEUTRAL_REPS,
            'n_fitness_seeds': N_FITNESS_SEEDS,
            'sigma_m': SIGMA_M,
            'gamma_min': GAMMA_MIN,
            'gamma_max': GAMMA_MAX,
            'tournament_k': TOURNAMENT_K,
            'max_step': MAX_STEP,
            'eval_timeout_s': EVAL_TIMEOUT_SECONDS,
        },
        'selection_reps': [{
            'replicate_id': r['replicate_id'],
            'gen0_fitness': r['generations'][0]['mean_fitness'],
            'genmax_fitness': r['generations'][-1]['mean_fitness'],
            'final_mean_gamma': r['generations'][-1]['mean_gamma'],
        } for r in selection_reps],
        'neutral_reps': [{
            'replicate_id': r['replicate_id'],
            'genmax_fitness': r['generations'][-1]['mean_fitness'],
            'final_mean_gamma': r['generations'][-1]['mean_gamma'],
        } for r in neutral_reps],
        'statistics': {
            'p_wilcoxon': float(p_val),
            'p_mannwhitney': float(p_neutral),
            'cohens_d': float(d),
        },
    }

    path = os.path.join(save_dir, 'phase1_results.json')
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
        description='Phase 1: Evolutionary Selection (single-process, v2)')
    parser.add_argument('--replicate', type=int, default=None,
                        help='Replicate index to run (0, 1, 2, ...)')
    parser.add_argument('--neutral', action='store_true',
                        help='Run as neutral drift control')
    parser.add_argument('--analyse', action='store_true',
                        help='Analyse existing results')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory (default: ../data/v2_confirmed/)')
    args = parser.parse_args()

    save_dir = args.results_dir if args.results_dir else os.path.join(
        _this_dir, '..', 'data', 'v2_confirmed')
    os.makedirs(save_dir, exist_ok=True)

    if args.analyse:
        analyse(save_dir)
    elif args.replicate is not None:
        baseline = load_phase0_baseline()
        rep_type = 'neutral' if args.neutral else 'selection'

        print(f"\n  Phase 1 (v2) — {rep_type} replicate {args.replicate}")
        print(f"  Baseline: gamma_0={baseline['gamma_0']:.5f}, "
              f"J={baseline['J']:.3f}, k_cat={baseline['k_cat']:.3f} "
              f"(source: {baseline['source']})")
        print(f"  Pop={POP_SIZE}, Gen={G_MAX}, Seeds={N_FITNESS_SEEDS}")
        print(f"  Safeguards: timeout={EVAL_TIMEOUT_SECONDS}s, max_step={MAX_STEP}, "
              f"NaN/Inf=ON")
        print(f"  Output: {save_dir}/phase1_rep{args.replicate}_{rep_type}.jsonl")

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
        print("  python3 phase1_evolution_v2.py --replicate 0              # selection rep 0")
        print("  python3 phase1_evolution_v2.py --replicate 0 --neutral    # neutral rep 0")
        print("  python3 phase1_evolution_v2.py --analyse                  # analyse results")
        print("  python3 phase1_evolution_v2.py --analyse --results-dir ../data/v1_pilot")
