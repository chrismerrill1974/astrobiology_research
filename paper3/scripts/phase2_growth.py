"""
Phase II: Growth Experiment for Paper 3.

Tests how reaction additions affect transient dimensional inflation.
Parallels Paper 2's growth experiment but starts from the enzyme-complex
coupled oscillator network instead of a bare Brusselator.

Two growth rules:
  - Random: autocatalytic reactions added without screening
  - Aligned: reactions filtered for oscillation preservation

Growth is "within-core only": added reactions involve species from a
single Brusselator core (Xi, Yi) plus new species Zj. No cross-core
or E-mediated additions.

Pre-registered predictions:
  1. η dilution persists (consistent with Papers 1-2)
  2. Aligned growth preserves D₂ > 1.2 significantly better than random
  3. Modular coupling amplifies alignment advantage vs Paper 2
  4. D₂ ceiling ~2.0 (5D system with one slow mode)
"""

import sys
import os
import json
import time
import copy
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Set, Dict

_this_dir = os.path.dirname(os.path.abspath(__file__))
_astro2_path = os.path.join(_this_dir, '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from pilot.pilot5b_enzyme_complex import EnzymeComplexParams, make_enzyme_complex_network
from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension, QualityFlag
from dimensional_opening.network_generator import GeneratedNetwork
from dimensional_opening.stoichiometry import StoichiometricAnalyzer
from dimensional_opening.oscillation_filter import check_oscillation, OscillationResult


# ── Configuration ────────────────────────────────────────────────────

CORE1_SPECIES = {'X1', 'Y1'}
CORE2_SPECIES = {'X2', 'Y2'}
FOOD_SPECIES = {'A', 'B'}
FORBIDDEN_SPECIES = {'A', 'B', 'Esrc', 'D1', 'W1', 'D2', 'W2', 'Ew', 'E', 'G', 'GE'}

PROJECTION_SPECIES = ['X1', 'Y1', 'X2', 'Y2', 'E']

# Parameter sets from Sweep A best points
PARAM_SETS = {
    'primary': EnzymeComplexParams(
        J=5.0, gamma=0.002, k_on=10.0, k_off=10.0,
        k_cat=0.3, G_total=1.0, label="J5_g0.002_kc0.3"),
    'replication1': EnzymeComplexParams(
        J=4.0, gamma=0.003, k_on=10.0, k_off=10.0,
        k_cat=0.2, G_total=1.0, label="J4_g0.003_kc0.2"),
    'replication2': EnzymeComplexParams(
        J=7.0, gamma=0.001, k_on=10.0, k_off=10.0,
        k_cat=0.5, G_total=1.0, label="J7_g0.001_kc0.5"),
}

N_TRAJ = {'primary': 50, 'replication1': 30, 'replication2': 30}
K_MAX = 5
MAX_CANDIDATES = 20
RATE_RANGE = (0.3, 1.5)

# Filter parameters (longer than default for enzyme-complex model)
# 500 time units = ~50 oscillator periods, sufficient for 5-sign-change test
FILTER_T_END = 500.0
FILTER_N_POINTS = 1000

# Analysis parameters
SIM_T_END = 10000.0
SIM_N_POINTS = 20000


# ── Reaction generator ──────────────────────────────────────────────

def _format_reaction(reactants: List[str], products: List[str]) -> str:
    """Format reaction as string: 'A + B -> C + D'."""
    return f"{' + '.join(sorted(reactants))} -> {' + '.join(sorted(products))}"


def generate_within_core_autocatalytic(
    existing_reactions: Set[str],
    core_id: int,
    existing_new_species: List[str],
    rng: np.random.Generator,
    rate_range: Tuple[float, float] = RATE_RANGE,
    max_attempts: int = 100,
) -> Optional[Tuple[str, float, Optional[str]]]:
    """
    Generate a within-core autocatalytic reaction.

    Returns (reaction_string, rate_constant, new_species_name_or_None)
    or None if unable to generate a valid reaction.
    """
    if core_id == 1:
        core_species = list(CORE1_SPECIES)
    else:
        core_species = list(CORE2_SPECIES)

    other_core = CORE2_SPECIES if core_id == 1 else CORE1_SPECIES

    # Pool of species available for this reaction
    available = core_species + list(existing_new_species)

    for _ in range(max_attempts):
        new_sp = None

        # Decide whether to create a new species (50% chance)
        if rng.random() < 0.5:
            n_existing = len(existing_new_species)
            new_sp = f"Z{n_existing}"
            candidate_pool = available + [new_sp]
        else:
            candidate_pool = available

        # Pick catalyst from available species (not food)
        non_food = [s for s in candidate_pool if s not in FOOD_SPECIES]
        if not non_food:
            continue
        catalyst = rng.choice(non_food)

        # Reactants: catalyst + substrate (substrate required to prevent
        # pure self-replication X -> 2X which causes solver blowup)
        reactants = [catalyst]
        substrate_pool = list(FOOD_SPECIES) + [s for s in candidate_pool if s != catalyst]
        if substrate_pool:
            substrate = rng.choice(substrate_pool)
            reactants.append(substrate)
        else:
            continue  # No valid substrate available

        # Products: 2 × catalyst + maybe byproduct
        products = [catalyst, catalyst]
        if rng.random() < 0.3 and len(candidate_pool) > 1:
            byproduct = rng.choice([s for s in candidate_pool if s != catalyst])
            products.append(byproduct)

        # Validate: no species from other core
        all_species_in_rxn = set(reactants) | set(products)
        if all_species_in_rxn & other_core:
            new_sp = None
            continue

        # Validate: no forbidden species (except food as substrate)
        forbidden_in_rxn = all_species_in_rxn & (FORBIDDEN_SPECIES - FOOD_SPECIES)
        if forbidden_in_rxn:
            new_sp = None
            continue

        rxn_str = _format_reaction(reactants, products)

        # Check not duplicate
        if rxn_str in existing_reactions:
            new_sp = None
            continue

        # Sample rate constant (log-uniform)
        rate = float(np.exp(rng.uniform(np.log(rate_range[0]), np.log(rate_range[1]))))

        return rxn_str, rate, new_sp

    return None


# ── Network extension ────────────────────────────────────────────────

def extend_network(
    base_net: GeneratedNetwork,
    added_reactions: List[str],
    added_rates: List[float],
    new_species: List[str],
    network_id: str,
) -> GeneratedNetwork:
    """Extend a GeneratedNetwork with additional reactions and species."""
    all_reactions = list(base_net.reactions) + added_reactions
    all_rates = list(base_net.rate_constants) + added_rates

    all_species = list(base_net.species)
    for sp in new_species:
        if sp not in all_species:
            all_species.append(sp)

    ic = dict(base_net.initial_concentrations)
    for sp in new_species:
        if sp not in ic:
            ic[sp] = 0.1

    return GeneratedNetwork(
        reactions=all_reactions,
        species=all_species,
        food_set=base_net.food_set,
        n_species=len(all_species),
        n_reactions=len(all_reactions),
        n_autocatalytic=base_net.n_autocatalytic + len(added_reactions),
        rate_constants=all_rates,
        initial_concentrations=ic,
        chemostat_species=dict(base_net.chemostat_species),
        network_id=network_id,
        is_autocatalytic=True,
        template=base_net.template,
        n_added_reactions=len(added_reactions),
    )


# ── Oscillation filter (adapted for enzyme-complex model) ───────────

def check_oscillation_extended(
    net: GeneratedNetwork,
    t_end: float = FILTER_T_END,
    n_points: int = FILTER_N_POINTS,
) -> OscillationResult:
    """Check oscillation with longer integration for enzyme-complex model."""
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
    except Exception:
        return OscillationResult(
            passes=False, cv=0.0, amplitude=0.0,
            sign_changes=0, boundedness_ratio=0.0,
            best_species_idx=-1, best_species_name="")

    if not result.success:
        return OscillationResult(
            passes=False, cv=0.0, amplitude=0.0,
            sign_changes=0, boundedness_ratio=0.0,
            best_species_idx=-1, best_species_name="")

    return check_oscillation(
        result.concentrations, result.time,
        species_names=result.species_names,
        food_species=net.food_set,
    )


# ── Growth functions ─────────────────────────────────────────────────

def grow_random(
    base_net: GeneratedNetwork,
    k: int,
    rng: np.random.Generator,
    trajectory_id: str,
) -> Tuple[GeneratedNetwork, List[str], List[str]]:
    """Add k random within-core autocatalytic reactions. No screening."""
    added_reactions = []
    added_rates = []
    new_species = []
    existing_rxns = set(base_net.reactions)

    for step in range(k):
        core_id = int(rng.choice([1, 2]))
        result = generate_within_core_autocatalytic(
            existing_rxns, core_id, new_species, rng)

        if result is None:
            break  # Could not generate valid reaction

        rxn_str, rate, new_sp = result
        added_reactions.append(rxn_str)
        added_rates.append(rate)
        existing_rxns.add(rxn_str)
        if new_sp is not None:
            new_species.append(new_sp)

    net = extend_network(
        base_net, added_reactions, added_rates, new_species,
        network_id=f"{trajectory_id}_random_k{k}")

    return net, added_reactions, new_species


def grow_aligned(
    base_net: GeneratedNetwork,
    k: int,
    rng: np.random.Generator,
    trajectory_id: str,
    max_candidates: int = MAX_CANDIDATES,
) -> Tuple[GeneratedNetwork, List[str], List[str], List[int], bool]:
    """
    Add k oscillation-preserving within-core autocatalytic reactions.

    Returns (network, added_reactions, new_species,
             candidates_per_step, early_terminated).
    """
    added_reactions = []
    added_rates = []
    new_species = []
    candidates_per_step = []
    existing_rxns = set(base_net.reactions)
    current_net = base_net
    early_terminated = False

    for step in range(k):
        found = False
        n_tried = 0

        for candidate_idx in range(max_candidates):
            n_tried += 1
            core_id = int(rng.choice([1, 2]))

            # Generate candidate reaction
            result = generate_within_core_autocatalytic(
                existing_rxns, core_id, new_species, rng)

            if result is None:
                continue

            rxn_str, rate, new_sp = result

            # Build candidate network
            candidate_new_sp = new_species + ([new_sp] if new_sp else [])
            candidate_net = extend_network(
                base_net,
                added_reactions + [rxn_str],
                added_rates + [rate],
                candidate_new_sp,
                network_id=f"{trajectory_id}_aligned_k{step+1}_candidate")

            # Test oscillation
            osc_result = check_oscillation_extended(candidate_net)

            if osc_result.passes:
                added_reactions.append(rxn_str)
                added_rates.append(rate)
                existing_rxns.add(rxn_str)
                if new_sp is not None:
                    new_species.append(new_sp)
                current_net = candidate_net
                found = True
                break

        candidates_per_step.append(n_tried)

        if not found:
            early_terminated = True
            break

    # Build final network
    final_net = extend_network(
        base_net, added_reactions, added_rates, new_species,
        network_id=f"{trajectory_id}_aligned_k{len(added_reactions)}")

    return final_net, added_reactions, new_species, candidates_per_step, early_terminated


# ── Analysis ─────────────────────────────────────────────────────────

def analyze_network(
    net: GeneratedNetwork,
    t_end: float = SIM_T_END,
    n_points: int = SIM_N_POINTS,
) -> Dict:
    """Simulate and analyze a grown network. Returns results dict."""
    sim = ReactionSimulator()
    stoich = StoichiometricAnalyzer()

    # Compute r_s
    try:
        stoich_result = stoich.from_reaction_strings(net.reactions)
        r_s = stoich_result.rank
    except Exception:
        r_s = None

    # Simulate
    try:
        network_obj = sim.build_network(net.reactions)
        sim_result = sim.simulate(
            network_obj,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=(0, t_end),
            n_points=n_points,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=net.chemostat_species,
        )
    except Exception as e:
        return {'regime': 'sim_failed', 'error': str(e),
                'D2_projected': float('nan'), 'D2_full': float('nan'),
                'r_s': r_s, 'eta_projected': float('nan')}

    if not sim_result.success:
        return {'regime': 'solver_failed',
                'D2_projected': float('nan'), 'D2_full': float('nan'),
                'r_s': r_s, 'eta_projected': float('nan')}

    t = sim_result.time
    c = sim_result.concentrations
    species = sim_result.species_names
    n_discard = len(t) // 2
    c_post = c[n_discard:]

    def col(name):
        return c_post[:, species.index(name)]

    # Check for fixed point
    try:
        cv_X1 = np.std(col('X1')) / max(np.mean(col('X1')), 1e-10)
        cv_X2 = np.std(col('X2')) / max(np.mean(col('X2')), 1e-10)
    except (ValueError, IndexError):
        return {'regime': 'extraction_failed',
                'D2_projected': float('nan'), 'D2_full': float('nan'),
                'r_s': r_s, 'eta_projected': float('nan')}

    if cv_X1 < 0.01 and cv_X2 < 0.01:
        return {'regime': 'fixed_point',
                'D2_projected': float('nan'), 'D2_full': float('nan'),
                'r_s': r_s, 'eta_projected': float('nan'),
                'oscillation_survives': False}

    # Projected D₂ on (X1, Y1, X2, Y2, E)
    cd = CorrelationDimension()
    try:
        proj_traj = np.column_stack([col(sp) for sp in PROJECTION_SPECIES])
        d2_proj_result = cd.compute(proj_traj)
        d2_proj = float(d2_proj_result.D2) if d2_proj_result.D2 is not None else float('nan')
    except Exception:
        d2_proj = float('nan')

    # Full D₂ on all non-monotonic, non-chemostatted species
    chemostatted = set(net.chemostat_species.keys()) if net.chemostat_species else set()
    waste = {'D1', 'W1', 'D2', 'W2', 'Ew'}
    dynamic_species = [sp for sp in species
                       if sp not in chemostatted and sp not in waste]

    try:
        full_cols = []
        for sp in dynamic_species:
            vals = col(sp)
            diffs = np.diff(vals)
            is_mono_inc = np.all(diffs >= -1e-10)
            is_mono_dec = np.all(diffs <= 1e-10)
            if not (is_mono_inc or is_mono_dec):
                full_cols.append(vals)

        if len(full_cols) >= 2:
            full_traj = np.column_stack(full_cols)
            d2_full_result = cd.compute(full_traj)
            d2_full = float(d2_full_result.D2) if d2_full_result.D2 is not None else float('nan')
        else:
            d2_full = float('nan')
    except Exception:
        d2_full = float('nan')

    # Phase correlation
    try:
        r_X1X2 = float(np.corrcoef(col('X1'), col('X2'))[0, 1])
    except Exception:
        r_X1X2 = float('nan')

    # Regime classification
    if np.isnan(d2_proj):
        regime = 'failed_d2'
    elif d2_proj > 1.2:
        regime = 'complex'
    else:
        regime = 'phase_locked'

    # η
    eta_proj = d2_proj / r_s if r_s and not np.isnan(d2_proj) else float('nan')
    eta_full = d2_full / r_s if r_s and not np.isnan(d2_full) else float('nan')

    # Oscillation survival
    osc_survives = cv_X1 > 0.03 or cv_X2 > 0.03

    return {
        'regime': regime,
        'D2_projected': d2_proj,
        'D2_full': d2_full,
        'r_s': r_s,
        'eta_projected': eta_proj,
        'eta_full': eta_full,
        'r_X1X2': r_X1X2,
        'oscillation_survives': osc_survives,
    }


# ── Main experiment loop ─────────────────────────────────────────────

def run_growth_experiment(
    param_set_name: str,
    params: EnzymeComplexParams,
    n_trajectories: int,
    k_max: int = K_MAX,
    max_candidates: int = MAX_CANDIDATES,
    seed_base: int = 42,
    verbose: bool = True,
) -> List[Dict]:
    """Run growth experiment for one parameter set."""
    results = []
    total_runs = (k_max + 1) * 2 * n_trajectories
    run_count = 0
    start_time = time.time()

    for rule in ['random', 'aligned']:
        for k in range(k_max + 1):
            for traj_idx in range(n_trajectories):
                run_count += 1
                seed = seed_base + traj_idx * 137
                rng = np.random.default_rng(seed + k * 10000 + (0 if rule == 'random' else 50000))

                # Build base network
                base_net = make_enzyme_complex_network(params, seed=seed)
                trajectory_id = f"{params.label}_t{traj_idx}_s{seed}"

                # Grow
                if k == 0:
                    # Baseline — no additions
                    grown_net = base_net
                    added_rxns = []
                    new_sp = []
                    candidates_per_step = []
                    early_terminated = False
                elif rule == 'random':
                    grown_net, added_rxns, new_sp = grow_random(
                        base_net, k, rng, trajectory_id)
                    candidates_per_step = []
                    early_terminated = False
                else:
                    grown_net, added_rxns, new_sp, candidates_per_step, early_terminated = \
                        grow_aligned(base_net, k, rng, trajectory_id, max_candidates)

                # Analyze
                analysis = analyze_network(grown_net)

                result = {
                    'param_set': param_set_name,
                    'param_label': params.label,
                    'k': k,
                    'rule': rule,
                    'trajectory_idx': traj_idx,
                    'seed': seed,
                    'n_reactions_total': len(grown_net.reactions),
                    'n_added': len(added_rxns),
                    'added_reactions': added_rxns,
                    'new_species': new_sp,
                    'early_terminated': early_terminated,
                    'candidates_per_step': candidates_per_step,
                    **analysis,
                }
                results.append(result)

                if verbose and run_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = run_count / elapsed if elapsed > 0 else 0
                    eta_mins = (total_runs - run_count) / rate / 60 if rate > 0 else 0
                    print(f"  [{param_set_name}] {run_count}/{total_runs} "
                          f"({elapsed/60:.1f} min, ~{eta_mins:.0f} min remaining) "
                          f"| {rule} k={k} t{traj_idx}: "
                          f"D2={result['D2_projected']:.3f} {result['regime']}",
                          flush=True)

    return results


def print_summary(results: List[Dict], param_label: str):
    """Print summary table for one parameter set."""
    print(f"\n{'='*80}")
    print(f"  GROWTH SUMMARY: {param_label}")
    print(f"{'='*80}")

    print(f"\n  {'rule':>8}  {'k':>3}  {'med D2':>7}  {'frac>1.2':>8}  "
          f"{'med η':>7}  {'osc_surv':>8}  {'n':>4}")
    print(f"  {'-'*55}")

    for rule in ['random', 'aligned']:
        for k in range(K_MAX + 1):
            cell = [r for r in results
                    if r['rule'] == rule and r['k'] == k
                    and r['param_label'] == param_label]

            d2s = [r['D2_projected'] for r in cell if not np.isnan(r['D2_projected'])]
            etas = [r['eta_projected'] for r in cell if not np.isnan(r['eta_projected'])]
            n_complex = sum(1 for r in cell if r.get('regime') == 'complex')
            n_osc = sum(1 for r in cell if r.get('oscillation_survives', False))
            n_valid = len(d2s)

            if n_valid > 0:
                med_d2 = np.median(d2s)
                frac_above = n_complex / len(cell) if cell else 0
                med_eta = np.median(etas) if etas else float('nan')
                osc_rate = n_osc / len(cell) if cell else 0
                print(f"  {rule:>8}  {k:>3}  {med_d2:>7.3f}  {frac_above:>8.1%}  "
                      f"{med_eta:>7.4f}  {osc_rate:>8.1%}  {n_valid:>4}")
            else:
                print(f"  {rule:>8}  {k:>3}  {'N/A':>7}  {'N/A':>8}  "
                      f"{'N/A':>7}  {'N/A':>8}  {0:>4}")

    # Aligned-specific: acceptance rates and early termination
    aligned_k5 = [r for r in results
                  if r['rule'] == 'aligned' and r['k'] == K_MAX
                  and r['param_label'] == param_label]
    if aligned_k5:
        n_early = sum(1 for r in aligned_k5 if r.get('early_terminated', False))
        print(f"\n  Aligned growth at k={K_MAX}: {n_early}/{len(aligned_k5)} early terminated")

        all_candidates = [r.get('candidates_per_step', []) for r in aligned_k5]
        if all_candidates and any(all_candidates):
            for step in range(K_MAX):
                step_counts = [c[step] for c in all_candidates
                               if len(c) > step]
                if step_counts:
                    print(f"    Step {step+1}: median {np.median(step_counts):.0f} "
                          f"candidates tried (max {max(step_counts)})")


def run_all(save_dir: str | None = None) -> Dict:
    """Run the full Phase II growth experiment."""
    print("\n" + "#" * 80, flush=True)
    print("# PHASE II: GROWTH EXPERIMENT", flush=True)
    print("#" * 80, flush=True)

    all_results = []
    total_start = time.time()

    for ps_name in ['primary', 'replication1', 'replication2']:
        params = PARAM_SETS[ps_name]
        n_traj = N_TRAJ[ps_name]

        print(f"\n{'='*80}", flush=True)
        print(f"  Parameter set: {ps_name} ({params.label})", flush=True)
        print(f"  N trajectories: {n_traj}", flush=True)
        print(f"{'='*80}", flush=True)

        results = run_growth_experiment(
            ps_name, params, n_traj,
            seed_base=42 if ps_name == 'primary' else (1000 if ps_name == 'replication1' else 2000))

        all_results.extend(results)
        print_summary(results, params.label)

        # Checkpoint save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f'phase2_growth_{ps_name}_checkpoint.json')
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'param_set': ps_name,
                    'param_label': params.label,
                    'n_trajectories': n_traj,
                    'results': results,
                }, f, indent=2, default=str)
            print(f"\n  Checkpoint saved to {checkpoint_path}", flush=True)

    total_time = time.time() - total_start

    # ── Final summary ────────────────────────────────────────────
    print(f"\n\n{'#'*80}")
    print(f"# FINAL GROWTH EXPERIMENT SUMMARY ({total_time:.0f}s / {total_time/60:.1f} min)")
    print(f"{'#'*80}")

    for ps_name in ['primary', 'replication1', 'replication2']:
        params = PARAM_SETS[ps_name]
        ps_results = [r for r in all_results if r['param_set'] == ps_name]
        print_summary(ps_results, params.label)

    # ── η slope analysis ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  η DILUTION SLOPES")
    print(f"{'='*80}")

    for ps_name in ['primary', 'replication1', 'replication2']:
        params = PARAM_SETS[ps_name]
        ps_results = [r for r in all_results if r['param_set'] == ps_name]

        for rule in ['random', 'aligned']:
            ks = []
            medians = []
            for k in range(K_MAX + 1):
                cell = [r for r in ps_results
                        if r['rule'] == rule and r['k'] == k]
                etas = [r['eta_projected'] for r in cell
                        if not np.isnan(r.get('eta_projected', float('nan')))]
                if etas:
                    ks.append(k)
                    medians.append(np.median(etas))

            if len(ks) >= 2:
                slope, intercept = np.polyfit(ks, medians, 1)
                print(f"  {params.label} {rule:>8}: slope = {slope:.4f}/rxn "
                      f"(η₀ = {intercept:.4f})")

    # ── D₂ > 1.2 retention ──────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  D₂ > 1.2 RETENTION (fraction complex)")
    print(f"{'='*80}")

    for ps_name in ['primary', 'replication1', 'replication2']:
        params = PARAM_SETS[ps_name]
        ps_results = [r for r in all_results if r['param_set'] == ps_name]

        for rule in ['random', 'aligned']:
            fracs = []
            for k in range(K_MAX + 1):
                cell = [r for r in ps_results
                        if r['rule'] == rule and r['k'] == k]
                if cell:
                    n_cpx = sum(1 for r in cell if r.get('regime') == 'complex')
                    fracs.append(n_cpx / len(cell))
                else:
                    fracs.append(0)

            frac_str = "  ".join(f"{f:.0%}" for f in fracs)
            print(f"  {params.label} {rule:>8}: k=0..{K_MAX}: {frac_str}")

    # ── Save ─────────────────────────────────────────────────────
    output = {
        'experiment': 'phase2_growth',
        'k_max': K_MAX,
        'max_candidates': MAX_CANDIDATES,
        'filter_t_end': FILTER_T_END,
        'sim_t_end': SIM_T_END,
        'runtime_seconds': total_time,
        'param_sets': {name: asdict(p) for name, p in PARAM_SETS.items()},
        'n_trajectories': N_TRAJ,
        'total_runs': len(all_results),
        'results': all_results,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'phase2_growth_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Full results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_all(save_dir=save_dir)
