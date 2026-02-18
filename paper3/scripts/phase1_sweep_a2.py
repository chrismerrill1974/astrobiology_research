"""
Sweep A2: Independent-reservoir causal control at γ = 0.001.

The decisive test: does splitting the shared E into independent E₁/E₂
kill dimensional inflation in the dominant γ=0.001 band?

Grid: J ∈ {3, 5, 7, 10} × γ = 0.001 × k_cat ∈ {0.1, 0.3, 0.5}
= 12 parameter sets × 5 seeds = 60 runs

Model: Each core gets its own energy pool (E_i), gate (G_i), and
complex (GE_i). The two cores are completely decoupled.

We compute D₂ on three projections:
  - 3D per-core: (X_i, Y_i, E_i) for each core independently
  - 6D combined: (X1, Y1, E1, X2, Y2, E2)
  - 4D oscillators-only: (X1, Y1, X2, Y2) — for comparison with shared-E runs
"""

import sys
import os
import json
import time
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
_astro2_path = os.path.join(_this_dir, '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension, QualityFlag
from dimensional_opening.network_generator import GeneratedNetwork
from pilot.pilot5b_enzyme_complex import EnzymeComplexParams


def make_independent_reservoir_network(
    p: EnzymeComplexParams,
    seed: int = 42,
    ic_perturbation: float = 0.05,
) -> GeneratedNetwork:
    """
    Two-core Brusselator with INDEPENDENT energy pools E1, E2.

    Each core i has its own Ei, Gi, GEi. No shared species.
    The two cores are completely decoupled.
    """
    rng = np.random.RandomState(seed)

    reactions = [
        # Core 1 Brusselator
        "A -> X1",
        "B + X1 -> Y1 + D1",
        "X1 + X1 + Y1 -> X1 + X1 + X1",
        "X1 -> W1",

        # Core 2 Brusselator
        "A -> X2",
        "B + X2 -> Y2 + D2",
        "X2 + X2 + Y2 -> X2 + X2 + X2",
        "X2 -> W2",

        # Energy pool 1 (independent)
        "Esrc -> Esrc + E1",
        "E1 -> Ew1",

        # Energy pool 2 (independent)
        "Esrc -> Esrc + E2",
        "E2 -> Ew2",

        # Gate 1
        "G1 + E1 -> GE1",
        "GE1 -> G1 + E1",

        # Gate 2
        "G2 + E2 -> GE2",
        "GE2 -> G2 + E2",

        # Gated autocatalysis — each core uses its OWN gate complex
        "X1 + X1 + Y1 + GE1 -> X1 + X1 + X1 + G1 + Ew1",
        "X2 + X2 + Y2 + GE2 -> X2 + X2 + X2 + G2 + Ew2",
    ]

    rate_constants = [
        # Core 1: A, B, autocatalysis, decay
        p.A, p.B, 1.0, 1.0,
        # Core 2: same
        p.A, p.B, 1.0, 1.0,
        # Energy inflow 1, leak 1
        p.J, p.gamma,
        # Energy inflow 2, leak 2
        p.J, p.gamma,
        # Gate 1: binding, unbinding
        p.k_on, p.k_off,
        # Gate 2: binding, unbinding
        p.k_on, p.k_off,
        # Gated autocatalysis for cores 1 & 2
        p.k_cat, p.k_cat,
    ]

    # Initial conditions — each pool identical
    E0 = p.J / max(p.gamma, 1e-6)
    GE0 = p.G_total * E0 / (p.K_d + E0)
    G0 = p.G_total - GE0

    ic = {
        'X1': 1.0, 'Y1': 1.0, 'D1': 0.0, 'W1': 0.0,
        'X2': 1.0 + rng.uniform(-ic_perturbation, ic_perturbation),
        'Y2': 1.0 + rng.uniform(-ic_perturbation, ic_perturbation),
        'D2': 0.0, 'W2': 0.0,
        'E1': E0, 'Ew1': 0.0, 'G1': G0, 'GE1': GE0,
        'E2': E0, 'Ew2': 0.0, 'G2': G0, 'GE2': GE0,
        'A': p.A, 'B': p.B, 'Esrc': 1.0,
    }

    chemostat = {'A': p.A, 'B': p.B, 'Esrc': 1.0}

    species = ['X1', 'Y1', 'D1', 'W1', 'X2', 'Y2', 'D2', 'W2',
               'E1', 'Ew1', 'G1', 'GE1',
               'E2', 'Ew2', 'G2', 'GE2',
               'A', 'B', 'Esrc']

    return GeneratedNetwork(
        reactions=reactions,
        species=species,
        food_set=['A', 'B', 'Esrc'],
        n_species=len(species),
        n_reactions=len(reactions),
        n_autocatalytic=2,
        rate_constants=rate_constants,
        initial_concentrations=ic,
        chemostat_species=chemostat,
        network_id=f"indep_ec_{p.label}_s{seed}",
        is_autocatalytic=True,
        template="IndependentReservoir_EnzymeComplex",
        n_added_reactions=0,
    )


def simulate_independent(
    p: EnzymeComplexParams,
    seed: int = 42,
    t_span: Tuple[float, float] = (0, 10000),
    n_points: int = 20000,
    verbose: bool = False,
) -> Dict:
    """Simulate independent-reservoir model, compute D₂ on multiple projections."""
    net = make_independent_reservoir_network(p, seed=seed)

    sim = ReactionSimulator()
    try:
        network_obj = sim.build_network(net.reactions)
        sim_result = sim.simulate(
            network_obj,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=t_span,
            n_points=n_points,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=net.chemostat_species,
        )
    except Exception as e:
        return {'label': p.label, 'seed': seed, 'regime': 'sim_failed',
                'D2_6d': float('nan'), 'D2_core1': float('nan'),
                'D2_core2': float('nan'), 'D2_4d': float('nan'),
                'error': str(e)}

    if not sim_result.success:
        return {'label': p.label, 'seed': seed, 'regime': 'solver_failed',
                'D2_6d': float('nan'), 'D2_core1': float('nan'),
                'D2_core2': float('nan'), 'D2_4d': float('nan')}

    t = sim_result.time
    c = sim_result.concentrations
    species = sim_result.species_names
    n_discard = len(t) // 2
    c_post = c[n_discard:]

    def idx(name):
        return species.index(name)

    try:
        X1 = c_post[:, idx('X1')]
        Y1 = c_post[:, idx('Y1')]
        X2 = c_post[:, idx('X2')]
        Y2 = c_post[:, idx('Y2')]
        E1 = c_post[:, idx('E1')]
        E2 = c_post[:, idx('E2')]
    except (ValueError, IndexError) as e:
        return {'label': p.label, 'seed': seed, 'regime': 'extraction_failed',
                'D2_6d': float('nan'), 'D2_core1': float('nan'),
                'D2_core2': float('nan'), 'D2_4d': float('nan')}

    # Check for fixed point
    cv_X1 = np.std(X1) / max(np.mean(X1), 1e-10)
    cv_X2 = np.std(X2) / max(np.mean(X2), 1e-10)

    if cv_X1 < 0.01 and cv_X2 < 0.01:
        return {'label': p.label, 'seed': seed, 'regime': 'fixed_point',
                'D2_6d': float('nan'), 'D2_core1': float('nan'),
                'D2_core2': float('nan'), 'D2_4d': float('nan'),
                'E1_mean': float(np.mean(E1)), 'E2_mean': float(np.mean(E2))}

    cd = CorrelationDimension()

    # D₂ on 6D combined: (X1, Y1, E1, X2, Y2, E2)
    traj_6d = np.column_stack([X1, Y1, E1, X2, Y2, E2])
    try:
        res_6d = cd.compute(traj_6d)
        D2_6d = float(res_6d.D2) if res_6d.D2 is not None else float('nan')
    except Exception:
        D2_6d = float('nan')

    # D₂ on 3D core 1: (X1, Y1, E1)
    traj_c1 = np.column_stack([X1, Y1, E1])
    try:
        res_c1 = cd.compute(traj_c1)
        D2_core1 = float(res_c1.D2) if res_c1.D2 is not None else float('nan')
    except Exception:
        D2_core1 = float('nan')

    # D₂ on 3D core 2: (X2, Y2, E2)
    traj_c2 = np.column_stack([X2, Y2, E2])
    try:
        res_c2 = cd.compute(traj_c2)
        D2_core2 = float(res_c2.D2) if res_c2.D2 is not None else float('nan')
    except Exception:
        D2_core2 = float('nan')

    # D₂ on 4D oscillators-only: (X1, Y1, X2, Y2) — comparable to shared model
    traj_4d = np.column_stack([X1, Y1, X2, Y2])
    try:
        res_4d = cd.compute(traj_4d)
        D2_4d = float(res_4d.D2) if res_4d.D2 is not None else float('nan')
    except Exception:
        D2_4d = float('nan')

    # Phase correlation
    r_X1X2 = float(np.corrcoef(X1, X2)[0, 1])

    # Regime classification (based on 6D D₂)
    if D2_6d > 1.2:
        regime = 'complex'
    elif np.isnan(D2_6d):
        regime = 'failed_d2'
    else:
        regime = 'phase_locked'

    return {
        'label': p.label,
        'seed': seed,
        'regime': regime,
        'D2_6d': D2_6d,
        'D2_core1': D2_core1,
        'D2_core2': D2_core2,
        'D2_4d': D2_4d,
        'r_X1X2': r_X1X2,
        'E1_mean': float(np.mean(E1)),
        'E2_mean': float(np.mean(E2)),
        'cv_X1': float(cv_X1),
        'cv_X2': float(cv_X2),
    }


def run_sweep_a2(
    n_seeds: int = 5,
    verbose: bool = True,
    save_dir: str | None = None,
) -> dict:
    """
    Sweep A2: Independent-reservoir causal control at γ=0.001.
    """
    print("\n" + "#" * 80)
    print("# SWEEP A2: Independent-Reservoir Causal Control (γ = 0.001)")
    print("#" * 80)

    J_values = [3.0, 5.0, 7.0, 10.0]
    gamma_values = [0.001]
    kcat_values = [0.1, 0.3, 0.5]

    K_ON = 10.0
    K_OFF = 10.0
    G_TOTAL = 1.0

    param_sets = []
    for J in J_values:
        for gamma in gamma_values:
            for k_cat in kcat_values:
                param_sets.append(EnzymeComplexParams(
                    J=J, gamma=gamma,
                    k_on=K_ON, k_off=K_OFF,
                    k_cat=k_cat, G_total=G_TOTAL,
                    label=f"J{J}_g{gamma}_kc{k_cat}",
                ))

    n_params = len(param_sets)
    n_total = n_params * n_seeds
    print(f"  Grid: {len(J_values)} J × {len(gamma_values)} γ × {len(kcat_values)} k_cat = {n_params} sets")
    print(f"  Seeds: {n_seeds}")
    print(f"  Total runs: {n_total}")
    print(f"  Model: INDEPENDENT reservoirs E₁/E₂ (decoupled cores)")
    print()

    results = []
    start = time.time()

    for i, p in enumerate(param_sets):
        for s in range(n_seeds):
            seed = 42 + s * 137
            r = simulate_independent(p, seed=seed, verbose=False)
            results.append(r)

            run_idx = i * n_seeds + s + 1
            elapsed = time.time() - start
            eta = max(0, elapsed / run_idx * (n_total - run_idx))

            if verbose and (run_idx % 5 == 0 or run_idx == n_total):
                d6 = f"{r['D2_6d']:.3f}" if not np.isnan(r['D2_6d']) else "N/A"
                dc1 = f"{r['D2_core1']:.3f}" if not np.isnan(r['D2_core1']) else "N/A"
                print(f"\r  {run_idx}/{n_total} ({p.label}, s{s+1}, "
                      f"D2_6d={d6}, D2_c1={dc1}) "
                      f"ETA: {eta:.0f}s    ", end='', flush=True)

    total_time = time.time() - start
    print(f"\n\n  Sweep A2 complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Summary ───────────────────────────────────────────────────────
    by_label = defaultdict(list)
    for r in results:
        by_label[r['label']].append(r)

    print(f"\n{'=' * 130}")
    print("SWEEP A2: Independent-Reservoir Results")
    print('=' * 130)
    print(f"{'Label':<22} {'J':>4} {'k_cat':>5}  "
          f"{'med D2_6d':>9} {'med D2_c1':>9} {'med D2_c2':>9} {'med D2_4d':>9}  "
          f"{'Cpx_6d':>6} {'r(X1,X2)':>9}  "
          f"{'E1_mean':>8}")
    print('-' * 130)

    # Also load Sweep A results for comparison
    sweep_a_path = os.path.join(_this_dir, 'results', 'phase1_sweep_a_results.json')
    sweep_a_lookup = {}
    if os.path.exists(sweep_a_path):
        with open(sweep_a_path) as f:
            sweep_a_data = json.load(f)
        for row in sweep_a_data.get('summary_rows', []):
            sweep_a_lookup[row['label']] = row

    summary_rows = []
    for label in sorted(by_label.keys(), key=lambda l: (
        float(l.split('_')[0][1:]),
        float(l.split('_')[2][2:]),
    )):
        runs = by_label[label]
        p_match = [p for p in param_sets if p.label == label][0]

        d2_6d = [r['D2_6d'] for r in runs if not np.isnan(r['D2_6d'])]
        d2_c1 = [r['D2_core1'] for r in runs if not np.isnan(r['D2_core1'])]
        d2_c2 = [r['D2_core2'] for r in runs if not np.isnan(r['D2_core2'])]
        d2_4d = [r['D2_4d'] for r in runs if not np.isnan(r['D2_4d'])]
        rs = [r['r_X1X2'] for r in runs if not np.isnan(r.get('r_X1X2', float('nan')))]
        es = [r.get('E1_mean', float('nan')) for r in runs]
        es = [e for e in es if not np.isnan(e)]

        med_6d = float(np.median(d2_6d)) if d2_6d else float('nan')
        med_c1 = float(np.median(d2_c1)) if d2_c1 else float('nan')
        med_c2 = float(np.median(d2_c2)) if d2_c2 else float('nan')
        med_4d = float(np.median(d2_4d)) if d2_4d else float('nan')
        med_r = float(np.median(rs)) if rs else float('nan')
        med_e = float(np.median(es)) if es else float('nan')
        n_cpx = sum(1 for r in runs if r['regime'] == 'complex')

        row = {
            'label': label, 'J': p_match.J, 'gamma': p_match.gamma, 'k_cat': p_match.k_cat,
            'med_D2_6d': med_6d, 'med_D2_core1': med_c1, 'med_D2_core2': med_c2,
            'med_D2_4d': med_4d, 'n_cpx_6d': n_cpx, 'med_r': med_r, 'med_e': med_e,
        }
        summary_rows.append(row)

        fmt = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A".rjust(5)

        # Sweep A comparison
        sa = sweep_a_lookup.get(label)
        sa_str = f"  (shared: {sa['med_d2']:.3f})" if sa else ""

        print(f"{label:<22} {p_match.J:>4.1f} {p_match.k_cat:>5.1f}  "
              f"{fmt(med_6d):>9} {fmt(med_c1):>9} {fmt(med_c2):>9} {fmt(med_4d):>9}  "
              f"{n_cpx:>6} {fmt(med_r):>9}  "
              f"{fmt(med_e):>8}{sa_str}")

    print('-' * 130)

    # ── Comparison table ──────────────────────────────────────────────
    print(f"\n  COMPARISON: Shared vs Independent reservoir (median D₂)")
    print(f"  {'Label':<22} {'Shared D₂':>10} {'Indep D₂_6d':>12} {'Indep D₂_c1':>12} {'Δ':>6}")
    print(f"  {'-'*65}")
    for row in summary_rows:
        sa = sweep_a_lookup.get(row['label'])
        if sa:
            shared = sa['med_d2']
            indep = row['med_D2_6d']
            delta = indep - shared if not (np.isnan(indep) or np.isnan(shared)) else float('nan')
            fmt_d = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A"
            print(f"  {row['label']:<22} {fmt_d(shared):>10} {fmt_d(indep):>12} "
                  f"{fmt_d(row['med_D2_core1']):>12} {fmt_d(delta):>6}")

    # ── Verdict ───────────────────────────────────────────────────────
    any_complex = any(r['n_cpx_6d'] > 0 for r in summary_rows)
    indep_d2s = [r['med_D2_6d'] for r in summary_rows if not np.isnan(r['med_D2_6d'])]
    max_indep = max(indep_d2s) if indep_d2s else float('nan')

    print(f"\n  Max independent D₂ (6D): {max_indep:.3f}" if not np.isnan(max_indep) else "  Max: N/A")

    if not any_complex:
        print(f"\n  VERDICT: D₂ > 1.2 DISAPPEARS with independent reservoirs.")
        print(f"  → Shared reservoir is CAUSALLY ESSENTIAL for dimensional inflation.")
        print(f"  → The 'slow shared resource coupling' narrative is confirmed.")
    else:
        cpx_labels = [r['label'] for r in summary_rows if r['n_cpx_6d'] > 0]
        print(f"\n  VERDICT: D₂ > 1.2 PERSISTS at {cpx_labels} with independent reservoirs.")
        print(f"  → Inflation may be intrinsic to each 3D subsystem.")
        print(f"  → The shared-reservoir claim needs revision.")

    print('=' * 130)

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        'sweep': 'A2',
        'description': 'Independent-reservoir causal control at gamma=0.001',
        'n_param_sets': n_params,
        'n_seeds': n_seeds,
        'n_total_runs': n_total,
        'runtime_seconds': total_time,
        'any_complex': any_complex,
        'max_D2_6d': float(max_indep) if not np.isnan(max_indep) else None,
        'summary_rows': summary_rows,
        'results': results,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'phase1_sweep_a2_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_sweep_a2(n_seeds=5, verbose=True, save_dir=save_dir)
