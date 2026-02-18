"""
Sweep A2b: Independent-reservoir causal control at moderate γ.

The DECISIVE test for the shared-coupling thesis. Sweep A2 showed that
at γ=0.001 (very slow E), inflation persists with independent reservoirs.
Now test the moderate-γ desynchronized regime where D₂ reaches its
highest values (up to 1.67) and sharing may be causally essential.

Grid: J ∈ {4, 5} × γ ∈ {0.002, 0.003} × k_cat ∈ {0.2, 0.3}
= 8 parameter sets × 5 seeds = 40 runs

Model: Same independent-reservoir architecture as Sweep A2
(each core has its own E_i, G_i, GE_i; cores completely decoupled).
"""

import sys
import os
import json
import time
import numpy as np
from collections import defaultdict

_this_dir = os.path.dirname(os.path.abspath(__file__))
_astro2_path = os.path.join(_this_dir, '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from pilot.pilot5b_enzyme_complex import EnzymeComplexParams
from pilot.phase1_sweep_a2 import simulate_independent


def run_sweep_a2b(
    n_seeds: int = 5,
    verbose: bool = True,
    save_dir: str | None = None,
) -> dict:
    """
    Sweep A2b: Independent-reservoir causal control at moderate γ.
    """
    print("\n" + "#" * 80)
    print("# SWEEP A2b: Independent-Reservoir Causal Control (moderate γ)")
    print("#" * 80)

    J_values = [4.0, 5.0]
    gamma_values = [0.002, 0.003]
    kcat_values = [0.2, 0.3]

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
    print(f"\n\n  Sweep A2b complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Summary ───────────────────────────────────────────────────────
    by_label = defaultdict(list)
    for r in results:
        by_label[r['label']].append(r)

    print(f"\n{'=' * 130}")
    print("SWEEP A2b: Independent-Reservoir Results (moderate γ)")
    print('=' * 130)
    print(f"{'Label':<22} {'J':>4} {'γ':>6} {'k_cat':>5}  "
          f"{'med D2_6d':>9} {'med D2_c1':>9} {'med D2_c2':>9} {'med D2_4d':>9}  "
          f"{'Cpx_6d':>6} {'r(X1,X2)':>9}  "
          f"{'E1_mean':>8}")
    print('-' * 130)

    # Load Sweep A results for comparison
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
        float(l.split('_')[1][1:]),
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

        print(f"{label:<22} {p_match.J:>4.1f} {p_match.gamma:>6.3f} {p_match.k_cat:>5.1f}  "
              f"{fmt(med_6d):>9} {fmt(med_c1):>9} {fmt(med_c2):>9} {fmt(med_4d):>9}  "
              f"{n_cpx:>6} {fmt(med_r):>9}  "
              f"{fmt(med_e):>8}{sa_str}")

    print('-' * 130)

    # ── Comparison table ──────────────────────────────────────────────
    print(f"\n  COMPARISON: Shared vs Independent reservoir at moderate γ (median D₂)")
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
        else:
            fmt_d = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A"
            print(f"  {row['label']:<22} {'(no data)':>10} {fmt_d(row['med_D2_6d']):>12} "
                  f"{fmt_d(row['med_D2_core1']):>12} {'N/A':>6}")

    # ── Verdict ───────────────────────────────────────────────────────
    any_complex = any(r['n_cpx_6d'] > 0 for r in summary_rows)
    indep_d2s = [r['med_D2_6d'] for r in summary_rows if not np.isnan(r['med_D2_6d'])]
    max_indep = max(indep_d2s) if indep_d2s else float('nan')

    # Points where shared had D₂ > 1.2
    shared_complex_labels = [row['label'] for row in summary_rows
                             if row['label'] in sweep_a_lookup
                             and sweep_a_lookup[row['label']]['med_d2'] > 1.2]

    print(f"\n  Max independent D₂ (6D): {max_indep:.3f}" if not np.isnan(max_indep) else "  Max: N/A")
    print(f"  Shared-reservoir D₂ > 1.2 at: {shared_complex_labels if shared_complex_labels else 'none'}")

    if not any_complex:
        print(f"\n  VERDICT: D₂ > 1.2 DISAPPEARS at moderate γ with independent reservoirs.")
        print(f"  → At moderate γ, the SHARED reservoir IS causally essential.")
        print(f"  → The thesis survives for the desynchronized regime:")
        print(f"    'Slow energy alone inflates D₂ to ~1.4; shared slow energy")
        print(f"     enables qualitatively richer inflation (D₂ up to 1.67)")
        print(f"     through inter-oscillator desynchronization.'")
    else:
        cpx_labels = [r['label'] for r in summary_rows if r['n_cpx_6d'] > 0]
        print(f"\n  VERDICT: D₂ > 1.2 PERSISTS at moderate γ with independent reservoirs.")
        print(f"  → Complex at: {cpx_labels}")
        print(f"  → The entire inflation phenomenon is intrinsic to each subsystem.")
        print(f"  → Thesis: 'Timescale separation is the sole mechanism.'")

    print('=' * 130)

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        'sweep': 'A2b',
        'description': 'Independent-reservoir causal control at moderate gamma (0.002-0.003)',
        'grid': {
            'J': J_values, 'gamma': gamma_values, 'k_cat': kcat_values,
        },
        'n_param_sets': n_params,
        'n_seeds': n_seeds,
        'n_total_runs': n_total,
        'runtime_seconds': total_time,
        'any_complex': any_complex,
        'max_D2_6d': float(max_indep) if not np.isnan(max_indep) else None,
        'shared_complex_labels': shared_complex_labels,
        'summary_rows': summary_rows,
        'results': results,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'phase1_sweep_a2b_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_sweep_a2b(n_seeds=5, verbose=True, save_dir=save_dir)
