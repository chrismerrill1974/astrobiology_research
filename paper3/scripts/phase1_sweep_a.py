"""
Phase I Sweep A: Primary parameter scan for Paper 3.

Grid: J ∈ {3, 4, 5, 7, 10} × γ ∈ {0.001, 0.002, 0.003, 0.005} × k_cat ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
Fixed: K_d = 1.0, G_total = 1.0, k_on = k_off = 10.0
Seeds: 5 per parameter set (100 × 5 = 500 runs)
Integration: t_span=[0, 10000], 20000 points, 50% transient discard

Reuses the validated Pilot 5b infrastructure (EnzymeComplexParams, simulate_and_analyze).
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

from pilot.pilot5b_enzyme_complex import EnzymeComplexParams, simulate_and_analyze


def run_sweep_a(
    n_seeds: int = 5,
    verbose: bool = True,
    save_dir: str | None = None,
) -> dict:
    """
    Phase I Sweep A: J × γ × k_cat grid with fixed gate parameters.
    """
    print("\n" + "#" * 80)
    print("# PHASE I — SWEEP A: Primary Parameter Scan")
    print("#" * 80)

    # Grid definition (from paper3_plan.tex)
    J_values = [3.0, 4.0, 5.0, 7.0, 10.0]
    gamma_values = [0.001, 0.002, 0.003, 0.005]
    kcat_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Fixed gate parameters
    K_ON = 10.0
    K_OFF = 10.0  # K_d = 1.0
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
    print(f"  Fixed: K_d=1.0, G_total=1.0, k_on=k_off=10.0")
    print()

    # Run sweep
    results = []
    start = time.time()

    for i, p in enumerate(param_sets):
        for s in range(n_seeds):
            seed = 42 + s * 137
            r = simulate_and_analyze(p, seed=seed, verbose=False)
            results.append(r)

            run_idx = i * n_seeds + s + 1
            elapsed = time.time() - start
            eta = max(0, elapsed / run_idx * (n_total - run_idx))

            if verbose and (run_idx % 10 == 0 or run_idx == n_total):
                d2_str = f"{r['D2']:.3f}" if not np.isnan(r['D2']) else "N/A"
                print(f"\r  {run_idx}/{n_total} ({p.label}, s{s+1}, "
                      f"{r['regime']}, D2={d2_str}) "
                      f"ETA: {eta:.0f}s    ", end='', flush=True)

    total_time = time.time() - start
    print(f"\n\n  Sweep A complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Aggregate by parameter set ────────────────────────────────────
    by_label = defaultdict(list)
    for r in results:
        by_label[r['label']].append(r)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("PHASE I SWEEP A: Results Summary")
    print('=' * 120)
    print(f"{'Label':<22} {'J':>4} {'γ':>6} {'k_cat':>5}  "
          f"{'FP':>3} {'Lock':>4} {'Cpx':>4}  "
          f"{'med D2':>7} {'max D2':>7} {'min D2':>7} {'r(X1,X2)':>9}  "
          f"{'E_mean':>8} {'GE_mean':>8}")
    print('-' * 120)

    # Collect summary rows for later analysis
    summary_rows = []
    complex_labels = []
    complex_count = 0

    for label in sorted(by_label.keys(), key=lambda l: (
        float(l.split('_')[0][1:]),   # J
        float(l.split('_')[1][1:]),   # gamma
        float(l.split('_')[2][2:]),   # k_cat
    )):
        runs = by_label[label]
        p_match = [p for p in param_sets if p.label == label][0]

        regimes = [r['regime'] for r in runs]
        n_fp = sum(1 for rg in regimes if rg == 'fixed_point')
        n_lock = sum(1 for rg in regimes if rg == 'phase_locked')
        n_cpx = sum(1 for rg in regimes if rg == 'complex')
        n_fail = sum(1 for rg in regimes if rg in ('sim_failed', 'solver_failed', 'failed_d2', 'extraction_failed', 'd2_failed'))

        d2s = [r['D2'] for r in runs if not np.isnan(r['D2'])]
        rs = [r['r_X1X2'] for r in runs if not np.isnan(r['r_X1X2'])]
        es = [r.get('E_mean', float('nan')) for r in runs]
        es = [e for e in es if e is not None and not np.isnan(e)]
        ges = [r.get('GE_mean', float('nan')) for r in runs]
        ges = [g for g in ges if g is not None and not np.isnan(g)]

        med_d2 = float(np.median(d2s)) if d2s else float('nan')
        max_d2 = float(max(d2s)) if d2s else float('nan')
        min_d2 = float(min(d2s)) if d2s else float('nan')
        med_r = float(np.median(rs)) if rs else float('nan')
        med_e = float(np.median(es)) if es else float('nan')
        med_ge = float(np.median(ges)) if ges else float('nan')

        row = {
            'label': label, 'J': p_match.J, 'gamma': p_match.gamma, 'k_cat': p_match.k_cat,
            'n_fp': n_fp, 'n_lock': n_lock, 'n_cpx': n_cpx, 'n_fail': n_fail,
            'med_d2': med_d2, 'max_d2': max_d2, 'min_d2': min_d2,
            'med_r': med_r, 'med_e': med_e, 'med_ge': med_ge,
            'frac_complex': n_cpx / max(n_cpx + n_lock + n_fp, 1),
        }
        summary_rows.append(row)

        if n_cpx > 0:
            complex_labels.append(label)
            complex_count += n_cpx

        # Print row
        cpx_str = f"*{n_cpx}" if n_cpx > 0 else f"{n_cpx}"
        fail_str = f" [{n_fail}F]" if n_fail > 0 else ""
        fmt = lambda v, f: f.format(v) if not np.isnan(v) else "N/A".rjust(len(f.format(0.0)))

        print(f"{label:<22} {p_match.J:>4.1f} {p_match.gamma:>6.3f} {p_match.k_cat:>5.1f}  "
              f"{n_fp:>3} {n_lock:>4} {cpx_str:>4}  "
              f"{fmt(med_d2, '{:.3f}'):>7} {fmt(max_d2, '{:.3f}'):>7} {fmt(min_d2, '{:.3f}'):>7} "
              f"{fmt(med_r, '{:.3f}'):>9}  "
              f"{fmt(med_e, '{:.1f}'):>8} {fmt(med_ge, '{:.3f}'):>8}{fail_str}")

    print('-' * 120)

    # ── Overall statistics ────────────────────────────────────────────
    all_d2 = [r['D2'] for r in results if not np.isnan(r['D2'])]
    n_complex_total = sum(1 for r in results if r['regime'] == 'complex')
    n_success = sum(1 for r in results if r['regime'] not in
                    ('sim_failed', 'solver_failed', 'extraction_failed', 'd2_failed'))

    print(f"\n  Total runs: {len(results)}, successful: {n_success}")
    print(f"  Max D₂: {max(all_d2):.3f}" if all_d2 else "  Max D₂: N/A")
    print(f"  Runs with D₂ > 1.2: {n_complex_total} / {n_success}")
    print(f"  Parameter sets with any D₂ > 1.2: {len(complex_labels)} / {n_params}")

    # ── D₂ > 1.2 detail ──────────────────────────────────────────────
    if complex_labels:
        print(f"\n  D₂ > 1.2 found at:")
        for label in complex_labels:
            row = [r for r in summary_rows if r['label'] == label][0]
            print(f"    {label}: med={row['med_d2']:.3f}, max={row['max_d2']:.3f}, "
                  f"{row['n_cpx']}/{n_seeds} seeds complex, r={row['med_r']:.3f}")

    # ── Baseline D₂ statistics (for threshold calibration) ───────────
    lc_d2s = [r['D2'] for r in results
              if r['regime'] == 'phase_locked' and not np.isnan(r['D2'])]
    if lc_d2s:
        lc_mean = np.mean(lc_d2s)
        lc_std = np.std(lc_d2s)
        lc_n = len(lc_d2s)
        print(f"\n  Baseline D₂ (phase-locked limit cycles, N={lc_n}):")
        print(f"    Mean ± SD: {lc_mean:.4f} ± {lc_std:.4f}")
        print(f"    Threshold 1.2 is {(1.2 - lc_mean) / max(lc_std, 1e-6):.1f}σ above baseline")

    # ── Heatmap data (J × γ slices) ──────────────────────────────────
    print(f"\n  Median D₂ heatmap (J × γ, best k_cat at each cell):")
    print(f"  {'':>6}", end='')
    for g in gamma_values:
        print(f"  γ={g:<6}", end='')
    print()

    for J in J_values:
        print(f"  J={J:<4.0f}", end='')
        for g in gamma_values:
            # Best k_cat at this (J, γ)
            cell_rows = [r for r in summary_rows
                         if abs(r['J'] - J) < 0.01 and abs(r['gamma'] - g) < 0.0001]
            if cell_rows:
                best = max(cell_rows, key=lambda r: r['med_d2'] if not np.isnan(r['med_d2']) else -1)
                d2 = best['med_d2']
                kc = best['k_cat']
                if not np.isnan(d2):
                    marker = "*" if d2 > 1.2 else " "
                    print(f"  {d2:.2f}{marker}({kc:.1f})", end='')
                else:
                    print(f"  {'N/A':>10}", end='')
            else:
                print(f"  {'---':>10}", end='')
        print()

    print('=' * 120)

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        'sweep': 'A',
        'description': 'Phase I Sweep A: J × γ × k_cat primary scan',
        'grid': {
            'J': J_values, 'gamma': gamma_values, 'k_cat': kcat_values,
            'fixed': {'K_d': 1.0, 'G_total': 1.0, 'k_on': 10.0, 'k_off': 10.0},
        },
        'n_param_sets': n_params,
        'n_seeds': n_seeds,
        'n_total_runs': n_total,
        'runtime_seconds': total_time,
        'n_complex_runs': n_complex_total,
        'n_complex_param_sets': len(complex_labels),
        'complex_labels': complex_labels,
        'max_d2': float(max(all_d2)) if all_d2 else None,
        'baseline_d2_mean': float(lc_mean) if lc_d2s else None,
        'baseline_d2_std': float(lc_std) if lc_d2s else None,
        'baseline_d2_n': lc_n if lc_d2s else 0,
        'summary_rows': summary_rows,
        'results': results,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'phase1_sweep_a_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_sweep_a(n_seeds=5, verbose=True, save_dir=save_dir)
