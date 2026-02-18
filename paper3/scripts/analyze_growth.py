"""
Quick analysis of Phase II Growth Experiment results.
Reads checkpoint JSONs and produces summary statistics.
"""

import json
import numpy as np
from collections import defaultdict

def load_checkpoint(path):
    with open(path) as f:
        data = json.load(f)
    return data['results']

def analyze(results, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  N results: {len(results)}")
    print(f"{'='*80}")

    # Group by (rule, k)
    groups = defaultdict(list)
    for r in results:
        groups[(r['rule'], r['k'])].append(r)

    # Print header
    print(f"\n  {'Rule':<10} {'k':>3}  {'N':>3}  {'D2_mean':>8}  {'D2_med':>8}  "
          f"{'D2>1.2':>6}  {'complex':>7}  {'locked':>7}  {'failed':>7}  "
          f"{'eta_mean':>8}  {'osc_surv':>8}")
    print(f"  {'-'*10} {'-'*3}  {'-'*3}  {'-'*8}  {'-'*8}  "
          f"{'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*8}  {'-'*8}")

    summary_rows = []

    for rule in ['random', 'aligned']:
        for k in range(6):
            key = (rule, k)
            if key not in groups:
                continue
            g = groups[key]
            n = len(g)

            d2_vals = [r['D2_projected'] for r in g if r['D2_projected'] is not None and not np.isnan(r['D2_projected'])]
            eta_vals = [r['eta_projected'] for r in g if r['eta_projected'] is not None and not np.isnan(r['eta_projected'])]

            regimes = [r['regime'] for r in g]
            n_complex = sum(1 for r in regimes if r == 'complex')
            n_locked = sum(1 for r in regimes if r == 'phase_locked')
            n_failed = sum(1 for r in regimes if r in ('solver_failed', 'sim_failed', 'failed_d2', 'fixed_point'))

            osc_surv = [r.get('oscillation_survives') for r in g]
            n_osc_true = sum(1 for o in osc_surv if o == 'True' or o is True)

            d2_above = sum(1 for d in d2_vals if d > 1.2)

            d2_mean = np.mean(d2_vals) if d2_vals else float('nan')
            d2_med = np.median(d2_vals) if d2_vals else float('nan')
            eta_mean = np.mean(eta_vals) if eta_vals else float('nan')

            print(f"  {rule:<10} {k:>3}  {n:>3}  {d2_mean:>8.3f}  {d2_med:>8.3f}  "
                  f"{d2_above:>4}/{len(d2_vals):<2} {n_complex:>5}/{n:<2} {n_locked:>5}/{n:<2} {n_failed:>5}/{n:<2} "
                  f"{eta_mean:>8.4f}  {n_osc_true:>5}/{n:<2}")

            summary_rows.append({
                'rule': rule, 'k': k, 'n': n,
                'd2_mean': d2_mean, 'd2_med': d2_med,
                'd2_above_1.2': d2_above, 'd2_valid': len(d2_vals),
                'n_complex': n_complex, 'n_locked': n_locked, 'n_failed': n_failed,
                'eta_mean': eta_mean, 'n_osc_survives': n_osc_true
            })

    # Key comparisons
    print(f"\n  --- Key comparisons ---")

    for k in range(1, 6):
        rand_g = groups.get(('random', k), [])
        alig_g = groups.get(('aligned', k), [])
        if not rand_g or not alig_g:
            continue

        rand_d2 = [r['D2_projected'] for r in rand_g if r['D2_projected'] is not None and not np.isnan(r['D2_projected'])]
        alig_d2 = [r['D2_projected'] for r in alig_g if r['D2_projected'] is not None and not np.isnan(r['D2_projected'])]

        rand_above = sum(1 for d in rand_d2 if d > 1.2) if rand_d2 else 0
        alig_above = sum(1 for d in alig_d2 if d > 1.2) if alig_d2 else 0

        rand_fail = sum(1 for r in rand_g if r['regime'] in ('solver_failed', 'sim_failed', 'failed_d2', 'fixed_point'))
        alig_fail = sum(1 for r in alig_g if r['regime'] in ('solver_failed', 'sim_failed', 'failed_d2', 'fixed_point'))

        print(f"  k={k}: Random D2>1.2: {rand_above}/{len(rand_d2)} valid ({len(rand_g)} total, {rand_fail} failed) | "
              f"Aligned D2>1.2: {alig_above}/{len(alig_d2)} valid ({len(alig_g)} total, {alig_fail} failed)")

    # Fisher exact test for k=1 (most informative)
    try:
        from scipy.stats import fisher_exact
        for k in [1, 2, 3]:
            rand_g = groups.get(('random', k), [])
            alig_g = groups.get(('aligned', k), [])
            if not rand_g or not alig_g:
                continue

            # Compare survival (complex regime) vs failure/collapse
            rand_complex = sum(1 for r in rand_g if r['regime'] == 'complex')
            rand_other = len(rand_g) - rand_complex
            alig_complex = sum(1 for r in alig_g if r['regime'] == 'complex')
            alig_other = len(alig_g) - alig_complex

            table = [[rand_complex, rand_other], [alig_complex, alig_other]]
            odds, p = fisher_exact(table)
            print(f"  Fisher exact k={k} (complex vs other): OR={odds:.3f}, p={p:.4f}")
            print(f"    Random: {rand_complex}/{len(rand_g)} complex, Aligned: {alig_complex}/{len(alig_g)} complex")
    except ImportError:
        print("  (scipy not available for Fisher test)")

    return summary_rows


if __name__ == '__main__':
    base = '/Users/chris/Documents/systems_analysis/astrobiology3/pilot/results'

    primary = load_checkpoint(f'{base}/phase2_growth_primary_checkpoint.json')
    rep1 = load_checkpoint(f'{base}/phase2_growth_replication1_checkpoint.json')

    s1 = analyze(primary, "PRIMARY (J=5, γ=0.002, k_cat=0.3)")
    s2 = analyze(rep1, "REPLICATION 1 (J=4, γ=0.003, k_cat=0.2)")

    # Combined analysis
    all_results = primary + rep1
    analyze(all_results, "COMBINED (primary + replication1)")

    # Baseline D2 summary
    baseline = [r for r in all_results if r['k'] == 0]
    d2_base = [r['D2_projected'] for r in baseline if r['D2_projected'] is not None and not np.isnan(r['D2_projected'])]
    print(f"\n\n  Baseline (k=0) D2: mean={np.mean(d2_base):.3f}, std={np.std(d2_base):.3f}, "
          f"min={np.min(d2_base):.3f}, max={np.max(d2_base):.3f}, N={len(d2_base)}")
