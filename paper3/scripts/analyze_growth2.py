"""
Extended analysis: survival rates and proper Fisher tests on failure vs survival.
The key signal is that aligned growth dramatically reduces catastrophic failure.
"""
import json
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact, mannwhitneyu

def load_checkpoint(path):
    with open(path) as f:
        return json.load(f)['results']

base = '/Users/chris/Documents/systems_analysis/astrobiology3/pilot/results'
all_results = load_checkpoint(f'{base}/phase2_growth_primary_checkpoint.json') + \
              load_checkpoint(f'{base}/phase2_growth_replication1_checkpoint.json')

print("="*80)
print("  SURVIVAL ANALYSIS: Aligned vs Random growth")
print("  (Combined: primary N=50 + replication1 N=30 = 80 trajectories)")
print("="*80)

groups = defaultdict(list)
for r in all_results:
    groups[(r['rule'], r['k'])].append(r)

print(f"\n  {'k':>3}  {'Rule':<10}  {'Survived':>10}  {'Failed':>10}  {'Survival%':>10}  {'Fisher p':>10}  {'OR':>10}")
print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

for k in range(6):
    rand = groups[('random', k)]
    alig = groups[('aligned', k)]

    # "survived" = got a valid D2 (not solver/sim failure)
    rand_surv = sum(1 for r in rand if r['regime'] not in ('solver_failed', 'sim_failed', 'failed_d2', 'fixed_point'))
    rand_fail = len(rand) - rand_surv
    alig_surv = sum(1 for r in alig if r['regime'] not in ('solver_failed', 'sim_failed', 'failed_d2', 'fixed_point'))
    alig_fail = len(alig) - alig_surv

    table = [[rand_surv, rand_fail], [alig_surv, alig_fail]]
    odds, p = fisher_exact(table)

    rand_pct = 100 * rand_surv / len(rand) if rand else 0
    alig_pct = 100 * alig_surv / len(alig) if alig else 0

    print(f"  {k:>3}  {'random':<10}  {rand_surv:>7}/{len(rand):<2}  {rand_fail:>7}/{len(rand):<2}  {rand_pct:>8.1f}%")
    print(f"  {k:>3}  {'aligned':<10}  {alig_surv:>7}/{len(alig):<2}  {alig_fail:>7}/{len(alig):<2}  {alig_pct:>8.1f}%  {p:>10.4f}  {odds:>10.3f}")
    print()

# η dilution analysis
print("\n" + "="*80)
print("  η DILUTION: eta decreases with k (prediction 1)")
print("="*80)

for rule in ['random', 'aligned']:
    print(f"\n  {rule.upper()}:")
    for k in range(6):
        g = groups[(rule, k)]
        eta_vals = [r['eta_projected'] for r in g if r['eta_projected'] is not None and not np.isnan(r['eta_projected'])]
        if eta_vals:
            print(f"    k={k}: η = {np.mean(eta_vals):.4f} ± {np.std(eta_vals):.4f} (N={len(eta_vals)})")
        else:
            print(f"    k={k}: no valid η values")

# D2 ceiling analysis
print("\n" + "="*80)
print("  D2 CEILING: max D2 values (prediction 4: ceiling ~2.0)")
print("="*80)

for rule in ['random', 'aligned']:
    print(f"\n  {rule.upper()}:")
    for k in range(6):
        g = groups[(rule, k)]
        d2_vals = [r['D2_projected'] for r in g if r['D2_projected'] is not None and not np.isnan(r['D2_projected'])]
        if d2_vals:
            print(f"    k={k}: max D2 = {max(d2_vals):.3f}, 90th pct = {np.percentile(d2_vals, 90):.3f}")

# Oscillation survival
print("\n" + "="*80)
print("  OSCILLATION SURVIVAL: aligned vs random")
print("="*80)

print(f"\n  {'k':>3}  {'random_osc':>12}  {'aligned_osc':>12}  {'Fisher_p':>10}")
for k in range(6):
    rand = groups[('random', k)]
    alig = groups[('aligned', k)]

    rand_osc = sum(1 for r in rand if r.get('oscillation_survives') in ('True', True))
    alig_osc = sum(1 for r in alig if r.get('oscillation_survives') in ('True', True))

    table = [[rand_osc, len(rand)-rand_osc], [alig_osc, len(alig)-alig_osc]]
    odds, p = fisher_exact(table)

    print(f"  {k:>3}  {rand_osc:>8}/{len(rand):<3}  {alig_osc:>8}/{len(alig):<3}  {p:>10.4f}")
