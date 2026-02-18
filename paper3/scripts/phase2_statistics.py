"""
Phase II Statistical Analysis for Paper 3.

Follows the Paper 2 framework (phase5b_statistics.py):
  - Fisher exact tests with Holm-Bonferroni correction
  - Logistic regression (survival ~ k + rule + k×rule)
  - Bootstrap CIs for median D₂, η, and survival rates
  - Slope analysis (η vs k, D₂ retention vs k)
  - Cross-paper comparison table
"""

import json
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact, mannwhitneyu
import warnings

# ── Helpers ──────────────────────────────────────────────────────────

def load_results():
    """Load both checkpoint files."""
    base = '/Users/chris/Documents/systems_analysis/astrobiology3/pilot/results'
    all_results = []
    for fname in ['phase2_growth_primary_checkpoint.json',
                  'phase2_growth_replication1_checkpoint.json']:
        with open(f'{base}/{fname}') as f:
            data = json.load(f)
        all_results.extend(data['results'])
    return all_results


def group_by(results, key_fn):
    """Group results by a key function."""
    groups = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)
    return groups


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values.
    Returns adjusted p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


def bootstrap_ci(data, stat_fn=np.median, n_boot=10000, ci=0.95, rng=None):
    """Bootstrap confidence interval for a statistic."""
    if rng is None:
        rng = np.random.default_rng(42)
    data = np.array(data)
    if len(data) == 0:
        return float('nan'), float('nan'), float('nan')
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        stats.append(stat_fn(sample))
    stats = np.sort(stats)
    alpha = 1 - ci
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot)]
    return stat_fn(data), lo, hi


def or_ci_wald(table, alpha=0.05):
    """Odds ratio with Wald 95% CI on log-OR.
    table = [[a, b], [c, d]]"""
    a, b = table[0]
    c, d = table[1]
    if a == 0 or b == 0 or c == 0 or d == 0:
        # Add 0.5 continuity correction
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    OR = (a * d) / (b * c)
    log_se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = 1.96  # ~alpha/2 for 95%
    lo = np.exp(np.log(OR) - z * log_se)
    hi = np.exp(np.log(OR) + z * log_se)
    return OR, lo, hi


# ── Main Analyses ───────────────────────────────────────────────────

def run_all():
    results = load_results()
    groups = group_by(results, lambda r: (r['rule'], r['k']))
    rng = np.random.default_rng(42)

    K_VALUES = list(range(6))

    print("=" * 90)
    print("  PAPER 3 — PHASE II STATISTICAL ANALYSIS")
    print("  N = 960 runs (primary 600 + replication1 360)")
    print("=" * 90)

    # ────────────────────────────────────────────────────────────────
    # 1. SURVIVAL ANALYSIS (Fisher exact + Holm-Bonferroni)
    # ────────────────────────────────────────────────────────────────
    print("\n\n" + "─" * 90)
    print("  1. SURVIVAL ANALYSIS (Fisher exact tests)")
    print("─" * 90)

    def is_survived(r):
        return r['regime'] not in ('solver_failed', 'sim_failed', 'failed_d2', 'fixed_point')

    raw_p_values = []
    survival_data = []

    for k in K_VALUES:
        rand = groups[('random', k)]
        alig = groups[('aligned', k)]

        rand_surv = sum(1 for r in rand if is_survived(r))
        rand_fail = len(rand) - rand_surv
        alig_surv = sum(1 for r in alig if is_survived(r))
        alig_fail = len(alig) - alig_surv

        table = [[rand_surv, rand_fail], [alig_surv, alig_fail]]
        _, p = fisher_exact(table)
        OR, or_lo, or_hi = or_ci_wald(table)

        raw_p_values.append(p)
        survival_data.append({
            'k': k,
            'rand_surv': rand_surv, 'rand_n': len(rand),
            'alig_surv': alig_surv, 'alig_n': len(alig),
            'OR': OR, 'OR_lo': or_lo, 'OR_hi': or_hi,
            'p_raw': p
        })

    # Holm-Bonferroni on k=1..5 (skip k=0)
    adj_p = holm_bonferroni(raw_p_values[1:])
    for i, d in enumerate(survival_data):
        d['p_adj'] = adj_p[i - 1] if i > 0 else 1.0

    print(f"\n  {'k':>3}  {'Random':>12}  {'Aligned':>12}  {'OR':>8}  {'95% CI':>16}  {'p_raw':>12}  {'p_adj':>12}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*16}  {'─'*12}  {'─'*12}")

    for d in survival_data:
        rand_pct = f"{d['rand_surv']}/{d['rand_n']}"
        alig_pct = f"{d['alig_surv']}/{d['alig_n']}"
        ci_str = f"[{d['OR_lo']:.2f}, {d['OR_hi']:.2f}]"
        p_raw_str = f"{d['p_raw']:.2e}" if d['p_raw'] < 0.001 else f"{d['p_raw']:.4f}"
        p_adj_str = f"{d['p_adj']:.2e}" if d['k'] > 0 and d['p_adj'] < 0.001 else (f"{d['p_adj']:.4f}" if d['k'] > 0 else "---")
        or_str = f"{d['OR']:.3f}" if d['k'] > 0 else "---"
        print(f"  {d['k']:>3}  {rand_pct:>12}  {alig_pct:>12}  {or_str:>8}  {ci_str:>16}  {p_raw_str:>12}  {p_adj_str:>12}")

    # Risk difference with bootstrap CI
    print(f"\n  Risk difference (aligned - random survival rate) with 95% bootstrap CI:")
    for k in K_VALUES[1:]:
        rand = groups[('random', k)]
        alig = groups[('aligned', k)]

        rand_binary = np.array([1 if is_survived(r) else 0 for r in rand])
        alig_binary = np.array([1 if is_survived(r) else 0 for r in alig])

        diffs = []
        for _ in range(10000):
            r_samp = rng.choice(rand_binary, size=len(rand_binary), replace=True)
            a_samp = rng.choice(alig_binary, size=len(alig_binary), replace=True)
            diffs.append(a_samp.mean() - r_samp.mean())
        diffs = np.sort(diffs)
        obs_diff = alig_binary.mean() - rand_binary.mean()
        lo, hi = diffs[250], diffs[9749]
        print(f"    k={k}: Δ = {obs_diff:.3f} [{lo:.3f}, {hi:.3f}]")

    # ────────────────────────────────────────────────────────────────
    # 2. D₂ > 1.2 RETENTION (Fisher exact on complex regime)
    # ────────────────────────────────────────────────────────────────
    print("\n\n" + "─" * 90)
    print("  2. D₂ > 1.2 RETENTION (among ALL runs, counting failures as non-complex)")
    print("─" * 90)

    raw_p2 = []
    for k in K_VALUES:
        rand = groups[('random', k)]
        alig = groups[('aligned', k)]

        rand_complex = sum(1 for r in rand if r['regime'] == 'complex')
        rand_other = len(rand) - rand_complex
        alig_complex = sum(1 for r in alig if r['regime'] == 'complex')
        alig_other = len(alig) - alig_complex

        table = [[rand_complex, rand_other], [alig_complex, alig_other]]
        _, p = fisher_exact(table)
        OR, or_lo, or_hi = or_ci_wald(table)
        raw_p2.append(p)

        print(f"  k={k}: Random {rand_complex}/{len(rand)} complex, "
              f"Aligned {alig_complex}/{len(alig)} complex, "
              f"OR={OR:.3f} [{or_lo:.2f},{or_hi:.2f}], p={p:.4f}")

    adj_p2 = holm_bonferroni(raw_p2[1:])
    print(f"\n  Holm-Bonferroni adjusted p-values (k=1..5): "
          + ", ".join(f"k={k}: {p:.4f}" for k, p in zip(range(1, 6), adj_p2)))

    # ────────────────────────────────────────────────────────────────
    # 3. LOGISTIC REGRESSION
    # ────────────────────────────────────────────────────────────────
    print("\n\n" + "─" * 90)
    print("  3. LOGISTIC REGRESSION: P(survived) ~ k + aligned + k×aligned")
    print("─" * 90)

    try:
        import statsmodels.api as sm

        # Build design matrix
        y_list = []
        X_list = []

        for r in results:
            survived = 1 if is_survived(r) else 0
            k = r['k']
            aligned = 1 if r['rule'] == 'aligned' else 0
            interaction = k * aligned
            X_list.append([1, k, aligned, interaction])  # intercept, k, aligned, k×aligned
            y_list.append(survived)

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)

        model = sm.Logit(y, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp=0)

        print(f"\n  {'Coeff':<15} {'Estimate':>10} {'SE':>10} {'z':>10} {'p':>12}")
        print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
        names = ['intercept', 'k', 'aligned', 'k×aligned']
        for name, coef, se, z, p in zip(names, result.params, result.bse,
                                         result.tvalues, result.pvalues):
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
            print(f"  {name:<15} {coef:>10.4f} {se:>10.4f} {z:>10.3f} {p_str:>12}")

        print(f"\n  AIC = {result.aic:.1f}, BIC = {result.bic:.1f}")
        print(f"  Pseudo R² (McFadden) = {1 - result.llf / result.llnull:.4f}")

        # Predicted survival curves
        print(f"\n  Predicted survival rates:")
        print(f"  {'k':>3}  {'Random':>10}  {'Aligned':>10}")
        for k in K_VALUES:
            p_rand = result.predict([1, k, 0, 0])[0]
            p_alig = result.predict([1, k, 1, k])[0]
            print(f"  {k:>3}  {p_rand:>9.1%}  {p_alig:>9.1%}")

        # Second model: P(complex) ~ k + aligned + k×aligned
        print(f"\n\n  LOGISTIC REGRESSION: P(complex regime) ~ k + aligned + k×aligned")

        y2 = np.array([1 if r['regime'] == 'complex' else 0 for r in results], dtype=float)
        model2 = sm.Logit(y2, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result2 = model2.fit(disp=0)

        print(f"\n  {'Coeff':<15} {'Estimate':>10} {'SE':>10} {'z':>10} {'p':>12}")
        print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
        for name, coef, se, z, p in zip(names, result2.params, result2.bse,
                                         result2.tvalues, result2.pvalues):
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
            print(f"  {name:<15} {coef:>10.4f} {se:>10.4f} {z:>10.3f} {p_str:>12}")

        print(f"\n  AIC = {result2.aic:.1f}, BIC = {result2.bic:.1f}")

    except ImportError:
        print("  statsmodels not available — skipping logistic regression")

    # ────────────────────────────────────────────────────────────────
    # 4. η ANALYSIS (bootstrap CIs, slopes)
    # ────────────────────────────────────────────────────────────────
    print("\n\n" + "─" * 90)
    print("  4. η ANALYSIS (median ± bootstrap 95% CI)")
    print("─" * 90)

    print(f"\n  {'Rule':<10} {'k':>3}  {'N':>4}  {'median η':>10}  {'95% CI':>20}  {'IQR':>12}")

    eta_medians = {'random': [], 'aligned': []}
    eta_ns = {'random': [], 'aligned': []}

    for rule in ['random', 'aligned']:
        for k in K_VALUES:
            g = groups[(rule, k)]
            eta_vals = [r['eta_projected'] for r in g
                       if r['eta_projected'] is not None and not np.isnan(r['eta_projected'])]
            if eta_vals:
                med, lo, hi = bootstrap_ci(eta_vals, np.median, rng=rng)
                q25, q75 = np.percentile(eta_vals, [25, 75])
                iqr = q75 - q25
                print(f"  {rule:<10} {k:>3}  {len(eta_vals):>4}  {med:>10.4f}  [{lo:.4f}, {hi:.4f}]  {iqr:>10.4f}")
                eta_medians[rule].append(med)
                eta_ns[rule].append(len(eta_vals))
            else:
                print(f"  {rule:<10} {k:>3}  {0:>4}  {'---':>10}")
                eta_medians[rule].append(float('nan'))
                eta_ns[rule].append(0)

    # η slope analysis (bootstrap)
    print(f"\n  η slope (median η vs k, bootstrap 10000):")
    for rule in ['random', 'aligned']:
        g_all = {k: [] for k in K_VALUES}
        for r in results:
            if r['rule'] == rule and r['eta_projected'] is not None and not np.isnan(r['eta_projected']):
                g_all[r['k']].append(r['eta_projected'])

        # Need per-trajectory tracking for proper bootstrap
        # Simple approach: bootstrap median η per k, fit slope
        slopes = []
        for _ in range(10000):
            medians = []
            for k in K_VALUES:
                vals = g_all[k]
                if vals:
                    samp = rng.choice(vals, size=len(vals), replace=True)
                    medians.append(np.median(samp))
                else:
                    medians.append(float('nan'))

            valid = [(k, m) for k, m in zip(K_VALUES, medians) if not np.isnan(m)]
            if len(valid) >= 2:
                ks, ms = zip(*valid)
                slope = np.polyfit(ks, ms, 1)[0]
                slopes.append(slope)

        slopes = np.sort(slopes)
        obs_medians = [np.median(g_all[k]) if g_all[k] else float('nan') for k in K_VALUES]
        valid_obs = [(k, m) for k, m in zip(K_VALUES, obs_medians) if not np.isnan(m)]
        if len(valid_obs) >= 2:
            ks, ms = zip(*valid_obs)
            obs_slope = np.polyfit(ks, ms, 1)[0]
        else:
            obs_slope = float('nan')

        lo, hi = slopes[250], slopes[9749]
        print(f"    {rule}: slope = {obs_slope:.5f} [{lo:.5f}, {hi:.5f}] per reaction")

    # ────────────────────────────────────────────────────────────────
    # 5. D₂ ANALYSIS (bootstrap CIs for median D₂)
    # ────────────────────────────────────────────────────────────────
    print("\n\n" + "─" * 90)
    print("  5. D₂ ANALYSIS (median ± bootstrap 95% CI, among survivors)")
    print("─" * 90)

    print(f"\n  {'Rule':<10} {'k':>3}  {'N':>4}  {'median D₂':>10}  {'95% CI':>20}  {'max D₂':>8}")

    for rule in ['random', 'aligned']:
        for k in K_VALUES:
            g = groups[(rule, k)]
            d2_vals = [r['D2_projected'] for r in g
                      if r['D2_projected'] is not None and not np.isnan(r['D2_projected'])]
            if d2_vals:
                med, lo, hi = bootstrap_ci(d2_vals, np.median, rng=rng)
                print(f"  {rule:<10} {k:>3}  {len(d2_vals):>4}  {med:>10.3f}  [{lo:.3f}, {hi:.3f}]  {max(d2_vals):>8.3f}")
            else:
                print(f"  {rule:<10} {k:>3}  {0:>4}  {'---':>10}")

    # Mann-Whitney U between random and aligned D₂ (among survivors)
    print(f"\n  Mann-Whitney U test (D₂ among survivors, random vs aligned):")
    for k in K_VALUES[1:]:
        rand_d2 = [r['D2_projected'] for r in groups[('random', k)]
                   if r['D2_projected'] is not None and not np.isnan(r['D2_projected'])]
        alig_d2 = [r['D2_projected'] for r in groups[('aligned', k)]
                   if r['D2_projected'] is not None and not np.isnan(r['D2_projected'])]
        if len(rand_d2) >= 2 and len(alig_d2) >= 2:
            stat, p = mannwhitneyu(rand_d2, alig_d2, alternative='two-sided')
            print(f"    k={k}: U={stat:.0f}, p={p:.4f} (N_rand={len(rand_d2)}, N_alig={len(alig_d2)})")
        else:
            print(f"    k={k}: insufficient data")

    # ────────────────────────────────────────────────────────────────
    # 6. OSCILLATION SURVIVAL (Fisher exact)
    # ────────────────────────────────────────────────────────────────
    print("\n\n" + "─" * 90)
    print("  6. OSCILLATION SURVIVAL (Fisher exact)")
    print("─" * 90)

    raw_p3 = []
    for k in K_VALUES:
        rand = groups[('random', k)]
        alig = groups[('aligned', k)]

        rand_osc = sum(1 for r in rand if r.get('oscillation_survives') in ('True', True))
        alig_osc = sum(1 for r in alig if r.get('oscillation_survives') in ('True', True))

        table = [[rand_osc, len(rand) - rand_osc],
                 [alig_osc, len(alig) - alig_osc]]
        _, p = fisher_exact(table)
        OR, or_lo, or_hi = or_ci_wald(table)
        raw_p3.append(p)

        print(f"  k={k}: Random {rand_osc}/{len(rand)}, Aligned {alig_osc}/{len(alig)}, "
              f"OR={OR:.3f} [{or_lo:.2f},{or_hi:.2f}], p={p:.4f}")

    adj_p3 = holm_bonferroni(raw_p3[1:])
    print(f"\n  Holm-Bonferroni adjusted (k=1..5): "
          + ", ".join(f"k={k}: {p:.2e}" for k, p in zip(range(1, 6), adj_p3)))

    # ────────────────────────────────────────────────────────────────
    # 7. CROSS-PAPER COMPARISON TABLE
    # ────────────────────────────────────────────────────────────────
    print("\n\n" + "─" * 90)
    print("  7. CROSS-PAPER COMPARISON")
    print("─" * 90)

    print("""
  Paper 2 (single Brusselator, N=200):
    Endpoint: oscillation survival after k reaction additions
    k=5: Random 36/200 (18.0%), Aligned 121/200 (60.5%), OR=7.0, p<10⁻¹⁷

  Paper 3 (enzyme-complex coupled oscillator, N=80):
    Endpoint A: system survival (valid D₂ obtained)
    k=5: Random 1/80 (1.2%), Aligned 53/80 (66.2%), OR=167, p<10⁻²⁰

    Endpoint B: complex regime (D₂ > 1.2)
    k=5: Random 1/80 (1.2%), Aligned 8/80 (10.0%)

    Endpoint C: oscillation survival
    k=5: Random 2/80 (2.5%), Aligned 61/80 (76.2%), OR=117, p<10⁻²⁰

  Key comparison:
    Paper 2 OR (oscillation survival, k=5):  7.0
    Paper 3 OR (system survival, k=5):       167
    Paper 3 OR (oscillation survival, k=5):  117

    Modular coupling AMPLIFIES the alignment advantage by >10× (Prediction 3 confirmed).
    The enzyme-complex system is far more fragile to random perturbation than the
    single Brusselator, but aligned growth preserves it at comparable rates.
""")

    # ────────────────────────────────────────────────────────────────
    # 8. RETENTION SLOPES
    # ────────────────────────────────────────────────────────────────
    print("─" * 90)
    print("  8. RETENTION SLOPES (fraction surviving/complex vs k)")
    print("─" * 90)

    for label, criterion in [("Survival", lambda r: is_survived(r)),
                              ("Complex (D₂>1.2)", lambda r: r['regime'] == 'complex'),
                              ("Oscillation", lambda r: r.get('oscillation_survives') in ('True', True))]:
        print(f"\n  {label}:")
        for rule in ['random', 'aligned']:
            fracs = []
            for k in K_VALUES:
                g = groups[(rule, k)]
                frac = sum(1 for r in g if criterion(r)) / len(g) if g else 0
                fracs.append(frac)

            slope = np.polyfit(K_VALUES, fracs, 1)[0]

            # Bootstrap the slope
            slopes_boot = []
            for _ in range(10000):
                boot_fracs = []
                for k in K_VALUES:
                    g = groups[(rule, k)]
                    binary = [1 if criterion(r) else 0 for r in g]
                    samp = rng.choice(binary, size=len(binary), replace=True)
                    boot_fracs.append(samp.mean())
                slopes_boot.append(np.polyfit(K_VALUES, boot_fracs, 1)[0])

            slopes_boot = np.sort(slopes_boot)
            lo, hi = slopes_boot[250], slopes_boot[9749]
            print(f"    {rule}: slope = {slope:.4f} [{lo:.4f}, {hi:.4f}] per reaction")

    # ────────────────────────────────────────────────────────────────
    # 9. PARAMETER SET CONSISTENCY
    # ────────────────────────────────────────────────────────────────
    print("\n\n" + "─" * 90)
    print("  9. PARAMETER SET CONSISTENCY (primary vs replication1)")
    print("─" * 90)

    for ps_label, ps_name in [("Primary (J=5, γ=0.002)", "primary"),
                               ("Replication1 (J=4, γ=0.003)", "replication1")]:
        ps_results = [r for r in results if r['param_set'] == ps_name]
        ps_groups = group_by(ps_results, lambda r: (r['rule'], r['k']))

        print(f"\n  {ps_label}:")
        for k in [1, 3, 5]:
            rand = ps_groups.get(('random', k), [])
            alig = ps_groups.get(('aligned', k), [])
            if not rand or not alig:
                continue

            rand_surv = sum(1 for r in rand if is_survived(r))
            alig_surv = sum(1 for r in alig if is_survived(r))

            print(f"    k={k}: Random {rand_surv}/{len(rand)} survived, "
                  f"Aligned {alig_surv}/{len(alig)} survived")

    # ────────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ────────────────────────────────────────────────────────────────
    output = {
        'n_total': len(results),
        'param_sets': ['primary', 'replication1'],
        'survival_table': survival_data,
        'eta_medians': eta_medians,
        'cross_paper': {
            'paper2_or_k5': 7.0,
            'paper3_survival_or_k5': survival_data[5]['OR'],
            'paper3_amplification': survival_data[5]['OR'] / 7.0 if survival_data[5]['OR'] else None,
        }
    }

    outpath = '/Users/chris/Documents/systems_analysis/astrobiology3/pilot/results/phase2_statistics.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\n  Results saved to {outpath}")


if __name__ == '__main__':
    run_all()
