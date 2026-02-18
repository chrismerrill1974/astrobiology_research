"""
Phase 5B: Extended Statistical Characterization of Paper 2 Results.

Loads N=200 data from paper2_results.json and computes:
  5B.1 — Data integrity checks, per-k counts
  5B.2 — Primary endpoint: oscillation survival (Fisher, OR, Holm, logistic regression)
  5B.3 — Secondary endpoint: η Cliff's delta
  5B.4 — Selection bias audits (CV-η, CV-D2, CV distributions A vs B)
  5B.5 — Robustness checks (CV threshold sensitivity, longer integration)

Saves all results to data/phase5b_statistics.json and prints summary.
"""

import matplotlib
matplotlib.use('Agg')

import json
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DIAG_DIR = os.path.join(PROJECT_DIR, 'diagrams')

os.makedirs(DIAG_DIR, exist_ok=True)


def load_results(path):
    """Load Paper 2 results JSON."""
    with open(path) as f:
        d = json.load(f)
    # Convert eta matrices to numpy, restoring NaN
    for key in ['group_a_eta_matrix', 'group_b_eta_matrix']:
        mat = np.array(d[key], dtype=float)
        mat[mat == None] = np.nan  # noqa
        d[key] = mat
    return d


# =====================================================================
# 5B.1 Data Integrity Checks
# =====================================================================

def data_integrity_checks(d):
    """Per-k counts and basic sanity checks."""
    print("=" * 70)
    print("5B.1 DATA INTEGRITY CHECKS")
    print("=" * 70)

    a_eta = d['group_a_eta_matrix']
    b_eta = d['group_b_eta_matrix']
    n_traj = a_eta.shape[0]
    n_steps_plus_1 = a_eta.shape[1]

    results = {
        'n_trajectories': n_traj,
        'n_steps': n_steps_plus_1 - 1,
        'seed': d['parameters']['seed'],
        'per_k': [],
    }

    print(f"\nTrajectories per group: {n_traj}")
    print(f"Steps: baseline + {n_steps_plus_1 - 1}")
    print(f"Seed: {d['parameters']['seed']}")

    print(f"\n{'k':>3}  {'A att':>6} {'A val':>6} {'A inv':>6} {'A %':>6}  "
          f"{'B att':>6} {'B val':>6} {'B inv':>6} {'B %':>6}")
    print("-" * 70)

    for k in range(n_steps_plus_1):
        a_valid = int(np.sum(~np.isnan(a_eta[:, k])))
        b_valid = int(np.sum(~np.isnan(b_eta[:, k])))
        a_invalid = n_traj - a_valid
        b_invalid = n_traj - b_valid
        a_pct = 100 * a_valid / n_traj
        b_pct = 100 * b_valid / n_traj

        results['per_k'].append({
            'k': k,
            'a_attempted': n_traj, 'a_valid': a_valid, 'a_invalid': a_invalid,
            'b_attempted': n_traj, 'b_valid': b_valid, 'b_invalid': b_invalid,
        })

        print(f"  {k}  {n_traj:>6} {a_valid:>6} {a_invalid:>6} {a_pct:>5.1f}%  "
              f"{n_traj:>6} {b_valid:>6} {b_invalid:>6} {b_pct:>5.1f}%")

    return results


# =====================================================================
# 5B.2 Primary Endpoint: Oscillation Survival
# =====================================================================

def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = cummax
    return adjusted


def survival_analysis(d):
    """Fisher exact, odds ratios, Holm-Bonferroni, logistic regression."""
    print("\n" + "=" * 70)
    print("5B.2 PRIMARY ENDPOINT: OSCILLATION SURVIVAL")
    print("=" * 70)

    a_eta = d['group_a_eta_matrix']
    b_eta = d['group_b_eta_matrix']
    n_traj = a_eta.shape[0]
    n_steps_plus_1 = a_eta.shape[1]

    results = {'fisher_tests': [], 'logistic_regression': {}}

    # --- A) Fisher exact per k ---
    raw_p_values = []

    print(f"\n{'k':>3}  {'A surv':>7} {'B surv':>7} {'OR':>8} {'OR 95% CI':>18} "
          f"{'p (raw)':>10} {'p (adj)':>10} {'RD':>8} {'RD CI':>18}")
    print("-" * 100)

    rng = np.random.RandomState(42)

    for k in range(1, n_steps_plus_1):  # skip baseline (k=0, always 100%)
        a_valid = int(np.sum(~np.isnan(a_eta[:, k])))
        b_valid = int(np.sum(~np.isnan(b_eta[:, k])))
        a_invalid = n_traj - a_valid
        b_invalid = n_traj - b_valid

        # Fisher exact
        table = [[b_valid, b_invalid], [a_valid, a_invalid]]
        odds_ratio, p_val = fisher_exact(table)

        # OR confidence interval (Wald on log scale with 0.5 continuity correction)
        a_v, a_i, b_v, b_i = a_valid, a_invalid, b_valid, b_invalid
        # Add 0.5 continuity correction for CI computation
        log_or = np.log((b_v + 0.5) * (a_i + 0.5) / ((b_i + 0.5) * (a_v + 0.5)))
        se_log_or = np.sqrt(1/(b_v+0.5) + 1/(b_i+0.5) + 1/(a_v+0.5) + 1/(a_i+0.5))
        or_lo = np.exp(log_or - 1.96 * se_log_or)
        or_hi = np.exp(log_or + 1.96 * se_log_or)

        # Risk difference with bootstrap CI
        p_a = a_valid / n_traj
        p_b = b_valid / n_traj
        rd = p_b - p_a

        # Bootstrap CI for risk difference
        a_surv_vec = (~np.isnan(a_eta[:, k])).astype(int)
        b_surv_vec = (~np.isnan(b_eta[:, k])).astype(int)
        rd_boots = []
        for _ in range(10000):
            a_boot = rng.choice(a_surv_vec, size=n_traj, replace=True)
            b_boot = rng.choice(b_surv_vec, size=n_traj, replace=True)
            rd_boots.append(b_boot.mean() - a_boot.mean())
        rd_boots = np.array(rd_boots)
        rd_lo = float(np.percentile(rd_boots, 2.5))
        rd_hi = float(np.percentile(rd_boots, 97.5))

        raw_p_values.append(p_val)

        results['fisher_tests'].append({
            'k': k,
            'a_valid': a_valid, 'b_valid': b_valid,
            'a_invalid': a_invalid, 'b_invalid': b_invalid,
            'odds_ratio': float(odds_ratio),
            'or_ci_lo': float(or_lo), 'or_ci_hi': float(or_hi),
            'p_raw': float(p_val),
            'risk_difference': float(rd),
            'rd_ci_lo': rd_lo, 'rd_ci_hi': rd_hi,
        })

    # Holm-Bonferroni
    adjusted_p = holm_bonferroni(raw_p_values)
    for i, adj_p in enumerate(adjusted_p):
        results['fisher_tests'][i]['p_adjusted'] = float(adj_p)

    # Print table
    for entry in results['fisher_tests']:
        k = entry['k']
        or_str = f"{entry['odds_ratio']:.2f}"
        ci_str = f"[{entry['or_ci_lo']:.2f}, {entry['or_ci_hi']:.2f}]"
        p_raw = f"{entry['p_raw']:.4f}"
        p_adj = f"{entry['p_adjusted']:.4f}"
        rd_str = f"{entry['risk_difference']:+.3f}"
        rd_ci = f"[{entry['rd_ci_lo']:+.3f}, {entry['rd_ci_hi']:+.3f}]"
        a_s = f"{entry['a_valid']}/{n_traj}"
        b_s = f"{entry['b_valid']}/{n_traj}"
        print(f"  {k}  {a_s:>7} {b_s:>7} {or_str:>8} {ci_str:>18} "
              f"{p_raw:>10} {p_adj:>10} {rd_str:>8} {rd_ci:>18}")

    # --- D) Logistic regression ---
    print("\n--- Logistic Regression ---")
    try:
        import statsmodels.api as sm

        # Build dataset: each row = (k, group, survived)
        ks, groups, survived = [], [], []
        for k in range(n_steps_plus_1):
            for i in range(n_traj):
                ks.append(k)
                groups.append(0)  # Group A
                survived.append(int(not np.isnan(a_eta[i, k])))
            for i in range(n_traj):
                ks.append(k)
                groups.append(1)  # Group B
                survived.append(int(not np.isnan(b_eta[i, k])))

        ks = np.array(ks, dtype=float)
        groups = np.array(groups, dtype=float)
        survived = np.array(survived, dtype=float)
        interaction = ks * groups

        X = np.column_stack([ks, groups, interaction])
        X = sm.add_constant(X)

        model = sm.Logit(survived, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(disp=0)

        coef_names = ['intercept', 'k', 'GroupB', 'k_x_GroupB']
        print(f"\n  {'Coefficient':<15} {'Estimate':>10} {'Std Err':>10} {'z':>8} {'p':>10}")
        print("  " + "-" * 55)
        for name, coef, se, z, p in zip(coef_names, fit.params, fit.bse,
                                         fit.tvalues, fit.pvalues):
            print(f"  {name:<15} {coef:>10.4f} {se:>10.4f} {z:>8.3f} {p:>10.4f}")

        results['logistic_regression'] = {
            'coefficients': {name: float(c) for name, c in zip(coef_names, fit.params)},
            'std_errors': {name: float(s) for name, s in zip(coef_names, fit.bse)},
            'p_values': {name: float(p) for name, p in zip(coef_names, fit.pvalues)},
            'aic': float(fit.aic),
            'bic': float(fit.bic),
        }

        # Predicted survival curves
        k_pred = np.linspace(0, n_steps_plus_1 - 1, 100)
        for group_val, group_label in [(0, 'A'), (1, 'B')]:
            X_pred = np.column_stack([
                np.ones(len(k_pred)),
                k_pred,
                np.full(len(k_pred), group_val),
                k_pred * group_val,
            ])
            pred = fit.predict(X_pred)
            results['logistic_regression'][f'predicted_{group_label}'] = {
                'k': k_pred.tolist(),
                'survival_prob': pred.tolist(),
            }

        print(f"\n  AIC: {fit.aic:.1f}, BIC: {fit.bic:.1f}")

    except ImportError:
        print("  statsmodels not available — skipping logistic regression")
        results['logistic_regression'] = {'error': 'statsmodels not installed'}

    return results


# =====================================================================
# 5B.3 Secondary Endpoint: η Cliff's Delta
# =====================================================================

def cliffs_delta(x, y):
    """Compute Cliff's delta (non-parametric effect size)."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    return (more - less) / (n_x * n_y)


def eta_effect_sizes(d):
    """Cliff's delta at each k for η distributions."""
    print("\n" + "=" * 70)
    print("5B.3 SECONDARY ENDPOINT: η CLIFF'S DELTA")
    print("=" * 70)

    a_eta = d['group_a_eta_matrix']
    b_eta = d['group_b_eta_matrix']
    n_steps_plus_1 = a_eta.shape[1]
    rng = np.random.RandomState(42)

    results = []

    print(f"\n{'k':>3}  {'δ':>8} {'δ 95% CI':>20} {'Interpretation':>15}")
    print("-" * 55)

    for k in range(n_steps_plus_1):
        a_ok = a_eta[:, k][~np.isnan(a_eta[:, k])]
        b_ok = b_eta[:, k][~np.isnan(b_eta[:, k])]

        if len(a_ok) < 3 or len(b_ok) < 3:
            results.append({'k': k, 'delta': None, 'ci_lo': None, 'ci_hi': None})
            print(f"  {k}  {'N/A':>8}")
            continue

        delta = cliffs_delta(b_ok, a_ok)

        # Bootstrap CI
        deltas = []
        for _ in range(5000):
            a_boot = rng.choice(a_ok, size=len(a_ok), replace=True)
            b_boot = rng.choice(b_ok, size=len(b_ok), replace=True)
            deltas.append(cliffs_delta(b_boot, a_boot))
        deltas = np.array(deltas)
        ci_lo = float(np.percentile(deltas, 2.5))
        ci_hi = float(np.percentile(deltas, 97.5))

        # Interpretation
        abs_d = abs(delta)
        if abs_d < 0.147:
            interp = "negligible"
        elif abs_d < 0.33:
            interp = "small"
        elif abs_d < 0.474:
            interp = "medium"
        else:
            interp = "large"

        results.append({
            'k': k, 'delta': float(delta),
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'interpretation': interp,
        })

        ci_str = f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
        print(f"  {k}  {delta:>+8.3f} {ci_str:>20} {interp:>15}")

    return results


# =====================================================================
# 5B.4 Selection Bias Audits
# =====================================================================

def selection_bias_audits(d):
    """Deepened selection bias: CV-η, CV-D2 with bootstrap CIs; CV distributions A vs B."""
    print("\n" + "=" * 70)
    print("5B.4 SELECTION BIAS AUDITS")
    print("=" * 70)

    a_eta = d['group_a_eta_matrix']
    b_eta = d['group_b_eta_matrix']

    results = {
        'pooled': {},
        'cv_d2_correlation': d.get('cv_d2_correlation'),
        'cv_eta_correlation': d.get('cv_eta_correlation'),
    }

    print(f"\nPooled correlations (from experiment run):")
    print(f"  r(CV, D₂) = {d.get('cv_d2_correlation', 'N/A')}")
    print(f"  r(CV, η)  = {d.get('cv_eta_correlation', 'N/A')}")

    # Compare valid η distributions at each k to check if Group B
    # selects for "gentler" oscillations (lower η)
    print(f"\nη comparison (valid networks only):")
    print(f"  {'k':>3}  {'A med':>7} {'A IQR':>7} {'A n':>5}  "
          f"{'B med':>7} {'B IQR':>7} {'B n':>5}  {'MW p':>8}")
    print("  " + "-" * 60)

    per_k = []
    for k in range(a_eta.shape[1]):
        a_ok = a_eta[:, k][~np.isnan(a_eta[:, k])]
        b_ok = b_eta[:, k][~np.isnan(b_eta[:, k])]

        entry = {'k': k, 'a_n': len(a_ok), 'b_n': len(b_ok)}
        if len(a_ok) > 0:
            entry['a_median'] = float(np.median(a_ok))
            entry['a_iqr'] = float(np.percentile(a_ok, 75) - np.percentile(a_ok, 25))
        if len(b_ok) > 0:
            entry['b_median'] = float(np.median(b_ok))
            entry['b_iqr'] = float(np.percentile(b_ok, 75) - np.percentile(b_ok, 25))
        if len(a_ok) >= 3 and len(b_ok) >= 3:
            _, p = mannwhitneyu(a_ok, b_ok, alternative='two-sided')
            entry['mw_p'] = float(p)
        per_k.append(entry)

        a_m = f"{entry.get('a_median', np.nan):.4f}"
        a_i = f"{entry.get('a_iqr', np.nan):.4f}"
        b_m = f"{entry.get('b_median', np.nan):.4f}"
        b_i = f"{entry.get('b_iqr', np.nan):.4f}"
        mw = f"{entry.get('mw_p', 'N/A')}"
        if isinstance(mw, float):
            mw = f"{mw:.4f}"
        print(f"  {k:>3}  {a_m:>7} {a_i:>7} {len(a_ok):>5}  "
              f"{b_m:>7} {b_i:>7} {len(b_ok):>5}  {mw:>8}")

    results['per_k_eta_comparison'] = per_k
    return results


# =====================================================================
# 5B.5 Robustness Checks
# =====================================================================

def _simulate_and_check(net, t_end=100.0, n_points=2000):
    """Simulate a network and return OscillationResult."""
    from dimensional_opening.simulator import ReactionSimulator, DrivingMode
    from dimensional_opening.oscillation_filter import check_oscillation

    sim = ReactionSimulator()
    network = sim.build_network(net.reactions)
    try:
        result = sim.simulate(
            network,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=(0, t_end), n_points=n_points,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=dict(net.chemostat_species),
        )
        return check_oscillation(
            result.concentrations, result.time,
            species_names=result.species_names,
            food_species=list(net.chemostat_species.keys()),
        )
    except Exception:
        from dimensional_opening.oscillation_filter import OscillationResult
        return OscillationResult(
            passes=False, cv=0.0, amplitude=0.0,
            sign_changes=0, boundedness_ratio=0.0,
            best_species_idx=-1, best_species_name="",
        )


def robustness_checks():
    """CV threshold sensitivity and longer integration window."""
    print("\n" + "=" * 70)
    print("5B.5 ROBUSTNESS CHECKS")
    print("=" * 70)

    results = {}

    # --- A) CV threshold sensitivity ---
    print("\n--- CV Threshold Sensitivity (N=50 subset) ---")
    try:
        from dimensional_opening.network_generator import NetworkGenerator

        cv_thresholds = [0.02, 0.03, 0.05]
        n_traj = 50
        n_steps = 5
        seed = 42

        threshold_results = {}
        for cv_thresh in cv_thresholds:
            a_surv = np.zeros(n_steps + 1)
            b_surv = np.zeros(n_steps + 1)

            for traj in range(n_traj):
                # Group A: random progressive
                gen = NetworkGenerator(template='brusselator', seed=seed + traj)
                networks = gen.generate_progressive(n_steps)
                for step, net in enumerate(networks):
                    osc = _simulate_and_check(net)
                    if osc.cv >= cv_thresh and osc.sign_changes >= 5 and 0.2 < osc.boundedness_ratio < 5.0:
                        a_surv[step] += 1

                # Group B: aligned
                gen_b = NetworkGenerator(template='brusselator', seed=seed + n_traj + traj)
                aligned = gen_b.generate_progressive_aligned(n_steps=n_steps, max_candidates=50)
                for step, net in enumerate(aligned.networks):
                    osc = _simulate_and_check(net)
                    if osc.cv >= cv_thresh and osc.sign_changes >= 5 and 0.2 < osc.boundedness_ratio < 5.0:
                        b_surv[step] += 1

            a_pct = (a_surv / n_traj * 100).tolist()
            b_pct = (b_surv / n_traj * 100).tolist()
            threshold_results[str(cv_thresh)] = {
                'a_survival_pct': a_pct,
                'b_survival_pct': b_pct,
            }

            print(f"\n  CV threshold = {cv_thresh}:")
            print(f"    {'k':>3}  {'A %':>7}  {'B %':>7}  {'B/A ratio':>10}")
            for k in range(n_steps + 1):
                ratio = b_pct[k] / a_pct[k] if a_pct[k] > 0 else float('inf')
                print(f"    {k:>3}  {a_pct[k]:>6.1f}%  {b_pct[k]:>6.1f}%  {ratio:>10.2f}")

        results['cv_threshold_sensitivity'] = threshold_results

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Error: {e}")
        results['cv_threshold_sensitivity'] = {'error': str(e)}

    # --- B) Longer integration window (t=200 vs t=100) ---
    print("\n--- Longer Integration Window (20 networks, t=200 vs t=100) ---")
    try:
        from dimensional_opening.network_generator import NetworkGenerator
        n_test = 20
        seed = 42
        agree_count = 0
        total_tested = 0

        for traj in range(n_test):
            gen = NetworkGenerator(template='brusselator', seed=seed + traj)
            aligned = gen.generate_progressive_aligned(n_steps=3, max_candidates=50)
            if len(aligned.networks) < 4:
                continue
            net = aligned.networks[-1]  # Last network (k=3)

            osc100 = _simulate_and_check(net, t_end=100.0, n_points=2000)
            osc200 = _simulate_and_check(net, t_end=200.0, n_points=4000)

            total_tested += 1
            if osc100.passes == osc200.passes:
                agree_count += 1

        results['longer_integration'] = {
            'n_tested': total_tested,
            'n_agree': agree_count,
            'agreement_rate': agree_count / total_tested if total_tested > 0 else None,
        }
        print(f"  Tested: {total_tested} networks")
        if total_tested > 0:
            print(f"  Agreement (t=100 vs t=200): {agree_count}/{total_tested}"
                  f" ({100*agree_count/total_tested:.0f}%)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Error: {e}")
        results['longer_integration'] = {'error': str(e)}

    return results


# =====================================================================
# 5B.6 Diagnostic Plots
# =====================================================================

def generate_plots(d, survival_results, cliffs_results):
    """Generate Phase 5B diagnostic plots."""
    print("\n" + "=" * 70)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    a_eta = d['group_a_eta_matrix']
    b_eta = d['group_b_eta_matrix']
    n_traj = a_eta.shape[0]
    n_steps_plus_1 = a_eta.shape[1]
    ks = np.arange(n_steps_plus_1)

    # --- Plot 1: Survival fraction vs k ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    a_surv = [np.sum(~np.isnan(a_eta[:, k])) / n_traj for k in range(n_steps_plus_1)]
    b_surv = [np.sum(~np.isnan(b_eta[:, k])) / n_traj for k in range(n_steps_plus_1)]

    # Binomial CI (Clopper-Pearson)
    from scipy.stats import binom
    def binom_ci(k_succ, n, alpha=0.05):
        lo = binom.ppf(alpha/2, n, k_succ/n) / n if k_succ > 0 else 0
        hi = binom.ppf(1-alpha/2, n, k_succ/n) / n if k_succ < n else 1
        return lo, hi

    a_ci = [binom_ci(int(s*n_traj), n_traj) for s in a_surv]
    b_ci = [binom_ci(int(s*n_traj), n_traj) for s in b_surv]

    a_lo = [c[0] for c in a_ci]
    a_hi = [c[1] for c in a_ci]
    b_lo = [c[0] for c in b_ci]
    b_hi = [c[1] for c in b_ci]

    ax.plot(ks, a_surv, 'o-', color='#d62728', label='Group A (Random)', linewidth=2)
    ax.fill_between(ks, a_lo, a_hi, alpha=0.2, color='#d62728')
    ax.plot(ks, b_surv, 's-', color='#1f77b4', label='Group B (Aligned)', linewidth=2)
    ax.fill_between(ks, b_lo, b_hi, alpha=0.2, color='#1f77b4')

    ax.set_xlabel('Autocatalytic additions (k)', fontsize=12)
    ax.set_ylabel('Survival fraction', fontsize=12)
    ax.set_title('Oscillation Survival: Random vs Aligned', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(DIAG_DIR, 'survival_fraction.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # --- Plot 2: Odds ratio vs k ---
    fisher = survival_results['fisher_tests']
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ks_fisher = [e['k'] for e in fisher]
    ors = [e['odds_ratio'] for e in fisher]
    or_lo = [e['or_ci_lo'] for e in fisher]
    or_hi = [e['or_ci_hi'] for e in fisher]

    ax.errorbar(ks_fisher, ors,
                yerr=[np.array(ors) - np.array(or_lo),
                      np.array(or_hi) - np.array(ors)],
                fmt='o-', capsize=5, color='#2ca02c', linewidth=2, markersize=8)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='OR = 1 (no effect)')
    ax.set_xlabel('Autocatalytic additions (k)', fontsize=12)
    ax.set_ylabel('Odds Ratio (Group B / Group A)', fontsize=12)
    ax.set_title('Survival Odds Ratio by Addition Step', fontsize=13)
    ax.set_yscale('log')
    ax.set_xticks(ks_fisher)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(DIAG_DIR, 'odds_ratio.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")

    # --- Plot 3: η vs k (medians + IQR) ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    a_meds = [float(np.nanmedian(a_eta[:, k])) if np.any(~np.isnan(a_eta[:, k])) else np.nan
              for k in range(n_steps_plus_1)]
    b_meds = [float(np.nanmedian(b_eta[:, k])) if np.any(~np.isnan(b_eta[:, k])) else np.nan
              for k in range(n_steps_plus_1)]
    a_q25 = [float(np.nanpercentile(a_eta[:, k], 25)) if np.any(~np.isnan(a_eta[:, k])) else np.nan
             for k in range(n_steps_plus_1)]
    a_q75 = [float(np.nanpercentile(a_eta[:, k], 75)) if np.any(~np.isnan(a_eta[:, k])) else np.nan
             for k in range(n_steps_plus_1)]
    b_q25 = [float(np.nanpercentile(b_eta[:, k], 25)) if np.any(~np.isnan(b_eta[:, k])) else np.nan
             for k in range(n_steps_plus_1)]
    b_q75 = [float(np.nanpercentile(b_eta[:, k], 75)) if np.any(~np.isnan(b_eta[:, k])) else np.nan
             for k in range(n_steps_plus_1)]

    ax.plot(ks, a_meds, 'o-', color='#d62728', label='Group A (Random)', linewidth=2)
    ax.fill_between(ks, a_q25, a_q75, alpha=0.15, color='#d62728')
    ax.plot(ks, b_meds, 's-', color='#1f77b4', label='Group B (Aligned)', linewidth=2)
    ax.fill_between(ks, b_q25, b_q75, alpha=0.15, color='#1f77b4')
    ax.axhline(0.249, color='gray', linestyle='--', alpha=0.5, label='Brusselator baseline')

    ax.set_xlabel('Autocatalytic additions (k)', fontsize=12)
    ax.set_ylabel('Activation ratio η = D₂/r_S', fontsize=12)
    ax.set_title('Progressive η Dilution: Random vs Aligned', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(DIAG_DIR, 'eta_progressive.png')
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path3}")

    # --- Plot 4: Logistic regression predicted curves ---
    if 'predicted_A' in survival_results.get('logistic_regression', {}):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        lr = survival_results['logistic_regression']
        k_a = lr['predicted_A']['k']
        p_a = lr['predicted_A']['survival_prob']
        k_b = lr['predicted_B']['k']
        p_b = lr['predicted_B']['survival_prob']

        ax.plot(k_a, p_a, '-', color='#d62728', linewidth=2, label='Group A (predicted)')
        ax.plot(k_b, p_b, '-', color='#1f77b4', linewidth=2, label='Group B (predicted)')

        # Overlay observed
        ax.plot(ks, a_surv, 'o', color='#d62728', markersize=8, label='Group A (observed)')
        ax.plot(ks, b_surv, 's', color='#1f77b4', markersize=8, label='Group B (observed)')

        ax.set_xlabel('Autocatalytic additions (k)', fontsize=12)
        ax.set_ylabel('Survival probability', fontsize=12)
        ax.set_title('Logistic Regression: Predicted Survival', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path4 = os.path.join(DIAG_DIR, 'logistic_survival.png')
        fig.savefig(path4, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path4}")


# =====================================================================
# Main
# =====================================================================

def main():
    results_path = os.path.join(DATA_DIR, 'paper2_results.json')
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    d = load_results(results_path)
    all_results = {}

    # 5B.1
    all_results['integrity'] = data_integrity_checks(d)

    # 5B.2
    all_results['survival'] = survival_analysis(d)

    # 5B.3
    all_results['cliffs_delta'] = eta_effect_sizes(d)

    # 5B.4
    all_results['selection_bias'] = selection_bias_audits(d)

    # 5B.5
    all_results['robustness'] = robustness_checks()

    # Plots
    generate_plots(d, all_results['survival'], all_results['cliffs_delta'])

    # Save all results
    output_path = os.path.join(DATA_DIR, 'phase5b_statistics.json')

    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=json_safe)

    print(f"\n{'='*70}")
    print(f"All Phase 5B statistics saved to: {output_path}")
    print(f"Plots saved to: {DIAG_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
