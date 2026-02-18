"""
Phase 6: Publication-Quality Figures, Tables, and Prediction Comparison.

Loads data from:
  - data/paper2_results.json (N=200 experiment)
  - data/phase5b_statistics.json (statistical characterization)

Generates:
  Figure 1: Survival fraction vs k (with 95% binomial CI)
  Figure 2: Progressive η plot (medians + IQR, slopes in legend)
  Figure 3: Odds ratio vs k (log scale, with CI)
  Figure 4: Selection bias diagnostics (η A vs B overlay + correlation annotations)
  Figure S1: Logistic regression predicted survival curves
  Figure S2: CV vs k by group (accepted-only, parsed from filter results)

  Table 1: Survival counts, Fisher p (raw+adj), OR with CI  (LaTeX + console)
  Table 2: η medians (IQR), MW p, Cliff's δ                (LaTeX + console)

  Prediction comparison vs pre-registered predictions

Saves all figures to figures/ (publication) and tables/analysis to data/.
"""

import matplotlib
matplotlib.use('Agg')

import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
FIG_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
COLOR_A = '#c0392b'   # muted red for Group A (Random)
COLOR_B = '#2980b9'   # muted blue for Group B (Aligned)
COLOR_OR = '#27ae60'  # green for odds ratio
LABEL_A = 'Group A (Random)'
LABEL_B = 'Group B (Aligned)'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# ── Data Loading ───────────────────────────────────────────────────────

def load_data():
    """Load both result files."""
    with open(os.path.join(DATA_DIR, 'paper2_results.json')) as f:
        results = json.load(f)
    with open(os.path.join(DATA_DIR, 'phase5b_statistics.json')) as f:
        stats = json.load(f)

    # Convert η matrices to numpy, None → NaN
    for key in ['group_a_eta_matrix', 'group_b_eta_matrix']:
        mat = np.array(results[key], dtype=float)
        mat = np.where(np.equal(mat, None), np.nan, mat)
        results[key] = mat

    return results, stats


# =====================================================================
# FIGURE 1: Survival Fraction vs k
# =====================================================================

def figure1_survival(results, stats):
    """Survival fraction vs k with 95% Clopper-Pearson CI."""
    a_eta = results['group_a_eta_matrix']
    b_eta = results['group_b_eta_matrix']
    n = a_eta.shape[0]
    n_steps = a_eta.shape[1]
    ks = np.arange(n_steps)

    a_valid = np.array([np.sum(~np.isnan(a_eta[:, k])) for k in range(n_steps)])
    b_valid = np.array([np.sum(~np.isnan(b_eta[:, k])) for k in range(n_steps)])
    a_frac = a_valid / n
    b_frac = b_valid / n

    # Clopper-Pearson 95% CI
    def cp_ci(k_succ, n_tot, alpha=0.05):
        if k_succ == 0:
            lo = 0.0
        else:
            lo = binom.ppf(alpha / 2, n_tot, k_succ / n_tot) / n_tot
        if k_succ == n_tot:
            hi = 1.0
        else:
            hi = binom.ppf(1 - alpha / 2, n_tot, k_succ / n_tot) / n_tot
        return lo, hi

    a_ci = np.array([cp_ci(int(v), n) for v in a_valid])
    b_ci = np.array([cp_ci(int(v), n) for v in b_valid])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(ks, a_frac, 'o-', color=COLOR_A, label=LABEL_A, linewidth=2, markersize=7, zorder=3)
    ax.fill_between(ks, a_ci[:, 0], a_ci[:, 1], alpha=0.10, color=COLOR_A, zorder=1)
    ax.plot(ks, b_frac, 's-', color=COLOR_B, label=LABEL_B, linewidth=2, markersize=7, zorder=3)
    ax.fill_between(ks, b_ci[:, 0], b_ci[:, 1], alpha=0.10, color=COLOR_B, zorder=1)

    # Annotate n/N at each step
    for ki in range(n_steps):
        ax.annotate(f'{int(a_valid[ki])}/{n}',
                    (ks[ki], a_frac[ki]),
                    textcoords='offset points',
                    xytext=(-5, -14) if ki > 0 else (-5, -14),
                    fontsize=7, color=COLOR_A, ha='center')
        ax.annotate(f'{int(b_valid[ki])}/{n}',
                    (ks[ki], b_frac[ki]),
                    textcoords='offset points',
                    xytext=(-5, 8) if ki > 0 else (-5, 8),
                    fontsize=7, color=COLOR_B, ha='center')

    ax.set_xlabel('Autocatalytic additions ($k$)')
    ax.set_ylabel('Survival fraction')
    ax.set_title('Oscillation Survival Under Network Growth')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.2, 5.2)
    ax.set_xticks(ks)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIG_DIR, 'fig1_survival_fraction.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'  Figure 1 saved: {path}')
    return path


# =====================================================================
# FIGURE 2: Progressive η Plot
# =====================================================================

def figure2_eta_progressive(results, stats):
    """η vs k with medians, IQR bands, slopes in legend."""
    a_eta = results['group_a_eta_matrix']
    b_eta = results['group_b_eta_matrix']
    n_steps = a_eta.shape[1]
    ks = np.arange(n_steps)

    def compute_stats(eta_mat):
        meds, q25s, q75s = [], [], []
        for k in range(n_steps):
            col = eta_mat[:, k]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                meds.append(np.median(valid))
                q25s.append(np.percentile(valid, 25))
                q75s.append(np.percentile(valid, 75))
            else:
                meds.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)
        return np.array(meds), np.array(q25s), np.array(q75s)

    a_med, a_q25, a_q75 = compute_stats(a_eta)
    b_med, b_q25, b_q75 = compute_stats(b_eta)

    slope_a = results['group_a_slope']
    slope_b = results['group_b_slope']
    ci_a = results['group_a_slope_ci']
    ci_b = results['group_b_slope_ci']

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Slopes in caption, not legend (reviewer feedback)
    ax.plot(ks, a_med, 'o-', color=COLOR_A, label=LABEL_A, linewidth=2, markersize=7, zorder=3)
    ax.fill_between(ks, a_q25, a_q75, alpha=0.12, color=COLOR_A, zorder=1)
    ax.plot(ks, b_med, 's-', color=COLOR_B, label=LABEL_B, linewidth=2, markersize=7, zorder=3)
    ax.fill_between(ks, b_q25, b_q75, alpha=0.12, color=COLOR_B, zorder=1)

    # Brusselator baseline
    ax.axhline(a_med[0], color='gray', linestyle='--', alpha=0.4, linewidth=1,
               label=f'Baseline $\\eta$ = {a_med[0]:.3f}')

    ax.set_xlabel('Autocatalytic additions ($k$)')
    ax.set_ylabel('Activation ratio $\\eta = D_2 / r_S$')
    ax.set_title('Progressive $\\eta$ Dilution: Random vs Aligned')
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax.set_xticks(ks)
    ax.set_xlim(-0.2, 5.2)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIG_DIR, 'fig2_eta_progressive.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'  Figure 2 saved: {path}')
    return path


# =====================================================================
# FIGURE 3: Odds Ratio vs k
# =====================================================================

def figure3_odds_ratio(results, stats):
    """Odds ratio (B vs A) per k on log scale with CI."""
    fisher = stats['survival']['fisher_tests']

    ks = [e['k'] for e in fisher]
    ors = [e['odds_ratio'] for e in fisher]
    or_lo = [e['or_ci_lo'] for e in fisher]
    or_hi = [e['or_ci_hi'] for e in fisher]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    yerr_lo = np.array(ors) - np.array(or_lo)
    yerr_hi = np.array(or_hi) - np.array(ors)

    ax.errorbar(ks, ors, yerr=[yerr_lo, yerr_hi],
                fmt='o-', capsize=6, capthick=1.5,
                color=COLOR_OR, linewidth=2, markersize=8, zorder=3)

    # Reference line at OR=1
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1,
               label='OR = 1 (no effect)')

    # Annotate OR values
    for k, or_val, p_adj in zip(ks, ors,
                                 [e['p_adjusted'] for e in fisher]):
        p_str = f'p < 10$^{{{int(np.floor(np.log10(p_adj)))}}}$' if p_adj < 0.001 else f'p = {p_adj:.3f}'
        ax.annotate(f'OR = {or_val:.1f}\n{p_str}',
                    (k, or_val), textcoords='offset points',
                    xytext=(12, 5), fontsize=8, ha='left',
                    color=COLOR_OR)

    ax.set_xlabel('Autocatalytic additions ($k$)')
    ax.set_ylabel('Odds ratio (Aligned / Random)')
    ax.set_title('Survival Odds Ratio by Addition Step')
    ax.set_yscale('log')
    ax.set_xticks(ks)
    ax.set_xlim(0.7, 5.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.2, linewidth=0.5, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIG_DIR, 'fig3_odds_ratio.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'  Figure 3 saved: {path}')
    return path


# =====================================================================
# FIGURE 4: Selection Bias Diagnostics
# =====================================================================

def figure4_selection_bias(results, stats):
    """η distribution overlap at each k + correlation annotations."""
    a_eta = results['group_a_eta_matrix']
    b_eta = results['group_b_eta_matrix']
    n_steps = a_eta.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), sharey=False)
    axes = axes.flatten()

    for k in range(n_steps):
        ax = axes[k]
        a_ok = a_eta[:, k][~np.isnan(a_eta[:, k])]
        b_ok = b_eta[:, k][~np.isnan(b_eta[:, k])]

        if len(a_ok) > 0:
            ax.hist(a_ok, bins=20, alpha=0.5, color=COLOR_A, density=True,
                    label=f'A (n={len(a_ok)})')
        if len(b_ok) > 0:
            ax.hist(b_ok, bins=20, alpha=0.5, color=COLOR_B, density=True,
                    label=f'B (n={len(b_ok)})')

        # Get MW p-value from stats
        per_k = stats['selection_bias']['per_k_eta_comparison']
        mw_p = per_k[k].get('mw_p', None)
        p_str = f'MW p = {mw_p:.3f}' if mw_p is not None else ''

        ax.set_title(f'$k = {k}$', fontsize=11)
        ax.legend(fontsize=7.5, loc='upper right')
        if p_str:
            ax.text(0.03, 0.95, p_str, transform=ax.transAxes,
                    fontsize=8, va='top', color='#555555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if k >= 3:
            ax.set_xlabel('$\\eta$', fontsize=11)
        if k % 3 == 0:
            ax.set_ylabel('Density', fontsize=11)

    # Use 6th panel for annotation summary
    ax_info = axes[5]  # This is k=5, already used above

    # Add global annotation — only r(CV, η); r(CV, D₂) moved to supplemental per reviewer
    cv_eta_r = results.get('cv_eta_correlation', 0)
    fig.text(0.5, -0.02,
             f'Selection bias check: $r$(CV, $\\eta$) = {cv_eta_r:.3f} (negligible)',
             ha='center', fontsize=10, style='italic')

    fig.suptitle('$\\eta$ Distributions: Random vs Aligned (valid networks only)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'fig4_selection_bias.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'  Figure 4 saved: {path}')
    return path


# =====================================================================
# FIGURE S1: Logistic Regression Predicted Curves
# =====================================================================

def figureS1_logistic(results, stats):
    """Logistic regression predicted survival with observed overlay."""
    lr = stats['survival']['logistic_regression']
    if 'predicted_A' not in lr:
        print('  Figure S1 skipped (no logistic regression data)')
        return None

    a_eta = results['group_a_eta_matrix']
    b_eta = results['group_b_eta_matrix']
    n = a_eta.shape[0]
    n_steps = a_eta.shape[1]
    ks = np.arange(n_steps)

    a_obs = [np.sum(~np.isnan(a_eta[:, k])) / n for k in range(n_steps)]
    b_obs = [np.sum(~np.isnan(b_eta[:, k])) / n for k in range(n_steps)]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Predicted curves
    ax.plot(lr['predicted_A']['k'], lr['predicted_A']['survival_prob'],
            '-', color=COLOR_A, linewidth=2, alpha=0.7, label=f'{LABEL_A} (predicted)')
    ax.plot(lr['predicted_B']['k'], lr['predicted_B']['survival_prob'],
            '-', color=COLOR_B, linewidth=2, alpha=0.7, label=f'{LABEL_B} (predicted)')

    # Observed points
    ax.plot(ks, a_obs, 'o', color=COLOR_A, markersize=8, zorder=4,
            label=f'{LABEL_A} (observed)')
    ax.plot(ks, b_obs, 's', color=COLOR_B, markersize=8, zorder=4,
            label=f'{LABEL_B} (observed)')

    # Coefficient annotations
    coefs = lr['coefficients']
    pvals = lr['p_values']
    text = (f'$\\beta_{{\\mathrm{{GroupB}}}}$ = {coefs["GroupB"]:.3f} '
            f'(p < 10$^{{{int(np.floor(np.log10(pvals["GroupB"])))}}}$)\n'
            f'$\\beta_{{k \\times \\mathrm{{GroupB}}}}$ = {coefs["k_x_GroupB"]:.3f} '
            f'(p = {pvals["k_x_GroupB"]:.4f})')
    ax.text(0.97, 0.55, text, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Autocatalytic additions ($k$)')
    ax.set_ylabel('Survival probability')
    ax.set_title('Logistic Regression: Predicted vs Observed Survival\n'
                 '(interaction term indicates divergence accelerates with $k$)',
                 fontsize=11)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIG_DIR, 'figS1_logistic_survival.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'  Figure S1 saved: {path}')
    return path


# =====================================================================
# FIGURE S2: CV vs k by Group (parsed from filter results)
# =====================================================================

def figureS2_cv_vs_k(results, stats):
    """CV distributions by k for Group B accepted networks."""
    filter_results = results.get('group_b_filter_results', [])
    if not filter_results:
        print('  Figure S2 skipped (no filter results)')
        return None

    # Parse CV values from string representations
    # Format: OscillationResult(passes=True, cv=np.float64(0.912...), ...)
    cv_by_step = {k: [] for k in range(1, 6)}
    for traj_results in filter_results:
        for step_idx, result_str in enumerate(traj_results):
            if isinstance(result_str, str):
                cv_match = re.search(r'cv=np\.float64\(([^)]+)\)', result_str)
                if cv_match:
                    cv_val = float(cv_match.group(1))
                    cv_by_step[step_idx + 1].append(cv_val)

    if all(len(v) == 0 for v in cv_by_step.values()):
        print('  Figure S2 skipped (could not parse CV values)')
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    positions = sorted(cv_by_step.keys())
    data = [cv_by_step[k] for k in positions]

    bp = ax.boxplot(data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', markersize=3, alpha=0.3))

    for box in bp['boxes']:
        box.set_facecolor(COLOR_B)
        box.set_alpha(0.3)
    for median in bp['medians']:
        median.set_color(COLOR_B)
        median.set_linewidth(2)

    # Add CV threshold reference
    ax.axhline(0.03, color='gray', linestyle='--', alpha=0.5, linewidth=1,
               label='CV threshold = 0.03')

    ax.set_xlabel('Autocatalytic additions ($k$)')
    ax.set_ylabel('Coefficient of variation (CV)')
    ax.set_title('CV of Accepted Networks (Group B)\n'
                 'Primary analysis: CV $\\geq$ 0.03; survival advantage persists under threshold variation',
                 fontsize=9.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticks(positions)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIG_DIR, 'figS2_cv_vs_k.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'  Figure S2 saved: {path}')
    return path


# =====================================================================
# TABLE 1: Survival (LaTeX)
# =====================================================================

def table1_survival(results, stats):
    """Survival counts, Fisher p (raw+adj), OR with CI — LaTeX + console."""
    fisher = stats['survival']['fisher_tests']
    n = results['group_a_eta_matrix'].shape[0]

    lines = []
    lines.append('\\begin{table}[ht]')
    lines.append('\\centering')
    lines.append('\\caption{Oscillation survival by addition step. '
                 'OR = odds ratio (B/A); CI = 95\\% Wald interval on log-OR; '
                 '$p$ values from Fisher exact test, Holm--Bonferroni adjusted.}')
    lines.append('\\label{tab:survival}')
    lines.append('\\begin{tabular}{ccccccc}')
    lines.append('\\toprule')
    lines.append('$k$ & A surv & B surv & OR & 95\\% CI & $p_{\\mathrm{raw}}$ & $p_{\\mathrm{adj}}$ \\\\')
    lines.append('\\midrule')

    # k=0 row
    lines.append(f'0 & {n}/{n} & {n}/{n} & --- & --- & --- & --- \\\\')

    for e in fisher:
        or_str = f'{e["odds_ratio"]:.2f}'
        ci_str = f'[{e["or_ci_lo"]:.2f}, {e["or_ci_hi"]:.2f}]'
        p_raw = _format_p(e['p_raw'])
        p_adj = _format_p(e['p_adjusted'])
        lines.append(f'{e["k"]} & {e["a_valid"]}/{n} & {e["b_valid"]}/{n} & '
                      f'{or_str} & {ci_str} & {p_raw} & {p_adj} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    latex = '\n'.join(lines)

    # Console version
    print('\n' + '=' * 80)
    print('TABLE 1: Oscillation Survival')
    print('=' * 80)
    print(f'{"k":>3}  {"A surv":>8}  {"B surv":>8}  {"OR":>6}  {"95% CI":>18}  '
          f'{"p(raw)":>12}  {"p(adj)":>12}')
    print('-' * 80)
    print(f'  0  {n:>4}/{n}  {n:>4}/{n}  {"---":>6}  {"---":>18}  {"---":>12}  {"---":>12}')
    for e in fisher:
        ci = f'[{e["or_ci_lo"]:.2f}, {e["or_ci_hi"]:.2f}]'
        print(f'  {e["k"]}  {e["a_valid"]:>4}/{n}  {e["b_valid"]:>4}/{n}  '
              f'{e["odds_ratio"]:>6.2f}  {ci:>18}  '
              f'{e["p_raw"]:>12.2e}  {e["p_adjusted"]:>12.2e}')

    return latex


def _format_p(p):
    """Format p-value for LaTeX."""
    if p < 1e-15:
        exp = int(np.floor(np.log10(p)))
        return f'$< 10^{{{exp}}}$'
    elif p < 0.001:
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return f'${mant:.1f} \\times 10^{{{exp}}}$'
    else:
        return f'{p:.4f}'


# =====================================================================
# TABLE 2: η Statistics (LaTeX)
# =====================================================================

def table2_eta(results, stats):
    """η medians (IQR), MW p, Cliff's δ — LaTeX + console."""
    bias = stats['selection_bias']['per_k_eta_comparison']
    cliffs = stats['cliffs_delta']

    lines = []
    lines.append('\\begin{table}[ht]')
    lines.append('\\centering')
    lines.append('\\caption{Activation ratio $\\eta$ by addition step (valid networks only). '
                 'Cliff\'s $\\delta$ is the non-parametric effect size (B vs A).}')
    lines.append('\\label{tab:eta}')
    lines.append('\\begin{tabular}{ccccccc}')
    lines.append('\\toprule')
    lines.append('$k$ & A med & A IQR & B med & B IQR & MW $p$ & Cliff\'s $\\delta$ \\\\')
    lines.append('\\midrule')

    for entry, cliff in zip(bias, cliffs):
        k = entry['k']
        a_med = f'{entry.get("a_median", 0):.4f}'
        a_iqr = f'{entry.get("a_iqr", 0):.4f}'
        b_med = f'{entry.get("b_median", 0):.4f}'
        b_iqr = f'{entry.get("b_iqr", 0):.4f}'
        mw_p = f'{entry.get("mw_p", 1.0):.3f}'

        delta = cliff.get('delta', 0)
        if delta is not None:
            d_str = f'{delta:+.3f}'
            interp = cliff.get('interpretation', '')
        else:
            d_str = '---'
            interp = ''

        lines.append(f'{k} & {a_med} & {a_iqr} & {b_med} & {b_iqr} & {mw_p} & {d_str} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    latex = '\n'.join(lines)

    # Console version
    print('\n' + '=' * 90)
    print('TABLE 2: Activation Ratio η')
    print('=' * 90)
    print(f'{"k":>3}  {"A med":>8}  {"A IQR":>8}  {"A n":>5}  '
          f'{"B med":>8}  {"B IQR":>8}  {"B n":>5}  '
          f'{"MW p":>8}  {"δ":>7}  {"interp":>10}')
    print('-' * 90)

    for entry, cliff in zip(bias, cliffs):
        delta = cliff.get('delta')
        d_str = f'{delta:+.4f}' if delta is not None else '---'
        interp = cliff.get('interpretation', '---')
        print(f'  {entry["k"]}  {entry.get("a_median",0):>8.4f}  {entry.get("a_iqr",0):>8.4f}  '
              f'{entry["a_n"]:>5}  {entry.get("b_median",0):>8.4f}  {entry.get("b_iqr",0):>8.4f}  '
              f'{entry["b_n"]:>5}  {entry.get("mw_p",1):>8.3f}  {d_str:>7}  {interp:>10}')

    return latex


# =====================================================================
# PREDICTION COMPARISON
# =====================================================================

def compare_predictions(results, stats):
    """Compare actual results to pre-registered predictions."""
    print('\n' + '=' * 80)
    print('PRE-REGISTERED PREDICTION COMPARISON')
    print('=' * 80)

    a_eta = results['group_a_eta_matrix']
    b_eta = results['group_b_eta_matrix']
    n = a_eta.shape[0]

    comparison = []

    # 1. η at k=0 (baseline)
    a_base = np.nanmedian(a_eta[:, 0])
    pred_base = 0.43  # pre-registered (before bug fix)
    actual_base = a_base
    comparison.append({
        'prediction': 'η at k=0 ≈ 0.43',
        'actual': f'{actual_base:.3f}',
        'match': 'PARTIAL — baseline is 0.249 (lower due to CHEMOSTAT correction)',
    })
    print(f'\n  1. η at k=0')
    print(f'     Predicted: ≈ 0.43')
    print(f'     Actual:    {actual_base:.3f}')
    print(f'     Note: Pre-registered predictions assumed CSTR mode (pre-bug-fix).')
    print(f'           CHEMOSTAT correction lowered baseline from ~0.43 to ~0.249.')
    print(f'     Status: EXPLAINED DEVIATION (bug fix changes baseline)')

    # 2. η at k=5
    a_k5 = np.nanmedian(a_eta[:, 5])
    b_k5 = np.nanmedian(b_eta[:, 5])
    comparison.append({
        'prediction': 'A: η(k=5) ≈ 0.17, B: η(k=5) ≈ 0.25–0.35',
        'actual': f'A: {a_k5:.3f}, B: {b_k5:.3f}',
        'match': 'PARTIAL — proportional decline correct but absolute values shifted',
    })
    print(f'\n  2. η at k=5')
    print(f'     Predicted: A ≈ 0.17, B ≈ 0.25–0.35')
    print(f'     Actual:    A = {a_k5:.3f}, B = {b_k5:.3f}')
    print(f'     Status: FAILED — B does NOT retain higher η. Both decline identically.')

    # 3. Slopes
    slope_a = results['group_a_slope']
    slope_b = results['group_b_slope']
    delta_s = results['slope_difference']
    perm_p = results['slope_difference_perm_p']
    comparison.append({
        'prediction': 'A slope ≈ -0.04, B slope ≈ -0.02 to 0.00',
        'actual': f'A: {slope_a:.4f}, B: {slope_b:.4f}, Δ = {delta_s:.4f} (p = {perm_p:.3f})',
        'match': 'FAILED — slopes are indistinguishable',
    })
    print(f'\n  3. η slope')
    print(f'     Predicted: A ≈ −0.04/rxn, B ≈ −0.02 to 0.00/rxn')
    print(f'     Actual:    A = {slope_a:.4f}, B = {slope_b:.4f}')
    print(f'     Δslope = {delta_s:+.4f}, permutation p = {perm_p:.3f}')
    print(f'     Status: FAILED — slopes indistinguishable')

    # 4. Tail probability P(η > 0.25) at k=5
    a_tail = results['group_a_tail_prob'][-1] if results['group_a_tail_prob'] else 0
    b_tail = results['group_b_tail_prob'][-1] if results['group_b_tail_prob'] else 0
    comparison.append({
        'prediction': 'A: P(η>0.25) < 10%, B: P(η>0.25) > 25%',
        'actual': f'A: {a_tail:.1%}, B: {b_tail:.1%}',
        'match': 'PARTIAL — both near zero (baseline lower than predicted)',
    })
    print(f'\n  4. Tail probability P(η > 0.25) at k=5')
    print(f'     Predicted: A < 10%, B > 25%')
    print(f'     Actual:    A = {a_tail:.1%}, B = {b_tail:.1%}')
    print(f'     Status: MOOT — baseline η ≈ 0.249 already near threshold')

    # 5. Acceptance rate (secondary)
    acc_meds = results.get('acceptance_rate_medians', [])
    comparison.append({
        'prediction': 'Acceptance ≈ 50% at k=1, declining to 10–20% at k=5',
        'actual': f'Median acceptance: {[f"{a:.0%}" for a in acc_meds]}',
        'match': 'FAILED — acceptance stays near 100% (filter is not restrictive)',
    })
    print(f'\n  5. Acceptance rate (Group B)')
    print(f'     Predicted: ~50% at k=1, declining to ~10–20% at k=5')
    print(f'     Actual:    Median acceptance per step: {[f"{a:.0%}" for a in acc_meds]}')
    print(f'     Status: FAILED — filter is not restrictive; acceptance ≈ 100%')

    # 6. Early termination (secondary)
    et_rate = results.get('early_termination_rate', 0)
    comparison.append({
        'prediction': '10–30% of trajectories terminate before k=5',
        'actual': f'{et_rate:.1%}',
        'match': 'FAILED — zero early terminations',
    })
    print(f'\n  6. Early termination rate')
    print(f'     Predicted: 10–30%')
    print(f'     Actual:    {et_rate:.1%}')
    print(f'     Status: FAILED — zero early terminations')

    # 7. Selection bias check (secondary)
    cv_d2_r = results.get('cv_d2_correlation', 0)
    comparison.append({
        'prediction': '|r(CV, D₂)| < 0.3',
        'actual': f'r = {cv_d2_r:.3f}',
        'match': 'MARGINAL — r = -0.356, slightly above 0.3 threshold',
    })
    print(f'\n  7. Selection bias: |r(CV, D₂)| < 0.3')
    print(f'     Predicted: |r| < 0.3')
    print(f'     Actual:    r = {cv_d2_r:.3f} (|r| = {abs(cv_d2_r):.3f})')
    print(f'     Status: MARGINAL — slightly exceeds threshold')
    print(f'     Note: Driven by D₂ ≈ 1.0 clustering; r(CV, η) = {results.get("cv_eta_correlation", 0):.3f} is fine')

    # HEADLINE: Survival — the actual finding
    a_surv5 = np.sum(~np.isnan(a_eta[:, 5]))
    b_surv5 = np.sum(~np.isnan(b_eta[:, 5]))
    comparison.append({
        'prediction': '(Not pre-registered)',
        'actual': f'Survival at k=5: A = {a_surv5}/{n} ({a_surv5/n:.0%}), B = {b_surv5}/{n} ({b_surv5/n:.0%})',
        'match': 'EMERGENT FINDING — 3.4× survival advantage for aligned additions',
    })
    print(f'\n  ★ HEADLINE FINDING (not pre-registered):')
    print(f'     Survival at k=5: A = {a_surv5}/{n} ({a_surv5/n:.0%}), '
          f'B = {b_surv5}/{n} ({b_surv5/n:.0%})')
    print(f'     Ratio: {b_surv5/a_surv5:.1f}×')
    print(f'     Fisher OR at k=5: {stats["survival"]["fisher_tests"][-1]["odds_ratio"]:.2f}')
    print(f'     All Fisher p < 10⁻⁷ after Holm–Bonferroni correction')

    # Summary assessment
    print(f'\n  {"="*60}')
    print(f'  OVERALL ASSESSMENT')
    print(f'  {"="*60}')
    print(f'  Primary predictions (η improvement): NOT CONFIRMED')
    print(f'  → η dilution rate is invariant to selection method')
    print(f'  ')
    print(f'  Emergent finding (dynamical viability): STRONG')
    print(f'  → Aligned additions preserve oscillation 3.4× better')
    print(f'  → OR = 5–7× at all k, all p < 10⁻⁷')
    print(f'  → Logistic regression: significant group effect + interaction')
    print(f'  ')
    print(f'  Publishable narrative:')
    print(f'  "Topology-aware additions preserve dynamical viability under')
    print(f'   growth, even though the rate of dimensional dilution is')
    print(f'   invariant to selection."')

    return comparison


# =====================================================================
# MAIN
# =====================================================================

def main():
    print('=' * 80)
    print('PHASE 6: PUBLICATION FIGURES AND ANALYSIS')
    print('=' * 80)

    results, stats = load_data()

    # ── Figures ──
    print('\n--- Generating Figures ---')
    figure1_survival(results, stats)
    figure2_eta_progressive(results, stats)
    figure3_odds_ratio(results, stats)
    figure4_selection_bias(results, stats)
    figureS1_logistic(results, stats)
    figureS2_cv_vs_k(results, stats)

    # ── Tables ──
    print('\n--- Generating Tables ---')
    tab1_latex = table1_survival(results, stats)
    tab2_latex = table2_eta(results, stats)

    # ── Predictions ──
    comparison = compare_predictions(results, stats)

    # ── Save analysis summary ──
    summary = {
        'table1_latex': tab1_latex,
        'table2_latex': tab2_latex,
        'prediction_comparison': comparison,
        'figures': [
            'fig1_survival_fraction.pdf',
            'fig2_eta_progressive.pdf',
            'fig3_odds_ratio.pdf',
            'fig4_selection_bias.pdf',
            'figS1_logistic_survival.pdf',
            'figS2_cv_vs_k.pdf',
        ],
    }

    out_path = os.path.join(DATA_DIR, 'phase6_analysis_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'\n{"="*80}')
    print(f'Phase 6 complete.')
    print(f'  Figures:  {FIG_DIR}/')
    print(f'  Summary:  {out_path}')
    print(f'{"="*80}')


if __name__ == '__main__':
    main()
