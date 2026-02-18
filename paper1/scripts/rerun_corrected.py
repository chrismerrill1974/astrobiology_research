"""
Paper 1 Rerun — Corrected Kinetics

Fixes applied:
  1. R[i,j] += coeff (was = coeff) — fixes repeated-species reactant parsing
  2. CHEMOSTAT mode (was CSTR) — required for oscillation with correct k*X²*Y

Reruns:
  - Experiment 1: Control vs Test (N=30 per group)
  - Experiment 2: Progressive autocatalysis (15 trajectories x 6 steps)
  - Generates updated figures
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import json
import os
import time
from datetime import datetime

from dimensional_opening.experiments import run_experiment_1, run_experiment_2
from dimensional_opening.activation_tracker import ActivationTracker

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 70)
    print("PAPER 1 RERUN — CORRECTED KINETICS (CHEMOSTAT MODE)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    total_start = time.time()

    # --- Experiment 1: Control vs Test ---
    print("\n>>> EXPERIMENT 1: Control vs Test")
    exp1 = run_experiment_1(
        n_networks=30,
        n_added_control=3,
        n_autocatalytic_test=2,
        n_random_test=1,
        template='brusselator',
        dilution_rate=0.1,  # kept for API compat but now ignored (chemostat)
        seed=42,
        verbose=True,
    )

    # --- Experiment 2: Progressive ---
    print("\n>>> EXPERIMENT 2: Progressive Autocatalysis")
    exp2 = run_experiment_2(
        n_trajectories=15,
        n_steps=5,
        template='brusselator',
        dilution_rate=0.1,
        seed=42,
        verbose=True,
    )

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")

    # --- Save results ---
    results = {
        'correction_note': {
            'bug': 'R[i,j] = coeff changed to R[i,j] += coeff in simulator._parse_reactions()',
            'effect': 'X+X+Y->3X now correctly computed as k*X^2*Y instead of k*X*Y',
            'mode_change': 'CSTR -> CHEMOSTAT (food species A,B clamped at fixed concentrations)',
            'reason': 'Corrected trimolecular kinetics require higher effective food concentrations',
        },
        'experiment_1': exp1.to_dict(),
        'experiment_2': {
            'name': exp2.name,
            'hypothesis': exp2.hypothesis,
            'n_trajectories': exp2.n_trajectories,
            'n_autocatalytic_range': exp2.n_autocatalytic_range,
            'eta_medians': [float(x) if not np.isnan(x) else None for x in exp2.eta_medians],
            'eta_lower': [float(x) if not np.isnan(x) else None for x in exp2.eta_lower],
            'eta_upper': [float(x) if not np.isnan(x) else None for x in exp2.eta_upper],
            'eta_slope': float(exp2.eta_slope) if not np.isnan(exp2.eta_slope) else None,
            'parameters': exp2.parameters,
        },
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(OUTPUT_DIR, 'rerun_corrected_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # --- Generate figures ---
    _plot_experiment_1(exp1)
    _plot_experiment_2(exp2)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY OF CORRECTED RESULTS")
    print("=" * 70)
    print(f"\nExperiment 1:")
    print(f"  Control: eta = {exp1.group1_eta_median:.4f} (IQR: {exp1.group1_eta_iqr:.4f}, n={exp1.group1_n})")
    print(f"  Test:    eta = {exp1.group2_eta_median:.4f} (IQR: {exp1.group2_eta_iqr:.4f}, n={exp1.group2_n})")
    print(f"  Delta:   {exp1.delta_eta:+.4f}")
    print(f"\nExperiment 2:")
    print(f"  Baseline eta: {exp2.eta_medians[0]:.4f}" if not np.isnan(exp2.eta_medians[0]) else "  Baseline eta: N/A")
    print(f"  Slope: {exp2.eta_slope:+.4f}/reaction" if not np.isnan(exp2.eta_slope) else "  Slope: N/A")
    for i, (med, lo, hi) in enumerate(zip(exp2.eta_medians, exp2.eta_lower, exp2.eta_upper)):
        if not np.isnan(med):
            print(f"  +{i} autocat: eta = {med:.4f} [{lo:.4f}, {hi:.4f}]")


def _plot_experiment_1(exp1):
    """Generate boxplot for Experiment 1."""
    import matplotlib.pyplot as plt
    from dimensional_opening.correlation_dimension import QualityFlag

    control_etas = [r.eta for r in exp1.results
                    if r.network_id.startswith('control')
                    and not r.skipped and r.quality != QualityFlag.FAILED
                    and not np.isnan(r.eta)]
    test_etas = [r.eta for r in exp1.results
                 if r.network_id.startswith('test')
                 and not r.skipped and r.quality != QualityFlag.FAILED
                 and not np.isnan(r.eta)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    bp = ax.boxplot([control_etas, test_etas],
                    labels=['Control\n(random)', 'Test\n(autocatalytic)'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#4ECDC4')
    bp['boxes'][1].set_facecolor('#FF6B6B')

    # Add individual data points
    for i, data in enumerate([control_etas, test_etas], 1):
        x = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.4, s=15, color='black', zorder=3)

    ax.set_ylabel(r'Activation ratio $\eta = D_2 / r_S$')
    ax.set_title('Control vs Test: Corrected Kinetics (Chemostat)')

    # Reference line
    if len(control_etas) > 0 or len(test_etas) > 0:
        all_etas = control_etas + test_etas
        if all_etas:
            ax.axhline(np.median(all_etas), color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'exp1_boxplot_corrected.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Figure saved: {path}")


def _plot_experiment_2(exp2):
    """Generate progressive eta plot for Experiment 2."""
    import matplotlib.pyplot as plt

    x = exp2.n_autocatalytic_range
    y = exp2.eta_medians
    lo = exp2.eta_lower
    hi = exp2.eta_upper

    # Filter out NaN
    valid = [(xi, yi, li, hi_i) for xi, yi, li, hi_i in zip(x, y, lo, hi)
             if not np.isnan(yi)]
    if not valid:
        print("WARNING: No valid data for progressive plot!")
        return

    vx, vy, vlo, vhi = zip(*valid)
    vx, vy, vlo, vhi = list(vx), list(vy), list(vlo), list(vhi)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.fill_between(vx, vlo, vhi, alpha=0.3, color='#FF6B6B', label='IQR')
    ax.plot(vx, vy, 'o-', color='#FF6B6B', markersize=8, label='Median')

    # Linear fit
    if len(vx) >= 2:
        slope, intercept = np.polyfit(vx, vy, 1)
        xfit = np.linspace(min(vx), max(vx), 100)
        ax.plot(xfit, slope * xfit + intercept, '--', color='gray',
                label=f'Slope = {slope:+.4f}/rxn')

    ax.set_xlabel('Number of additional autocatalytic reactions')
    ax.set_ylabel(r'Activation ratio $\eta = D_2 / r_S$')
    ax.set_title('Progressive Autocatalysis: Corrected Kinetics (Chemostat)')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'progressive_eta_corrected.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Figure saved: {path}")


if __name__ == '__main__':
    main()
