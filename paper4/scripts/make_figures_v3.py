#!/usr/bin/env python3
"""
Generate figures for V3a (reversed selection) and V3b (multi-parameter evolution).

Produces publication-quality PNGs comparing V3a/V3b to V2 results.
No new simulations — purely data visualization.

Usage:
    python3 make_figures_v3.py                         # all figures
    python3 make_figures_v3.py --v3a-only              # V3a figures only
    python3 make_figures_v3.py --v3b-only              # V3b figures only
"""

import argparse
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_this_dir = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_this_dir, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# V2 results for comparison
V2_DIR = os.path.join(_this_dir, '..', 'data', 'v2_confirmed')
# Phase 0 baseline
BASELINE_DIR = os.path.join(_this_dir, '..', 'data')

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_replicates(results_dir, prefix, types):
    """Load JSONL replicates with given prefix and types."""
    data = {}
    for rep_type in types:
        reps = []
        pattern = os.path.join(results_dir, f'{prefix}_rep*_{rep_type}.jsonl')
        files = sorted(glob.glob(pattern))
        for path in files:
            reps.append(load_jsonl(path))
        data[rep_type] = reps
        print(f'  Loaded {len(reps)} {rep_type} replicates ({prefix})')
    return data


def load_v2_replicates():
    """Load V2 results for comparison."""
    if not os.path.isdir(V2_DIR):
        print(f'  WARNING: V2 results not found at {V2_DIR}')
        return {'selection': [], 'neutral': []}
    return load_replicates(V2_DIR, 'phase1', ['selection', 'neutral'])


# ── V3a Figures ──────────────────────────────────────────────────────

def fig_v3a_tau_trajectories():
    """V3a: reversed vs neutral tau trajectories, with V2 selection for reference."""
    v3a_dir = os.path.join(_this_dir, '..', 'data', 'v3a_reversed')
    if not os.path.isdir(v3a_dir):
        print('  Skipping V3a tau trajectories (no data/v3a_reversed/)')
        return

    v3a = load_replicates(v3a_dir, 'v3a', ['reversed', 'neutral'])
    v2 = load_v2_replicates()

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    alpha = 0.3

    # V2 selection (reference, grey)
    for i, rep in enumerate(v2.get('selection', [])):
        gens = [r['generation'] for r in rep]
        taus = [r['mean_fitness'] for r in rep]
        label = 'V2 selection' if i == 0 else None
        ax.plot(gens, taus, color='0.7', alpha=alpha, linewidth=0.6, label=label)

    # V3a reversed (red, solid)
    for i, rep in enumerate(v3a.get('reversed', [])):
        gens = [r['generation'] for r in rep]
        taus = [r['mean_fitness'] for r in rep]
        label = 'V3a reversed' if i == 0 else None
        ax.plot(gens, taus, color='#d73027', alpha=0.5, linewidth=0.8, label=label)

    # V3a neutral (blue, dashed)
    for i, rep in enumerate(v3a.get('neutral', [])):
        gens = [r['generation'] for r in rep]
        taus = [r['mean_fitness'] for r in rep]
        label = 'Neutral' if i == 0 else None
        ax.plot(gens, taus, color='#4575b4', alpha=0.5, linewidth=0.8,
                linestyle='--', label=label)

    # Means
    for rep_type, color, ls in [('reversed', '#d73027', '-'), ('neutral', '#4575b4', '--')]:
        reps = v3a.get(rep_type, [])
        if len(reps) >= 2:
            max_gen = min(len(r) for r in reps)
            mean_tau = np.mean([[r[g]['mean_fitness'] for g in range(max_gen)]
                                for r in reps], axis=0)
            ax.plot(range(max_gen), mean_tau, color=color, linewidth=2.0, linestyle=ls)

    ax.set_xlabel('Generation')
    ax.set_ylabel(r'Mean population $\tau_{>1.2}$')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 5)

    fig.savefig(os.path.join(FIGURES_DIR, 'fig_v3a_tau_trajectories.png'))
    plt.close(fig)
    print('  fig_v3a_tau_trajectories.png')


def fig_v3a_gamma_evolution():
    """V3a: gamma trajectories (reversed should push gamma UP)."""
    v3a_dir = os.path.join(_this_dir, '..', 'data', 'v3a_reversed')
    if not os.path.isdir(v3a_dir):
        print('  Skipping V3a gamma evolution (no data/v3a_reversed/)')
        return

    v3a = load_replicates(v3a_dir, 'v3a', ['reversed', 'neutral'])
    v2 = load_v2_replicates()

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    alpha = 0.3

    # V2 selection (grey reference)
    for i, rep in enumerate(v2.get('selection', [])):
        gens = [r['generation'] for r in rep]
        gammas = [r['mean_gamma'] for r in rep]
        label = 'V2 selection' if i == 0 else None
        ax.plot(gens, gammas, color='0.7', alpha=alpha, linewidth=0.6, label=label)

    # V3a reversed (red)
    for i, rep in enumerate(v3a.get('reversed', [])):
        gens = [r['generation'] for r in rep]
        gammas = [r['mean_gamma'] for r in rep]
        label = 'V3a reversed' if i == 0 else None
        ax.plot(gens, gammas, color='#d73027', alpha=0.5, linewidth=0.8, label=label)

    # V3a neutral (blue dashed)
    for i, rep in enumerate(v3a.get('neutral', [])):
        gens = [r['generation'] for r in rep]
        gammas = [r['mean_gamma'] for r in rep]
        label = 'Neutral' if i == 0 else None
        ax.plot(gens, gammas, color='#4575b4', alpha=0.5, linewidth=0.8,
                linestyle='--', label=label)

    ax.set_xlabel('Generation')
    ax.set_ylabel(r'Mean population $\gamma$')
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.9, fontsize=7)
    ax.set_xlim(0, 40)

    fig.savefig(os.path.join(FIGURES_DIR, 'fig_v3a_gamma_evolution.png'))
    plt.close(fig)
    print('  fig_v3a_gamma_evolution.png')


# ── V3b Figures ──────────────────────────────────────────────────────

def fig_v3b_tau_trajectories():
    """V3b: multi-param selection vs neutral, with V2 for reference."""
    v3b_dir = os.path.join(_this_dir, '..', 'data', 'v3b_multi_param')
    if not os.path.isdir(v3b_dir):
        print('  Skipping V3b tau trajectories (no data/v3b_multi_param/)')
        return

    v3b = load_replicates(v3b_dir, 'v3b', ['selection', 'neutral'])
    v2 = load_v2_replicates()

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    alpha = 0.3

    # V2 selection (grey reference)
    for i, rep in enumerate(v2.get('selection', [])):
        gens = [r['generation'] for r in rep]
        taus = [r['mean_fitness'] for r in rep]
        label = 'V2 (gamma only)' if i == 0 else None
        ax.plot(gens, taus, color='0.7', alpha=alpha, linewidth=0.6, label=label)

    # V3b selection (green)
    for i, rep in enumerate(v3b.get('selection', [])):
        gens = [r['generation'] for r in rep]
        taus = [r['mean_fitness'] for r in rep]
        label = 'V3b multi-param' if i == 0 else None
        ax.plot(gens, taus, color='#1a9850', alpha=0.5, linewidth=0.8, label=label)

    # V3b neutral (blue dashed)
    for i, rep in enumerate(v3b.get('neutral', [])):
        gens = [r['generation'] for r in rep]
        taus = [r['mean_fitness'] for r in rep]
        label = 'Neutral' if i == 0 else None
        ax.plot(gens, taus, color='#4575b4', alpha=0.5, linewidth=0.8,
                linestyle='--', label=label)

    # Means
    for rep_type, color, ls in [('selection', '#1a9850', '-'), ('neutral', '#4575b4', '--')]:
        reps = v3b.get(rep_type, [])
        if len(reps) >= 2:
            max_gen = min(len(r) for r in reps)
            mean_tau = np.mean([[r[g]['mean_fitness'] for g in range(max_gen)]
                                for r in reps], axis=0)
            ax.plot(range(max_gen), mean_tau, color=color, linewidth=2.0, linestyle=ls)

    ax.set_xlabel('Generation')
    ax.set_ylabel(r'Mean population $\tau_{>1.2}$')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=7)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 5)

    fig.savefig(os.path.join(FIGURES_DIR, 'fig_v3b_tau_trajectories.png'))
    plt.close(fig)
    print('  fig_v3b_tau_trajectories.png')


def fig_v3b_parameter_trajectories():
    """V3b: all three parameter trajectories (gamma, J, k_cat) in subplots."""
    v3b_dir = os.path.join(_this_dir, '..', 'data', 'v3b_multi_param')
    if not os.path.isdir(v3b_dir):
        print('  Skipping V3b parameter trajectories (no data/v3b_multi_param/)')
        return

    v3b = load_replicates(v3b_dir, 'v3b', ['selection', 'neutral'])

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.0), sharex=True)

    params = [
        ('mean_gamma', r'$\gamma$', True),
        ('mean_J', r'$J$', False),
        ('mean_k_cat', r'$k_{\mathrm{cat}}$', False),
    ]

    for ax, (key, label, log_scale) in zip(axes, params):
        for i, rep in enumerate(v3b.get('selection', [])):
            gens = [r['generation'] for r in rep]
            vals = [r[key] for r in rep]
            lbl = 'Selection' if i == 0 else None
            ax.plot(gens, vals, color='#1a9850', alpha=0.4, linewidth=0.8, label=lbl)

        for i, rep in enumerate(v3b.get('neutral', [])):
            gens = [r['generation'] for r in rep]
            vals = [r[key] for r in rep]
            lbl = 'Neutral' if i == 0 else None
            ax.plot(gens, vals, color='#4575b4', alpha=0.4, linewidth=0.8,
                    linestyle='--', label=lbl)

        # Means
        for rep_type, color, ls in [('selection', '#1a9850', '-'), ('neutral', '#4575b4', '--')]:
            reps = v3b.get(rep_type, [])
            if len(reps) >= 2:
                max_gen = min(len(r) for r in reps)
                mean_vals = np.mean([[r[g][key] for g in range(max_gen)]
                                     for r in reps], axis=0)
                ax.plot(range(max_gen), mean_vals, color=color, linewidth=2.0, linestyle=ls)

        ax.set_ylabel(label)
        if log_scale:
            ax.set_yscale('log')
        ax.legend(loc='best', framealpha=0.9, fontsize=6)

    axes[-1].set_xlabel('Generation')
    axes[-1].set_xlim(0, 40)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_v3b_parameter_trajectories.png'))
    plt.close(fig)
    print('  fig_v3b_parameter_trajectories.png')


# ── Combined comparison figure ───────────────────────────────────────

def fig_combined_comparison():
    """Side-by-side: V2 selection vs V3a reversed vs V3b multi-param (final gen boxplots)."""
    v3a_dir = os.path.join(_this_dir, '..', 'data', 'v3a_reversed')
    v3b_dir = os.path.join(_this_dir, '..', 'data', 'v3b_multi_param')

    conditions = []
    labels = []

    # V2
    v2 = load_v2_replicates()
    if v2['selection']:
        v2_final = [rep[-1]['mean_fitness'] for rep in v2['selection']]
        conditions.append(v2_final)
        labels.append('V2\nselection')

    # V3a reversed
    if os.path.isdir(v3a_dir):
        v3a = load_replicates(v3a_dir, 'v3a', ['reversed', 'neutral'])
        if v3a['reversed']:
            v3a_final = [rep[-1]['mean_fitness'] for rep in v3a['reversed']]
            conditions.append(v3a_final)
            labels.append('V3a\nreversed')

    # V3b multi-param
    if os.path.isdir(v3b_dir):
        v3b = load_replicates(v3b_dir, 'v3b', ['selection', 'neutral'])
        if v3b['selection']:
            v3b_final = [rep[-1]['mean_fitness'] for rep in v3b['selection']]
            conditions.append(v3b_final)
            labels.append('V3b\nmulti-param')

    # Neutral (any available)
    neutral_data = []
    for src in [v2, v3a if os.path.isdir(v3a_dir) else {'neutral': []},
                v3b if os.path.isdir(v3b_dir) else {'neutral': []}]:
        for rep in src.get('neutral', []):
            if rep:
                neutral_data.append(rep[-1]['mean_fitness'])
    if neutral_data:
        conditions.append(neutral_data)
        labels.append('Neutral\n(pooled)')

    if len(conditions) < 2:
        print('  Skipping combined comparison (insufficient data)')
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    colors = ['#2166ac', '#d73027', '#1a9850', '0.6'][:len(conditions)]
    bp = ax.boxplot(conditions, labels=labels, patch_artist=True,
                    widths=0.5, showfliers=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(r'Final mean $\tau_{>1.2}$ (gen 40)')
    ax.axhline(y=1.22, color='k', linestyle=':', linewidth=0.6, alpha=0.4,
               label='V2 gen 0 baseline')

    fig.savefig(os.path.join(FIGURES_DIR, 'fig_combined_comparison.png'))
    plt.close(fig)
    print('  fig_combined_comparison.png')


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate V3 figures')
    parser.add_argument('--v3a-only', action='store_true')
    parser.add_argument('--v3b-only', action='store_true')
    args = parser.parse_args()

    print(f'Generating V3 figures...')

    if not args.v3b_only:
        fig_v3a_tau_trajectories()
        fig_v3a_gamma_evolution()

    if not args.v3a_only:
        fig_v3b_tau_trajectories()
        fig_v3b_parameter_trajectories()

    fig_combined_comparison()
    print(f'Done. Figures in: {FIGURES_DIR}')
