#!/usr/bin/env python3
"""
Generate figures for Paper 4: Evolutionary Selection on Transient Dynamical Retention Time.

Reads JSONL data from data/v2_confirmed/ (or data/v1_pilot/) and produces
publication-quality PNGs in figures/. No new simulations — purely data visualization.

Usage:
    python3 make_figures.py                                # default: v2_confirmed
    python3 make_figures.py --results-dir ../data/v1_pilot # use V1 pilot data
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

# Phase 0 baseline is always in data/ (not versioned)
BASELINE_DIR = os.path.join(_this_dir, '..', 'data')
FIGURES_DIR = os.path.join(_this_dir, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Matplotlib style for publication
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_phase1_replicates(results_dir):
    """Load all Phase 1 JSONL files, grouped by type.

    Auto-discovers replicate count from files present on disk.
    """
    selection = []
    neutral = []

    # Discover selection replicates
    sel_files = sorted(glob.glob(os.path.join(results_dir, 'phase1_rep*_selection.jsonl')))
    for path in sel_files:
        selection.append(load_jsonl(path))

    # Discover neutral replicates
    neu_files = sorted(glob.glob(os.path.join(results_dir, 'phase1_rep*_neutral.jsonl')))
    for path in neu_files:
        neutral.append(load_jsonl(path))

    print(f'  Loaded {len(selection)} selection + {len(neutral)} neutral replicates from {results_dir}')
    return selection, neutral


def figure1_tau_trajectories(results_dir):
    """
    Figure 1: Mean population tau vs generation for all replicates.
    Selection (solid blue) vs neutral (dashed red).
    """
    selection, neutral = load_phase1_replicates(results_dir)
    n_sel = len(selection)
    n_neu = len(neutral)
    n_total = n_sel + n_neu

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # Adjust alpha for readability: more replicates -> more transparent individuals
    ind_alpha = 0.6 if n_total <= 8 else 0.35

    # Plot individual replicates
    for i, rep_data in enumerate(selection):
        gens = [r['generation'] for r in rep_data]
        taus = [r['mean_fitness'] for r in rep_data]
        label = 'Selection' if i == 0 else None
        ax.plot(gens, taus, color='#2166ac', alpha=ind_alpha, linewidth=0.8, label=label)

    for i, rep_data in enumerate(neutral):
        gens = [r['generation'] for r in rep_data]
        taus = [r['mean_fitness'] for r in rep_data]
        label = 'Neutral' if i == 0 else None
        ax.plot(gens, taus, color='#b2182b', alpha=ind_alpha, linewidth=0.8,
                linestyle='--', label=label)

    # Compute and plot means (bold lines)
    if len(selection) >= 2:
        max_gen = min(len(rep) for rep in selection)
        sel_mean = np.mean([[rep[g]['mean_fitness'] for g in range(max_gen)]
                            for rep in selection], axis=0)
        gens = list(range(max_gen))
        ax.plot(gens, sel_mean, color='#2166ac', linewidth=2.0, alpha=1.0)

    if len(neutral) >= 2:
        max_gen = min(len(rep) for rep in neutral)
        neu_mean = np.mean([[rep[g]['mean_fitness'] for g in range(max_gen)]
                            for rep in neutral], axis=0)
        gens = list(range(max_gen))
        ax.plot(gens, neu_mean, color='#b2182b', linewidth=2.0,
                linestyle='--', alpha=1.0)

    ax.set_xlabel('Generation')
    ax.set_ylabel(r'Mean population $\tau_{>1.2}$')
    ax.legend(loc='upper left', framealpha=0.9)

    # Dynamic axis limits
    all_taus = []
    for rep_data in selection + neutral:
        all_taus.extend(r['mean_fitness'] for r in rep_data)
    max_gen_all = max(r['generation'] for rep in selection + neutral for r in rep)
    ax.set_xlim(0, max_gen_all)
    ax.set_ylim(0, max(4.5, max(all_taus) * 1.1))

    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_tau_trajectories.png'))
    plt.close(fig)
    print('  fig1_tau_trajectories.png')


def figure2_gamma_evolution(results_dir):
    """
    Figure 2: Mean population gamma vs generation (log scale Y-axis).
    Selection (solid blue) vs neutral (dashed red).
    """
    selection, neutral = load_phase1_replicates(results_dir)
    n_total = len(selection) + len(neutral)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ind_alpha = 0.6 if n_total <= 8 else 0.35

    for i, rep_data in enumerate(selection):
        gens = [r['generation'] for r in rep_data]
        gammas = [r['mean_gamma'] for r in rep_data]
        label = 'Selection' if i == 0 else None
        ax.plot(gens, gammas, color='#2166ac', alpha=ind_alpha, linewidth=0.8, label=label)

    for i, rep_data in enumerate(neutral):
        gens = [r['generation'] for r in rep_data]
        gammas = [r['mean_gamma'] for r in rep_data]
        label = 'Neutral' if i == 0 else None
        ax.plot(gens, gammas, color='#b2182b', alpha=ind_alpha, linewidth=0.8,
                linestyle='--', label=label)

    # Means
    if len(selection) >= 2:
        max_gen = min(len(rep) for rep in selection)
        sel_mean = np.mean([[rep[g]['mean_gamma'] for g in range(max_gen)]
                            for rep in selection], axis=0)
        ax.plot(range(max_gen), sel_mean, color='#2166ac', linewidth=2.0)

    if len(neutral) >= 2:
        max_gen = min(len(rep) for rep in neutral)
        neu_mean = np.mean([[rep[g]['mean_gamma'] for g in range(max_gen)]
                            for rep in neutral], axis=0)
        ax.plot(range(max_gen), neu_mean, color='#b2182b', linewidth=2.0,
                linestyle='--')

    ax.set_xlabel('Generation')
    ax.set_ylabel(r'Mean population $\gamma$')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9)

    # Dynamic x-limit
    max_gen_all = max(r['generation'] for rep in selection + neutral for r in rep)
    ax.set_xlim(0, max_gen_all)

    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_gamma_evolution.png'))
    plt.close(fig)
    print('  fig2_gamma_evolution.png')


def figure3_baseline_landscape():
    """
    Figure 3: Phase 0 baseline — gamma vs mean tau, colored by J.
    Shows the fitness landscape that selection exploits.
    Always reads from data/ (baseline is not versioned).
    """
    phase0_path = os.path.join(BASELINE_DIR, 'phase0_incremental.jsonl')
    data = load_jsonl(phase0_path)

    gammas = np.array([d['gamma'] for d in data])
    mean_taus = np.array([d['mean_tau'] for d in data])
    js = np.array([d['J'] for d in data])

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    sc = ax.scatter(gammas, mean_taus, c=js, cmap='viridis', s=12, alpha=0.7,
                    edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, label='$J$', shrink=0.85)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'Mean $\tau_{>1.2}$')

    # Mark baseline
    ax.axvline(x=0.00223, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.annotate(r'$\gamma_0$', xy=(0.00223, 3.5), fontsize=7, ha='right',
                color='0.3')

    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_baseline_landscape.png'))
    plt.close(fig)
    print('  fig3_baseline_landscape.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Paper 4 figures')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory (default: ../data/v2_confirmed/)')
    args = parser.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = os.path.join(_this_dir, '..', 'data', 'v2_confirmed')

    print(f'Generating Paper 4 figures from {results_dir}...')
    figure1_tau_trajectories(results_dir)
    figure2_gamma_evolution(results_dir)
    figure3_baseline_landscape()
    print('Done. Figures in:', FIGURES_DIR)
