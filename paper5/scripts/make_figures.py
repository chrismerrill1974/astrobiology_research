#!/usr/bin/env python3
"""
Paper 5 figure generation.

Produces:
  figures/fig1_rw_distribution.png   — Phase 1 R_w bar chart (named topologies + random ensemble)
  figures/fig2_drift_curves.png      — Phase 2 R_w vs sigma_d for 4 candidates
  figures/fig3_predictor.png         — E-coupler net flow vs R_w (structural predictor)
  figures/fig4_random_bimodality.png — Histogram of R_w across 20 random wirings
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────

PROJECT = '/Users/chris/Documents/systems_analysis/astrobiology_paper5'
P1_DIR = os.path.join(PROJECT, 'results_phase1')
P2_DIR = os.path.join(PROJECT, 'results_phase2')
FIG_DIR = os.path.join(PROJECT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, '..', 'astrobiology_research', 'paper3', 'scripts'))
sys.path.insert(0, os.path.join(PROJECT, '..', 'astrobiology_research', 'shared'))

# ── Style ────────────────────────────────────────────────────────────

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
    'savefig.pad_inches': 0.05,
})

GROUP_A_COLOR = '#4878CF'   # blue
GROUP_B_COLOR = '#D65F5F'   # red
RANDOM_COLOR = '#6ACC65'    # green
DEAD_COLOR = '#B0B0B0'      # gray
BASELINE_COLOR = '#4878CF'


# ── Load Phase 1 data ───────────────────────────────────────────────

def load_phase1():
    named = {}
    for topo in ['T0', 'T1a', 'T1b', 'T2', 'T3', 'T4', 'T5', 'T6']:
        fname = os.path.join(P1_DIR, f'screen_{topo}.jsonl')
        rows = [json.loads(l) for l in open(fname)]
        n = len(rows)
        n_pos = sum(1 for r in rows if r['tau'] > 0)
        rw = n_pos / n if n > 0 else 0
        mean_tau = sum(r['tau'] for r in rows) / n if n > 0 else 0
        named[topo] = {'R_w': rw, 'mean_tau': mean_tau, 'n': n}

    random_wirings = {}
    for i in range(20):
        fname = os.path.join(P1_DIR, f'screen_T1_rw{i}.jsonl')
        if not os.path.exists(fname):
            continue
        rows = [json.loads(l) for l in open(fname)]
        n = len(rows)
        if n == 0:
            continue
        n_pos = sum(1 for r in rows if r['tau'] > 0)
        rw = n_pos / n
        mean_tau = sum(r['tau'] for r in rows) / n
        random_wirings[i] = {'R_w': rw, 'mean_tau': mean_tau, 'n': n}

    return named, random_wirings


def load_phase2():
    results = {}
    SIGMA_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for topo in ['T0', 'T6', 'rw16', 'rw4']:
        fname = os.path.join(P2_DIR, f'drift_{topo}.jsonl')
        rows = [json.loads(l) for l in open(fname)]
        by_sigma = defaultdict(list)
        for r in rows:
            sd = round(r['sigma_d'], 3)
            by_sigma[sd].append(r)
        rw_curve = []
        for sd in SIGMA_LEVELS:
            sd_rows = by_sigma.get(sd, [])
            n = len(sd_rows)
            if n == 0:
                rw_curve.append(0)
                continue
            n_pos = sum(1 for r in sd_rows if r['tau'] > 0)
            rw_curve.append(n_pos / n)
        results[topo] = rw_curve
    return SIGMA_LEVELS, results


def load_predictor_data():
    """Reconstruct E-coupler net flow for each random wiring."""
    from pilot5b_enzyme_complex import EnzymeComplexParams
    from paper5_topology_library import make_t1_random_control

    p = EnzymeComplexParams()
    data = []
    for i in range(20):
        fname = os.path.join(P1_DIR, f'screen_T1_rw{i}.jsonl')
        if not os.path.exists(fname):
            continue
        rows = [json.loads(l) for l in open(fname)]
        n = len(rows)
        if n == 0:
            continue
        n_pos = sum(1 for r in rows if r['tau'] > 0)
        rw = n_pos / n

        rng_seed = 5000 + i * 97
        net = make_t1_random_control(p, seed=42, rng_seed=rng_seed)
        motif_rxns = net.reactions[14:]

        # Parse motif graph
        adj_in = defaultdict(int)
        adj_out = defaultdict(int)
        e_coupler = None
        for rxn in motif_rxns:
            rxn_s = str(rxn)
            if '+ E ->' in rxn_s:
                e_coupler = rxn_s.split('+')[0].strip()
                continue
            parts = rxn_s.split(' -> ')
            if len(parts) == 2:
                src = parts[0].strip()
                dst = parts[1].strip()
                adj_out[src] += 1
                adj_in[dst] += 1

        f_net = adj_in.get(e_coupler, 0) - adj_out.get(e_coupler, 0)
        data.append({
            'rw_idx': i, 'R_w': rw, 'f_net': f_net,
            'alive': rw > 0, 'e_coupler': e_coupler,
        })
    return data


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Phase 1 R_w bar chart
# ══════════════════════════════════════════════════════════════════════

def fig1_rw_distribution(named, random_wirings):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    # Named topologies
    topos = ['T0', 'T1a', 'T1b', 'T2', 'T3', 'T4', 'T5', 'T6']
    rw_vals = [named[t]['R_w'] for t in topos]
    colors = [GROUP_A_COLOR, GROUP_A_COLOR, GROUP_A_COLOR,
              GROUP_B_COLOR, GROUP_B_COLOR, GROUP_B_COLOR,
              GROUP_B_COLOR, GROUP_B_COLOR]

    x = np.arange(len(topos))
    bars = ax.bar(x, rw_vals, color=colors, edgecolor='black', linewidth=0.5, width=0.7)

    # Random ensemble mean + std as shaded band
    rw_random = [random_wirings[i]['R_w'] for i in sorted(random_wirings)]
    rw_mean = np.mean(rw_random)
    rw_std = np.std(rw_random)
    ax.axhspan(rw_mean - rw_std, rw_mean + rw_std, color=RANDOM_COLOR, alpha=0.2)
    ax.axhline(rw_mean, color=RANDOM_COLOR, linewidth=1.2, linestyle='--',
               label=f'Random ensemble mean')

    ax.set_xticks(x)
    ax.set_xticklabels(topos, rotation=45, ha='right')
    ax.set_ylabel('Ridge width $R_w$')
    ax.set_ylim(0, 0.5)

    legend_elements = [
        Patch(facecolor=GROUP_A_COLOR, edgecolor='black', linewidth=0.5, label='Group A (controls)'),
        Patch(facecolor=GROUP_B_COLOR, edgecolor='black', linewidth=0.5, label='Group B (designed)'),
        Patch(facecolor=RANDOM_COLOR, alpha=0.3, label='Random ensemble $\\mu \\pm \\sigma$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'fig1_rw_distribution.png'))
    plt.close(fig)
    print('  fig1_rw_distribution.png')


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Phase 2 drift curves
# ══════════════════════════════════════════════════════════════════════

def fig2_drift_curves(sigma_levels, drift_results):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    styles = {
        'T0':   {'color': BASELINE_COLOR, 'marker': 'o', 'label': 'T0 (baseline)'},
        'T6':   {'color': GROUP_B_COLOR,  'marker': 's', 'label': 'T6 (nested loop)'},
        'rw16': {'color': RANDOM_COLOR,   'marker': '^', 'label': 'rw16 (best random)'},
        'rw4':  {'color': '#E5AE38',      'marker': 'D', 'label': 'rw4 (2nd random)'},
    }

    for topo, curve in drift_results.items():
        s = styles[topo]
        ax.plot(sigma_levels, curve, marker=s['marker'], color=s['color'],
                label=s['label'], linewidth=1.2, markersize=4)

    ax.set_xlabel('Drift magnitude $\\sigma_d$')
    ax.set_ylabel('Ridge width $R_w$')
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'fig2_drift_curves.png'))
    plt.close(fig)
    print('  fig2_drift_curves.png')


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Structural predictor scatter
# ══════════════════════════════════════════════════════════════════════

def fig3_predictor(pred_data):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    alive = [d for d in pred_data if d['alive']]
    dead = [d for d in pred_data if not d['alive']]

    # Jitter for overlapping points
    rng = np.random.RandomState(42)

    for group, color, label in [(alive, RANDOM_COLOR, 'Alive ($R_w > 0$)'),
                                 (dead, DEAD_COLOR, 'Dead ($R_w = 0$)')]:
        fnet = [d['f_net'] + rng.uniform(-0.15, 0.15) for d in group]
        rw = [d['R_w'] for d in group]
        ax.scatter(fnet, rw, c=color, edgecolor='black', linewidth=0.5,
                   s=35, label=label, zorder=3)

    # Threshold line
    ax.axvline(0.5, color='red', linewidth=1, linestyle='--', alpha=0.7,
               label='$f_{\\mathrm{net}} = 1$ threshold')

    ax.set_xlabel('E-coupler net flow $f_{\\mathrm{net}}$')
    ax.set_ylabel('Ridge width $R_w$')
    ax.set_xlim(-5, 3.5)
    ax.set_ylim(-0.02, 0.45)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'fig3_predictor.png'))
    plt.close(fig)
    print('  fig3_predictor.png')


# ══════════════════════════════════════════════════════════════════════
# Figure 4: Random ensemble R_w histogram
# ══════════════════════════════════════════════════════════════════════

def fig4_random_bimodality(random_wirings):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    rw_vals = [random_wirings[i]['R_w'] for i in sorted(random_wirings)]

    bins = np.arange(-0.025, 0.475, 0.05)
    alive_vals = [v for v in rw_vals if v > 0]
    dead_vals = [v for v in rw_vals if v == 0]

    ax.hist(dead_vals, bins=bins, color=DEAD_COLOR, edgecolor='black',
            linewidth=0.5, label=f'Dead ($n={len(dead_vals)}$)')
    ax.hist(alive_vals, bins=bins, color=RANDOM_COLOR, edgecolor='black',
            linewidth=0.5, label=f'Alive ($n={len(alive_vals)}$)', alpha=0.8)

    # T0 baseline marker
    ax.axvline(0.372, color=BASELINE_COLOR, linewidth=1.5, linestyle='--',
               label='T0 baseline')

    ax.set_xlabel('Ridge width $R_w$')
    ax.set_ylabel('Count')
    ax.set_xlim(-0.05, 0.45)
    ax.legend(loc='upper right', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'fig4_random_bimodality.png'))
    plt.close(fig)
    print('  fig4_random_bimodality.png')


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Loading Phase 1 data...')
    named, random_wirings = load_phase1()

    print('Loading Phase 2 data...')
    sigma_levels, drift_results = load_phase2()

    print('Loading predictor data...')
    pred_data = load_predictor_data()

    print('Generating figures...')
    fig1_rw_distribution(named, random_wirings)
    fig2_drift_curves(sigma_levels, drift_results)
    fig3_predictor(pred_data)
    fig4_random_bimodality(random_wirings)

    print(f'\nAll figures saved to {FIG_DIR}/')
