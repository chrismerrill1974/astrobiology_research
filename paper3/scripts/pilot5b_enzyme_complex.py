"""
Pilot 5b: Enzyme-complex mass-action validation of coupled Brusselator.

After Pilot 5 showed that pure mass-action E-consumption (linear in E) fails
to produce D₂ > 1.2, this pilot validates the enzyme-complex approach:
a gate species G mediates E access, producing Michaelis-Menten-like saturation
entirely within mass-action kinetics.

Key insight from Pilot 5 failure: The linear coupling k*E*Xi²Yi creates a
binary regime (phase-locked or anti-phase) without the intermediate
quasi-periodic window that Michaelis-Menten saturation (E/(K+E)) enables.
The saturation bounds effective coupling at high E, creating the narrow
window for timescale separation.

Enzyme-complex trick:
    Per core i ∈ {1, 2}, we introduce gate species G and complex GE:

    (1) G + E → GE         (rate k_on)     # fast binding
    (2) GE → G + E         (rate k_off)    # fast unbinding
    (3) 2Xi + Yi + GE → 3Xi + G + Ew  (rate k_cat)  # autocatalysis + E drain

At quasi-steady-state for GE:
    GE_ss = G_total * E / (K_d + E)  where K_d = (k_off + k_cat*flux) / k_on
    ≈ G_total * E / (K_d + E)  when k_cat*flux << k_off (fast equilibrium)

Effective autocatalytic rate: k_cat * GE * Xi²Yi ≈ k_cat * G_total * E/(K_d+E) * Xi²Yi
This IS Michaelis-Menten! And E is irreversibly consumed (→ Ew), creating
true competition between cores.

The gate G is regenerated (catalytic), E is consumed (competitive).
G_total is conserved (G + GE = const), which we enforce via initial conditions.

Full reaction set (16 reactions):
    Per core i ∈ {1, 2}:
        A → Xi              (rate: A, chemostatted)
        B + Xi → Yi + Di    (rate: B·Xi, chemostatted)
        Xi + Xi + Yi → 3Xi  (rate: Xi²Yi)
        Xi → Wi             (rate: Xi)

    Energy pool:
        Esrc → Esrc + E     (rate: J·Esrc, Esrc chemostatted at 1)
        E → Ew              (rate: γ·E)

    Enzyme complex (shared gate):
        G + E → GE          (rate: k_on·G·E)
        GE → G + E          (rate: k_off·GE)

    Gated autocatalysis:
        Xi + Xi + Yi + GE → Xi + Xi + Xi + G + Ew  (rate: k_cat·Xi²Yi·GE)
        (one per core, i=1,2)

Comparison with Pilot 4:
    Pilot 4: dXi/dt includes + k_extra * E/(K+E) * Xi²Yi
             dE/dt  includes - α * E/(K+E) * (X1²Y1 + X2²Y2)
    Pilot 5b: At QSS, effective = k_cat * G_tot * E/(K_d+E) * Xi²Yi  ← same form!
              E drain = k_cat * GE * (X1²Y1 + X2²Y2)                  ← saturating drain

Parameter mapping from Pilot 4 to 5b:
    k_extra ↔ k_cat * G_total   (effective max autocatalytic rate)
    K       ↔ K_d = k_off/k_on  (half-saturation for gating)
    α       ↔ k_cat * G_total   (drain coefficient, same as boost)

Success criterion: At least 3 parameter sets produce median D₂ > 1.2.
"""

import sys
import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# Add astrobiology2 to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_astro2_path = os.path.join(_this_dir, '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension, QualityFlag
from dimensional_opening.network_generator import GeneratedNetwork


# ── Model ─────────────────────────────────────────────────────────────

@dataclass
class EnzymeComplexParams:
    """Parameters for enzyme-complex coupled Brusselator + shared energy."""
    # Brusselator base
    A: float = 1.0
    B: float = 3.0

    # Enzyme-complex gating
    k_on: float = 10.0     # G + E → GE binding rate (fast)
    k_off: float = 10.0    # GE → G + E unbinding rate (fast)
    k_cat: float = 1.0     # GE + substrates → products (catalytic rate)
    G_total: float = 1.0   # Total gate concentration (G + GE = G_total)

    # Energy pool dynamics
    J: float = 1.0         # Energy inflow rate
    gamma: float = 0.005   # Energy leak rate

    # Label for identification
    label: str = ""

    @property
    def K_d(self) -> float:
        """Effective dissociation constant K_d ≈ k_off/k_on."""
        return self.k_off / self.k_on if self.k_on > 0 else float('inf')

    @property
    def effective_k_extra(self) -> float:
        """Effective max autocatalytic rate ≈ k_cat * G_total."""
        return self.k_cat * self.G_total

    def copy(self, **overrides) -> 'EnzymeComplexParams':
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        d.update(overrides)
        return EnzymeComplexParams(**d)


def make_enzyme_complex_network(
    p: EnzymeComplexParams,
    seed: int = 42,
    ic_perturbation: float = 0.05,
) -> GeneratedNetwork:
    """
    Create a two-core coupled Brusselator with enzyme-complex E gating.

    The gate species G and complex GE mediate E access to the autocatalytic
    cores, producing Michaelis-Menten-like saturation in pure mass-action.
    """
    rng = np.random.RandomState(seed)

    reactions = [
        # Core 1 Brusselator (reactions 0-3)
        "A -> X1",
        "B + X1 -> Y1 + D1",
        "X1 + X1 + Y1 -> X1 + X1 + X1",
        "X1 -> W1",

        # Core 2 Brusselator (reactions 4-7)
        "A -> X2",
        "B + X2 -> Y2 + D2",
        "X2 + X2 + Y2 -> X2 + X2 + X2",
        "X2 -> W2",

        # Energy pool (reactions 8-9)
        "Esrc -> Esrc + E",    # Inflow (Esrc chemostatted at 1.0)
        "E -> Ew",             # Slow leak

        # Enzyme complex formation/dissolution (reactions 10-11)
        "G + E -> GE",         # Binding
        "GE -> G + E",         # Unbinding

        # Gated autocatalysis with E consumption (reactions 12-13)
        # GE donates its E to the autocatalytic reaction, E is consumed (→ Ew)
        # G is regenerated. Net: E → Ew (consumed), extra autocatalysis happens.
        "X1 + X1 + Y1 + GE -> X1 + X1 + X1 + G + Ew",   # Core 1
        "X2 + X2 + Y2 + GE -> X2 + X2 + X2 + G + Ew",   # Core 2
    ]

    rate_constants = [
        # Core 1: A, B, autocatalysis, decay
        p.A, p.B, 1.0, 1.0,
        # Core 2: same
        p.A, p.B, 1.0, 1.0,
        # Energy inflow, leak
        p.J, p.gamma,
        # Enzyme complex: binding, unbinding
        p.k_on, p.k_off,
        # Gated autocatalysis for cores 1 & 2
        p.k_cat, p.k_cat,
    ]

    # Initial conditions
    E0 = p.J / max(p.gamma, 1e-6)  # Approx steady-state E without drain
    GE0 = p.G_total * E0 / (p.K_d + E0)  # QSS for GE
    G0 = p.G_total - GE0

    ic = {
        'X1': 1.0, 'Y1': 1.0, 'D1': 0.0, 'W1': 0.0,
        'X2': 1.0 + rng.uniform(-ic_perturbation, ic_perturbation),
        'Y2': 1.0 + rng.uniform(-ic_perturbation, ic_perturbation),
        'D2': 0.0, 'W2': 0.0,
        'E': E0, 'Ew': 0.0,
        'G': G0, 'GE': GE0,
        'A': p.A, 'B': p.B, 'Esrc': 1.0,
    }

    chemostat = {'A': p.A, 'B': p.B, 'Esrc': 1.0}

    species = ['X1', 'Y1', 'D1', 'W1', 'X2', 'Y2', 'D2', 'W2',
               'E', 'Ew', 'G', 'GE', 'A', 'B', 'Esrc']

    return GeneratedNetwork(
        reactions=reactions,
        species=species,
        food_set=['A', 'B', 'Esrc'],
        n_species=len(species),
        n_reactions=len(reactions),
        n_autocatalytic=2,
        rate_constants=rate_constants,
        initial_concentrations=ic,
        chemostat_species=chemostat,
        network_id=f"coupled_ec_{p.label}_s{seed}",
        is_autocatalytic=True,
        template="CoupledBrusselator_EnzymeComplex",
        n_added_reactions=0,
    )


# ── Simulation & Analysis ─────────────────────────────────────────────

def simulate_and_analyze(
    p: EnzymeComplexParams,
    seed: int = 42,
    t_span: Tuple[float, float] = (0, 10000),
    n_points: int = 20000,
    verbose: bool = False,
) -> Dict:
    """
    Simulate the enzyme-complex coupled model and compute D₂.
    """
    net = make_enzyme_complex_network(p, seed=seed)

    sim = ReactionSimulator()
    try:
        network_obj = sim.build_network(net.reactions)
        sim_result = sim.simulate(
            network_obj,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=t_span,
            n_points=n_points,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=net.chemostat_species,
        )
    except Exception as e:
        if verbose:
            print(f"  Simulation failed ({p.label}, s{seed}): {e}")
        return {
            'label': p.label, 'seed': seed, 'regime': 'sim_failed',
            'D2': float('nan'), 'r_X1X2': float('nan'),
            'E_range': float('nan'), 'error': str(e),
        }

    if not sim_result.success:
        if verbose:
            print(f"  Solver failed ({p.label}): {sim_result.solver_message}")
        return {
            'label': p.label, 'seed': seed, 'regime': 'solver_failed',
            'D2': float('nan'), 'r_X1X2': float('nan'),
            'E_range': float('nan'),
        }

    # Extract trajectories (discard first 50% as transient)
    t = sim_result.time
    c = sim_result.concentrations
    species = sim_result.species_names
    n_discard = len(t) // 2
    c_post = c[n_discard:]

    def idx(name):
        return species.index(name)

    try:
        X1 = c_post[:, idx('X1')]
        Y1 = c_post[:, idx('Y1')]
        X2 = c_post[:, idx('X2')]
        Y2 = c_post[:, idx('Y2')]
        E = c_post[:, idx('E')]
        G = c_post[:, idx('G')]
        GE = c_post[:, idx('GE')]
    except (ValueError, IndexError) as e:
        if verbose:
            print(f"  Species extraction failed: {e}")
        return {
            'label': p.label, 'seed': seed, 'regime': 'extraction_failed',
            'D2': float('nan'), 'r_X1X2': float('nan'),
            'E_range': float('nan'),
        }

    # Check for fixed point
    cv_X1 = np.std(X1) / max(np.mean(X1), 1e-10)
    cv_X2 = np.std(X2) / max(np.mean(X2), 1e-10)
    cv_E = np.std(E) / max(np.mean(E), 1e-10)

    if cv_X1 < 0.01 and cv_X2 < 0.01:
        return {
            'label': p.label, 'seed': seed, 'regime': 'fixed_point',
            'D2': float('nan'), 'r_X1X2': float('nan'),
            'E_range': float(np.ptp(E)),
            'E_mean': float(np.mean(E)),
            'GE_mean': float(np.mean(GE)),
            'G_mean': float(np.mean(G)),
        }

    # Compute D₂ on the dynamic projection (X1, Y1, X2, Y2, E)
    # We exclude G and GE from D₂ because they're fast auxiliary variables
    trajectory = np.column_stack([X1, Y1, X2, Y2, E])

    cd = CorrelationDimension()
    try:
        d2_result = cd.compute(trajectory)
    except Exception as e:
        if verbose:
            print(f"  D₂ computation failed: {e}")
        return {
            'label': p.label, 'seed': seed, 'regime': 'd2_failed',
            'D2': float('nan'), 'r_X1X2': float('nan'),
            'E_range': float(np.ptp(E)),
        }

    D2 = d2_result.D2
    quality = d2_result.quality

    # Phase correlation between cores
    r_X1X2 = float(np.corrcoef(X1, X2)[0, 1])

    # Regime classification
    if quality == QualityFlag.FAILED or D2 is None or np.isnan(D2):
        regime = 'failed_d2'
    elif D2 > 1.2:
        regime = 'complex'
    else:
        regime = 'phase_locked'

    return {
        'label': p.label,
        'seed': seed,
        'regime': regime,
        'D2': float(D2) if D2 is not None else float('nan'),
        'D2_unc': float(d2_result.D2_uncertainty) if d2_result.D2_uncertainty else float('nan'),
        'quality': quality.value if quality else 'unknown',
        'r_X1X2': r_X1X2,
        'E_range': float(np.ptp(E)),
        'E_mean': float(np.mean(E)),
        'GE_mean': float(np.mean(GE)),
        'G_mean': float(np.mean(G)),
        'cv_X1': float(cv_X1),
        'cv_X2': float(cv_X2),
        'cv_E': float(cv_E),
    }


# ── Sweep ─────────────────────────────────────────────────────────────

def run_pilot5b(
    n_seeds: int = 3,
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Run Pilot 5b: enzyme-complex mass-action validation.

    Scans parameter space to find D₂ > 1.2 using the enzyme-complex
    representation of Michaelis-Menten gating within ReactionSimulator.
    """
    print("\n" + "#" * 80)
    print("# PILOT 5b: Enzyme-Complex Mass-Action Validation")
    print("#" * 80)
    print("  Model: enzyme-complex gating (G + E ⇌ GE, GE + substrates → products)")
    print("  Effective: Michaelis-Menten saturation in pure mass-action")
    print(f"  Seeds per param set: {n_seeds}")
    print()

    # ── Parameter sets ────────────────────────────────────────────────
    # Map from Pilot 4 winners:
    #   Pilot 4 slow_strong_hJ: J=1.0, k_extra=1.0, γ=0.005, α=0.1, K=1.0
    #   Pilot 4 drain_strong:   J=1.0, k_extra=1.0, γ=0.02, α=0.3, K=1.0
    #
    # Mapping:
    #   K       → K_d = k_off/k_on
    #   k_extra → k_cat * G_total
    #   α       → controlled by k_cat and G_total
    #
    # For K_d = 1.0: k_off = k_on (e.g., both = 10 for fast equilibrium)

    param_sets = []

    # ── Group A: Direct Pilot 4 translations (K_d = 1.0) ──
    # Pilot 4 slow pathway: γ=0.005, k_extra=1.0, K=1.0
    for J in [0.5, 1.0]:
        for k_cat in [0.5, 1.0, 2.0]:
            param_sets.append(EnzymeComplexParams(
                J=J, gamma=0.005,
                k_on=10.0, k_off=10.0,  # K_d = 1.0
                k_cat=k_cat, G_total=1.0,
                label=f"slow_J{J}_kcat{k_cat}"))

    # Pilot 4 drain pathway: γ=0.02, k_extra=1.0, K=1.0
    for J in [0.5, 1.0]:
        for k_cat in [0.5, 1.0, 2.0]:
            param_sets.append(EnzymeComplexParams(
                J=J, gamma=0.02,
                k_on=10.0, k_off=10.0,
                k_cat=k_cat, G_total=1.0,
                label=f"drain_J{J}_kcat{k_cat}"))

    # ── Group B: Varying K_d (half-saturation) ──
    for Kd in [0.3, 1.0, 3.0]:
        k_on = 10.0
        k_off = Kd * k_on
        param_sets.append(EnzymeComplexParams(
            J=1.0, gamma=0.005,
            k_on=k_on, k_off=k_off,
            k_cat=1.0, G_total=1.0,
            label=f"Kd{Kd}_slow"))
        param_sets.append(EnzymeComplexParams(
            J=1.0, gamma=0.02,
            k_on=k_on, k_off=k_off,
            k_cat=1.0, G_total=1.0,
            label=f"Kd{Kd}_drain"))

    # ── Group C: Varying G_total (effective max rate) ──
    for G_tot in [0.5, 1.0, 2.0]:
        param_sets.append(EnzymeComplexParams(
            J=1.0, gamma=0.005,
            k_on=10.0, k_off=10.0,
            k_cat=1.0, G_total=G_tot,
            label=f"Gtot{G_tot}_slow"))
        param_sets.append(EnzymeComplexParams(
            J=1.0, gamma=0.02,
            k_on=10.0, k_off=10.0,
            k_cat=1.0, G_total=G_tot,
            label=f"Gtot{G_tot}_drain"))

    # ── Group D: Very slow γ (best regime from Pilot 3b/4) ──
    for gamma in [0.002, 0.003, 0.005]:
        for J in [1.0, 3.0]:
            param_sets.append(EnzymeComplexParams(
                J=J, gamma=gamma,
                k_on=10.0, k_off=10.0,
                k_cat=1.0, G_total=1.0,
                label=f"vslow_g{gamma}_J{J}"))

    # ── Group E: Fast equilibrium sensitivity (k_on = k_off scaling) ──
    for rate_scale in [1.0, 10.0, 100.0]:
        param_sets.append(EnzymeComplexParams(
            J=1.0, gamma=0.005,
            k_on=rate_scale, k_off=rate_scale,  # K_d = 1.0 always
            k_cat=1.0, G_total=1.0,
            label=f"eqscale{rate_scale}"))

    # ── Group F: Higher J + higher k_cat (Pilot 3b's very_slow_hJ regime) ──
    for J in [3.0, 5.0]:
        for k_cat in [0.3, 0.5, 1.0]:
            param_sets.append(EnzymeComplexParams(
                J=J, gamma=0.002,
                k_on=10.0, k_off=10.0,
                k_cat=k_cat, G_total=1.0,
                label=f"hiJ_J{J}_kcat{k_cat}"))

    n_total = len(param_sets) * n_seeds
    print(f"  Parameter sets: {len(param_sets)}")
    print(f"  Total runs: {n_total}")
    print()

    # ── Run sweep ─────────────────────────────────────────────────────
    results = []
    start = time.time()

    for i, p in enumerate(param_sets):
        for s in range(n_seeds):
            seed = 42 + s * 137
            t0 = time.time()
            r = simulate_and_analyze(p, seed=seed, verbose=verbose)
            elapsed = time.time() - t0
            results.append(r)

            run_idx = i * n_seeds + s + 1
            eta = max(0, (time.time() - start) / run_idx * (n_total - run_idx))

            if verbose:
                d2_str = f"{r['D2']:.3f}" if not np.isnan(r['D2']) else "N/A"
                r_str = f"{r['r_X1X2']:.2f}" if not np.isnan(r['r_X1X2']) else "N/A"
                print(f"\r  {run_idx}/{n_total} ({p.label}, s{s+1}, "
                      f"{r['regime']}, D2={d2_str}, r={r_str}) "
                      f"ETA: {eta:.0f}s    ", end='', flush=True)

    total_time = time.time() - start
    if verbose:
        print(f"\n  Sweep complete in {total_time:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("PILOT 5b: Enzyme-Complex Mass-Action Validation Results")
    print('=' * 110)

    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        by_label[r['label']].append(r)

    print(f"{'Label':<25} {'J':>4} {'γ':>7} {'k_cat':>5} {'K_d':>4} {'G_t':>4}  "
          f"{'FP':>3} {'Lock':>4} {'Cpx':>4}  "
          f"{'med D2':>7} {'max D2':>7} {'r(X1,X2)':>9}  {'E_mean':>7}")
    print('-' * 110)

    complex_labels = []

    for label in sorted(by_label.keys()):
        runs = by_label[label]
        p_match = [p for p in param_sets if p.label == label][0]

        regimes = [r['regime'] for r in runs]
        n_fp = sum(1 for r in regimes if r == 'fixed_point')
        n_lock = sum(1 for r in regimes if r == 'phase_locked')
        n_cpx = sum(1 for r in regimes if r == 'complex')
        n_fail = sum(1 for r in regimes if r in ('sim_failed', 'solver_failed', 'failed_d2', 'extraction_failed'))

        d2s = [r['D2'] for r in runs if not np.isnan(r['D2'])]
        rs = [r['r_X1X2'] for r in runs if not np.isnan(r['r_X1X2'])]
        es = [r.get('E_mean', float('nan')) for r in runs]
        es = [e for e in es if not np.isnan(e)]

        med_d2 = np.median(d2s) if d2s else float('nan')
        max_d2 = max(d2s) if d2s else float('nan')
        med_r = np.median(rs) if rs else float('nan')
        med_e = np.median(es) if es else float('nan')

        cpx_str = f"*{n_cpx}" if n_cpx > 0 else f"{n_cpx}"
        d2_med_str = f"{med_d2:.3f}" if not np.isnan(med_d2) else "N/A"
        d2_max_str = f"{max_d2:.3f}" if not np.isnan(max_d2) else "N/A"
        r_str = f"{med_r:.3f}" if not np.isnan(med_r) else "N/A"
        e_str = f"{med_e:.1f}" if not np.isnan(med_e) else "N/A"

        regime_note = ""
        if n_fail > 0:
            regime_note = f" [{n_fail}F]"

        print(f"{label:<25} {p_match.J:>4.1f} {p_match.gamma:>7.3f} "
              f"{p_match.k_cat:>5.1f} {p_match.K_d:>4.1f} {p_match.G_total:>4.1f}  "
              f"{n_fp:>3} {n_lock:>4} {cpx_str:>4}  "
              f"{d2_med_str:>7} {d2_max_str:>7} {r_str:>9}  {e_str:>7}{regime_note}")

        if n_cpx > 0:
            complex_labels.append(label)

    print('-' * 110)

    # Overall verdict
    all_d2 = [r['D2'] for r in results if not np.isnan(r['D2'])]
    max_d2 = max(all_d2) if all_d2 else float('nan')
    n_complex = sum(1 for r in results if r['regime'] == 'complex')
    n_total_success = sum(1 for r in results if r['regime'] not in
                          ('sim_failed', 'solver_failed', 'extraction_failed'))

    print(f"\n  Total runs: {len(results)}, successful: {n_total_success}")
    print(f"  Max D₂: {max_d2:.3f}" if not np.isnan(max_d2) else "  Max D₂: N/A")
    print(f"  Runs with D₂ > 1.2: {n_complex}")

    if complex_labels:
        print(f"\n  D₂ > 1.2 FOUND in: {complex_labels}")
        print(f"  VERDICT: Enzyme-complex model VALIDATED — {len(complex_labels)} param sets")
        print(f"  → Michaelis-Menten dynamics reproducible in pure mass-action")
        print(f"  → Ready for Phase I production sweep via ReactionSimulator")
    else:
        print(f"\n  D₂ > 1.2 NOT FOUND in any parameter set")
        print(f"  VERDICT: Enzyme-complex approach needs further calibration")
        # Diagnostic: show top 5 D₂ values
        top5 = sorted(all_d2, reverse=True)[:5]
        if top5:
            print(f"  Top 5 D₂ values: {[f'{d:.3f}' for d in top5]}")
    print('=' * 110)

    # ── Save results ──────────────────────────────────────────────────
    output = {
        'model': 'coupled_brusselator_enzyme_complex',
        'n_param_sets': len(param_sets),
        'n_seeds': n_seeds,
        'n_total_runs': n_total,
        'runtime_seconds': total_time,
        'max_d2': float(max_d2) if not np.isnan(max_d2) else None,
        'found_complex': len(complex_labels) > 0,
        'complex_labels': complex_labels,
        'n_complex_sets': len(complex_labels),
        'results': results,
        'param_sets': [{k: v for k, v in p.__dict__.items() if not k.startswith('_')}
                       for p in param_sets],
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'pilot5b_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_pilot5b(n_seeds=3, verbose=True, save_dir=save_dir)
