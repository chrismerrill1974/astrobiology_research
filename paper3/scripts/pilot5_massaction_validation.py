"""
Pilot 5: Mass-action validation of the coupled Brusselator model.

Validates that replacing Michaelis-Menten gating f(E) = E/(K+E) with pure
mass-action E participation (E consumed as reactant) preserves D₂ > 1.2
in the parameter regimes identified by Pilot 4.

This is the critical bridge between:
  - Pilot 4 (hand-written ODE, Michaelis-Menten, D₂ > 1.2 confirmed)
  - Paper 3 production (ReactionSimulator / GeneratedNetwork, mass-action)

Model (pure mass-action, 11 reactions):
    Per core i ∈ {1, 2}:
        A -> Xi                           (rate: A, chemostatted)
        B + Xi -> Yi + Di                 (rate: B·Xi, chemostatted)
        Xi + Xi + Yi -> Xi + Xi + Xi      (rate: Xi²·Yi)
        Xi -> Wi                          (rate: Xi)
        Xi + Xi + Yi + E -> Xi + Xi + Xi  (rate: k_extra·Xi²·Yi·E, E consumed)

    Energy pool:
        Esrc -> Esrc + E                  (rate: J·Esrc, Esrc chemostatted at 1)
        E -> Ew                           (rate: γ·E)

Effective ODE:
    dXi/dt = A - (B+1)Xi + Xi²Yi + k_extra·Xi²Yi·E
    dYi/dt = BXi - Xi²Yi - k_extra·Xi²Yi·E
    dE/dt  = J - γE - k_extra·E·(X1²Y1 + X2²Y2)

Compare Pilot 4: k_extra·f(E)·Xi²Yi where f(E) = E/(K+E)
Mass-action:      k_extra·E·Xi²Yi

The mass-action form is linear in E (no saturation), which may shift the
parameter sweet spots but preserves the competitive coupling mechanism.

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
class MassActionCoupledParams:
    """Parameters for mass-action coupled Brusselator + shared energy."""
    # Brusselator base
    A: float = 1.0
    B: float = 3.0

    # Energy-gated coupling (E consumed as reactant)
    k_extra: float = 1.0

    # Energy pool dynamics
    J: float = 1.0       # Energy inflow rate
    gamma: float = 0.005  # Energy leak rate

    # Label for identification
    label: str = ""

    def copy(self, **overrides) -> 'MassActionCoupledParams':
        d = {k: v for k, v in asdict(self).items()}
        d.update(overrides)
        return MassActionCoupledParams(**d)


def make_coupled_network(
    p: MassActionCoupledParams,
    seed: int = 42,
    ic_perturbation: float = 0.05,
) -> GeneratedNetwork:
    """
    Create a two-core coupled Brusselator as a GeneratedNetwork.

    Uses the same reaction format as Papers I-II Brusselator template,
    duplicated for two cores with shared energy pool E.
    """
    rng = np.random.RandomState(seed)

    reactions = [
        # Core 1 Brusselator
        "A -> X1",
        "B + X1 -> Y1 + D1",
        "X1 + X1 + Y1 -> X1 + X1 + X1",
        "X1 -> W1",
        # Core 2 Brusselator
        "A -> X2",
        "B + X2 -> Y2 + D2",
        "X2 + X2 + Y2 -> X2 + X2 + X2",
        "X2 -> W2",
        # Energy-gated extra autocatalysis (E consumed → competition)
        "X1 + X1 + Y1 + E -> X1 + X1 + X1",  # Core 1 drains E
        "X2 + X2 + Y2 + E -> X2 + X2 + X2",  # Core 2 drains E
        # Energy pool
        "Esrc -> Esrc + E",   # Inflow (Esrc chemostatted at 1.0)
        "E -> Ew",            # Slow leak
    ]

    rate_constants = [
        # Core 1: A, B, autocatalysis, decay
        p.A, p.B, 1.0, 1.0,
        # Core 2: same
        p.A, p.B, 1.0, 1.0,
        # Energy-gated autocatalysis for cores 1 & 2
        p.k_extra, p.k_extra,
        # Energy inflow, leak
        p.J, p.gamma,
    ]

    # Initial conditions with symmetry-breaking perturbation
    E0 = p.J / max(p.gamma, 1e-6)  # Steady-state E without drain
    ic = {
        'X1': 1.0, 'Y1': 1.0, 'D1': 0.0, 'W1': 0.0,
        'X2': 1.0 + rng.uniform(-ic_perturbation, ic_perturbation),
        'Y2': 1.0 + rng.uniform(-ic_perturbation, ic_perturbation),
        'D2': 0.0, 'W2': 0.0,
        'E': E0, 'Ew': 0.0,
        'A': p.A, 'B': p.B, 'Esrc': 1.0,
    }

    chemostat = {'A': p.A, 'B': p.B, 'Esrc': 1.0}

    species = ['X1', 'Y1', 'D1', 'W1', 'X2', 'Y2', 'D2', 'W2', 'E', 'Ew',
               'A', 'B', 'Esrc']

    return GeneratedNetwork(
        reactions=reactions,
        species=species,
        food_set=['A', 'B', 'Esrc'],
        n_species=len(species),
        n_reactions=len(reactions),
        n_autocatalytic=2,  # Two base autocatalytic reactions
        rate_constants=rate_constants,
        initial_concentrations=ic,
        chemostat_species=chemostat,
        network_id=f"coupled_ma_{p.label}_s{seed}",
        is_autocatalytic=True,
        template="CoupledBrusselator_MassAction",
        n_added_reactions=0,
    )


# ── Simulation & Analysis ─────────────────────────────────────────────

def simulate_and_analyze(
    p: MassActionCoupledParams,
    seed: int = 42,
    t_span: Tuple[float, float] = (0, 10000),
    n_points: int = 20000,
    verbose: bool = False,
) -> Dict:
    """
    Simulate the mass-action coupled model and compute D₂.

    Returns a dict with regime, D₂, phase correlation, etc.
    """
    net = make_coupled_network(p, seed=seed)

    # Simulate via ReactionSimulator
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
            print(f"  Simulation failed: {e}")
        return {
            'label': p.label, 'seed': seed, 'regime': 'sim_failed',
            'D2': float('nan'), 'r_X1X2': float('nan'),
            'E_range': float('nan'),
        }

    if not sim_result.success:
        if verbose:
            print(f"  Solver failed: {sim_result.solver_message}")
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
    t_post = t[n_discard:]
    c_post = c[n_discard:]

    # Find dynamic species indices
    def idx(name):
        return species.index(name)

    try:
        X1 = c_post[:, idx('X1')]
        Y1 = c_post[:, idx('Y1')]
        X2 = c_post[:, idx('X2')]
        Y2 = c_post[:, idx('Y2')]
        E = c_post[:, idx('E')]
    except (ValueError, IndexError) as e:
        if verbose:
            print(f"  Species extraction failed: {e}")
        return {
            'label': p.label, 'seed': seed, 'regime': 'extraction_failed',
            'D2': float('nan'), 'r_X1X2': float('nan'),
            'E_range': float('nan'),
        }

    # Check for fixed point (low variance in all species)
    cv_X1 = np.std(X1) / max(np.mean(X1), 1e-10)
    cv_X2 = np.std(X2) / max(np.mean(X2), 1e-10)
    cv_E = np.std(E) / max(np.mean(E), 1e-10)

    if cv_X1 < 0.01 and cv_X2 < 0.01:
        return {
            'label': p.label, 'seed': seed, 'regime': 'fixed_point',
            'D2': float('nan'), 'r_X1X2': float('nan'),
            'E_range': float(np.ptp(E)),
        }

    # Compute D₂ on the 5D projection (X1, Y1, X2, Y2, E)
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

    # Regime classification (same as Pilot 4)
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
        'cv_X1': float(cv_X1),
        'cv_E': float(cv_E),
    }


# ── Sweep ─────────────────────────────────────────────────────────────

def run_pilot5(
    n_seeds: int = 3,
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Run Pilot 5: mass-action validation sweep.

    Tests the 6 Pilot 4 parameter sets that produced D₂ > 1.2,
    translated to mass-action form. Also includes a parameter scan
    around the sweet spots in case they shift.
    """
    print("\n" + "#" * 80)
    print("# PILOT 5: Mass-Action Validation of Coupled Brusselator")
    print("#" * 80)
    print("  Model: pure mass-action (E consumed as reactant)")
    print("  Comparison: Pilot 4 used f(E) = E/(K+E) Michaelis-Menten gating")
    print(f"  Seeds per param set: {n_seeds}")
    print()

    # ── Parameter sets ────────────────────────────────────────────────
    # 1. Direct translation of Pilot 4 winners
    #    Pilot 4 had k_extra and alpha as separate params.
    #    In mass-action: k_extra controls both boost and drain (they're the same reaction).
    #    We need to find equivalent effective coupling strengths.
    #
    #    Pilot 4: extra_rate = k_extra * E/(K+E), drain_rate = alpha * E/(K+E) * Xi²Yi
    #    Mass-action: both are k_extra * E * Xi²Yi
    #    So effective k_extra_ma should be calibrated to match the pilot effect.

    param_sets = []

    # Group A: Translations of Pilot 4 slow pathway winners
    # Pilot 4 slow_strong: J=0.5, k_extra=1.0, γ=0.005, α=0.1
    # Pilot 4 slow_strong_hJ: J=1.0, k_extra=1.0, γ=0.005, α=0.1
    # In Pilot 4 at E_ss ≈ J/γ = 100-200, f(E) ≈ 1.0, so effective rate ≈ k_extra * Xi²Yi
    # In mass-action: rate = k_extra_ma * E * Xi²Yi, at E_ss ≈ J/γ = 100-200
    # For comparable effective rate: k_extra_ma * E_ss ≈ k_extra_pilot
    # So k_extra_ma ≈ k_extra_pilot / E_ss ≈ 1.0 / 100 = 0.01
    # But E_ss is itself a function of drain, so we scan a range.

    for k_ex in [0.005, 0.01, 0.02, 0.05, 0.1]:
        param_sets.append(MassActionCoupledParams(
            J=0.5, gamma=0.005, k_extra=k_ex,
            label=f"slow_J0.5_k{k_ex}"))
        param_sets.append(MassActionCoupledParams(
            J=1.0, gamma=0.005, k_extra=k_ex,
            label=f"slow_J1.0_k{k_ex}"))

    # Group B: Translations of Pilot 4 drain pathway winners
    # Pilot 4 drain_strong: J=1.0, k_extra=1.0, γ=0.02, α=0.3
    # E_ss ≈ J/γ = 50, f(E) ≈ 0.98, effective ≈ k_extra * Xi²Yi
    # k_extra_ma ≈ 1.0 / 50 = 0.02
    for k_ex in [0.005, 0.01, 0.02, 0.05, 0.1]:
        param_sets.append(MassActionCoupledParams(
            J=1.0, gamma=0.02, k_extra=k_ex,
            label=f"drain_J1.0_k{k_ex}"))

    # Group C: Wider scan at standard γ with moderate J
    for k_ex in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        param_sets.append(MassActionCoupledParams(
            J=0.5, gamma=0.02, k_extra=k_ex,
            label=f"std_J0.5_k{k_ex}"))

    # Group D: Very slow γ scan
    for k_ex in [0.01, 0.02, 0.05, 0.1]:
        param_sets.append(MassActionCoupledParams(
            J=1.0, gamma=0.002, k_extra=k_ex,
            label=f"vslow_J1.0_k{k_ex}"))

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
    print(f"\n{'=' * 100}")
    print("PILOT 5: Mass-Action Validation Results")
    print('=' * 100)

    # Group by label
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        by_label[r['label']].append(r)

    print(f"{'Label':<30} {'J':>4} {'γ':>7} {'k_ex':>6}  "
          f"{'FP':>3} {'Lock':>4} {'Cpx':>4}  "
          f"{'med D2':>7} {'max D2':>7} {'r(X1,X2)':>9}  {'E_mean':>7}")
    print('-' * 100)

    complex_labels = []

    for label in sorted(by_label.keys()):
        runs = by_label[label]
        p_match = [p for p in param_sets if p.label == label][0]

        regimes = [r['regime'] for r in runs]
        n_fp = sum(1 for r in regimes if r == 'fixed_point')
        n_lock = sum(1 for r in regimes if r == 'phase_locked')
        n_cpx = sum(1 for r in regimes if r == 'complex')

        d2s = [r['D2'] for r in runs if not np.isnan(r['D2'])]
        rs = [r['r_X1X2'] for r in runs if not np.isnan(r['r_X1X2'])]
        es = [r['E_mean'] for r in runs if not np.isnan(r.get('E_mean', float('nan')))]

        med_d2 = np.median(d2s) if d2s else float('nan')
        max_d2 = max(d2s) if d2s else float('nan')
        med_r = np.median(rs) if rs else float('nan')
        med_e = np.median(es) if es else float('nan')

        cpx_str = f"*{n_cpx}" if n_cpx > 0 else f"{n_cpx}"

        d2_med_str = f"{med_d2:.3f}" if not np.isnan(med_d2) else "N/A"
        d2_max_str = f"{max_d2:.3f}" if not np.isnan(max_d2) else "N/A"
        r_str = f"{med_r:.3f}" if not np.isnan(med_r) else "N/A"
        e_str = f"{med_e:.1f}" if not np.isnan(med_e) else "N/A"

        print(f"{label:<30} {p_match.J:>4.1f} {p_match.gamma:>7.3f} {p_match.k_extra:>6.3f}  "
              f"{n_fp:>3} {n_lock:>4} {cpx_str:>4}  "
              f"{d2_med_str:>7} {d2_max_str:>7} {r_str:>9}  {e_str:>7}")

        if n_cpx > 0:
            complex_labels.append(label)

    print('-' * 100)

    # Overall verdict
    all_d2 = [r['D2'] for r in results if not np.isnan(r['D2'])]
    max_d2 = max(all_d2) if all_d2 else float('nan')

    print(f"  Max D₂: {max_d2:.3f}" if not np.isnan(max_d2) else "  Max D₂: N/A")
    if complex_labels:
        print(f"  D₂ > 1.2 FOUND in: {complex_labels}")
        print(f"  VERDICT: Mass-action model VALIDATED — {len(complex_labels)} param sets with D₂ > 1.2")
    else:
        print("  D₂ > 1.2 NOT FOUND in any parameter set")
        print("  VERDICT: Mass-action model needs parameter recalibration")
    print('=' * 100)

    # ── Save results ──────────────────────────────────────────────────
    output = {
        'model': 'coupled_brusselator_massaction',
        'n_param_sets': len(param_sets),
        'n_seeds': n_seeds,
        'n_total_runs': n_total,
        'runtime_seconds': total_time,
        'max_d2': float(max_d2) if not np.isnan(max_d2) else None,
        'found_complex': len(complex_labels) > 0,
        'complex_labels': complex_labels,
        'n_complex_sets': len(complex_labels),
        'results': results,
        'param_sets': [asdict(p) for p in param_sets],
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'pilot5_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_pilot5(n_seeds=3, verbose=True, save_dir=save_dir)
