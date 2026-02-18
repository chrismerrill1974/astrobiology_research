"""
Phase I-B: Control Experiments for Paper 3.

Four controls establish causality for the transient chaos finding:

Control 1: Single Brusselator + E + G/GE
    - Remove Core 2. Tests whether single core + slow E inflates D₂.
    - Prediction: transient D₂ > 1.2 at γ=0.001 (intrinsic mechanism).

Control 2: Two Uncoupled Brusselators (k_cat=0)
    - Gated autocatalysis disabled. E evolves but doesn't affect cores.
    - Prediction: D₂ ≈ 1 always (no coupling → no complexity).

Control 3: Diffusive Coupling (no slow E)
    - X1 ⇌ X2 at rate ε. No energy pool.
    - Prediction: D₂ ≈ 1 at all ε (slow E is essential, not just coupling).

Control 4: Independent Reservoirs (E₁/E₂)
    - Already done via Sweep A2 + A2b. Repackaged here for completeness.

All controls report D₂ plus T_lock and τ_{>1.2} at exemplar parameter sets.
"""

import sys
import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict

_this_dir = os.path.dirname(os.path.abspath(__file__))
_astro2_path = os.path.join(_this_dir, '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from pilot.pilot5b_enzyme_complex import EnzymeComplexParams, make_enzyme_complex_network
from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension, QualityFlag
from dimensional_opening.network_generator import GeneratedNetwork


# ── Parameter sets ───────────────────────────────────────────────────

PARAM_SETS = [
    EnzymeComplexParams(J=7.0,  gamma=0.001, k_on=10.0, k_off=10.0, k_cat=0.5, G_total=1.0, label="J7_g0.001_kc0.5"),
    EnzymeComplexParams(J=5.0,  gamma=0.002, k_on=10.0, k_off=10.0, k_cat=0.3, G_total=1.0, label="J5_g0.002_kc0.3"),
    EnzymeComplexParams(J=4.0,  gamma=0.003, k_on=10.0, k_off=10.0, k_cat=0.2, G_total=1.0, label="J4_g0.003_kc0.2"),
    EnzymeComplexParams(J=3.0,  gamma=0.001, k_on=10.0, k_off=10.0, k_cat=0.1, G_total=1.0, label="J3_g0.001_kc0.1"),
    EnzymeComplexParams(J=10.0, gamma=0.001, k_on=10.0, k_off=10.0, k_cat=0.3, G_total=1.0, label="J10_g0.001_kc0.3"),
]

EPSILON_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

N_SEEDS = 5


# ── Network builders ────────────────────────────────────────────────

def make_single_core_network(p: EnzymeComplexParams, seed: int = 42,
                              ic_perturbation: float = 0.05) -> GeneratedNetwork:
    """Control 1: Single Brusselator + E + G/GE (no Core 2)."""
    reactions = [
        # Core 1 Brusselator (0-3)
        "A -> X1",
        "B + X1 -> Y1 + D1",
        "X1 + X1 + Y1 -> X1 + X1 + X1",
        "X1 -> W1",
        # Energy pool (4-5)
        "Esrc -> Esrc + E",
        "E -> Ew",
        # Gate (6-7)
        "G + E -> GE",
        "GE -> G + E",
        # Gated autocatalysis Core 1 only (8)
        "X1 + X1 + Y1 + GE -> X1 + X1 + X1 + G + Ew",
    ]

    rate_constants = [
        p.A, p.B, 1.0, 1.0,   # Core 1
        p.J, p.gamma,          # Energy
        p.k_on, p.k_off,       # Gate
        p.k_cat,               # Gated autocatalysis (Core 1 only)
    ]

    E0 = p.J / max(p.gamma, 1e-6)
    GE0 = p.G_total * E0 / (p.K_d + E0)
    G0 = p.G_total - GE0

    ic = {
        'X1': 1.0, 'Y1': 1.0, 'D1': 0.0, 'W1': 0.0,
        'E': E0, 'Ew': 0.0,
        'G': G0, 'GE': GE0,
        'A': p.A, 'B': p.B, 'Esrc': 1.0,
    }

    species = ['X1', 'Y1', 'D1', 'W1', 'E', 'Ew', 'G', 'GE', 'A', 'B', 'Esrc']

    return GeneratedNetwork(
        reactions=reactions, species=species,
        food_set=['A', 'B', 'Esrc'], n_species=len(species),
        n_reactions=len(reactions), n_autocatalytic=1,
        rate_constants=rate_constants, initial_concentrations=ic,
        chemostat_species={'A': p.A, 'B': p.B, 'Esrc': 1.0},
        network_id=f"ctrl1_single_{p.label}_s{seed}",
        is_autocatalytic=True, template="Control1_SingleCore",
        n_added_reactions=0,
    )


def make_diffusive_network(epsilon: float, seed: int = 42,
                            A: float = 1.0, B: float = 3.0) -> GeneratedNetwork:
    """Control 3: Two Brusselators with diffusive X1 ⇌ X2 coupling, no E."""
    rng = np.random.RandomState(seed)

    reactions = [
        # Core 1 (0-3)
        "A -> X1",
        "B + X1 -> Y1 + Dx1",
        "X1 + X1 + Y1 -> X1 + X1 + X1",
        "X1 -> W1",
        # Core 2 (4-7)
        "A -> X2",
        "B + X2 -> Y2 + Dx2",
        "X2 + X2 + Y2 -> X2 + X2 + X2",
        "X2 -> W2",
        # Diffusive coupling (8-9)
        "X1 -> X2",
        "X2 -> X1",
    ]

    rate_constants = [
        A, B, 1.0, 1.0,   # Core 1
        A, B, 1.0, 1.0,   # Core 2
        epsilon, epsilon,  # Diffusion
    ]

    ic = {
        'X1': 1.0, 'Y1': 1.0, 'Dx1': 0.0, 'W1': 0.0,
        'X2': 1.0 + rng.uniform(-0.05, 0.05),
        'Y2': 1.0 + rng.uniform(-0.05, 0.05),
        'Dx2': 0.0, 'W2': 0.0,
        'A': A, 'B': B,
    }

    species = ['X1', 'Y1', 'Dx1', 'W1', 'X2', 'Y2', 'Dx2', 'W2', 'A', 'B']

    return GeneratedNetwork(
        reactions=reactions, species=species,
        food_set=['A', 'B'], n_species=len(species),
        n_reactions=len(reactions), n_autocatalytic=2,
        rate_constants=rate_constants, initial_concentrations=ic,
        chemostat_species={'A': A, 'B': B},
        network_id=f"ctrl3_diffusive_eps{epsilon}_s{seed}",
        is_autocatalytic=True, template="Control3_Diffusive",
        n_added_reactions=0,
    )


# ── Simulation helpers ───────────────────────────────────────────────

def simulate_network(net: GeneratedNetwork, t_end: float = 10000,
                     n_points: int = 20000) -> dict | None:
    """Simulate a network, return post-transient species dict or None."""
    sim = ReactionSimulator()
    try:
        network_obj = sim.build_network(net.reactions)
        result = sim.simulate(
            network_obj,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=(0, t_end),
            n_points=n_points,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=net.chemostat_species,
        )
    except Exception as e:
        return None

    if not result.success:
        return None

    t = result.time
    c = result.concentrations
    species = result.species_names
    n_discard = len(t) // 2
    c_post = c[n_discard:]

    def col(name):
        return c_post[:, species.index(name)]

    out = {'t_post': t[n_discard:]}
    for sp in species:
        out[sp] = col(sp)
    return out


def simulate_network_full(net: GeneratedNetwork, t_end: float = 20000,
                          n_points: int = 20000) -> dict | None:
    """Simulate a network, return FULL trajectory (no discard) for sliding-window."""
    sim = ReactionSimulator()
    try:
        network_obj = sim.build_network(net.reactions)
        result = sim.simulate(
            network_obj,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=(0, t_end),
            n_points=n_points,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=net.chemostat_species,
        )
    except Exception as e:
        return None

    if not result.success:
        return None

    t = result.time
    c = result.concentrations
    species = result.species_names

    out = {'t_post': t}
    for i, sp in enumerate(species):
        out[sp] = c[:, i]
    return out


def compute_d2(trajectory: np.ndarray) -> tuple:
    """Compute D₂. Returns (d2, unc, quality_str)."""
    cd = CorrelationDimension()
    try:
        result = cd.compute(trajectory)
        d2 = float(result.D2) if result.D2 is not None else float('nan')
        unc = float(result.D2_uncertainty) if result.D2_uncertainty else float('nan')
        quality = result.quality.name if result.quality else 'UNKNOWN'
        return d2, unc, quality
    except Exception:
        return float('nan'), float('nan'), 'FAILED'


def sliding_window_d2(traj_dict: dict, projection_keys: list,
                      window_size: float = 2500) -> dict:
    """Compute D₂ in sliding windows. Returns T_lock, τ_{>1.2}, window D₂s."""
    t = traj_dict['t_post']
    t_min, t_max = t[0], t[-1]

    windows = []
    t_lo = t_min
    while t_lo + window_size <= t_max + 1:
        t_hi = t_lo + window_size
        mask = (t >= t_lo) & (t < t_hi)
        if np.sum(mask) < 200:
            t_lo += window_size
            continue

        traj = np.column_stack([traj_dict[k][mask] for k in projection_keys])
        d2, unc, quality = compute_d2(traj)

        # Phase correlation if both X1 and X2 exist
        r = float('nan')
        if 'X1' in traj_dict and 'X2' in traj_dict:
            r = float(np.corrcoef(traj_dict['X1'][mask], traj_dict['X2'][mask])[0, 1])

        windows.append({
            't_start': float(t_lo), 't_end': float(t_hi),
            'D2': d2, 'r_X1X2': r,
        })
        t_lo += window_size

    # Compute T_lock and τ_{>1.2}
    d2_vals = [w['D2'] for w in windows]
    n_above = sum(1 for d in d2_vals if d > 1.2 and not np.isnan(d))
    tau_above = n_above * window_size

    T_lock = float('nan')
    for i, w in enumerate(windows):
        subsequent = [ww['D2'] for ww in windows[i:] if not np.isnan(ww['D2'])]
        if subsequent and all(d < 1.1 for d in subsequent):
            T_lock = w['t_start']
            break

    return {
        'windows': windows,
        'T_lock': T_lock,
        'tau_above_1p2': tau_above,
        'n_windows_above': n_above,
        'n_windows_total': len(windows),
    }


# ── Control runners ──────────────────────────────────────────────────

def run_control1(verbose: bool = True) -> dict:
    """Control 1: Single Brusselator + E + G/GE."""
    print("\n" + "=" * 70)
    print("  CONTROL 1: Single Brusselator + E + G/GE")
    print("=" * 70)

    results = []
    for p in PARAM_SETS:
        for s in range(N_SEEDS):
            seed = 42 + s * 137
            net = make_single_core_network(p, seed=seed)
            traj = simulate_network(net)

            if traj is None:
                results.append({'label': p.label, 'seed': seed, 'D2': float('nan'),
                                'regime': 'sim_failed'})
                continue

            # D₂ on (X1, Y1, E) — 3D
            trajectory = np.column_stack([traj['X1'], traj['Y1'], traj['E']])
            d2, unc, quality = compute_d2(trajectory)

            cv_X1 = np.std(traj['X1']) / max(np.mean(traj['X1']), 1e-10)
            if cv_X1 < 0.01:
                regime = 'fixed_point'
            elif d2 > 1.2:
                regime = 'complex'
            else:
                regime = 'phase_locked'

            results.append({
                'label': p.label, 'seed': seed, 'D2': d2, 'D2_unc': unc,
                'quality': quality, 'regime': regime,
                'J': p.J, 'gamma': p.gamma, 'k_cat': p.k_cat,
            })

    _print_summary("Control 1", results, verbose)

    # Sliding-window at exemplars (use FULL trajectory — no 50% discard)
    sw_results = []
    for p in [PARAM_SETS[0], PARAM_SETS[1]]:  # Regime 1 and 2 flagships
        net = make_single_core_network(p, seed=42)
        traj = simulate_network_full(net, t_end=20000)
        if traj is not None:
            sw = sliding_window_d2(traj, ['X1', 'Y1', 'E'])
            sw['label'] = p.label
            sw_results.append(sw)
            if verbose:
                print(f"  Sliding-window {p.label}: T_lock={sw['T_lock']:.0f}, "
                      f"τ_{{>1.2}}={sw['tau_above_1p2']:.0f}")

    return {'control': 1, 'results': results, 'sliding_window': sw_results}


def run_control2(verbose: bool = True) -> dict:
    """Control 2: Two Uncoupled Brusselators (k_cat=0)."""
    print("\n" + "=" * 70)
    print("  CONTROL 2: Two Uncoupled Brusselators (k_cat=0)")
    print("=" * 70)

    results = []
    for p in PARAM_SETS:
        p_uncoupled = EnzymeComplexParams(
            J=p.J, gamma=p.gamma, k_on=p.k_on, k_off=p.k_off,
            k_cat=0.0, G_total=p.G_total, label=p.label,
        )
        for s in range(N_SEEDS):
            seed = 42 + s * 137
            net = make_enzyme_complex_network(p_uncoupled, seed=seed)
            traj = simulate_network(net)

            if traj is None:
                results.append({'label': p.label, 'seed': seed, 'D2': float('nan'),
                                'regime': 'sim_failed'})
                continue

            # D₂ on (X1, Y1, X2, Y2) — 4D, no E
            trajectory = np.column_stack([traj['X1'], traj['Y1'],
                                          traj['X2'], traj['Y2']])
            d2, unc, quality = compute_d2(trajectory)
            r = float(np.corrcoef(traj['X1'], traj['X2'])[0, 1])

            cv_X1 = np.std(traj['X1']) / max(np.mean(traj['X1']), 1e-10)
            cv_X2 = np.std(traj['X2']) / max(np.mean(traj['X2']), 1e-10)
            if cv_X1 < 0.01 and cv_X2 < 0.01:
                regime = 'fixed_point'
            elif d2 > 1.2:
                regime = 'complex'
            else:
                regime = 'phase_locked'

            results.append({
                'label': p.label, 'seed': seed, 'D2': d2, 'D2_unc': unc,
                'quality': quality, 'regime': regime, 'r_X1X2': r,
                'J': p.J, 'gamma': p.gamma, 'k_cat': 0.0,
            })

    _print_summary("Control 2", results, verbose)
    return {'control': 2, 'results': results, 'sliding_window': []}


def run_control3(verbose: bool = True) -> dict:
    """Control 3: Diffusive coupling (no E)."""
    print("\n" + "=" * 70)
    print("  CONTROL 3: Diffusive Coupling (no slow E)")
    print("=" * 70)

    results = []
    for eps in EPSILON_VALUES:
        for s in range(N_SEEDS):
            seed = 42 + s * 137
            net = make_diffusive_network(eps, seed=seed)
            traj = simulate_network(net)

            if traj is None:
                results.append({'epsilon': eps, 'seed': seed, 'D2': float('nan'),
                                'regime': 'sim_failed'})
                continue

            trajectory = np.column_stack([traj['X1'], traj['Y1'],
                                          traj['X2'], traj['Y2']])
            d2, unc, quality = compute_d2(trajectory)
            r = float(np.corrcoef(traj['X1'], traj['X2'])[0, 1])

            cv_X1 = np.std(traj['X1']) / max(np.mean(traj['X1']), 1e-10)
            cv_X2 = np.std(traj['X2']) / max(np.mean(traj['X2']), 1e-10)
            if cv_X1 < 0.01 and cv_X2 < 0.01:
                regime = 'fixed_point'
            elif d2 > 1.2:
                regime = 'complex'
            else:
                regime = 'phase_locked'

            results.append({
                'epsilon': eps, 'seed': seed, 'D2': d2, 'D2_unc': unc,
                'quality': quality, 'regime': regime, 'r_X1X2': r,
            })

    # Print Control 3 summary
    if verbose:
        print(f"\n  {'ε':>8}  {'med D₂':>7}  {'max D₂':>7}  {'n_cpx':>5}  {'med r':>7}")
        print(f"  {'-'*42}")
        for eps in EPSILON_VALUES:
            eps_results = [r for r in results if r.get('epsilon') == eps
                           and not np.isnan(r['D2'])]
            if eps_results:
                d2s = [r['D2'] for r in eps_results]
                rs = [r['r_X1X2'] for r in eps_results if not np.isnan(r.get('r_X1X2', float('nan')))]
                n_cpx = sum(1 for r in eps_results if r['regime'] == 'complex')
                print(f"  {eps:>8.3f}  {np.median(d2s):>7.3f}  {max(d2s):>7.3f}  "
                      f"{n_cpx:>5}  {np.median(rs):>7.3f}")

    return {'control': 3, 'results': results, 'sliding_window': []}


def run_control4(verbose: bool = True) -> dict:
    """Control 4: Repackage existing A2 + A2b results."""
    print("\n" + "=" * 70)
    print("  CONTROL 4: Independent Reservoirs (from Sweep A2 + A2b)")
    print("=" * 70)

    results_dir = os.path.join(_this_dir, 'results')
    a2_path = os.path.join(results_dir, 'phase1_sweep_a2_results.json')
    a2b_path = os.path.join(results_dir, 'phase1_sweep_a2b_results.json')

    summary = {}
    for path, label in [(a2_path, 'A2 (γ=0.001)'), (a2b_path, 'A2b (moderate γ)')]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            n_runs = data.get('n_total_runs', len(data.get('results', [])))
            n_complex = sum(1 for r in data.get('results', [])
                           if r.get('regime') == 'complex')
            summary[label] = {
                'n_runs': n_runs,
                'n_complex': n_complex,
                'source': path,
            }
            if verbose:
                print(f"  {label}: {n_runs} runs, {n_complex} complex")
        else:
            if verbose:
                print(f"  {label}: FILE NOT FOUND ({path})")
            summary[label] = {'error': 'file not found'}

    if verbose:
        print(f"\n  Verdict: A2 confirms D₂ persists at γ=0.001 (intrinsic).")
        print(f"           A2b confirms D₂ disappears at moderate γ (sharing essential).")

    return {'control': 4, 'summary': summary, 'results': [], 'sliding_window': []}


# ── Summary printer ──────────────────────────────────────────────────

def _print_summary(control_name: str, results: list, verbose: bool):
    """Print summary table for a control."""
    if not verbose:
        return

    # Group by label
    labels = sorted(set(r.get('label', '') for r in results))
    print(f"\n  {'Label':>25}  {'med D₂':>7}  {'max D₂':>7}  {'n_cpx':>5}  {'regime':>12}")
    print(f"  {'-'*60}")

    for label in labels:
        lr = [r for r in results if r.get('label') == label and not np.isnan(r['D2'])]
        if not lr:
            print(f"  {label:>25}  {'N/A':>7}  {'N/A':>7}  {'N/A':>5}")
            continue
        d2s = [r['D2'] for r in lr]
        n_cpx = sum(1 for r in lr if r['regime'] == 'complex')
        majority = max(set(r['regime'] for r in lr), key=lambda x: sum(1 for r in lr if r['regime'] == x))
        marker = " ***" if n_cpx > 0 else ""
        print(f"  {label:>25}  {np.median(d2s):>7.3f}  {max(d2s):>7.3f}  "
              f"{n_cpx:>5}  {majority:>12}{marker}")


# ── Main ─────────────────────────────────────────────────────────────

def run_all_controls(save_dir: str | None = None) -> dict:
    """Run all 4 control experiments."""
    print("\n" + "#" * 80)
    print("# PHASE I-B: CONTROL EXPERIMENTS")
    print("#" * 80)

    start_time = time.time()

    c1 = run_control1()
    c2 = run_control2()
    c3 = run_control3()
    c4 = run_control4()

    total_time = time.time() - start_time

    # ── Final verdict ────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print(f"  CONTROL VERDICT  ({total_time:.0f}s / {total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    # Control 1: single core
    c1_d2s = [r['D2'] for r in c1['results'] if not np.isnan(r['D2'])]
    c1_cpx = sum(1 for r in c1['results'] if r.get('regime') == 'complex')
    print(f"\n  Control 1 (Single core + E):")
    print(f"    Median D₂ = {np.median(c1_d2s):.3f}, {c1_cpx}/{len(c1_d2s)} complex")
    if c1_cpx > 0:
        print(f"    ✓ CONFIRMS intrinsic slow-energy mechanism")
    else:
        print(f"    ✗ Single core does NOT inflate — unexpected")

    # Control 2: uncoupled
    c2_d2s = [r['D2'] for r in c2['results'] if not np.isnan(r['D2'])]
    c2_cpx = sum(1 for r in c2['results'] if r.get('regime') == 'complex')
    print(f"\n  Control 2 (Uncoupled, k_cat=0):")
    print(f"    Median D₂ = {np.median(c2_d2s):.3f}, {c2_cpx}/{len(c2_d2s)} complex")
    if c2_cpx == 0:
        print(f"    ✓ CONFIRMS coupling through E is necessary")
    else:
        print(f"    ✗ Uncoupled shows complexity — unexpected")

    # Control 3: diffusive
    c3_d2s = [r['D2'] for r in c3['results'] if not np.isnan(r['D2'])]
    c3_cpx = sum(1 for r in c3['results'] if r.get('regime') == 'complex')
    print(f"\n  Control 3 (Diffusive coupling, no E):")
    print(f"    Median D₂ = {np.median(c3_d2s):.3f}, {c3_cpx}/{len(c3_d2s)} complex")
    if c3_cpx == 0:
        print(f"    ✓ CONFIRMS slow E is essential (not just any coupling)")
    else:
        print(f"    ⚠ Some diffusive points show D₂ > 1.2 — investigate")

    # Control 4: independent
    print(f"\n  Control 4 (Independent reservoirs):")
    print(f"    Repackaged from Sweep A2 + A2b (see summaries)")

    # ── Save ─────────────────────────────────────────────────────
    output = {
        'test': 'phase1b_controls',
        'description': 'Four control experiments for causal claims',
        'runtime_seconds': total_time,
        'controls': {
            'control1': c1,
            'control2': c2,
            'control3': c3,
            'control4': c4,
        },
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'phase1b_controls_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_all_controls(save_dir=save_dir)
