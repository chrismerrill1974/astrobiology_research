"""
Phase 1 Dynamical Diagnostics: D₂ convergence + Lyapunov exponents.

Two tests at two exemplar parameter sets:

1. D₂ convergence (time-doubling): T ∈ {5000, 10000, 20000}
   Pass criterion: |D₂(2T) - D₂(T)| < 0.05 at both doublings.
   Purpose: rule out transient beating / finite-time artifacts.

2. Lyapunov exponent (Rosenstein algorithm):
   λ₁ > 0 (CI excludes 0) → chaos
   λ₁ ≈ 0 (CI includes 0) → quasi-periodic (torus)
   λ₁ < 0 → limit cycle
   Purpose: distinguish torus from strange attractor.

Exemplar points:
  - Regime 2 (shared-reservoir desynchronization):
    J=5, γ=0.002, k_cat=0.3  (shared D₂ ≈ 1.558, r ≈ 0)
  - Regime 1 (intrinsic slow-fast):
    J=7, γ=0.001, k_cat=0.5  (shared D₂ ≈ 1.542, r ≈ 1)
"""

import sys
import os
import json
import time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_astro2_path = os.path.join(_this_dir, '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from pilot.pilot5b_enzyme_complex import (
    EnzymeComplexParams, make_enzyme_complex_network, simulate_and_analyze,
)
from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension, QualityFlag

try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False
    print("WARNING: nolds not installed. Lyapunov exponents will be skipped.")


# ── Helper: simulate and extract raw trajectory ─────────────────────

def simulate_raw(p: EnzymeComplexParams, seed: int, t_end: float,
                 n_points: int = 20000) -> dict | None:
    """Run simulation and return raw post-transient trajectories."""
    net = make_enzyme_complex_network(p, seed=seed)
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
        print(f"  Simulation failed: {e}")
        return None

    if not result.success:
        print(f"  Solver failed: {result.solver_message}")
        return None

    t = result.time
    c = result.concentrations
    species = result.species_names
    n_discard = len(t) // 2
    c_post = c[n_discard:]

    def idx(name):
        return species.index(name)

    return {
        'X1': c_post[:, idx('X1')],
        'Y1': c_post[:, idx('Y1')],
        'X2': c_post[:, idx('X2')],
        'Y2': c_post[:, idx('Y2')],
        'E':  c_post[:, idx('E')],
        't_post': t[n_discard:],
    }


# ── D₂ Convergence Test ─────────────────────────────────────────────

def d2_convergence_test(p: EnzymeComplexParams, seed: int = 42,
                        T_values=None) -> dict:
    """
    Run D₂ at multiple integration times using simulate_and_analyze.
    Pass criterion: |D₂(2T) - D₂(T)| < 0.05 at each doubling.
    """
    if T_values is None:
        T_values = [5_000, 10_000, 20_000]

    print(f"\n  D₂ convergence test: {p.label}, seed={seed}")
    print(f"  T values: {T_values}")

    results = []
    for T in T_values:
        print(f"    T={T:>6}...", end='', flush=True)
        t0 = time.time()
        r = simulate_and_analyze(p, seed=seed, t_span=(0, T),
                                  n_points=20000, verbose=False)
        elapsed = time.time() - t0
        d2 = r['D2']
        unc = r.get('D2_unc', float('nan'))
        regime = r['regime']
        print(f" D₂={d2:.4f} ({regime}) [{elapsed:.1f}s]")
        results.append({
            'T': T, 'D2': float(d2), 'unc': float(unc),
            'regime': regime, 'time_s': elapsed,
        })

    # Check convergence
    deltas = []
    for i in range(len(results) - 1):
        d1 = results[i]['D2']
        d2_val = results[i + 1]['D2']
        if not (np.isnan(d1) or np.isnan(d2_val)):
            delta = abs(d2_val - d1)
            deltas.append(delta)
            results[i]['delta_next'] = delta
        else:
            deltas.append(float('nan'))
            results[i]['delta_next'] = float('nan')

    passed = all(d < 0.05 for d in deltas if not np.isnan(d)) and len(deltas) > 0
    print(f"    Deltas: {[f'{d:.4f}' for d in deltas]}")
    print(f"    CONVERGENCE: {'PASS ✓' if passed else 'FAIL ✗'}")

    return {
        'label': p.label, 'seed': seed,
        'T_values': T_values,
        'results': results,
        'deltas': [float(d) for d in deltas],
        'converged': passed,
    }


# ── Lyapunov Exponent (Rosenstein) ───────────────────────────────────

def mutual_information_delay(x: np.ndarray, max_lag: int = 100) -> int:
    """Estimate embedding delay from first minimum of mutual information."""
    n = len(x)
    n_bins = max(10, int(np.sqrt(n / 5)))

    mi_values = []
    for lag in range(1, min(max_lag + 1, n // 4)):
        x1 = x[:-lag]
        x2 = x[lag:]
        c_xy, _, _ = np.histogram2d(x1, x2, bins=n_bins)
        c_xy = c_xy / c_xy.sum()
        c_x = c_xy.sum(axis=1)
        c_y = c_xy.sum(axis=0)

        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if c_xy[i, j] > 0 and c_x[i] > 0 and c_y[j] > 0:
                    mi += c_xy[i, j] * np.log(c_xy[i, j] / (c_x[i] * c_y[j]))
        mi_values.append(mi)

    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] <= mi_values[i + 1]:
            return i + 1

    # Fallback: first zero-crossing of autocorrelation
    x_centered = x - np.mean(x)
    acf = np.correlate(x_centered[:min(n, 5000)], x_centered[:min(n, 5000)], 'full')
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0]
    for i in range(1, min(max_lag, len(acf))):
        if acf[i] <= 0:
            return i

    return max_lag // 2


def lyapunov_rosenstein(traj_dict: dict, m: int = 5, n_bootstrap: int = 200) -> dict:
    """
    Compute largest Lyapunov exponent using Rosenstein algorithm.
    Uses most variable component of (X1,Y1,X2,Y2,E).
    Returns λ₁ with bootstrapped 95% CI.
    """
    if not HAS_NOLDS:
        return {'lambda1': float('nan'), 'ci_lo': float('nan'),
                'ci_hi': float('nan'), 'interpretation': 'SKIPPED (nolds not installed)'}

    components = [traj_dict['X1'], traj_dict['Y1'], traj_dict['X2'],
                  traj_dict['Y2'], traj_dict['E']]
    cvs = [np.std(c) / max(np.mean(np.abs(c)), 1e-10) for c in components]
    best_comp = components[np.argmax(cvs)]

    delay = mutual_information_delay(best_comp)
    print(f"    Embedding: m={m}, delay={delay}")

    # Subsample if very long
    max_pts = 10000
    if len(best_comp) > max_pts:
        step = len(best_comp) // max_pts
        data = best_comp[::step]
        print(f"    Subsampled: {len(best_comp)} → {len(data)} points")
    else:
        data = best_comp

    try:
        lam = float(nolds.lyap_r(data, emb_dim=m, lag=delay, min_tsep=delay))
    except Exception as e:
        print(f"    Lyapunov computation failed: {e}")
        return {'lambda1': float('nan'), 'ci_lo': float('nan'),
                'ci_hi': float('nan'), 'interpretation': f'FAILED: {e}'}

    # Block bootstrap CI
    block_size = max(100, delay * 10)
    n_blocks = max(1, len(data) // block_size)
    bootstrap_lams = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        blocks = [data[i * block_size:(i + 1) * block_size]
                  for i in range(n_blocks)]
        chosen = rng.choice(len(blocks), size=len(blocks), replace=True)
        boot_series = np.concatenate([blocks[c] for c in chosen])
        try:
            boot_lam = float(nolds.lyap_r(boot_series, emb_dim=m, lag=delay,
                                           min_tsep=delay))
            bootstrap_lams.append(boot_lam)
        except Exception:
            continue

    if len(bootstrap_lams) >= 20:
        ci_lo = float(np.percentile(bootstrap_lams, 2.5))
        ci_hi = float(np.percentile(bootstrap_lams, 97.5))
    else:
        ci_lo = float('nan')
        ci_hi = float('nan')

    if ci_lo > 0 and not np.isnan(ci_lo):
        interp = 'CHAOS (λ₁ > 0, CI excludes 0)'
    elif ci_hi < 0 and not np.isnan(ci_hi):
        interp = 'LIMIT CYCLE (λ₁ < 0)'
    elif not np.isnan(ci_lo):
        interp = 'QUASI-PERIODIC (λ₁ ≈ 0, CI includes 0)'
    else:
        interp = 'INDETERMINATE (insufficient bootstrap samples)'

    print(f"    λ₁ = {lam:.6f}  95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"    Interpretation: {interp}")
    print(f"    Bootstrap: {len(bootstrap_lams)}/{n_bootstrap} successful")

    return {
        'lambda1': lam, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'delay': delay, 'emb_dim': m,
        'n_bootstrap': len(bootstrap_lams),
        'interpretation': interp,
    }


# ── Main ─────────────────────────────────────────────────────────────

def run_diagnostics(save_dir: str | None = None) -> dict:
    """Run full dynamical diagnostics on both exemplar points."""

    print("\n" + "#" * 80)
    print("# DYNAMICAL DIAGNOSTICS: D₂ Convergence + Lyapunov Exponents")
    print("#" * 80)

    exemplars = [
        EnzymeComplexParams(
            J=5.0, gamma=0.002, k_on=10.0, k_off=10.0,
            k_cat=0.3, G_total=1.0,
            label="Regime2_J5_g0.002_kc0.3",
        ),
        EnzymeComplexParams(
            J=7.0, gamma=0.001, k_on=10.0, k_off=10.0,
            k_cat=0.5, G_total=1.0,
            label="Regime1_J7_g0.001_kc0.5",
        ),
    ]

    seeds = [42, 179]
    all_results = {}
    start_time = time.time()

    for p in exemplars:
        print(f"\n{'=' * 70}")
        print(f"  Exemplar: {p.label}")
        print(f"  J={p.J}, γ={p.gamma}, k_cat={p.k_cat}")
        print(f"{'=' * 70}")

        exemplar_results = {
            'params': {'J': p.J, 'gamma': p.gamma, 'k_cat': p.k_cat},
            'convergence': [],
            'lyapunov': [],
        }

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")

            # 1. D₂ convergence
            conv = d2_convergence_test(p, seed=seed)
            exemplar_results['convergence'].append(conv)

            # 2. Lyapunov (use standard T=10000 trajectory)
            print(f"\n  Lyapunov exponent (T=10000):")
            traj = simulate_raw(p, seed=seed, t_end=10000, n_points=20000)
            if traj is not None:
                lyap = lyapunov_rosenstein(traj)
                lyap['seed'] = seed
                exemplar_results['lyapunov'].append(lyap)
            else:
                exemplar_results['lyapunov'].append({
                    'seed': seed, 'lambda1': float('nan'),
                    'interpretation': 'SIM_FAILED'
                })

        # Summary
        conv_pass = sum(1 for c in exemplar_results['convergence'] if c['converged'])
        n_conv = len(exemplar_results['convergence'])
        lyap_vals = [l['lambda1'] for l in exemplar_results['lyapunov']
                     if not np.isnan(l.get('lambda1', float('nan')))]

        print(f"\n  ── Summary: {p.label} ──")
        print(f"  D₂ convergence: {conv_pass}/{n_conv} seeds passed")
        if lyap_vals:
            print(f"  λ₁ values: {[f'{v:.6f}' for v in lyap_vals]}")
            print(f"  Interpretations: {[l['interpretation'] for l in exemplar_results['lyapunov']]}")

        all_results[p.label] = exemplar_results

    total_time = time.time() - start_time

    # ── Final verdict ────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print(f"  FINAL VERDICT  (total time: {total_time:.1f}s / {total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    for label, res in all_results.items():
        conv_pass = sum(1 for c in res['convergence'] if c['converged'])
        n_conv = len(res['convergence'])

        # D₂ at T=10000 (index 1 in [5000,10000,20000])
        d2_refs = []
        for c in res['convergence']:
            for r in c['results']:
                if r['T'] == 10000 and not np.isnan(r['D2']):
                    d2_refs.append(r['D2'])
        d2_str = f"D₂(T=10k)={np.median(d2_refs):.3f}" if d2_refs else "D₂=N/A"

        # D₂ at T=20000
        d2_20k = []
        for c in res['convergence']:
            for r in c['results']:
                if r['T'] == 20000 and not np.isnan(r['D2']):
                    d2_20k.append(r['D2'])
        d2_20k_str = f"D₂(T=20k)={np.median(d2_20k):.3f}" if d2_20k else ""

        from collections import Counter
        lyap_interps = [l['interpretation'] for l in res['lyapunov']]
        majority = Counter(lyap_interps).most_common(1)[0][0] if lyap_interps else "N/A"

        conv_str = "CONVERGED ✓" if conv_pass == n_conv else f"FAIL ({conv_pass}/{n_conv})"
        print(f"    {label}:")
        print(f"      {d2_str}  {d2_20k_str}")
        print(f"      Convergence: {conv_str}")
        print(f"      Lyapunov: {majority}")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        'test': 'dynamical_diagnostics',
        'description': 'D₂ convergence (T-doubling) + Lyapunov (Rosenstein) at exemplar points',
        'runtime_seconds': total_time,
        'exemplars': {},
    }
    for label, res in all_results.items():
        output['exemplars'][label] = {
            'params': res['params'],
            'convergence': res['convergence'],
            'lyapunov': res['lyapunov'],
        }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'phase1_diagnostics_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {path}")

    return output


if __name__ == '__main__':
    save_dir = os.path.join(_this_dir, 'results')
    run_diagnostics(save_dir=save_dir)
