"""
Pilot 3b: True slow–fast bursting (Structural Upgrade 1).

Diagnosis from Pilot 3: E equilibrated too fast (~1-2% modulation),
acting as parametric perturbation rather than a slow manifold.

Fix: Make the ORIGINAL autocatalysis energy-dependent (Convention B).
When E depletes, oscillations quench. When E slowly recovers, they re-ignite.
This creates relaxation oscillations on two timescales → bursting / MMOs.

Model:
    dX/dt = A - (B+1)X + k_a * f(E) * X²Y
    dY/dt = BX - k_a * f(E) * X²Y
    dE/dt = J - γE - αEX²Y                     (drain ∝ autocatalytic flux, not just X)

    f(E) = E / (K + E)

Convention B: At J=0, E→0, f(E)→0, autocatalysis dies → fixed point.
The oscillator NEEDS energy to oscillate. This is the key structural change.

Key differences from Pilot 3:
- f(E) gates the ORIGINAL autocatalytic step (not an additive extra)
- E drain is proportional to autocatalytic flux (E*X²Y), not just E*X
- γ is very small (slow recovery, timescale ~100-500)
- α is large enough to create 30-50% E modulation
"""

import sys
import os
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# Add astrobiology2 to path
_astro2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from dimensional_opening.correlation_dimension import CorrelationDimension, QualityFlag
from dimensional_opening.oscillation_filter import check_oscillation


# ── Model ─────────────────────────────────────────────────────────────

@dataclass
class BurstingParams:
    """Parameters for the bursting Brusselator (Convention B)."""
    # Brusselator base
    A: float = 1.0
    B: float = 3.0
    k_a: float = 1.0       # Base autocatalytic rate (gated by f(E))

    # Energy dynamics — slow recovery, strong drain
    J: float = 1.0          # Energy inflow (the drive knob)
    gamma: float = 0.01     # Very slow leak (timescale ~100)
    alpha: float = 0.3      # Strong drain ∝ autocatalytic flux
    K: float = 1.0          # Half-saturation

    # Initial conditions
    X0: float = 1.0
    Y0: float = 1.0
    E0: float = 5.0         # Start with energy available

    def copy(self, **overrides) -> 'BurstingParams':
        d = asdict(self)
        d.update(overrides)
        return BurstingParams(**d)


def bursting_rhs(t, state, p: BurstingParams):
    """RHS for the bursting Brusselator."""
    X, Y, E = state

    # Clamp to non-negative
    X = max(X, 0.0)
    Y = max(Y, 0.0)
    E = max(E, 0.0)

    # Gating
    fE = E / (p.K + E)

    # Energy-gated autocatalytic flux
    auto_flux = p.k_a * fE * X * X * Y

    dX = p.A - (p.B + 1) * X + auto_flux
    dY = p.B * X - auto_flux

    # E drain proportional to autocatalytic flux (the activity that consumes energy)
    dE = p.J - p.gamma * E - p.alpha * auto_flux

    return [dX, dY, dE]


def simulate_bursting(
    params: BurstingParams,
    t_end: float = 5000.0,
    n_points: int = 50000,
    remove_transient: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the bursting Brusselator.

    Long integration time needed for slow–fast systems.
    """
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        bursting_rhs,
        (0, t_end),
        [params.X0, params.Y0, params.E0],
        args=(params,),
        method='LSODA',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.5,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    n_discard = int(remove_transient * len(sol.t))
    t = sol.t[n_discard:]
    traj = sol.y[:, n_discard:].T

    return t, traj


# ── Classification ────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Result of a single bursting simulation."""
    J: float
    gamma: float
    alpha: float
    K: float
    success: bool
    regime: str
    D2: float
    D2_uncertainty: float
    quality: str
    osc_passes: bool
    osc_cv: float
    osc_sign_changes: int
    X_range: float
    E_mean: float
    E_range: float
    E_modulation_pct: float   # (E_max - E_min) / E_mean * 100


def classify_run(params: BurstingParams, cd: CorrelationDimension) -> RunResult:
    """Simulate and classify."""
    base = dict(J=params.J, gamma=params.gamma, alpha=params.alpha, K=params.K)

    try:
        t, traj = simulate_bursting(params)
    except RuntimeError:
        return RunResult(
            **base, success=False, regime="failed",
            D2=np.nan, D2_uncertainty=np.nan, quality="FAILED",
            osc_passes=False, osc_cv=0, osc_sign_changes=0,
            X_range=0, E_mean=0, E_range=0, E_modulation_pct=0,
        )

    X, Y, E = traj[:, 0], traj[:, 1], traj[:, 2]

    if np.any(np.isnan(traj)) or np.any(np.abs(traj) > 1e6):
        return RunResult(
            **base, success=False, regime="blowup",
            D2=np.nan, D2_uncertainty=np.nan, quality="FAILED",
            osc_passes=False, osc_cv=0, osc_sign_changes=0,
            X_range=0, E_mean=0, E_range=0, E_modulation_pct=0,
        )

    X_range = float(np.max(X) - np.min(X))
    E_mean = float(np.mean(E))
    E_range = float(np.max(E) - np.min(E))
    E_mod = (E_range / E_mean * 100) if E_mean > 1e-6 else 0.0

    # Check oscillation on X, Y
    osc = check_oscillation(
        concentrations=traj[:, :2],
        time=t,
        species_names=['X', 'Y'],
        food_species=[],
    )

    if not osc.passes:
        return RunResult(
            **base, success=True, regime="fixed_point",
            D2=np.nan, D2_uncertainty=np.nan, quality="N/A",
            osc_passes=False, osc_cv=osc.cv, osc_sign_changes=osc.sign_changes,
            X_range=X_range, E_mean=E_mean, E_range=E_range,
            E_modulation_pct=E_mod,
        )

    # Compute D₂ on full 3D trajectory
    try:
        d2_result = cd.compute(traj, random_state=42)
        D2 = float(d2_result.D2)
        D2_unc = float(d2_result.D2_uncertainty)
        quality = d2_result.quality.value
    except Exception:
        D2 = np.nan
        D2_unc = np.nan
        quality = "FAILED"

    regime = "complex" if (not np.isnan(D2) and D2 > 1.2) else "limit_cycle"

    return RunResult(
        **base, success=True, regime=regime,
        D2=D2, D2_uncertainty=D2_unc, quality=quality,
        osc_passes=True, osc_cv=osc.cv, osc_sign_changes=osc.sign_changes,
        X_range=X_range, E_mean=E_mean, E_range=E_range,
        E_modulation_pct=E_mod,
    )


# ── Sweep ─────────────────────────────────────────────────────────────

@dataclass
class SweepRow:
    """One row of the parameter sweep."""
    label: str
    J: float
    gamma: float
    alpha: float
    K: float
    n_runs: int
    n_failed: int
    n_blowup: int
    n_fixed_point: int
    n_limit_cycle: int
    n_complex: int
    median_D2: float
    max_D2: float
    median_E_mod_pct: float
    max_E_mod_pct: float


def run_bursting_sweep(
    verbose: bool = True,
) -> List[SweepRow]:
    """
    Sweep parameters targeting the bursting regime.

    Strategy: vary (J, gamma, alpha) to find the window where
    E modulation is 30-50% and D₂ > 1.2.
    """
    cd = CorrelationDimension()
    rng = np.random.default_rng(42)

    # Parameter grid: each row is (label, J, gamma, alpha, K)
    # Logic: gamma controls recovery speed, alpha controls drain strength,
    # J controls how much energy is available
    param_grid = [
        # Baseline: moderate drain, slow recovery
        ("base",           1.0, 0.01, 0.3, 1.0),
        # More energy
        ("high_J",         3.0, 0.01, 0.3, 1.0),
        ("very_high_J",    5.0, 0.01, 0.3, 1.0),
        # Stronger drain (should create deeper E dips)
        ("strong_drain",   1.0, 0.01, 1.0, 1.0),
        ("strong_drain_hJ", 3.0, 0.01, 1.0, 1.0),
        # Very slow recovery (timescale ~500)
        ("very_slow",      1.0, 0.002, 0.3, 1.0),
        ("very_slow_hJ",   3.0, 0.002, 0.3, 1.0),
        # Sharp gating (low K = switch-like)
        ("sharp_gate",     1.0, 0.01, 0.3, 0.1),
        ("sharp_gate_hJ",  3.0, 0.01, 0.3, 0.1),
        # Combo: slow + strong drain + sharp
        ("combo1",         2.0, 0.005, 0.5, 0.3),
        ("combo2",         3.0, 0.005, 1.0, 0.3),
        ("combo3",         5.0, 0.003, 0.5, 0.2),
        # Near-quenching: barely enough energy to sustain oscillation
        ("near_quench1",   0.3, 0.01, 0.5, 0.5),
        ("near_quench2",   0.5, 0.005, 0.5, 0.5),
        ("near_quench3",   0.5, 0.01, 1.0, 0.3),
    ]

    n_seeds = 3
    rows = []
    total = len(param_grid) * n_seeds
    done = 0
    start_time = time.time()

    for label, J, gamma, alpha, K in param_grid:
        results = []

        for seed_idx in range(n_seeds):
            X0 = 1.0 + 0.1 * rng.standard_normal()
            Y0 = 1.0 + 0.1 * rng.standard_normal()
            E0 = max(0.1, J / (gamma + 0.01) + rng.standard_normal())

            params = BurstingParams(
                J=J, gamma=gamma, alpha=alpha, K=K,
                X0=X0, Y0=Y0, E0=E0,
            )
            result = classify_run(params, cd)
            results.append(result)

            done += 1
            if verbose:
                elapsed = time.time() - start_time
                eta_sec = (total - done) * elapsed / done if done > 0 else 0
                e_mod = f"{result.E_modulation_pct:.1f}%" if result.success else "N/A"
                print(f"\r  Bursting: {done}/{total} "
                      f"({label}, seed {seed_idx+1}/{n_seeds}, "
                      f"regime={result.regime}, D2={result.D2:.3f}, "
                      f"E_mod={e_mod}) "
                      f"ETA: {eta_sec:.0f}s    ",
                      end="", flush=True)

        d2_vals = [r.D2 for r in results if not np.isnan(r.D2)]
        e_mods = [r.E_modulation_pct for r in results if r.success]

        rows.append(SweepRow(
            label=label, J=J, gamma=gamma, alpha=alpha, K=K,
            n_runs=n_seeds,
            n_failed=sum(1 for r in results if r.regime == "failed"),
            n_blowup=sum(1 for r in results if r.regime == "blowup"),
            n_fixed_point=sum(1 for r in results if r.regime == "fixed_point"),
            n_limit_cycle=sum(1 for r in results if r.regime == "limit_cycle"),
            n_complex=sum(1 for r in results if r.regime == "complex"),
            median_D2=float(np.median(d2_vals)) if d2_vals else np.nan,
            max_D2=float(np.max(d2_vals)) if d2_vals else np.nan,
            median_E_mod_pct=float(np.median(e_mods)) if e_mods else np.nan,
            max_E_mod_pct=float(np.max(e_mods)) if e_mods else np.nan,
        ))

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  Sweep complete in {elapsed:.1f}s")

    return rows


def print_sweep_table(rows: List[SweepRow]):
    """Print formatted results."""
    print("\n" + "=" * 110)
    print("PILOT 3b: Bursting Brusselator (Convention B — energy gates original autocatalysis)")
    print("=" * 110)
    print(f"{'Label':<18} {'J':>4} {'γ':>6} {'α':>5} {'K':>4}  "
          f"{'FP':>3} {'LC':>3} {'Cpx':>3}  "
          f"{'med D2':>7} {'max D2':>7}  "
          f"{'E mod%':>7} {'max E%':>7}")
    print("-" * 110)

    for r in rows:
        d2_str = f"{r.median_D2:.3f}" if not np.isnan(r.median_D2) else "  N/A"
        mx_str = f"{r.max_D2:.3f}" if not np.isnan(r.max_D2) else "  N/A"
        em_str = f"{r.median_E_mod_pct:.1f}" if not np.isnan(r.median_E_mod_pct) else "  N/A"
        mx_em = f"{r.max_E_mod_pct:.1f}" if not np.isnan(r.max_E_mod_pct) else "  N/A"

        # Highlight complex runs
        cpx_str = f" {r.n_complex}" if r.n_complex == 0 else f"*{r.n_complex}"

        print(f"{r.label:<18} {r.J:>4.1f} {r.gamma:>6.3f} {r.alpha:>5.1f} {r.K:>4.1f}  "
              f"{r.n_fixed_point:>3} {r.n_limit_cycle:>3} {cpx_str:>3}  "
              f"{d2_str:>7} {mx_str:>7}  "
              f"{em_str:>7} {mx_em:>7}")

    # Verdict
    any_complex = any(r.n_complex > 0 for r in rows)
    max_d2 = max((r.max_D2 for r in rows if not np.isnan(r.max_D2)), default=0)
    max_emod = max((r.max_E_mod_pct for r in rows if not np.isnan(r.max_E_mod_pct)), default=0)

    print("-" * 110)
    print(f"  Max D₂ seen: {max_d2:.3f}")
    print(f"  Max E modulation: {max_emod:.1f}%")

    if any_complex:
        complex_labels = [r.label for r in rows if r.n_complex > 0]
        print(f"  D₂ > 1.2 FOUND in: {complex_labels}")
        print(f"  VERDICT: Bursting mechanism produces higher-dimensional dynamics!")
    else:
        if max_emod > 20:
            print(f"  E modulation reached {max_emod:.1f}% but D₂ still ≤ 1.2")
            print(f"  VERDICT: Strong modulation but no dimensional increase — "
                  f"try coupled oscillators (structural upgrade 2)")
        else:
            print(f"  E modulation only {max_emod:.1f}% — still slaved")
            print(f"  VERDICT: Need even slower γ or stronger α")

    print("=" * 110)


# ── Entry point ───────────────────────────────────────────────────────

def run_pilot3b(
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """Run Pilot 3b: bursting Brusselator sweep."""
    if verbose:
        print("\n" + "#" * 80)
        print("# PILOT 3b: True Slow–Fast Bursting (Structural Upgrade 1)")
        print("#" * 80)
        print("  Convention B: f(E) gates ORIGINAL autocatalysis")
        print("  E drain ∝ autocatalytic flux (E·X²Y)")
        print("  Goal: E modulation 30-50%, D₂ > 1.2")

    rows = run_bursting_sweep(verbose=verbose)
    print_sweep_table(rows)

    any_complex = any(r.n_complex > 0 for r in rows)
    max_d2 = max((r.max_D2 for r in rows if not np.isnan(r.max_D2)), default=0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        data = {
            'rows': [asdict(r) for r in rows],
            'found_complex': any_complex,
            'max_d2': max_d2,
        }
        path = os.path.join(save_dir, 'pilot3b_results.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2,
                      default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
        if verbose:
            print(f"\n  Results saved to {path}")

    return {'rows': rows, 'found_complex': any_complex, 'max_d2': max_d2}
