"""
Pilot 4: Two coupled Brusselator oscillators sharing a slow energy pool.

This is the modular-coupling path (Hypothesis B):
Dimensional inflation requires competing oscillatory modules, not just
energy injection into a single oscillator.

Model (5 state variables):
    For i = 1, 2:
        dXi/dt = A - (B+1)Xi + Xi²Yi + k_extra * f(E) * Xi²Yi
        dYi/dt = BXi - Xi²Yi - k_extra * f(E) * Xi²Yi

    dE/dt = J - γE - α * f(E) * (X1²Y1 + X2²Y2)

    f(E) = E / (K + E)

Convention A: At J=0, E→0, f(E)→0, extra term vanishes → two uncoupled
standard Brusselators. Energy coupling is the ONLY interaction channel.

Expected regimes:
    - Phase locking (D₂ ≈ 1) — both oscillators sync
    - Phase drift / quasi-periodicity (D₂ ≈ 2) — incommensurate frequencies
    - Intermittent switching / chaos (D₂ > 1)
    - Collapse to fixed point (too much damping)
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


# ── Model ─────────────────────────────────────────────────────────────

@dataclass
class CoupledParams:
    """Parameters for two coupled Brusselators + shared energy pool."""
    # Brusselator base (same for both cores)
    A: float = 1.0
    B: float = 3.0

    # Energy-gated coupling
    k_extra: float = 0.5    # Extra autocatalytic rate when energized
    K: float = 1.0           # Half-saturation for gating

    # Energy pool dynamics
    J: float = 0.5           # Energy inflow (drive knob)
    gamma: float = 0.02      # Slow energy leak
    alpha: float = 0.1       # Energy drain per autocatalytic flux

    # Initial conditions (slightly different per core to break symmetry)
    X1_0: float = 1.0
    Y1_0: float = 1.0
    X2_0: float = 1.1       # Perturbation to break sync
    Y2_0: float = 0.9
    E0: float = 5.0

    def copy(self, **overrides) -> 'CoupledParams':
        d = asdict(self)
        d.update(overrides)
        return CoupledParams(**d)


def coupled_rhs(t, state, p: CoupledParams):
    """RHS for coupled Brusselators + shared energy."""
    X1, Y1, X2, Y2, E = state

    # Clamp non-negative
    X1 = max(X1, 0.0)
    Y1 = max(Y1, 0.0)
    X2 = max(X2, 0.0)
    Y2 = max(Y2, 0.0)
    E = max(E, 0.0)

    fE = E / (p.K + E)

    # Core 1 autocatalytic flux
    auto1 = X1 * X1 * Y1
    extra1 = p.k_extra * fE * auto1

    # Core 2 autocatalytic flux
    auto2 = X2 * X2 * Y2
    extra2 = p.k_extra * fE * auto2

    # Core 1 ODEs
    dX1 = p.A - (p.B + 1) * X1 + auto1 + extra1
    dY1 = p.B * X1 - auto1 - extra1

    # Core 2 ODEs
    dX2 = p.A - (p.B + 1) * X2 + auto2 + extra2
    dY2 = p.B * X2 - auto2 - extra2

    # Shared energy pool
    dE = p.J - p.gamma * E - p.alpha * fE * (auto1 + auto2)

    return [dX1, dY1, dX2, dY2, dE]


def simulate_coupled(
    params: CoupledParams,
    t_end: float = 3000.0,
    n_points: int = 40000,
    remove_transient: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the coupled system."""
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        coupled_rhs,
        (0, t_end),
        [params.X1_0, params.Y1_0, params.X2_0, params.Y2_0, params.E0],
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
    traj = sol.y[:, n_discard:].T  # (n_times, 5)

    return t, traj


# ── Classification ────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Result of a single coupled simulation."""
    J: float
    k_extra: float
    gamma: float
    alpha: float
    K: float
    success: bool
    regime: str           # "blowup", "fixed_point", "phase_locked", "complex", "failed"
    D2: float
    D2_uncertainty: float
    quality: str
    X1_range: float
    X2_range: float
    E_mean: float
    E_range: float
    phase_corr: float     # Pearson correlation between X1 and X2 (1 = locked, 0 = drift)


def _check_oscillating(x: np.ndarray) -> bool:
    """Quick check: does a 1D signal oscillate?"""
    if np.max(np.abs(x)) < 1e-12:
        return False
    mean_x = np.mean(x)
    if mean_x < 1e-12:
        return False
    cv = np.std(x) / mean_x
    if cv < 0.03:
        return False
    # Check sign changes in smoothed derivative
    dx = np.diff(x)
    if len(dx) >= 5:
        kernel = np.ones(5) / 5.0
        dx_smooth = np.convolve(dx, kernel, mode='valid')
    else:
        dx_smooth = dx
    signs = np.sign(dx_smooth)
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return False
    sign_changes = int(np.sum(np.abs(np.diff(nonzero)) > 0))
    return sign_changes >= 5


def classify_run(params: CoupledParams, cd: CorrelationDimension) -> RunResult:
    """Simulate and classify."""
    base = dict(
        J=params.J, k_extra=params.k_extra,
        gamma=params.gamma, alpha=params.alpha, K=params.K,
    )

    try:
        t, traj = simulate_coupled(params)
    except RuntimeError:
        return RunResult(
            **base, success=False, regime="failed",
            D2=np.nan, D2_uncertainty=np.nan, quality="FAILED",
            X1_range=0, X2_range=0, E_mean=0, E_range=0, phase_corr=0,
        )

    X1, Y1, X2, Y2, E = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3], traj[:, 4]

    if np.any(np.isnan(traj)) or np.any(np.abs(traj) > 1e6):
        return RunResult(
            **base, success=False, regime="blowup",
            D2=np.nan, D2_uncertainty=np.nan, quality="FAILED",
            X1_range=0, X2_range=0, E_mean=0, E_range=0, phase_corr=0,
        )

    X1_range = float(np.max(X1) - np.min(X1))
    X2_range = float(np.max(X2) - np.min(X2))
    E_mean = float(np.mean(E))
    E_range = float(np.max(E) - np.min(E))

    # Phase correlation: Pearson r between X1 and X2
    if np.std(X1) > 1e-10 and np.std(X2) > 1e-10:
        phase_corr = float(np.corrcoef(X1, X2)[0, 1])
    else:
        phase_corr = np.nan

    # Check if either core oscillates
    osc1 = _check_oscillating(X1)
    osc2 = _check_oscillating(X2)

    if not osc1 and not osc2:
        return RunResult(
            **base, success=True, regime="fixed_point",
            D2=np.nan, D2_uncertainty=np.nan, quality="N/A",
            X1_range=X1_range, X2_range=X2_range,
            E_mean=E_mean, E_range=E_range, phase_corr=phase_corr,
        )

    # Compute D₂ on full 5D trajectory (X1, Y1, X2, Y2, E)
    try:
        d2_result = cd.compute(traj, random_state=42)
        D2 = float(d2_result.D2)
        D2_unc = float(d2_result.D2_uncertainty)
        quality = d2_result.quality.value
    except Exception:
        D2 = np.nan
        D2_unc = np.nan
        quality = "FAILED"

    # Classify
    if np.isnan(D2):
        regime = "phase_locked"
    elif D2 > 1.2:
        regime = "complex"
    elif not np.isnan(phase_corr) and abs(phase_corr) > 0.95:
        regime = "phase_locked"
    else:
        regime = "phase_locked"  # D₂ ≈ 1 but not necessarily correlated

    return RunResult(
        **base, success=True, regime=regime,
        D2=D2, D2_uncertainty=D2_unc, quality=quality,
        X1_range=X1_range, X2_range=X2_range,
        E_mean=E_mean, E_range=E_range, phase_corr=phase_corr,
    )


# ── Sweep ─────────────────────────────────────────────────────────────

@dataclass
class SweepRow:
    """One row of results."""
    label: str
    J: float
    k_extra: float
    gamma: float
    alpha: float
    K: float
    n_runs: int
    n_fixed_point: int
    n_phase_locked: int
    n_complex: int
    n_failed: int
    median_D2: float
    max_D2: float
    median_phase_corr: float
    median_E_range: float


def run_coupled_sweep(verbose: bool = True) -> List[SweepRow]:
    """
    Sweep parameters for the coupled oscillator system.

    The two key axes:
    - k_extra: coupling strength through energy (how much E matters)
    - J: how much energy is available

    Also vary gamma (E timescale) and alpha (drain strength).
    """
    cd = CorrelationDimension()
    rng = np.random.default_rng(42)

    # Parameter grid: (label, J, k_extra, gamma, alpha, K)
    param_grid = [
        # Baseline: no coupling (J=0)
        ("no_drive",       0.0,  0.5, 0.02, 0.1, 1.0),

        # Weak coupling, varying J
        ("weak_lowJ",      0.3,  0.3, 0.02, 0.1, 1.0),
        ("weak_midJ",      0.5,  0.3, 0.02, 0.1, 1.0),
        ("weak_hiJ",       1.0,  0.3, 0.02, 0.1, 1.0),

        # Medium coupling, varying J
        ("med_lowJ",       0.3,  0.5, 0.02, 0.1, 1.0),
        ("med_midJ",       0.5,  0.5, 0.02, 0.1, 1.0),
        ("med_hiJ",        1.0,  0.5, 0.02, 0.1, 1.0),
        ("med_vhiJ",       2.0,  0.5, 0.02, 0.1, 1.0),

        # Strong coupling
        ("strong_lowJ",    0.3,  1.0, 0.02, 0.1, 1.0),
        ("strong_midJ",    0.5,  1.0, 0.02, 0.1, 1.0),
        ("strong_hiJ",     1.0,  1.0, 0.02, 0.1, 1.0),
        ("strong_vhiJ",    2.0,  1.0, 0.02, 0.1, 1.0),

        # Very strong coupling
        ("vstrong_midJ",   0.5,  2.0, 0.02, 0.1, 1.0),
        ("vstrong_hiJ",    1.0,  2.0, 0.02, 0.1, 1.0),

        # Slower energy (gamma smaller)
        ("slow_med",       0.5,  0.5, 0.005, 0.1, 1.0),
        ("slow_strong",    0.5,  1.0, 0.005, 0.1, 1.0),
        ("slow_strong_hJ", 1.0,  1.0, 0.005, 0.1, 1.0),

        # Stronger drain
        ("drain_med",      0.5,  0.5, 0.02, 0.3, 1.0),
        ("drain_strong",   1.0,  1.0, 0.02, 0.3, 1.0),

        # Low K (sharper gating)
        ("sharp_med",      0.5,  0.5, 0.02, 0.1, 0.3),
        ("sharp_strong",   1.0,  1.0, 0.02, 0.1, 0.3),
    ]

    n_seeds = 3
    rows = []
    total = len(param_grid) * n_seeds
    done = 0
    start_time = time.time()

    for label, J, k_extra, gamma, alpha, K in param_grid:
        results = []

        for seed_idx in range(n_seeds):
            # Randomize initial conditions to break symmetry
            X1_0 = 1.0 + 0.2 * rng.standard_normal()
            Y1_0 = 1.0 + 0.2 * rng.standard_normal()
            X2_0 = 1.0 + 0.2 * rng.standard_normal()
            Y2_0 = 1.0 + 0.2 * rng.standard_normal()
            E0 = max(0.1, J / (gamma + 0.01) * 0.5 + rng.standard_normal())

            params = CoupledParams(
                J=J, k_extra=k_extra, gamma=gamma, alpha=alpha, K=K,
                X1_0=X1_0, Y1_0=Y1_0, X2_0=X2_0, Y2_0=Y2_0, E0=E0,
            )
            result = classify_run(params, cd)
            results.append(result)

            done += 1
            if verbose:
                elapsed = time.time() - start_time
                eta_sec = (total - done) * elapsed / done if done > 0 else 0
                pc = f"{result.phase_corr:.2f}" if not np.isnan(result.phase_corr) else "N/A"
                print(f"\r  Coupled: {done}/{total} "
                      f"({label}, s{seed_idx+1}, "
                      f"{result.regime}, D2={result.D2:.3f}, "
                      f"r={pc}) "
                      f"ETA: {eta_sec:.0f}s    ",
                      end="", flush=True)

        d2_vals = [r.D2 for r in results if not np.isnan(r.D2)]
        pc_vals = [r.phase_corr for r in results if not np.isnan(r.phase_corr)]
        er_vals = [r.E_range for r in results if r.success]

        rows.append(SweepRow(
            label=label, J=J, k_extra=k_extra, gamma=gamma, alpha=alpha, K=K,
            n_runs=n_seeds,
            n_fixed_point=sum(1 for r in results if r.regime == "fixed_point"),
            n_phase_locked=sum(1 for r in results if r.regime == "phase_locked"),
            n_complex=sum(1 for r in results if r.regime == "complex"),
            n_failed=sum(1 for r in results if r.regime in ("failed", "blowup")),
            median_D2=float(np.median(d2_vals)) if d2_vals else np.nan,
            max_D2=float(np.max(d2_vals)) if d2_vals else np.nan,
            median_phase_corr=float(np.median(pc_vals)) if pc_vals else np.nan,
            median_E_range=float(np.median(er_vals)) if er_vals else np.nan,
        ))

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  Sweep complete in {elapsed:.1f}s")

    return rows


def print_sweep_table(rows: List[SweepRow]):
    """Print formatted results."""
    print("\n" + "=" * 115)
    print("PILOT 4: Two Coupled Brusselators + Shared Energy Pool")
    print("=" * 115)
    print(f"{'Label':<18} {'J':>4} {'k_ex':>5} {'γ':>6} {'α':>4} {'K':>4}  "
          f"{'FP':>3} {'Lock':>4} {'Cpx':>3}  "
          f"{'med D2':>7} {'max D2':>7}  "
          f"{'r(X1,X2)':>8}  {'E range':>7}")
    print("-" * 115)

    for r in rows:
        d2_str = f"{r.median_D2:.3f}" if not np.isnan(r.median_D2) else "  N/A"
        mx_str = f"{r.max_D2:.3f}" if not np.isnan(r.max_D2) else "  N/A"
        pc_str = f"{r.median_phase_corr:.3f}" if not np.isnan(r.median_phase_corr) else "   N/A"
        er_str = f"{r.median_E_range:.2f}" if not np.isnan(r.median_E_range) else "  N/A"
        cpx_str = f" {r.n_complex}" if r.n_complex == 0 else f"*{r.n_complex}"

        print(f"{r.label:<18} {r.J:>4.1f} {r.k_extra:>5.1f} {r.gamma:>6.3f} {r.alpha:>4.1f} {r.K:>4.1f}  "
              f"{r.n_fixed_point:>3} {r.n_phase_locked:>4} {cpx_str:>3}  "
              f"{d2_str:>7} {mx_str:>7}  "
              f"{pc_str:>8}  {er_str:>7}")

    any_complex = any(r.n_complex > 0 for r in rows)
    max_d2 = max((r.max_D2 for r in rows if not np.isnan(r.max_D2)), default=0)

    print("-" * 115)
    print(f"  Max D₂: {max_d2:.3f}")

    if any_complex:
        cx_labels = [r.label for r in rows if r.n_complex > 0]
        print(f"  D₂ > 1.2 FOUND in: {cx_labels}")
        print(f"  VERDICT: Modular coupling produces higher-dimensional dynamics!")
    else:
        # Check for desynchronization (low phase correlation)
        desync = [r for r in rows if not np.isnan(r.median_phase_corr) and abs(r.median_phase_corr) < 0.8]
        if desync:
            ds_labels = [r.label for r in desync]
            print(f"  Desynchronization seen in: {ds_labels}")
            print(f"  VERDICT: Oscillators decouple but D₂ stays ≤ 1.2 — may need finer grid or longer integration")
        else:
            print(f"  VERDICT: Oscillators remain phase-locked at all params — need stronger symmetry breaking")

    print("=" * 115)


# ── Entry point ───────────────────────────────────────────────────────

def run_pilot4(
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """Run Pilot 4: coupled oscillators."""
    if verbose:
        print("\n" + "#" * 80)
        print("# PILOT 4: Two Coupled Brusselators + Shared Energy Pool")
        print("#" * 80)
        print("  State: (X1, Y1, X2, Y2, E)")
        print("  Coupling: shared energy pool drained by both cores")
        print("  Goal: phase drift / quasi-periodicity → D₂ > 1.2")

    rows = run_coupled_sweep(verbose=verbose)
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
        path = os.path.join(save_dir, 'pilot4_results.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2,
                      default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
        if verbose:
            print(f"\n  Results saved to {path}")

    return {'rows': rows, 'found_complex': any_complex, 'max_d2': max_d2}
