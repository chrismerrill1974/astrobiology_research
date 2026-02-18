"""
Pilot 3: Slow-energy feedback model.

Replaces the static Ehi/Elo carriers (Pilots 1–2) with a dynamic energy
variable E(t) that creates slow–fast coupling with the Brusselator.

Model:
    dX/dt = A - (B+1)X + X²Y + k_extra * f(E) * X²Y     (original + energy-gated autocatalysis)
    dY/dt = BX - X²Y - k_extra * f(E) * X²Y               (complementary)
    dE/dt = J - γE - αEX                                    (slow energy with activity drain)

    f(E) = E / (K + E)    (Michaelis-Menten gating)

Convention A: At J=0, E→0, f(E)→0, extra term vanishes → recovers standard Brusselator.

Goal: Scan J to find windows where D₂ > 1.2 (torus / MMO / chaos).
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


# ── Model parameters ─────────────────────────────────────────────────

@dataclass
class SlowEnergyParams:
    """Parameters for the slow-energy Brusselator."""
    # Brusselator base
    A: float = 1.0
    B: float = 3.0

    # Energy dynamics
    J: float = 0.0          # Drive strength (the knob)
    gamma: float = 0.05     # Energy leak rate (timescale ~20)
    alpha: float = 0.05     # Activity-dependent energy drain
    K: float = 1.0          # Half-saturation for gating
    k_extra: float = 1.0    # Extra autocatalytic rate when fully energized

    # Initial conditions
    X0: float = 1.0
    Y0: float = 1.0
    E0: float = 0.0         # Start with no energy

    def copy(self, **overrides) -> 'SlowEnergyParams':
        d = asdict(self)
        d.update(overrides)
        return SlowEnergyParams(**d)


def slow_energy_rhs(t, state, params: SlowEnergyParams):
    """Right-hand side for the slow-energy Brusselator."""
    X, Y, E = state

    # Clamp to non-negative (numerical safety)
    X = max(X, 0.0)
    Y = max(Y, 0.0)
    E = max(E, 0.0)

    # Gating function
    fE = E / (params.K + E)

    # Brusselator + energy-gated extra autocatalysis
    auto_base = X * X * Y
    auto_extra = params.k_extra * fE * X * X * Y

    dX = params.A - (params.B + 1) * X + auto_base + auto_extra
    dY = params.B * X - auto_base - auto_extra
    dE = params.J - params.gamma * E - params.alpha * E * X

    return [dX, dY, dE]


# ── Simulation ────────────────────────────────────────────────────────

def simulate_slow_energy(
    params: SlowEnergyParams,
    t_end: float = 2000.0,
    n_points: int = 20000,
    remove_transient: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the slow-energy Brusselator.

    Parameters
    ----------
    params : SlowEnergyParams
    t_end : float
        Total integration time. Needs to be long for slow-fast systems.
    n_points : int
        Number of output points.
    remove_transient : float
        Fraction of trajectory to discard as transient.

    Returns
    -------
    time : np.ndarray, shape (n_post_transient,)
    trajectory : np.ndarray, shape (n_post_transient, 3)
        Columns: X, Y, E
    """
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        slow_energy_rhs,
        (0, t_end),
        [params.X0, params.Y0, params.E0],
        args=(params,),
        method='LSODA',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=1.0,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Remove transient
    n_discard = int(remove_transient * len(sol.t))
    t = sol.t[n_discard:]
    traj = sol.y[:, n_discard:].T  # shape (n_times, 3)

    return t, traj


# ── Classification ────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Result of a single simulation at one J value."""
    J: float
    success: bool
    regime: str           # "blowup", "fixed_point", "limit_cycle", "complex", "failed"
    D2: float
    D2_uncertainty: float
    quality: str
    osc_passes: bool
    osc_cv: float
    osc_sign_changes: int
    X_range: float        # max(X) - min(X) in post-transient
    E_mean: float         # mean E in post-transient
    E_range: float        # max(E) - min(E)


def classify_run(params: SlowEnergyParams, cd: CorrelationDimension) -> RunResult:
    """
    Simulate at given params and classify the regime.
    """
    J = params.J

    # Simulate
    try:
        t, traj = simulate_slow_energy(params)
    except RuntimeError:
        return RunResult(
            J=J, success=False, regime="failed",
            D2=np.nan, D2_uncertainty=np.nan, quality="FAILED",
            osc_passes=False, osc_cv=0, osc_sign_changes=0,
            X_range=0, E_mean=0, E_range=0,
        )

    X = traj[:, 0]
    Y = traj[:, 1]
    E = traj[:, 2]

    # Check for blowup
    if np.any(np.isnan(traj)) or np.any(np.abs(traj) > 1e6):
        return RunResult(
            J=J, success=False, regime="blowup",
            D2=np.nan, D2_uncertainty=np.nan, quality="FAILED",
            osc_passes=False, osc_cv=0, osc_sign_changes=0,
            X_range=0, E_mean=0, E_range=0,
        )

    X_range = float(np.max(X) - np.min(X))
    E_mean = float(np.mean(E))
    E_range = float(np.max(E) - np.min(E))

    # Check oscillation on X, Y (not E — it's the slow variable)
    osc_result = check_oscillation(
        concentrations=traj[:, :2],  # X, Y only
        time=t,
        species_names=['X', 'Y'],
        food_species=[],
    )

    if not osc_result.passes:
        regime = "fixed_point"
        return RunResult(
            J=J, success=True, regime=regime,
            D2=np.nan, D2_uncertainty=np.nan, quality="N/A",
            osc_passes=False, osc_cv=osc_result.cv,
            osc_sign_changes=osc_result.sign_changes,
            X_range=X_range, E_mean=E_mean, E_range=E_range,
        )

    # Oscillating — compute D₂ on full (X, Y, E) trajectory
    try:
        d2_result = cd.compute(traj, random_state=42)
        D2 = float(d2_result.D2)
        D2_unc = float(d2_result.D2_uncertainty)
        quality = d2_result.quality.value
    except Exception:
        D2 = np.nan
        D2_unc = np.nan
        quality = "FAILED"

    if np.isnan(D2):
        regime = "limit_cycle"  # oscillating but D₂ measurement failed
    elif D2 > 1.2:
        regime = "complex"
    else:
        regime = "limit_cycle"

    return RunResult(
        J=J, success=True, regime=regime,
        D2=D2, D2_uncertainty=D2_unc, quality=quality,
        osc_passes=True, osc_cv=osc_result.cv,
        osc_sign_changes=osc_result.sign_changes,
        X_range=X_range, E_mean=E_mean, E_range=E_range,
    )


# ── J sweep ───────────────────────────────────────────────────────────

@dataclass
class JLevelSummary:
    """Summary at one J level."""
    J: float
    n_runs: int
    n_failed: int
    n_blowup: int
    n_fixed_point: int
    n_limit_cycle: int
    n_complex: int
    median_D2: float
    iqr_D2: float
    max_D2: float
    median_E_mean: float
    median_E_range: float


def run_j_sweep(
    J_values: List[float] = None,
    n_seeds: int = 5,
    base_params: SlowEnergyParams = None,
    verbose: bool = True,
) -> List[JLevelSummary]:
    """
    Sweep J values and classify regime at each.

    Uses multiple seeds (perturbed initial conditions) per J level.
    """
    if J_values is None:
        J_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    if base_params is None:
        base_params = SlowEnergyParams()

    cd = CorrelationDimension()
    rng = np.random.default_rng(42)

    summaries = []
    total = len(J_values) * n_seeds
    done = 0
    start_time = time.time()

    for J in J_values:
        results = []

        for seed_idx in range(n_seeds):
            # Perturb initial conditions slightly
            X0 = 1.0 + 0.1 * rng.standard_normal()
            Y0 = 1.0 + 0.1 * rng.standard_normal()
            E0 = max(0.0, 0.1 * rng.standard_normal())

            params = base_params.copy(J=J, X0=X0, Y0=Y0, E0=E0)
            result = classify_run(params, cd)
            results.append(result)

            done += 1
            if verbose:
                elapsed = time.time() - start_time
                eta_sec = (total - done) * elapsed / done if done > 0 else 0
                print(f"\r  J-sweep: {done}/{total} "
                      f"(J={J:.2f}, seed {seed_idx+1}/{n_seeds}, "
                      f"regime={result.regime}, D2={result.D2:.3f}) "
                      f"ETA: {eta_sec:.0f}s    ",
                      end="", flush=True)

        # Summarize
        d2_vals = [r.D2 for r in results if not np.isnan(r.D2)]
        e_means = [r.E_mean for r in results if r.success]
        e_ranges = [r.E_range for r in results if r.success]

        summaries.append(JLevelSummary(
            J=J,
            n_runs=n_seeds,
            n_failed=sum(1 for r in results if r.regime == "failed"),
            n_blowup=sum(1 for r in results if r.regime == "blowup"),
            n_fixed_point=sum(1 for r in results if r.regime == "fixed_point"),
            n_limit_cycle=sum(1 for r in results if r.regime == "limit_cycle"),
            n_complex=sum(1 for r in results if r.regime == "complex"),
            median_D2=float(np.median(d2_vals)) if d2_vals else np.nan,
            iqr_D2=float(np.percentile(d2_vals, 75) - np.percentile(d2_vals, 25)) if len(d2_vals) >= 2 else np.nan,
            max_D2=float(np.max(d2_vals)) if d2_vals else np.nan,
            median_E_mean=float(np.median(e_means)) if e_means else np.nan,
            median_E_range=float(np.median(e_ranges)) if e_ranges else np.nan,
        ))

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  J-sweep complete in {elapsed:.1f}s")

    return summaries


# ── Parameter exploration ─────────────────────────────────────────────

def run_param_exploration(
    verbose: bool = True,
) -> List[JLevelSummary]:
    """
    If the default parameters don't produce D₂>1.2, try alternative
    parameter sets (different gamma, alpha, k_extra).
    """
    param_sets = [
        ("default",   SlowEnergyParams(gamma=0.05, alpha=0.05, K=1.0, k_extra=1.0)),
        ("slow_leak", SlowEnergyParams(gamma=0.02, alpha=0.05, K=1.0, k_extra=1.5)),
        ("strong_gate", SlowEnergyParams(gamma=0.05, alpha=0.1, K=0.5, k_extra=2.0)),
        ("very_slow",  SlowEnergyParams(gamma=0.01, alpha=0.02, K=1.0, k_extra=1.0)),
    ]

    # Focused J values for exploration
    J_values = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]

    best_summaries = []
    best_label = ""
    best_max_d2 = 0

    for label, base_params in param_sets:
        if verbose:
            print(f"\n  --- Param set: {label} "
                  f"(γ={base_params.gamma}, α={base_params.alpha}, "
                  f"K={base_params.K}, k_extra={base_params.k_extra}) ---")

        summaries = run_j_sweep(
            J_values=J_values,
            n_seeds=3,  # Quick exploration
            base_params=base_params,
            verbose=verbose,
        )

        max_d2 = max((s.max_D2 for s in summaries if not np.isnan(s.max_D2)), default=0)
        n_complex = sum(s.n_complex for s in summaries)

        if verbose:
            print(f"  Max D2 seen: {max_d2:.3f}, complex runs: {n_complex}")

        if max_d2 > best_max_d2:
            best_max_d2 = max_d2
            best_summaries = summaries
            best_label = label

    if verbose:
        print(f"\n  Best param set: {best_label} (max D2 = {best_max_d2:.3f})")

    return best_summaries


# ── Printing ──────────────────────────────────────────────────────────

def print_sweep_table(summaries: List[JLevelSummary]):
    """Print formatted table."""
    print("\n" + "=" * 95)
    print("PILOT 3: Slow-Energy Drive Window (J scan at k=0)")
    print("=" * 95)
    print(f"{'J':>5}  {'N':>3}  {'Fail':>4}  {'Blow':>4}  {'FP':>4}  {'LC':>4}  "
          f"{'Cplx':>4}  {'med D2':>7}  {'max D2':>7}  {'IQR':>6}  "
          f"{'E mean':>7}  {'E range':>8}")
    print("-" * 95)

    for s in summaries:
        d2_str = f"{s.median_D2:.3f}" if not np.isnan(s.median_D2) else "  N/A"
        mx_str = f"{s.max_D2:.3f}" if not np.isnan(s.max_D2) else "  N/A"
        iqr_str = f"{s.iqr_D2:.3f}" if not np.isnan(s.iqr_D2) else " N/A"
        em_str = f"{s.median_E_mean:.2f}" if not np.isnan(s.median_E_mean) else "  N/A"
        er_str = f"{s.median_E_range:.3f}" if not np.isnan(s.median_E_range) else "   N/A"
        print(f"{s.J:>5.2f}  {s.n_runs:>3}  {s.n_failed:>4}  {s.n_blowup:>4}  "
              f"{s.n_fixed_point:>4}  {s.n_limit_cycle:>4}  {s.n_complex:>4}  "
              f"{d2_str:>7}  {mx_str:>7}  {iqr_str:>6}  {em_str:>7}  {er_str:>8}")

    # Verdict
    any_complex = any(s.n_complex > 0 for s in summaries)
    max_d2 = max((s.max_D2 for s in summaries if not np.isnan(s.max_D2)), default=0)
    print("-" * 95)
    if any_complex:
        complex_J = [s.J for s in summaries if s.n_complex > 0]
        print(f"  D₂ > 1.2 FOUND at J = {complex_J}  (max D₂ = {max_d2:.3f})")
        print(f"  VERDICT: Slow-energy model produces higher-dimensional dynamics")
    else:
        print(f"  D₂ > 1.2 NOT FOUND  (max D₂ = {max_d2:.3f})")
        print(f"  VERDICT: Adjust parameters or try coupled oscillators")
    print("=" * 95)


# ── Main entry point ──────────────────────────────────────────────────

def run_pilot3(
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Run Pilot 3: slow-energy feedback model.

    Phase 1: Default parameters, fine J grid
    Phase 2: If no D₂>1.2 found, explore alternative parameter sets

    Returns dict with results.
    """
    if verbose:
        print("\n" + "#" * 80)
        print("# PILOT 3: Slow-Energy Feedback Model")
        print("#" * 80)
        print("  Model: dE/dt = J - γE - αEX")
        print("  Gating: v_extra = k_extra * E/(K+E) * X²Y")
        print("  Default: γ=0.05, α=0.05, K=1.0, k_extra=1.0")

    # Phase 1: Default parameters, wide J sweep
    if verbose:
        print("\n>>> Phase 1: Default parameters, J sweep")

    summaries = run_j_sweep(
        J_values=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
        n_seeds=5,
        verbose=verbose,
    )
    print_sweep_table(summaries)

    any_complex = any(s.n_complex > 0 for s in summaries)
    max_d2 = max((s.max_D2 for s in summaries if not np.isnan(s.max_D2)), default=0)

    # Phase 2: Parameter exploration if needed
    explore_summaries = None
    if not any_complex:
        if verbose:
            print("\n>>> Phase 2: No D₂>1.2 found — exploring alternative parameters")
        explore_summaries = run_param_exploration(verbose=verbose)
        if explore_summaries:
            print_sweep_table(explore_summaries)
    else:
        if verbose:
            print("\n>>> Phase 2: Skipped (D₂>1.2 already found)")

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        data = {
            'phase1': [asdict(s) for s in summaries],
            'phase2': [asdict(s) for s in explore_summaries] if explore_summaries else None,
            'found_complex': any_complex,
            'max_d2': max_d2,
        }
        path = os.path.join(save_dir, 'pilot3_results.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2,
                      default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
        if verbose:
            print(f"\n  Results saved to {path}")

    return {
        'phase1': summaries,
        'phase2': explore_summaries,
        'found_complex': any_complex,
        'max_d2': max_d2,
    }
