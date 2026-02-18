"""
Oscillation survival filter for Paper 2.

Implements the locked oscillation-preservation criterion from the research plan.
A network passes the filter if at least one non-food, non-waste species
satisfies ALL three criteria:
  1. Boundedness: 0.2 < c(t_end)/c(t_mid) < 5
  2. Non-monotonic: >= 5 sign changes in smoothed dc/dt
  3. Amplitude: CV > 0.03

The filter returns a binary outcome (pass/fail) plus diagnostic metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from .simulator import (
    ReactionSimulator,
    DrivingMode,
    SimulationResult,
)
from .network_generator import GeneratedNetwork


@dataclass
class OscillationResult:
    """Result of oscillation filter evaluation."""
    passes: bool
    cv: float                    # CV of best species
    amplitude: float             # Mean amplitude of best species
    sign_changes: int            # Sign changes in smoothed dc/dt of best species
    boundedness_ratio: float     # c(t_end)/c(t_mid) of best species
    best_species_idx: int        # Index of best species (-1 if none pass)
    best_species_name: str = ""  # Name of best species


def check_oscillation(
    concentrations: np.ndarray,
    time: np.ndarray,
    species_names: Optional[List[str]] = None,
    food_species: Optional[List[str]] = None,
) -> OscillationResult:
    """
    Check whether a simulated network exhibits oscillatory dynamics.

    Evaluates the second half of the trajectory (post-transient) for each
    non-food, non-waste species. A species is considered oscillatory if it
    satisfies all three locked criteria.

    Parameters
    ----------
    concentrations : np.ndarray
        Shape (n_times, n_species). Full simulation trajectory.
    time : np.ndarray
        Shape (n_times,). Time points.
    species_names : list of str, optional
        Names of species. Used to exclude food species.
    food_species : list of str, optional
        Names of food/clamped species to exclude from evaluation.

    Returns
    -------
    OscillationResult
        Filter outcome with diagnostic metrics.
    """
    if food_species is None:
        food_species = []

    n_times, n_species = concentrations.shape
    t_mid_idx = n_times // 2

    # Analysis window: second half of trajectory
    t_window = time[t_mid_idx:]
    c_window = concentrations[t_mid_idx:]

    best_result = OscillationResult(
        passes=False, cv=0.0, amplitude=0.0,
        sign_changes=0, boundedness_ratio=0.0,
        best_species_idx=-1,
    )
    best_score = -1  # Track best candidate by sign_changes

    for i in range(n_species):
        # Skip food species
        if species_names and i < len(species_names):
            if species_names[i] in food_species:
                continue

        c_i = c_window[:, i]

        # Skip species with negligible concentration
        if np.max(np.abs(c_i)) < 1e-12:
            continue

        # Check if species is a pure sink (monotonically increasing waste)
        dc = np.diff(c_i)
        if np.all(dc >= -1e-12):
            continue

        # --- Criterion 1: Boundedness ---
        c_end = c_i[-1]
        c_mid = c_i[0]  # Start of analysis window = midpoint of full sim
        if c_mid < 1e-12:
            boundedness = float('inf')
        else:
            boundedness = c_end / c_mid
        passes_boundedness = 0.2 < boundedness < 5.0

        # --- Criterion 2: Non-monotonic (sign changes in smoothed dc/dt) ---
        dt = np.diff(t_window)
        dc_dt = np.diff(c_i) / dt

        # 5-point moving average smoothing
        if len(dc_dt) >= 5:
            kernel = np.ones(5) / 5.0
            dc_dt_smooth = np.convolve(dc_dt, kernel, mode='valid')
        else:
            dc_dt_smooth = dc_dt

        sign_changes = _count_sign_changes(dc_dt_smooth)
        passes_nonmonotonic = sign_changes >= 5

        # --- Criterion 3: Amplitude (CV) ---
        mean_c = np.mean(c_i)
        if mean_c < 1e-12:
            cv = 0.0
        else:
            cv = np.std(c_i) / mean_c
        passes_amplitude = cv > 0.03

        # Mean amplitude (peak-to-trough)
        amplitude = np.max(c_i) - np.min(c_i)

        # Does this species pass all three?
        all_pass = passes_boundedness and passes_nonmonotonic and passes_amplitude

        # Track best species (by sign changes as primary, CV as tiebreaker)
        score = sign_changes * 1000 + cv * 100
        if all_pass and score > best_score:
            best_score = score
            name = species_names[i] if species_names and i < len(species_names) else ""
            best_result = OscillationResult(
                passes=True,
                cv=cv,
                amplitude=amplitude,
                sign_changes=sign_changes,
                boundedness_ratio=boundedness,
                best_species_idx=i,
                best_species_name=name,
            )

    # If no species passed, report metrics from the best candidate anyway
    if not best_result.passes:
        # Find best non-food species by sign changes for diagnostic reporting
        for i in range(n_species):
            if species_names and i < len(species_names):
                if species_names[i] in food_species:
                    continue
            c_i = c_window[:, i]
            if np.max(np.abs(c_i)) < 1e-12:
                continue

            c_end = c_i[-1]
            c_mid = c_i[0]
            boundedness = c_end / c_mid if c_mid > 1e-12 else float('inf')

            dt = np.diff(t_window)
            dc_dt = np.diff(c_i) / dt
            if len(dc_dt) >= 5:
                kernel = np.ones(5) / 5.0
                dc_dt_smooth = np.convolve(dc_dt, kernel, mode='valid')
            else:
                dc_dt_smooth = dc_dt
            sign_changes = _count_sign_changes(dc_dt_smooth)

            mean_c = np.mean(c_i)
            cv = np.std(c_i) / mean_c if mean_c > 1e-12 else 0.0
            amplitude = np.max(c_i) - np.min(c_i)

            score = sign_changes * 1000 + cv * 100
            if score > best_score:
                best_score = score
                name = species_names[i] if species_names and i < len(species_names) else ""
                best_result = OscillationResult(
                    passes=False,
                    cv=cv,
                    amplitude=amplitude,
                    sign_changes=sign_changes,
                    boundedness_ratio=boundedness,
                    best_species_idx=i,
                    best_species_name=name,
                )

    return best_result


def passes_oscillation_filter(
    net: GeneratedNetwork,
    dilution_rate: float = 0.1,
    t_end: float = 100.0,
    n_points: int = 2000,
) -> OscillationResult:
    """
    Convenience function: simulate a network and check for oscillation.

    Parameters
    ----------
    net : GeneratedNetwork
        Network to evaluate.
    dilution_rate : float
        CSTR dilution rate (default 0.1, matching Paper 1).
    t_end : float
        Simulation end time (default 100).
    n_points : int
        Number of output time points.

    Returns
    -------
    OscillationResult
        Filter outcome with diagnostic metrics.
    """
    sim = ReactionSimulator()
    network = sim.build_network(net.reactions)

    try:
        result = sim.simulate(
            network,
            rate_constants=net.rate_constants,
            initial_concentrations=net.initial_concentrations,
            t_span=(0, t_end),
            n_points=n_points,
            driving_mode=DrivingMode.CHEMOSTAT,
            chemostat_species=dict(net.chemostat_species),
        )
    except Exception:
        return OscillationResult(
            passes=False, cv=0.0, amplitude=0.0,
            sign_changes=0, boundedness_ratio=0.0,
            best_species_idx=-1,
        )

    if not result.success:
        return OscillationResult(
            passes=False, cv=0.0, amplitude=0.0,
            sign_changes=0, boundedness_ratio=0.0,
            best_species_idx=-1,
        )

    return check_oscillation(
        concentrations=result.concentrations,
        time=result.time,
        species_names=result.species_names,
        food_species=net.food_set,
    )


def _count_sign_changes(x: np.ndarray) -> int:
    """Count the number of sign changes in array x."""
    if len(x) < 2:
        return 0
    signs = np.sign(x)
    # Remove zeros (treat as continuation of previous sign)
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(nonzero)) > 0))
