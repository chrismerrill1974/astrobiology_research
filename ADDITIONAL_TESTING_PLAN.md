# Additional Testing Plan for `dimensional_opening` Shared Code

**Date:** 2026-02-25
**Context:** Pre-Paper 5 hardening of the shared library used across all four existing papers.
**Scope:** `shared/dimensional_opening/` modules and `shared/tests/`

---

## Overview

The existing test suite (~100+ cases across 7 files) provides solid happy-path coverage of the
core pipeline: stoichiometry -> simulation -> correlation dimension -> activation tracker. However,
several categories of gaps remain that could produce **silent wrong answers**, **crashes on
realistic edge cases**, or **unreproducible statistical claims**. This plan addresses those gaps
in three phases, ordered by risk to scientific correctness.

---

## Phase 1 — Guard Against Silent Wrong Answers

These tests protect the pipeline seams where data flows between modules. A bug here
would not crash — it would quietly produce incorrect eta values.

### 1.1 `correlation_dimension.py`: D2 exceeding embedding dimension

**What:** When the estimated D2 is greater than the number of columns in the input trajectory,
the result is physically meaningless. This can happen with noisy or short data.

**Why it matters:** An inflated D2 directly inflates eta = D2 / r_S, which is the central
quantity in all four papers.

**Tests to add (in `test_correlation_dimension.py`):**
- Compute D2 on a 2D trajectory from a known 2D system. Verify D2 <= 2 or quality is
  MARGINAL/FAILED.
- Construct a pathological case (e.g., short noisy 2D data) that could produce D2 > 2.
  Verify the result's quality flag is not GOOD.
- Add a parameterized test: for dim in [2, 3, 5], generate uniform hypercube data and
  verify D2 does not exceed dim + 0.5 (with tolerance for finite-size effects).

**Estimated effort:** 3 tests, small.

### 1.2 `correlation_dimension.py`: `compute_from_simulation()` direct testing

**What:** This method bridges `SimulationResult` to D2 computation. It handles transient
removal and species selection internally. Currently tested only indirectly through
`ActivationTracker`.

**Why it matters:** If transient removal or species extraction has a bug, every downstream
eta value is wrong — and no existing test would catch it because the tracker adds its own
layer of processing.

**Tests to add (in `test_correlation_dimension.py`):**
- Run Brusselator simulation, call `compute_from_simulation()` directly. Verify D2 ~ 1.0
  (limit cycle).
- Pass a simulation result with `remove_transient=0.9` and verify the trajectory used is
  only the final 10%.
- Pass a simulation result where all species are monotonic (e.g., pure decay A -> B -> C
  with no driving). Verify the method handles it gracefully (FAILED quality or appropriate
  skip).

**Estimated effort:** 3 tests, small.

### 1.3 `activation_tracker.py`: Trajectory extraction / species filtering

**What:** The tracker automatically excludes monotonically increasing or decreasing species
(waste products, depleted feedstocks) before computing D2. This filtering logic determines
which dimensions of the trajectory are used.

**Why it matters:** Over-filtering loses real dynamical dimensions and deflates D2.
Under-filtering includes accumulator species that add a trivial linear dimension, inflating D2.

**Tests to add (in `test_activation_tracker.py`):**
- Create a simulation with 2 oscillating species + 1 monotonically increasing waste species.
  Verify the waste species is excluded from the D2 trajectory.
- Create a simulation with 2 oscillating species + 1 constant (chemostatted) species.
  Verify the constant species is excluded.
- Create a simulation where ALL species are monotonic. Verify the tracker returns a skipped
  result with an appropriate skip_reason, not a crash.
- Create a simulation where only 1 non-monotonic species remains after filtering. Verify
  behavior (D2 of a 1D trajectory should be ~1 or skipped if below minimum dimension
  threshold).

**Estimated effort:** 4 tests, medium.

### 1.4 `simulator.py`: ODE solver failure handling

**What:** When `scipy.integrate.solve_ivp` returns `sol.success = False`, the simulator
currently warns but still returns the (potentially garbage) trajectory.

**Why it matters:** Downstream code trusts `result.success` but the D2 computation does not
check it — a failed simulation's noisy partial trajectory could produce a plausible-looking
but meaningless D2.

**Tests to add (in `test_simulator.py`):**
- Construct a system known to cause solver failure (e.g., extremely stiff rates like
  k1=1e6, k2=1e-6 with tight tolerances and short max_step). Verify `result.success`
  is False.
- Verify the `ActivationTracker` skips networks where simulation fails (integration test,
  could go in `test_activation_tracker.py`).

**Estimated effort:** 2 tests, small.

### 1.5 `activation_tracker.py`: `species_to_track` parameter

**What:** Callers can override automatic species selection by passing an explicit list of
species names. This is used in paper scripts for reproducibility.

**Why it matters:** If a species name is misspelled or absent, the current code may crash
with an unhelpful IndexError — or worse, silently select wrong species.

**Tests to add (in `test_activation_tracker.py`):**
- Pass `species_to_track=["X", "Y"]` for a Brusselator analysis. Verify only those species
  contribute to D2.
- Pass `species_to_track=["nonexistent"]`. Verify a clear error or skip, not an IndexError.
- Pass `species_to_track=["X"]` (single species). Verify the result is either a valid 1D
  D2 or a graceful skip.

**Estimated effort:** 3 tests, small.

---

## Phase 2 — Edge Cases Likely to Appear in Paper 5

These cover scenarios that have not yet arisen in Papers 1-4 but become plausible as
network complexity grows.

### 2.1 `stoichiometry.py`: Larger networks

**What:** All existing stoichiometry tests use systems with 2-6 species and 1-4 reactions.
Paper 5 may involve networks with 10+ species.

**Tests to add (in `test_stoichiometry.py`):**
- 10-species, 12-reaction network with known rank (construct from block-diagonal
  stoichiometric matrix so rank is analytically known). Verify rank and conservation
  law count.
- Network where SVD tolerance matters: construct a matrix with a singular value just
  above and just below the default tolerance. Verify rank changes appropriately.

**Estimated effort:** 2 tests, small.

### 2.2 `simulator.py`: Stiff system accuracy

**What:** Current tests only cover non-stiff or mildly stiff systems. Widely separated
rate constants (common in larger networks) create stiffness.

**Tests to add (in `test_simulator.py`):**
- Two-timescale system: fast equilibrium A <-> B (k=1000, k_rev=1000) coupled to slow
  C production B -> C (k=0.01). Run with default settings. Verify mass conservation
  holds to rtol=1e-4 and C grows at the expected slow rate.
- Same system with deliberately loose tolerances (rtol=1e-2). Verify that mass
  conservation degrades (documents the sensitivity).

**Estimated effort:** 2 tests, small.

### 2.3 `simulator.py`: `remove_transient` boundary values

**What:** Only tested at `remove_transient=0.5`. Extreme values could produce empty or
near-empty trajectories.

**Tests to add (in `test_simulator.py`):**
- `remove_transient=0.0`: verify full trajectory is returned (no points removed).
- `remove_transient=0.99` with 100 points: verify at least 1 point remains and no crash.
- `remove_transient=1.0`: verify behavior (should this be an error? currently untested).

**Estimated effort:** 3 tests, small.

### 2.4 `oscillation_filter.py`: NaN and negative concentrations

**What:** Stiff or divergent simulations can produce NaN or negative concentrations. The
filter should fail gracefully.

**Tests to add (in `test_oscillation_filter.py`):**
- Trajectory with NaN values scattered in one species. Verify `passes=False`, no crash.
- Trajectory with negative concentrations (e.g., from solver overshoot). Verify
  `passes=False`, no crash.

**Estimated effort:** 2 tests, small.

### 2.5 `network_generator.py`: Species name collisions

**What:** Extra species are auto-named Z0, Z1, Z2, etc. If a template already uses those
names, reactions would be silently corrupted — the "extra" species and the template species
would be conflated.

**Tests to add (in `test_network_generator.py`):**
- Create a custom template that uses species named "Z0" and "Z1". Attempt to generate a
  network with `n_extra_species=4`. Verify either an error is raised or the names are
  de-conflicted.
- Verify that default templates (Brusselator, Oregonator) do NOT use any Z-prefixed names
  (regression guard).

**Estimated effort:** 2 tests, small.

### 2.6 `network_generator.py`: Duplicate reaction detection

**What:** The generator tries to avoid adding duplicate reactions, but uses string-level
comparison. Reactions with reordered reactants (e.g., `A + B -> C` vs `B + A -> C`) may
not be detected as duplicates.

**Tests to add (in `test_network_generator.py`):**
- Generate a large batch (50+ networks) with `n_added=5`. Verify no network contains
  two reactions with identical net stoichiometry.
- Construct two reaction strings that differ only in reactant order. Verify the generator's
  internal duplicate check catches them (or document that it doesn't).

**Estimated effort:** 2 tests, small.

### 2.7 `activation_tracker.py`: Ensemble with n_runs=1

**What:** `analyze_network_ensemble()` is designed for multiple runs, but `n_runs=1` is a
natural edge case (e.g., quick sanity check).

**Tests to add (in `test_activation_tracker.py`):**
- Call `analyze_network_ensemble()` with `n_runs=1` on the Brusselator. Verify it returns
  a valid result without crashing on variance/median calculations.

**Estimated effort:** 1 test, small.

---

## Phase 3 — Robustness, Error Handling, and Statistical Code

Lower risk to scientific conclusions but important for long-term maintainability.

### 3.1 `experiments.py`: Smoke tests for all experiment functions

**What:** `experiments.py` is the largest module (~1200 lines) with **zero** unit tests. It
contains four experiment runners and complex statistical analysis code (Mann-Whitney U,
KS test, Fisher exact test, bootstrap CIs, permutation tests).

**Why it matters:** These functions produce the tables and p-values reported in the papers.
A regression in the statistical code would directly affect published claims.

**Tests to add (new file `test_experiments.py`):**
- `run_experiment_1()` with `n_networks=2, n_added_control=1, n_autocatalytic_test=1`.
  Verify it returns an `ExperimentResult` with correct field types and `n_total` counts.
- `run_experiment_2()` with `n_trajectories=2, n_steps=2`. Verify it returns a
  `ProgressiveResult` with the expected number of steps.
- `run_experiment_3()` with `dilution_rates=[0.1], n_networks=2`. Verify it returns a
  `DrivingResult`.
- `run_experiment_paper2()` with `n_trajectories=2, n_steps=2, max_candidates=10`.
  Verify it returns a `Paper2Result` with populated statistical fields.
- `_compute_paper2_statistics()` with known input arrays. Verify p-values are in [0, 1]
  and effect sizes have correct signs.
- `_compute_paper2_statistics()` with identical groups. Verify p-values are ~1.0 (no
  significant difference).
- `_compute_paper2_statistics()` with empty arrays. Verify graceful handling (NaN or
  error, not crash).
- `Paper2Result.to_dict()` round-trip: verify JSON-serializable and all keys present.

**Estimated effort:** 8 tests, medium. These will be slow (~30-60s each for experiment
runners) so mark them with `@pytest.mark.slow`.

### 3.2 `activation_tracker.py`: Checkpoint error handling

**What:** `load_results_json()` has no test for malformed or truncated JSON files. Long
batch runs that crash mid-checkpoint could leave corrupted files.

**Tests to add (in `test_activation_tracker.py`):**
- Write a truncated JSON file (valid JSON prefix, cut mid-object). Verify
  `load_results_json()` raises a clear error, not a cryptic decoder exception.
- Write a JSON file with missing required fields. Verify behavior (error or partial load).
- Write an empty file. Verify behavior.

**Estimated effort:** 3 tests, small.

### 3.3 `activation_tracker.py`: Empty batch

**Tests to add (in `test_activation_tracker.py`):**
- Call `analyze_batch([])`. Verify it returns a `BatchResult` with `n_total=0` and no
  crash.

**Estimated effort:** 1 test, small.

### 3.4 `correlation_dimension.py`: Diagnostic plot with FAILED result

**Tests to add (in `test_correlation_dimension.py`):**
- Compute D2 on a constant trajectory (quality=FAILED). Call `plot_diagnostics()` on the
  result. Verify no crash (some arrays may be empty/None).

**Estimated effort:** 1 test, small.

### 3.5 `simulator.py`: Negative rate constant validation

**Tests to add (in `test_simulator.py`):**
- Pass a negative rate constant. Verify either a `ValueError` is raised or the behavior
  is documented. (If no validation exists, this test documents the gap and can be updated
  once validation is added.)

**Estimated effort:** 1 test, small.

---

## Summary Table

| Phase | Module | New Tests | Risk Addressed | Effort |
|-------|--------|-----------|----------------|--------|
| 1.1 | correlation_dimension | 3 | Inflated D2 / eta | Small |
| 1.2 | correlation_dimension | 3 | Pipeline seam bug | Small |
| 1.3 | activation_tracker | 4 | Wrong species in D2 | Medium |
| 1.4 | simulator | 2 | Garbage trajectories | Small |
| 1.5 | activation_tracker | 3 | Silent wrong species | Small |
| 2.1 | stoichiometry | 2 | Rank errors at scale | Small |
| 2.2 | simulator | 2 | Stiff system errors | Small |
| 2.3 | simulator | 3 | Empty trajectory | Small |
| 2.4 | oscillation_filter | 2 | NaN crash | Small |
| 2.5 | network_generator | 2 | Name collision | Small |
| 2.6 | network_generator | 2 | Doubled rates | Small |
| 2.7 | activation_tracker | 1 | Ensemble edge case | Small |
| 3.1 | experiments | 8 | Statistical regressions | Medium |
| 3.2 | activation_tracker | 3 | Checkpoint corruption | Small |
| 3.3 | activation_tracker | 1 | Empty batch crash | Small |
| 3.4 | correlation_dimension | 1 | Plot crash | Small |
| 3.5 | simulator | 1 | Negative rates | Small |
| **Total** | | **43** | | |

---

## Execution Notes

- **Phase 1** should be completed before starting Paper 5 analysis. These tests protect
  against the most dangerous failure mode: silent wrong answers in eta.
- **Phase 2** can be done incrementally as Paper 5 work reveals which edge cases actually
  arise (larger networks, stiffer systems, etc.).
- **Phase 3** is maintenance work that can be scheduled independently. The `experiments.py`
  smoke tests (3.1) are the most valuable item here — they are the only tests that would
  catch regressions in published statistical claims.
- All new tests should go in `shared/tests/` alongside the existing test files, with
  `test_experiments.py` as the only new file.
- Slow integration tests (Phase 3.1) should be marked `@pytest.mark.slow` so they can
  be excluded from fast CI runs.
