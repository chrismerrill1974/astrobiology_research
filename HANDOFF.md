# Handoff: Additional Testing for `dimensional_opening` Shared Code

## Current Status

**Phase 1 — Complete.** 18 new tests added, all passing.

**Phase 2 — Complete.** 15 new tests added, all passing. Full suite: 205 passed, 4 skipped, 0 failures.

**Phase 3 — Not started.** Robustness, error handling, and statistical code coverage.

## What Was Done

### Phase 1: Guard against silent wrong answers in eta

- `test_correlation_dimension.py`: `TestD2EmbeddingDimensionBound` (4 tests), `TestComputeFromSimulation` (3 tests)
- `test_activation_tracker.py`: `TestTrajectoryExtraction` (4 tests), `TestSimulationFailureSkip` (1 test), `TestSpeciesToTrack` (3 tests)
- `test_simulator.py`: `TestSolverFailureHandling` (3 tests)

### Phase 2: Edge cases for Paper 5

- `test_stoichiometry.py`: `TestLargerNetworks` (3 tests) — 10-species networks, SVD tolerance boundary
- `test_simulator.py`: `TestStiffSystemAccuracy` (2 tests), `TestRemoveTransientBoundary` (3 tests)
- `test_oscillation_filter.py`: `TestNaNAndNegativeConcentrations` (2 tests)
- `test_network_generator.py`: `TestSpeciesNameCollisions` (2 tests), `TestDuplicateReactionDetection` (2 tests)
- `test_activation_tracker.py`: `TestEnsembleEdgeCases` (1 test) — n_runs=1 fallback

## What Remains (Phase 3)

Per `ADDITIONAL_TESTING_PLAN.md`:

- 3.1: `experiments.py` smoke tests (8 tests) — the largest gap, zero coverage on ~1200 lines
- 3.2: Checkpoint error handling (3 tests)
- 3.3: Empty batch (1 test)
- 3.4: Diagnostic plot with FAILED result (1 test)
- 3.5: Negative rate constant validation (1 test)

## How to Run Tests

```bash
cd /Users/chris/Documents/systems_analysis/astrobiology_research/shared
pytest tests/ -v
```

## Key Files

- Plan: `ADDITIONAL_TESTING_PLAN.md`
- Shared code: `shared/dimensional_opening/`
- Tests: `shared/tests/`
