# Handoff: Additional Testing for `dimensional_opening` Shared Code

## Current Status

**Phase 1 — Complete.** 18 new tests added, all passing. No regressions (190 passed, 4 skipped).

**Phase 2 — In Progress.** Edge cases likely to appear in Paper 5.

## What Was Done (Phase 1)

Tests guarding against silent wrong answers in eta. Added to existing test files in `shared/tests/`:

- `test_correlation_dimension.py`: `TestD2EmbeddingDimensionBound` (4 tests), `TestComputeFromSimulation` (3 tests)
- `test_activation_tracker.py`: `TestTrajectoryExtraction` (4 tests), `TestSimulationFailureSkip` (1 test), `TestSpeciesToTrack` (3 tests)
- `test_simulator.py`: `TestSolverFailureHandling` (3 tests)

## What Is Being Done (Phase 2)

Edge cases for Paper 5 scenarios, per `ADDITIONAL_TESTING_PLAN.md` sections 2.1–2.7:

- 2.1: Larger networks in stoichiometry (10+ species)
- 2.2: Stiff system accuracy in simulator
- 2.3: `remove_transient` boundary values
- 2.4: NaN/negative concentrations in oscillation filter
- 2.5: Species name collisions in network generator
- 2.6: Duplicate reaction detection in network generator
- 2.7: Ensemble with n_runs=1 in activation tracker

## What Remains (Phase 3)

Per the plan: `experiments.py` smoke tests, checkpoint error handling, empty batch, diagnostic plot with FAILED result, negative rate constant validation.

## How to Run Tests

```bash
cd /Users/chris/Documents/systems_analysis/astrobiology_research/shared
pytest tests/ -v
```

## Key Files

- Plan: `/Users/chris/Documents/systems_analysis/astrobiology_research/ADDITIONAL_TESTING_PLAN.md`
- Shared code: `shared/dimensional_opening/`
- Tests: `shared/tests/`
