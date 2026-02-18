# Installation Guide

## Requirements

- **Python**: 3.9 or later
- **Operating System**: macOS, Linux, or Windows

## Python Dependencies

Install all required packages:

```bash
pip install numpy scipy matplotlib
```

### Core dependencies

| Package | Used for | Minimum version |
|---------|----------|-----------------|
| numpy | Array operations, linear algebra | 1.21+ |
| scipy | ODE integration (`solve_ivp`), statistics (`fisher_exact`, `bootstrap`) | 1.7+ |
| matplotlib | Figure generation | 3.5+ |

### Optional dependencies

| Package | Used for |
|---------|----------|
| nolds | Lyapunov exponent estimation (Paper 3 diagnostics) |
| statsmodels | Logistic regression (Paper 3 statistics) |

Install optional dependencies:
```bash
pip install nolds statsmodels
```

## Installing the Shared Library

The `dimensional_opening` package in `shared/` is used by Papers 2 and 3. Install it in development mode:

```bash
cd shared
pip install -e .
```

This makes `from dimensional_opening import ...` available system-wide.

Paper 1 scripts are self-contained and do not require the shared library.

## Compiling the Papers

All three manuscripts use RevTeX 4-2. To compile:

```bash
cd paper1/paper && pdflatex dimensional_activation_paper.tex && pdflatex dimensional_activation_paper.tex
cd paper2/paper && pdflatex paper2_feedback_aligned_autocatalysis.tex && pdflatex paper2_feedback_aligned_autocatalysis.tex
cd paper3/paper && pdflatex paper3_transient_ecology.tex && pdflatex paper3_transient_ecology.tex
```

(Two passes resolve cross-references.)

Requires a LaTeX distribution with `revtex4-2`. On macOS: `brew install --cask mactex`. On Ubuntu: `sudo apt install texlive-publishers texlive-science`.

## Datasets

### Included in the repository

All JSON result files under `paper*/data/` are included. These are the primary experimental outputs and are sufficient to reproduce all figures and statistical analyses.

| Paper | Files | Total size |
|-------|-------|------------|
| Paper 1 | 1 JSON | ~50 KB |
| Paper 2 | 5 JSON | ~2 MB |
| Paper 3 | 15 JSON | ~5 MB |

### Not included

The following are **not** included due to size:

1. **Compiled PDFs** — Generate these by running `pdflatex` on the `.tex` files (see above).

2. **Paper 3 growth experiment raw trajectories** — The full Phase II growth experiment (960 runs) produces large intermediate trajectory data. The checkpoint JSON files containing the final results *are* included (`phase2_growth_primary_checkpoint.json` and `phase2_growth_replication1_checkpoint.json`). To regenerate the raw data:
   ```bash
   cd paper3/scripts && python phase2_growth.py
   ```
   Note: This takes approximately 12-16 hours on a single core.

3. **Paper 3 replication2 data** — The third replication batch (J=7, γ=0.001 with aligned filter) was not completed due to computational intractability of the oscillation filter at strong coupling. The 960 runs from primary + replication1 provide sufficient statistical power.

## Reproducing Results

### Paper 1
```bash
cd paper1/scripts
python rerun_corrected.py          # Main experiments (~10 min)
python figure2.py                   # Generate Figure 2
python figure3.py                   # Generate Figure 3
```

### Paper 2
```bash
cd paper2/scripts
python run_paper2.py               # Main experiments (~2 hours)
python phase5b_statistics.py       # Statistical analysis
python phase6_figures.py           # Generate all figures
python selection_bias_check.py     # Supplementary analysis
python transient_decay_check.py    # Supplementary analysis
```

### Paper 3
```bash
cd paper3/scripts

# Pilot experiments (run in sequence, ~1 hour total)
python run_pilots.py

# Phase 1: Parameter sweeps and controls
python phase1_sweep_a.py           # 100-set sweep (~30 min)
python phase1_sweep_a2.py          # γ=0.001 persistence test
python phase1_sweep_a2b.py         # Moderate-γ sharing test
python phase1b_controls.py         # 4 causal controls

# Phase 1: Diagnostics
python phase1_diagnostics.py       # D₂ convergence + Lyapunov (~40 min)
python phase1_sliding_window.py    # Sliding-window D₂ analysis

# Phase 2: Growth experiment
python phase2_growth.py            # 960 growth runs (~12-16 hours)
python analyze_growth.py           # Basic growth analysis
python analyze_growth2.py          # Extended survival analysis
python phase2_statistics.py        # Full statistical analysis
```

## Running Tests

```bash
cd shared
pip install -e .
pip install pytest
pytest tests/ -v
```
