# Astrobiology Research: Dynamical Activation in Autocatalytic Networks

A trilogy of papers investigating how random chemical networks can develop and maintain dynamical complexity — a prerequisite for the emergence of life-like behavior.

## The Trilogy

### Paper 1: Dynamical Activation in Autocatalytic Chemical Networks
**File:** `paper1/paper/dimensional_activation_paper.tex`

Establishes the activation ratio η = D₂/rₛ (correlation dimension normalized by stoichiometric rank) as a measure of dynamical complexity. Shows that random autocatalytic additions to a Brusselator oscillator progressively dilute η toward zero — each new reaction adds stoichiometric degrees of freedom faster than the dynamics can exploit them.

### Paper 2: Feedback-Aligned Autocatalysis Preserves Dynamical Viability but Not Activation
**File:** `paper2/paper/paper2_feedback_aligned_autocatalysis.tex`

Tests whether oscillation-aligned chemical elaboration (adding only reactions that preserve oscillatory behavior) can maintain η. Finds that alignment preserves oscillation survival (OR ≈ 7 vs random) but does not prevent η dilution. D₂ remains pinned near 1.0 (the limit cycle floor) regardless of network growth strategy.

### Paper 3: Transient Dimensional Inflation via Timescale Separation in Energy-Coupled Chemical Oscillators
**File:** `paper3/paper/paper3_transient_ecology.tex`

Identifies the mechanism that breaks the D₂ ≈ 1 floor: timescale separation via a slow energy variable. An enzyme-complex mass-action model with energy coupling produces transient chaos (D₂ up to 1.7, λ₁ > 0) lasting 500–750 oscillator periods before phase-locking. Two distinct inflation mechanisms are identified: intrinsic slow-fast dynamics and shared-reservoir desynchronization. Growth experiments confirm catastrophic fragility (survival OR = 0.006 at k = 5 random additions).

## Repository Structure

```
astrobiology_research/
├── paper1/
│   ├── scripts/          # Simulation and figure generation
│   ├── data/             # Experiment results (JSON)
│   ├── figures/          # Publication figures (PNG)
│   └── paper/            # LaTeX manuscript
├── paper2/
│   ├── scripts/          # Simulation, statistics, figures
│   ├── data/             # Experiment results (JSON)
│   ├── figures/          # Publication figures (PDF + PNG)
│   └── paper/            # LaTeX manuscript
├── paper3/
│   ├── scripts/          # Pilots, sweeps, controls, growth, statistics
│   ├── data/             # Experiment results (JSON)
│   ├── paper/            # LaTeX manuscript
│   └── plan/             # Research plan document
└── shared/
    ├── dimensional_opening/  # Core simulation library
    │   ├── stoichiometry.py        # Stoichiometric matrix analysis
    │   ├── network_generator.py    # Random autocatalytic network generation
    │   ├── simulator.py            # ODE integration (ReactionSimulator)
    │   ├── correlation_dimension.py # D₂ estimation (Grassberger-Procaccia)
    │   ├── oscillation_filter.py   # Oscillation detection and filtering
    │   ├── activation_tracker.py   # η = D₂/rₛ tracking
    │   ├── experiments.py          # High-level experiment runners
    │   ├── visualization.py        # Plotting utilities
    │   └── validation.py           # Numerical validation checks
    └── tests/                # Unit tests for shared library
```

## Key Concepts

| Symbol | Meaning |
|--------|---------|
| D₂ | Correlation dimension (Grassberger-Procaccia) |
| rₛ | Stoichiometric rank (rank of stoichiometric matrix) |
| η = D₂/rₛ | Activation ratio — fraction of available dimensions used |
| T_lock | Time at which D₂ drops below 1.1 permanently |
| τ_{>1.2} | Duration with D₂ > 1.2 (exploratory phase) |
| λ₁ | Largest Lyapunov exponent |
| γ | Energy dissipation rate |
| kc | Inter-oscillator coupling strength |
| J | Number of chemical species |

## Quick Start

See [INSTALL_GUIDE.md](INSTALL_GUIDE.md) for setup instructions.

**Run Paper 1 experiments:**
```bash
cd paper1/scripts && python rerun_corrected.py
```

**Run Paper 2 experiments:**
```bash
cd paper2/scripts && python run_paper2.py
```

**Run Paper 3 pilot experiments:**
```bash
cd paper3/scripts && python run_pilots.py
```

**Run shared library tests:**
```bash
cd shared && pip install -e . && pytest tests/
```

## Author

Christopher Merrill (Independent Researcher), with computational collaboration from Claude (Anthropic) and ChatGPT (OpenAI).
