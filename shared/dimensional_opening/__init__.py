"""
Dimensional Opening: Correlation dimension analysis for prebiotic chemistry.

A pipeline to test whether dynamical activation increases relative to
stoichiometric degrees of freedom across transitions associated with
autocatalytic organization.
"""

from .stoichiometry import (
    StoichiometricAnalyzer,
    StoichiometricAnalysis,
    compute_rank,
    get_conservation_laws,
)

from .simulator import (
    ReactionSimulator,
    ReactionNetwork,
    SimulationResult,
    DrivingMode,
    simulate_reactions,
)

from .correlation_dimension import (
    CorrelationDimension,
    CorrelationDimensionResult,
    EnsembleResult,
    QualityFlag,
    compute_activation_ratio,
    compute_D2_ensemble,
)

from .activation_tracker import (
    ActivationTracker,
    ActivationResult,
    BatchResult,
    save_results_csv,
    load_results_json,
)

from .visualization import (
    plot_time_series,
    plot_scaling_regime,
    plot_eta_vs_complexity,
    plot_D2_vs_rS,
    plot_eta_distribution,
    plot_comparison,
    plot_diagnostic_panel,
    plot_batch_summary,
)

from .validation import (
    ValidationResult,
    ValidationReport,
    run_validation,
    print_validation_table,
    generate_lorenz,
    generate_rossler,
    generate_henon,
)

from .network_generator import (
    NetworkGenerator,
    GeneratedNetwork,
    OscillatorTemplate,
    AlignedProgressiveResult,
    BRUSSELATOR,
    OREGONATOR,
    TEMPLATES,
)

from .oscillation_filter import (
    OscillationResult,
    check_oscillation,
    passes_oscillation_filter,
)

from .experiments import (
    ExperimentResult,
    ProgressiveResult,
    DrivingResult,
    Paper2Result,
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_all_experiments,
    run_experiment_paper2,
)

__version__ = "0.1.0"
__all__ = [
    # Stoichiometry (Step 1)
    "StoichiometricAnalyzer",
    "StoichiometricAnalysis",
    "compute_rank",
    "get_conservation_laws",
    # Simulator (Step 2)
    "ReactionSimulator",
    "ReactionNetwork",
    "SimulationResult",
    "DrivingMode",
    "simulate_reactions",
    # Correlation Dimension (Step 4)
    "CorrelationDimension",
    "CorrelationDimensionResult",
    "QualityFlag",
    "compute_activation_ratio",
    # Ensemble (Step 5)
    "EnsembleResult",
    "compute_D2_ensemble",
    # Activation Tracker (Step 6)
    "ActivationTracker",
    "ActivationResult",
    "BatchResult",
    "save_results_csv",
    "load_results_json",
    # Visualization (Step 6)
    "plot_time_series",
    "plot_scaling_regime",
    "plot_eta_vs_complexity",
    "plot_D2_vs_rS",
    "plot_eta_distribution",
    "plot_comparison",
    "plot_diagnostic_panel",
    "plot_batch_summary",
    # Validation (Step 7)
    "ValidationResult",
    "ValidationReport",
    "run_validation",
    "print_validation_table",
    "generate_lorenz",
    "generate_rossler",
    "generate_henon",
    # Network Generator (Step 7)
    "NetworkGenerator",
    "GeneratedNetwork",
    "OscillatorTemplate",
    "BRUSSELATOR",
    "OREGONATOR",
    "TEMPLATES",
    "AlignedProgressiveResult",
    # Oscillation Filter (Paper 2)
    "OscillationResult",
    "check_oscillation",
    "passes_oscillation_filter",
    # Experiments (Step 7)
    "ExperimentResult",
    "ProgressiveResult",
    "DrivingResult",
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",
    "run_all_experiments",
    "Paper2Result",
    "run_experiment_paper2",
]
