"""
Visualization module for dimensional activation analysis.

Step 6 of the Dimensional Activation pipeline.

Produces publication-quality plots:
- Time series panels
- Scaling plots with regime overlay
- η vs network complexity
- D2 vs r_S scatter with η contours
- Distribution comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, List, Tuple, Union
from pathlib import Path

from .activation_tracker import ActivationResult, BatchResult
from .correlation_dimension import CorrelationDimensionResult, QualityFlag
from .simulator import SimulationResult


# Publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_time_series(
    sim_result: SimulationResult,
    species: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot concentration time series.
    
    Parameters
    ----------
    sim_result : SimulationResult
        Simulation output.
    species : list of str, optional
        Species to plot (None = all).
    ax : Axes, optional
        Matplotlib axes (creates new if None).
    title : str, optional
        Plot title.
        
    Returns
    -------
    Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    t = sim_result.time
    c = sim_result.concentrations
    names = sim_result.species_names
    
    if species is None:
        species = names
    
    for sp in species:
        if sp in names:
            idx = names.index(sp)
            ax.plot(t, c[:, idx], label=sp, linewidth=1.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xlim(t[0], t[-1])
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_scaling_regime(
    d2_result: CorrelationDimensionResult,
    ax: Optional[Axes] = None,
    show_local_slope: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot log C(r) vs log r with scaling regime highlighted.
    
    Parameters
    ----------
    d2_result : CorrelationDimensionResult
        Correlation dimension result.
    ax : Axes, optional
        Matplotlib axes.
    show_local_slope : bool
        Show local slope curve.
    title : str, optional
        Plot title.
        
    Returns
    -------
    Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    r = d2_result.r_values
    C = d2_result.C_values
    
    # Filter valid points
    valid = (C > 0) & np.isfinite(C)
    log_r = np.log10(r[valid])
    log_C = np.log10(C[valid])
    
    # Plot all data
    ax.plot(log_r, log_C, 'o', markersize=3, alpha=0.5, color='gray', label='Data')
    
    # Highlight scaling regime
    r_min, r_max = d2_result.scaling_range
    if r_min > 0 and r_max > 0:
        in_regime = (r >= r_min) & (r <= r_max) & valid
        if np.any(in_regime):
            ax.plot(
                np.log10(r[in_regime]), 
                np.log10(C[in_regime]),
                'o', markersize=5, color='C0', label='Scaling regime'
            )
            
            # Plot fit line
            D2 = d2_result.D2
            intercept = d2_result.fit_intercept
            log_r_fit = np.linspace(np.log10(r_min), np.log10(r_max), 50)
            log_C_fit = D2 * log_r_fit + intercept / np.log(10)
            ax.plot(log_r_fit, log_C_fit, 'r-', linewidth=2, 
                   label=f'Fit: D₂={D2:.2f}')
    
    # Local slope (secondary y-axis)
    if show_local_slope and len(d2_result.local_slopes) > 0:
        ax2 = ax.twinx()
        slopes = d2_result.local_slopes
        valid_slopes = np.isfinite(slopes)
        if np.any(valid_slopes):
            # Local slopes are at midpoints of r bins
            log_r_mid = (log_r[:-1] + log_r[1:]) / 2 if len(log_r) > 1 else log_r
            # Make sure lengths match
            if len(log_r_mid) == len(slopes):
                ax2.plot(
                    log_r_mid[valid_slopes],
                    slopes[valid_slopes],
                    '--', color='C1', alpha=0.7, linewidth=1,
                    label='Local slope'
                )
                ax2.set_ylabel('Local slope', color='C1')
                ax2.tick_params(axis='y', labelcolor='C1')
                ax2.set_ylim(0, 5)
    
    ax.set_xlabel('log₁₀(r)')
    ax.set_ylabel('log₁₀(C(r))')
    ax.legend(loc='lower right')
    
    if title:
        ax.set_title(title)
    elif d2_result.quality != QualityFlag.FAILED:
        ax.set_title(f'D₂ = {d2_result.D2:.3f} ± {d2_result.D2_uncertainty:.3f} ({d2_result.quality.value})')
    else:
        ax.set_title('No scaling regime found')
    
    return ax


def plot_eta_vs_complexity(
    results: List[ActivationResult],
    complexity_metric: str = 'n_reactions',
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot η vs network complexity.
    
    Parameters
    ----------
    results : list of ActivationResult
        Activation analysis results.
    complexity_metric : str
        'n_reactions', 'n_species', or 'r_S'.
    ax : Axes, optional
        Matplotlib axes.
    title : str, optional
        Plot title.
        
    Returns
    -------
    Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    # Filter successful results
    successful = [r for r in results if not r.skipped and r.quality != QualityFlag.FAILED]
    
    if len(successful) == 0:
        ax.text(0.5, 0.5, 'No successful results', ha='center', va='center', 
               transform=ax.transAxes)
        return ax
    
    # Get complexity values
    if complexity_metric == 'n_reactions':
        x = [r.n_reactions for r in successful]
        xlabel = 'Number of reactions'
    elif complexity_metric == 'n_species':
        x = [r.n_species for r in successful]
        xlabel = 'Number of species'
    elif complexity_metric == 'r_S':
        x = [r.r_S for r in successful]
        xlabel = 'Stoichiometric rank (r_S)'
    else:
        raise ValueError(f"Unknown metric: {complexity_metric}")
    
    etas = [r.eta for r in successful]
    
    # Color by quality
    colors = ['C0' if r.quality == QualityFlag.GOOD else 'C1' for r in successful]
    
    ax.scatter(x, etas, c=colors, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('η = D₂ / r_S')
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='η = 1')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='η = 0.5')
    
    # Legend for quality
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', 
               markersize=8, label='GOOD'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C1',
               markersize=8, label='MARGINAL'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_D2_vs_rS(
    results: List[ActivationResult],
    ax: Optional[Axes] = None,
    show_contours: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot D2 vs r_S scatter with η = const contours.
    
    Parameters
    ----------
    results : list of ActivationResult
        Activation analysis results.
    ax : Axes, optional
        Matplotlib axes.
    show_contours : bool
        Show η = 0.1, 0.5, 1.0 lines.
    title : str, optional
        Plot title.
        
    Returns
    -------
    Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # Filter successful
    successful = [r for r in results if not r.skipped and r.quality != QualityFlag.FAILED]
    
    if len(successful) == 0:
        ax.text(0.5, 0.5, 'No successful results', ha='center', va='center',
               transform=ax.transAxes)
        return ax
    
    r_S = np.array([r.r_S for r in successful])
    D2 = np.array([r.D2 for r in successful])
    
    # Color by quality
    colors = ['C0' if r.quality == QualityFlag.GOOD else 'C1' for r in successful]
    
    ax.scatter(r_S, D2, c=colors, alpha=0.7, edgecolors='k', linewidths=0.5, s=50)
    
    # η contours
    if show_contours:
        r_range = np.linspace(1, max(r_S) + 1, 100)
        for eta, style in [(0.1, ':'), (0.5, '--'), (1.0, '-')]:
            ax.plot(r_range, eta * r_range, style, color='gray', alpha=0.5,
                   label=f'η = {eta}')
    
    ax.set_xlabel('Stoichiometric rank (r_S)')
    ax.set_ylabel('Correlation dimension (D₂)')
    ax.set_xlim(0, max(r_S) + 1)
    ax.set_ylim(0, max(D2) + 0.5)
    ax.legend(loc='upper left')
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_eta_distribution(
    results: List[ActivationResult],
    ax: Optional[Axes] = None,
    bins: int = 15,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot distribution of η values.
    
    Parameters
    ----------
    results : list of ActivationResult
        Activation analysis results.
    ax : Axes, optional
        Matplotlib axes.
    bins : int
        Number of histogram bins.
    title : str, optional
        Plot title.
        
    Returns
    -------
    Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    # Filter successful
    successful = [r for r in results if not r.skipped and r.quality != QualityFlag.FAILED]
    
    if len(successful) == 0:
        ax.text(0.5, 0.5, 'No successful results', ha='center', va='center',
               transform=ax.transAxes)
        return ax
    
    etas = [r.eta for r in successful]
    
    ax.hist(etas, bins=bins, range=(0, 1.2), edgecolor='black', alpha=0.7)
    
    # Summary stats
    median = np.median(etas)
    iqr = np.percentile(etas, 75) - np.percentile(etas, 25)
    ax.axvline(median, color='red', linestyle='--', linewidth=2,
              label=f'Median: {median:.2f}')
    
    ax.set_xlabel('η = D₂ / r_S')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1.2)
    ax.legend()
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'η distribution (n={len(successful)}, median={median:.2f}, IQR={iqr:.2f})')
    
    return ax


def plot_comparison(
    group1: List[ActivationResult],
    group2: List[ActivationResult],
    label1: str = 'Group 1',
    label2: str = 'Group 2',
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """
    Compare η distributions between two groups.
    
    Parameters
    ----------
    group1, group2 : list of ActivationResult
        Two groups to compare.
    label1, label2 : str
        Labels for groups.
    ax : Axes, optional
        Matplotlib axes.
    title : str, optional
        Plot title.
        
    Returns
    -------
    Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    # Filter successful
    eta1 = [r.eta for r in group1 if not r.skipped and r.quality != QualityFlag.FAILED]
    eta2 = [r.eta for r in group2 if not r.skipped and r.quality != QualityFlag.FAILED]
    
    if len(eta1) == 0 or len(eta2) == 0:
        ax.text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center',
               transform=ax.transAxes)
        return ax
    
    # Box plot
    bp = ax.boxplot([eta1, eta2], labels=[label1, label2], patch_artist=True)
    bp['boxes'][0].set_facecolor('C0')
    bp['boxes'][1].set_facecolor('C1')
    
    ax.set_ylabel('η = D₂ / r_S')
    ax.set_ylim(0, 1.2)
    
    # Effect size
    median1, median2 = np.median(eta1), np.median(eta2)
    delta = median2 - median1
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Δη = {delta:+.3f} ({label2} - {label1})')
    
    # Add medians to plot
    ax.text(0.02, 0.98, f'{label1}: median={median1:.2f}, n={len(eta1)}\n'
                         f'{label2}: median={median2:.2f}, n={len(eta2)}',
           transform=ax.transAxes, va='top', fontsize=9)
    
    return ax


def plot_diagnostic_panel(
    sim_result: SimulationResult,
    d2_result: CorrelationDimensionResult,
    species: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Create diagnostic panel with time series and scaling plot.
    
    Parameters
    ----------
    sim_result : SimulationResult
        Simulation output.
    d2_result : CorrelationDimensionResult
        Correlation dimension result.
    species : list of str, optional
        Species to show in time series.
    save_path : str, optional
        Path to save figure.
    title : str, optional
        Overall title.
        
    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_time_series(sim_result, species=species, ax=axes[0], title='Dynamics')
    plot_scaling_regime(d2_result, ax=axes[1])
    
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_batch_summary(
    batch: BatchResult,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create summary figure for batch analysis.
    
    Parameters
    ----------
    batch : BatchResult
        Batch analysis results.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    results = batch.results
    
    # Top left: η vs n_reactions
    plot_eta_vs_complexity(results, complexity_metric='n_reactions', ax=axes[0, 0])
    
    # Top right: D2 vs r_S
    plot_D2_vs_rS(results, ax=axes[0, 1])
    
    # Bottom left: η distribution
    plot_eta_distribution(results, ax=axes[1, 0])
    
    # Bottom right: summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    text = (
        f"Batch Summary\n"
        f"{'='*40}\n"
        f"Total networks: {batch.n_total}\n"
        f"Analyzed: {batch.n_analyzed}\n"
        f"Successful: {batch.n_successful}\n"
        f"\n"
        f"η Statistics (successful only):\n"
        f"  Median: {batch.eta_median:.3f}\n"
        f"  IQR: {batch.eta_iqr:.3f}\n"
        f"  Mean: {batch.eta_mean:.3f}\n"
        f"  Std: {batch.eta_std:.3f}\n"
        f"\n"
        f"D₂ median: {batch.D2_median:.3f}\n"
        f"r_S range: {batch.r_S_range}\n"
    )
    ax.text(0.1, 0.9, text, transform=ax.transAxes, va='top', 
           fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig
