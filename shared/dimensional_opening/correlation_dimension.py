"""
Correlation dimension calculator using Grassberger-Procaccia algorithm.

Step 4 of the Dimensional Activation pipeline.

Computes D2 (correlation dimension) from multivariate time series with:
- Explicit scaling regime selection rules
- Adaptive Theiler window estimation
- Quality flags for reliability assessment
- Mandatory diagnostic plots

Critical design choices from the plan:
- Native multivariate trajectories (no Takens embedding by default)
- Hard-coded scaling selection rules (auditable)
- Uncertainty from repeated simulations, not iid bootstrap
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import warnings


class QualityFlag(Enum):
    """Quality assessment for D2 estimate."""
    GOOD = "good"           # Clear scaling regime, low uncertainty
    MARGINAL = "marginal"   # Scaling regime found but borderline
    FAILED = "failed"       # No valid scaling regime


@dataclass
class CorrelationDimensionResult:
    """Results from correlation dimension estimation."""
    
    # Core result
    D2: float                               # Correlation dimension estimate
    D2_uncertainty: float                   # Uncertainty (std from fits or NaN)
    quality: QualityFlag                    # Quality assessment
    
    # Scaling regime info
    scaling_range: Tuple[float, float]      # (r_min, r_max) of valid regime
    n_points_in_regime: int                 # Number of r values used
    r_squared: float                        # R² of linear fit in scaling regime
    fit_intercept: float                    # Intercept from log-log fit
    
    # Correlation integral data
    r_values: np.ndarray                    # Distance scales
    C_values: np.ndarray                    # C(r) values
    local_slopes: np.ndarray                # d(log C)/d(log r)
    
    # Metadata
    n_trajectory_points: int
    n_pairs: int
    theiler_window: int
    embedding_dimension: int                # m (native dimension of trajectory)
    
    # Diagnostic info
    failure_reason: Optional[str] = None
    random_state: Optional[int] = None      # RNG seed used for pair sampling
    
    def __repr__(self) -> str:
        return (
            f"CorrelationDimensionResult(\n"
            f"  D2={self.D2:.3f} ± {self.D2_uncertainty:.3f},\n"
            f"  quality={self.quality.value},\n"
            f"  scaling_range=[{self.scaling_range[0]:.2e}, {self.scaling_range[1]:.2e}],\n"
            f"  R²={self.r_squared:.4f},\n"
            f"  n_points={self.n_trajectory_points}, theiler_window={self.theiler_window}\n"
            f")"
        )


class CorrelationDimension:
    """
    Grassberger-Procaccia correlation dimension estimator.
    
    Computes D2 from the scaling of the correlation integral:
        C(r) ~ r^D2  as r -> 0
    
    Uses explicit, auditable rules for scaling regime selection
    as specified in the implementation plan.
    
    Examples
    --------
    >>> cd = CorrelationDimension()
    >>> 
    >>> # From simulation result
    >>> result = cd.compute(trajectory, theiler_window='auto')
    >>> print(f"D2 = {result.D2:.2f}, quality = {result.quality.value}")
    >>>
    >>> # Generate diagnostic plot
    >>> cd.plot_diagnostics(result, save_path="d2_diagnostic.png")
    """
    
    # Default scaling regime parameters (from plan)
    DEFAULT_C_MIN = 1e-3        # Minimum C(r) for adequate statistics
    DEFAULT_C_MAX = 0.1         # Maximum C(r) to avoid saturation
    DEFAULT_SLOPE_TOLERANCE = 0.3   # Max deviation in local slope
    DEFAULT_MIN_CONSECUTIVE = 5     # Min consecutive bins for valid regime
    DEFAULT_MAX_UNCERTAINTY = 0.5   # Max acceptable D2 uncertainty
    
    def __init__(
        self,
        n_r_bins: int = 50,
        c_min: float = DEFAULT_C_MIN,
        c_max: float = DEFAULT_C_MAX,
        slope_tolerance: float = DEFAULT_SLOPE_TOLERANCE,
        min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
        max_uncertainty: float = DEFAULT_MAX_UNCERTAINTY,
    ):
        """
        Initialize correlation dimension estimator.
        
        Parameters
        ----------
        n_r_bins : int
            Number of distance bins for C(r).
        c_min, c_max : float
            Valid range for C(r) in scaling regime.
        slope_tolerance : float
            Maximum allowed deviation in local slope (δ in plan).
        min_consecutive : int
            Minimum consecutive bins with stable slope (k in plan).
        max_uncertainty : float
            Maximum D2 uncertainty for GOOD quality.
        """
        self.n_r_bins = n_r_bins
        self.c_min = c_min
        self.c_max = c_max
        self.slope_tolerance = slope_tolerance
        self.min_consecutive = min_consecutive
        self.max_uncertainty = max_uncertainty
    
    def compute(
        self,
        trajectory: ArrayLike,
        theiler_window: Optional[int] = None,
        max_pairs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> CorrelationDimensionResult:
        """
        Compute correlation dimension from trajectory.
        
        Parameters
        ----------
        trajectory : array_like
            Trajectory data, shape (n_times, n_dimensions).
            For chemical networks: (n_times, n_species).
        theiler_window : int or None
            Minimum temporal separation between pairs.
            If None, uses adaptive estimation (first zero-crossing of ACF).
        max_pairs : int or None
            Maximum number of pairs to use (for computational efficiency).
            If None, uses all valid pairs.
        random_state : int or None
            Random seed for reproducible pair sampling when max_pairs is used.
            If None, sampling is non-deterministic.
            
        Returns
        -------
        CorrelationDimensionResult
            Results including D2, quality flag, and diagnostic data.
        """
        X = np.asarray(trajectory)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        N, m = X.shape  # N time points, m dimensions
        
        # Estimate Theiler window if not provided
        if theiler_window is None:
            theiler_window = self._estimate_theiler_window(X)
        
        # Clamp Theiler window to [5, N/5] - always enforce minimum of 5
        # to avoid temporal correlations (per plan specification)
        if theiler_window < 5:
            warnings.warn(
                f"Theiler window {theiler_window} < 5; clamping to 5 to avoid "
                f"temporal correlations."
            )
        theiler_window = max(5, min(theiler_window, N // 5))
        
        # Compute pairwise distances (respecting Theiler window)
        distances, n_pairs = self._compute_distances(X, theiler_window, max_pairs, random_state)
        
        if n_pairs < 100:
            return self._failed_result(
                X, theiler_window, m,
                f"Insufficient pairs after Theiler exclusion: {n_pairs}",
                random_state=random_state,
            )
        
        # Compute correlation integral
        r_values, C_values = self._compute_correlation_integral(distances)
        
        # Compute local slopes
        local_slopes = self._compute_local_slopes(r_values, C_values)
        
        # Find scaling regime
        regime = self._find_scaling_regime(r_values, C_values, local_slopes)
        
        if regime is None:
            return CorrelationDimensionResult(
                D2=np.nan,
                D2_uncertainty=np.nan,
                quality=QualityFlag.FAILED,
                scaling_range=(np.nan, np.nan),
                n_points_in_regime=0,
                r_squared=np.nan,
                fit_intercept=np.nan,
                r_values=r_values,
                C_values=C_values,
                local_slopes=local_slopes,
                n_trajectory_points=N,
                n_pairs=n_pairs,
                theiler_window=theiler_window,
                embedding_dimension=m,
                failure_reason="No valid scaling regime found",
                random_state=random_state,
            )
        
        # Fit D2 in scaling regime
        i_start, i_end = regime
        D2, D2_unc, r_squared, fit_intercept = self._fit_dimension(
            r_values[i_start:i_end+1], 
            C_values[i_start:i_end+1]
        )
        
        # Assess quality
        quality = self._assess_quality(D2_unc, r_squared, i_end - i_start + 1)
        
        return CorrelationDimensionResult(
            D2=D2,
            D2_uncertainty=D2_unc,
            quality=quality,
            scaling_range=(r_values[i_start], r_values[i_end]),
            n_points_in_regime=i_end - i_start + 1,
            r_squared=r_squared,
            fit_intercept=fit_intercept,
            r_values=r_values,
            C_values=C_values,
            local_slopes=local_slopes,
            n_trajectory_points=N,
            n_pairs=n_pairs,
            theiler_window=theiler_window,
            embedding_dimension=m,
            random_state=random_state,
        )
    
    def compute_from_simulation(
        self,
        sim_result,
        remove_transient: float = 0.5,
        theiler_window: Optional[int] = None,
    ) -> CorrelationDimensionResult:
        """
        Compute D2 directly from a SimulationResult.
        
        Parameters
        ----------
        sim_result : SimulationResult
            Output from ReactionSimulator.
        remove_transient : float
            Fraction of trajectory to discard (0 to 1).
        theiler_window : int or None
            Theiler window (None for auto).
            
        Returns
        -------
        CorrelationDimensionResult
        """
        c = sim_result.concentrations
        
        # Remove transient
        if remove_transient > 0:
            n_remove = int(len(c) * remove_transient)
            c = c[n_remove:]
        
        return self.compute(c, theiler_window=theiler_window)
    
    def _estimate_theiler_window(self, X: np.ndarray) -> int:
        """
        Estimate Theiler window from first zero-crossing of autocorrelation.
        
        Uses the first principal component or most variable species.
        Returns w = ceil(tau_c / dt), clamped to [5, N/5].
        """
        N = len(X)
        
        # Use most variable dimension
        variances = np.var(X, axis=0)
        best_dim = np.argmax(variances)
        x = X[:, best_dim]
        
        # Compute autocorrelation
        x_centered = x - np.mean(x)
        if np.std(x_centered) < 1e-10:
            return 5  # Default for constant signal
        
        acf = np.correlate(x_centered, x_centered, mode='full')
        acf = acf[N-1:]  # Keep positive lags only
        acf = acf / acf[0]  # Normalize
        
        # Find first zero crossing
        zero_crossings = np.where(np.diff(np.sign(acf)))[0]
        if len(zero_crossings) > 0:
            tau_c = zero_crossings[0]
        else:
            # No zero crossing: use 1/e decay
            below_threshold = np.where(acf < 1/np.e)[0]
            tau_c = below_threshold[0] if len(below_threshold) > 0 else N // 10
        
        # Clamp to [5, N/5]
        w = max(5, min(int(tau_c), N // 5))
        return w
    
    def _compute_distances(
        self,
        X: np.ndarray,
        theiler_window: int,
        max_pairs: Optional[int],
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute pairwise distances respecting Theiler window.
        
        If max_pairs is specified, samples pairs BEFORE computing distances
        to avoid O(N²) memory/compute for large trajectories.
        
        Parameters
        ----------
        X : np.ndarray
            Trajectory data, shape (N, m).
        theiler_window : int
            Minimum temporal separation between pairs.
        max_pairs : int or None
            Maximum pairs to compute.
        random_state : int or None
            Seed for reproducible pair sampling.
        
        Returns
        -------
        distances : np.ndarray
            Flattened array of valid distances.
        n_pairs : int
            Number of pairs.
        """
        N = len(X)
        
        # Total number of valid pairs (respecting Theiler window)
        # Valid pairs: j - i >= theiler_window, where i < j
        # For each i in [0, N-1-w], valid j in [i+w, N-1]
        # Count = sum_{i=0}^{N-1-w} (N-1 - (i+w) + 1) = sum_{i=0}^{N-1-w} (N - i - w)
        #       = (N-w) * (N-w+1) / 2 - (N-w)*(N-w-1)/2 ... simplified:
        n_valid_total = (N - theiler_window) * (N - theiler_window + 1) // 2
        
        if n_valid_total == 0:
            return np.array([]), 0
        
        # Determine how many pairs to compute
        if max_pairs is not None and max_pairs < n_valid_total:
            n_pairs_to_compute = max_pairs
            use_sampling = True
        else:
            n_pairs_to_compute = n_valid_total
            use_sampling = False
        
        # Create RNG with explicit seed for reproducibility
        rng = np.random.default_rng(random_state)
        
        if use_sampling:
            # Sample pairs efficiently without generating all indices
            # Use vectorized sampling with rejection for invalid pairs
            pairs_set = set()
            
            while len(pairs_set) < n_pairs_to_compute:
                # Sample i uniformly from valid range
                batch_size = min((n_pairs_to_compute - len(pairs_set)) * 3, 10000)
                i_samples = rng.integers(0, N - theiler_window, size=batch_size)
                
                # For each i, sample j uniformly from [i + theiler_window, N)
                # We sample offset in [theiler_window, N - i) for each i
                # Use rejection: sample offset up to max possible, reject invalid
                max_possible_offset = N - 1  # Maximum offset if i=0
                offsets = rng.integers(theiler_window, max_possible_offset + 1, size=batch_size)
                
                for i, offset in zip(i_samples, offsets):
                    j = i + offset
                    if j < N and (i, j) not in pairs_set:
                        pairs_set.add((i, j))
                        if len(pairs_set) >= n_pairs_to_compute:
                            break
            
            pairs = np.array(list(pairs_set))
            distances = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
            
        else:
            # For smaller N or when computing all pairs, use vectorized approach
            # but avoid the O(N²) python loop
            if N <= 5000:
                # For moderate N, pdist is fast enough
                from scipy.spatial.distance import pdist
                all_distances = pdist(X, metric='euclidean')
                
                # Create mask using vectorized operations
                # pdist index k corresponds to (i,j) where i<j
                # k = i*N - i*(i+1)/2 + j - i - 1
                # We need |j - i| >= theiler_window
                
                # Generate all (i,j) pairs efficiently
                idx = np.triu_indices(N, k=1)
                time_sep = idx[1] - idx[0]
                valid_mask = time_sep >= theiler_window
                
                distances = all_distances[valid_mask]
            else:
                # For large N, generate valid pairs directly and compute distances
                # This avoids creating the full N² distance matrix
                pairs_list = []
                for i in range(N - theiler_window):
                    j_start = i + theiler_window
                    j_indices = np.arange(j_start, N)
                    i_indices = np.full(len(j_indices), i)
                    pairs_list.append(np.column_stack([i_indices, j_indices]))
                
                pairs = np.vstack(pairs_list)
                distances = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
        
        return distances, len(distances)
    
    def _compute_correlation_integral(
        self,
        distances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute correlation integral C(r) = (# pairs with d < r) / (total pairs).
        """
        # Use logarithmically spaced r values
        r_min = np.percentile(distances, 1)  # Avoid r=0
        r_max = np.percentile(distances, 99)
        
        # Handle edge case of constant distances
        if r_min <= 0 or not np.isfinite(r_min):
            positive_distances = distances[distances > 0]
            r_min = positive_distances.min() if len(positive_distances) > 0 else 1e-10
        if r_max <= r_min or not np.isfinite(r_max):
            r_max = r_min * 10 if r_min > 0 else 1.0
        
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), self.n_r_bins)
        
        # Compute C(r) for each r
        n_total = len(distances)
        C_values = np.array([np.sum(distances < r) / n_total for r in r_values])
        
        return r_values, C_values
    
    def _compute_local_slopes(
        self,
        r_values: np.ndarray,
        C_values: np.ndarray,
    ) -> np.ndarray:
        """
        Compute local slopes d(log C)/d(log r).
        """
        # Avoid log(0)
        valid = (C_values > 0) & (r_values > 0)
        log_r = np.log(r_values)
        log_C = np.full_like(C_values, np.nan)
        log_C[valid] = np.log(C_values[valid])
        
        # Central differences
        slopes = np.full_like(log_C, np.nan)
        for i in range(1, len(log_r) - 1):
            if valid[i-1] and valid[i+1]:
                slopes[i] = (log_C[i+1] - log_C[i-1]) / (log_r[i+1] - log_r[i-1])
        
        return slopes
    
    def _find_scaling_regime(
        self,
        r_values: np.ndarray,
        C_values: np.ndarray,
        local_slopes: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """
        Find scaling regime using explicit rules from the plan.
        
        Rules:
        1. C(r) ∈ [c_min, c_max] (adequate statistics, avoid saturation)
        2. Local slope stable within δ over k consecutive bins
        3. Return longest such regime
        
        Returns (i_start, i_end) indices or None if no valid regime.
        """
        n = len(r_values)
        
        # Find indices where C is in valid range
        valid_C = (C_values >= self.c_min) & (C_values <= self.c_max)
        
        # Find runs of consecutive valid indices with stable slope
        best_run = None
        best_length = 0
        
        i = 0
        while i < n:
            if not valid_C[i] or np.isnan(local_slopes[i]):
                i += 1
                continue
            
            # Start a potential run
            run_start = i
            run_slopes = [local_slopes[i]]
            
            j = i + 1
            while j < n and valid_C[j] and not np.isnan(local_slopes[j]):
                # Check if slope is stable
                new_slope = local_slopes[j]
                median_slope = np.median(run_slopes)
                
                if abs(new_slope - median_slope) <= self.slope_tolerance:
                    run_slopes.append(new_slope)
                    j += 1
                else:
                    break
            
            run_end = j - 1
            run_length = run_end - run_start + 1
            
            if run_length >= self.min_consecutive and run_length > best_length:
                best_run = (run_start, run_end)
                best_length = run_length
            
            i = j
        
        return best_run
    
    def _fit_dimension(
        self,
        r_values: np.ndarray,
        C_values: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Fit D2 via linear regression on log-log plot.
        
        Returns (D2, uncertainty, R², intercept).
        """
        valid = (r_values > 0) & (C_values > 0)
        log_r = np.log(r_values[valid])
        log_C = np.log(C_values[valid])
        
        if len(log_r) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        # Linear regression: log_C = D2 * log_r + intercept
        coeffs = np.polyfit(log_r, log_C, 1)
        D2 = coeffs[0]
        intercept = coeffs[1]
        
        # R² calculation
        log_C_fit = np.polyval(coeffs, log_r)
        ss_res = np.sum((log_C - log_C_fit) ** 2)
        ss_tot = np.sum((log_C - np.mean(log_C)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Uncertainty from residuals
        if len(log_r) > 2:
            residuals = log_C - log_C_fit
            std_err = np.sqrt(ss_res / (len(log_r) - 2))
            # Standard error of slope
            x_var = np.sum((log_r - np.mean(log_r)) ** 2)
            D2_unc = std_err / np.sqrt(x_var) if x_var > 0 else np.nan
        else:
            D2_unc = np.nan
        
        return D2, D2_unc, r_squared, intercept
    
    def _assess_quality(
        self,
        D2_unc: float,
        r_squared: float,
        n_points: int,
    ) -> QualityFlag:
        """
        Assess quality of D2 estimate.
        """
        if np.isnan(D2_unc) or np.isnan(r_squared):
            return QualityFlag.FAILED
        
        if D2_unc < self.max_uncertainty and r_squared > 0.99 and n_points >= 10:
            return QualityFlag.GOOD
        elif D2_unc < self.max_uncertainty * 2 and r_squared > 0.95 and n_points >= 5:
            return QualityFlag.MARGINAL
        else:
            return QualityFlag.FAILED
    
    def _failed_result(
        self,
        X: np.ndarray,
        theiler_window: int,
        m: int,
        reason: str,
        random_state: Optional[int] = None,
    ) -> CorrelationDimensionResult:
        """Create a failed result with reason."""
        return CorrelationDimensionResult(
            D2=np.nan,
            D2_uncertainty=np.nan,
            quality=QualityFlag.FAILED,
            scaling_range=(np.nan, np.nan),
            n_points_in_regime=0,
            r_squared=np.nan,
            fit_intercept=np.nan,
            r_values=np.array([]),
            C_values=np.array([]),
            local_slopes=np.array([]),
            n_trajectory_points=len(X),
            n_pairs=0,
            theiler_window=theiler_window,
            embedding_dimension=m,
            failure_reason=reason,
            random_state=random_state,
        )
    
    def plot_diagnostics(
        self,
        result: CorrelationDimensionResult,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Generate diagnostic plot for D2 estimation.
        
        Creates a two-panel figure:
        - Top: log C(r) vs log r with scaling regime highlighted
        - Bottom: local slope vs log r
        
        Parameters
        ----------
        result : CorrelationDimensionResult
            Result from compute().
        save_path : str, optional
            Path to save figure.
        show : bool
            Whether to display the figure.
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        r = result.r_values
        C = result.C_values
        slopes = result.local_slopes
        
        valid = (r > 0) & (C > 0)
        
        # Top panel: log C vs log r
        ax1.plot(np.log10(r[valid]), np.log10(C[valid]), 'b.-', alpha=0.7, label='C(r)')
        
        # Highlight scaling regime
        if not np.isnan(result.scaling_range[0]):
            r_min, r_max = result.scaling_range
            regime_mask = (r >= r_min) & (r <= r_max) & valid
            ax1.plot(
                np.log10(r[regime_mask]), 
                np.log10(C[regime_mask]), 
                'r-', linewidth=3, label=f'Scaling regime (D₂={result.D2:.2f})'
            )
            
            # Add fit line using actual regression coefficients
            # fit: ln(C) = D2 * ln(r) + intercept
            # convert to log10: log10(C) = D2 * log10(r) + intercept / ln(10)
            log_r_fit = np.log10(r[regime_mask])
            log_C_fit = result.D2 * log_r_fit + result.fit_intercept / np.log(10)
            ax1.plot(log_r_fit, log_C_fit, 'k--', linewidth=1, alpha=0.7, label='Fit')
        
        # Mark C_min and C_max
        ax1.axhline(np.log10(self.c_min), color='gray', linestyle=':', alpha=0.5, label=f'C_min={self.c_min}')
        ax1.axhline(np.log10(self.c_max), color='gray', linestyle=':', alpha=0.5, label=f'C_max={self.c_max}')
        
        ax1.set_ylabel('log₁₀ C(r)')
        ax1.set_title(f'Correlation Integral (Quality: {result.quality.value})')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom panel: local slopes
        valid_slopes = ~np.isnan(slopes)
        ax2.plot(np.log10(r[valid_slopes]), slopes[valid_slopes], 'b.-', alpha=0.7)
        
        # Highlight scaling regime
        if not np.isnan(result.scaling_range[0]):
            r_min, r_max = result.scaling_range
            regime_mask = (r >= r_min) & (r <= r_max) & valid_slopes
            ax2.plot(
                np.log10(r[regime_mask]), 
                slopes[regime_mask], 
                'r-', linewidth=3
            )
        
        ax2.axhline(result.D2, color='r', linestyle='--', alpha=0.7, label=f'D₂={result.D2:.2f}')
        
        ax2.set_xlabel('log₁₀ r')
        ax2.set_ylabel('Local slope')
        ax2.set_title('Local Slope d(log C)/d(log r)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


def compute_activation_ratio(
    D2: float,
    r_S: int,
) -> float:
    """
    Compute activation ratio η = D2 / r_S.
    
    Parameters
    ----------
    D2 : float
        Correlation dimension.
    r_S : int
        Stoichiometric rank.
        
    Returns
    -------
    float
        Activation ratio η. Returns NaN if r_S < 2 or D2 is NaN.
    """
    if r_S < 2 or np.isnan(D2):
        return np.nan
    return D2 / r_S


@dataclass
class EnsembleResult:
    """Results from ensemble D2 estimation.
    
    MARGINAL results are included in statistics by design, as they often
    represent finite-data limitations rather than estimator failure. Only
    FAILED results are excluded from D2 aggregation.
    """
    
    # Summary statistics
    D2_median: float                    # Median D2 across successful runs
    D2_iqr: float                       # Interquartile range
    D2_mean: float                      # Mean D2
    D2_std: float                       # Standard deviation
    
    # Quality summary
    n_total: int                        # Total number of trajectories
    n_successful: int                   # Number with GOOD or MARGINAL quality
    n_good: int                         # Number with GOOD quality
    n_marginal: int                     # Number with MARGINAL quality
    n_failed: int                       # Number with FAILED quality
    success_rate: float                 # Fraction successful
    
    # Individual results
    D2_values: List[float]              # All successful D2 values
    results: List[CorrelationDimensionResult]  # All individual results
    
    # Metadata
    theiler_window: int                 # Theiler window used
    random_state: Optional[int] = None  # RNG seed if used
    
    @property
    def qualities(self) -> List[QualityFlag]:
        """Quality flags for all trajectories (convenience accessor)."""
        return [r.quality for r in self.results]
    
    def __repr__(self) -> str:
        return (
            f"EnsembleResult(\n"
            f"  D2 = {self.D2_median:.3f} (IQR: {self.D2_iqr:.3f}),\n"
            f"  n_successful = {self.n_successful}/{self.n_total} "
            f"({self.success_rate:.0%}),\n"
            f"  quality: {self.n_good} good, {self.n_marginal} marginal, "
            f"{self.n_failed} failed\n"
            f")"
        )


def compute_D2_ensemble(
    trajectories: List[np.ndarray],
    theiler_window: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> EnsembleResult:
    """
    Compute D2 across multiple trajectories for uncertainty estimation.
    
    This is the recommended approach per the plan: uncertainty from
    repeated simulations, not iid bootstrap of points.
    
    Parameters
    ----------
    trajectories : list of array
        List of trajectory arrays, each shape (n_times, n_dims).
    theiler_window : int, optional
        Theiler window (None for auto, estimated from first trajectory).
    random_state : int, optional
        Base random seed. Each trajectory uses random_state + i for
        reproducibility.
    **kwargs
        Additional arguments to CorrelationDimension().
        
    Returns
    -------
    EnsembleResult
        Summary statistics and individual results.
    
    Examples
    --------
    >>> trajectories = [generate_trajectory(seed=i) for i in range(10)]
    >>> result = compute_D2_ensemble(trajectories, random_state=42)
    >>> print(f"D2 = {result.D2_median:.2f} ± {result.D2_iqr:.2f}")
    """
    if len(trajectories) == 0:
        return EnsembleResult(
            D2_median=np.nan, D2_iqr=np.nan, D2_mean=np.nan, D2_std=np.nan,
            n_total=0, n_successful=0, n_good=0, n_marginal=0, n_failed=0,
            success_rate=0.0, D2_values=[], results=[],
            theiler_window=0, random_state=random_state,
        )
    
    cd = CorrelationDimension(**{k: v for k, v in kwargs.items() 
                                  if k in CorrelationDimension.__init__.__code__.co_varnames})
    
    # Estimate Theiler window from first trajectory if not provided
    if theiler_window is None:
        theiler_window = cd._estimate_theiler_window(np.asarray(trajectories[0]))
    
    results = []
    D2_values = []
    n_good = 0
    n_marginal = 0
    n_failed = 0
    
    for i, traj in enumerate(trajectories):
        # Use reproducible seed for each trajectory
        seed = (random_state + i) if random_state is not None else None
        
        result = cd.compute(traj, theiler_window=theiler_window, random_state=seed)
        results.append(result)
        
        if result.quality == QualityFlag.GOOD:
            D2_values.append(result.D2)
            n_good += 1
        elif result.quality == QualityFlag.MARGINAL:
            D2_values.append(result.D2)
            n_marginal += 1
        else:
            n_failed += 1
    
    n_total = len(trajectories)
    n_successful = n_good + n_marginal
    
    if n_successful == 0:
        return EnsembleResult(
            D2_median=np.nan, D2_iqr=np.nan, D2_mean=np.nan, D2_std=np.nan,
            n_total=n_total, n_successful=0, n_good=0, n_marginal=0, n_failed=n_failed,
            success_rate=0.0, D2_values=[], results=results,
            theiler_window=theiler_window, random_state=random_state,
        )
    
    D2_arr = np.array(D2_values)
    
    return EnsembleResult(
        D2_median=np.median(D2_arr),
        D2_iqr=np.percentile(D2_arr, 75) - np.percentile(D2_arr, 25),
        D2_mean=np.mean(D2_arr),
        D2_std=np.std(D2_arr),
        n_total=n_total,
        n_successful=n_successful,
        n_good=n_good,
        n_marginal=n_marginal,
        n_failed=n_failed,
        success_rate=n_successful / n_total,
        D2_values=D2_values,
        results=results,
        theiler_window=theiler_window,
        random_state=random_state,
    )
