"""
Validation module for dimensional activation pipeline.

Step 7 Part 1: Validate D2 estimates against known systems.

Benchmarks:
- Lorenz attractor: D2 ≈ 2.05
- Rössler attractor: D2 ≈ 1.99  
- Hénon map: D2 ≈ 1.22
- Geometric objects: line (1), disk (2), cube (3)
- Brusselator limit cycle: D2 ≈ 1
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .correlation_dimension import CorrelationDimension, CorrelationDimensionResult, QualityFlag


@dataclass
class ValidationResult:
    """Result from a single validation benchmark."""
    name: str
    expected_D2: float
    measured_D2: float
    measured_uncertainty: float
    quality: QualityFlag
    error: float
    relative_error: float
    passed: bool
    
    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.name}: D2 = {self.measured_D2:.3f} +/- {self.measured_uncertainty:.3f} "
            f"(expected {self.expected_D2:.2f}, error {self.error:+.3f}) [{status}]"
        )


@dataclass
class ValidationReport:
    """Complete validation report."""
    results: List[ValidationResult]
    n_passed: int
    n_failed: int
    n_total: int
    
    def __repr__(self) -> str:
        lines = ["=" * 60, "VALIDATION REPORT", "=" * 60]
        for r in self.results:
            lines.append(str(r))
        lines.append("-" * 60)
        lines.append(f"Passed: {self.n_passed}/{self.n_total}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            'n_passed': self.n_passed,
            'n_failed': self.n_failed,
            'n_total': self.n_total,
            'results': [
                {
                    'name': r.name,
                    'expected_D2': r.expected_D2,
                    'measured_D2': r.measured_D2,
                    'error': r.error,
                    'passed': r.passed,
                }
                for r in self.results
            ]
        }


def generate_lorenz(n_points: int = 10000, dt: float = 0.01, 
                    transient: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate Lorenz attractor. Expected D2 ≈ 2.05"""
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    rng = np.random.default_rng(seed)
    x, y, z = rng.uniform(-1, 1, 3) + np.array([1.0, 1.0, 1.0])
    
    trajectory = []
    for i in range(n_points + transient):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        if i >= transient:
            trajectory.append([x, y, z])
    return np.array(trajectory)


def generate_rossler(n_points: int = 10000, dt: float = 0.05,
                     transient: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate Rössler attractor. Expected D2 ≈ 1.99"""
    a, b, c = 0.2, 0.2, 5.7
    rng = np.random.default_rng(seed)
    x, y, z = rng.uniform(-1, 1, 3) + np.array([1.0, 1.0, 1.0])
    
    trajectory = []
    for i in range(n_points + transient):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        if i >= transient:
            trajectory.append([x, y, z])
    return np.array(trajectory)


def generate_henon(n_points: int = 10000, transient: int = 1000,
                   seed: int = 42) -> np.ndarray:
    """Generate Hénon map. Expected D2 ≈ 1.22"""
    a, b = 1.4, 0.3
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-0.1, 0.1, 2)
    
    trajectory = []
    for i in range(n_points + transient):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        if i >= transient:
            trajectory.append([x, y])
    return np.array(trajectory)


def generate_uniform_cube(n_points: int = 5000, dim: int = 3,
                          seed: int = 42) -> np.ndarray:
    """Generate IID uniform in unit cube. Expected D2 = dim"""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n_points, dim))


def generate_uniform_line(n_points: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate IID uniform on line. Expected D2 = 1"""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n_points)
    return np.column_stack([t, t, t])


def generate_uniform_disk(n_points: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate IID uniform on disk. Expected D2 = 2"""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n_points))
    theta = rng.uniform(0, 2*np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(n_points)
    return np.column_stack([x, y, z])


def generate_brusselator(n_points: int = 5000, transient: int = 2500,
                         seed: int = 42) -> np.ndarray:
    """Generate Brusselator limit cycle. Expected D2 ≈ 1"""
    from scipy.integrate import odeint
    
    A, B = 1.0, 3.0
    def brusselator(y, t):
        X, Y = y
        dX = A - (B + 1) * X + X**2 * Y
        dY = B * X - X**2 * Y
        return [dX, dY]
    
    rng = np.random.default_rng(seed)
    y0 = [1.0 + 0.1 * rng.standard_normal(), 
          1.0 + 0.1 * rng.standard_normal()]
    t = np.linspace(0, 200, n_points + transient)
    sol = odeint(brusselator, y0, t)
    return sol[transient:]


def run_validation(
    tolerance: float = 0.15,
    verbose: bool = True,
    random_state: int = 42,
) -> ValidationReport:
    """Run full validation suite."""
    cd = CorrelationDimension()
    results = []
    
    benchmarks = [
        ("Lorenz attractor", generate_lorenz, 2.05, 
         {'n_points': 10000, 'seed': random_state}),
        ("Rossler attractor", generate_rossler, 1.99,
         {'n_points': 10000, 'seed': random_state}),
        ("Henon map", generate_henon, 1.22,
         {'n_points': 10000, 'seed': random_state}),
        ("Uniform cube (3D)", generate_uniform_cube, 3.0,
         {'n_points': 5000, 'dim': 3, 'seed': random_state}),
        ("Uniform line", generate_uniform_line, 1.0,
         {'n_points': 5000, 'seed': random_state}),
        ("Uniform disk", generate_uniform_disk, 2.0,
         {'n_points': 5000, 'seed': random_state}),
        ("Brusselator limit cycle", generate_brusselator, 1.0,
         {'n_points': 5000, 'seed': random_state}),
    ]
    
    for name, generator, expected, kwargs in benchmarks:
        if verbose:
            print(f"Validating {name}...", end=" ")
        
        try:
            trajectory = generator(**kwargs)
            
            if "Uniform" in name:
                d2_result = cd.compute(trajectory, theiler_window=1, 
                                       random_state=random_state)
            else:
                d2_result = cd.compute(trajectory, random_state=random_state)
            
            measured = d2_result.D2
            uncertainty = d2_result.D2_uncertainty
            
            if np.isnan(measured):
                error, rel_error, passed = np.nan, np.nan, False
            else:
                error = measured - expected
                rel_error = abs(error) / expected
                passed = rel_error <= tolerance
            
            result = ValidationResult(
                name=name, expected_D2=expected, measured_D2=measured,
                measured_uncertainty=uncertainty, quality=d2_result.quality,
                error=error, relative_error=rel_error, passed=passed,
            )
        except Exception as e:
            result = ValidationResult(
                name=name, expected_D2=expected, measured_D2=np.nan,
                measured_uncertainty=np.nan, quality=QualityFlag.FAILED,
                error=np.nan, relative_error=np.nan, passed=False,
            )
            if verbose:
                print(f"ERROR: {e}")
        
        results.append(result)
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            if np.isnan(result.measured_D2):
                print(f"{status} FAILED")
            else:
                print(f"{status} D2={result.measured_D2:.3f} (expected {expected:.2f})")
    
    n_passed = sum(1 for r in results if r.passed)
    return ValidationReport(results=results, n_passed=n_passed, 
                           n_failed=len(results)-n_passed, n_total=len(results))


def print_validation_table(report: ValidationReport) -> str:
    """Format validation report as markdown table."""
    lines = [
        "| System | Expected D2 | Measured D2 | Error | Status |",
        "|--------|-------------|-------------|-------|--------|",
    ]
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        if np.isnan(r.measured_D2):
            measured, error = "N/A", "N/A"
        else:
            measured = f"{r.measured_D2:.3f} +/- {r.measured_uncertainty:.3f}"
            error = f"{r.error:+.3f}"
        lines.append(f"| {r.name} | {r.expected_D2:.2f} | {measured} | {error} | {status} |")
    lines.append(f"\n**Passed: {report.n_passed}/{report.n_total}**")
    return "\n".join(lines)
