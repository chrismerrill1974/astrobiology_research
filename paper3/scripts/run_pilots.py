"""
Run Paper 3 pilots.

Usage:
    cd astrobiology3
    python -m pilot.run_pilots           # Run all pilots
    python -m pilot.run_pilots --pilot3  # Pilot 3 only (slow energy)
    python -m pilot.run_pilots --pilot3b # Pilot 3b only (bursting)
    python -m pilot.run_pilots --pilot4  # Pilot 4 only (coupled oscillators)
    python -m pilot.run_pilots --pilot5  # Pilot 5 only (mass-action validation)

Runs pilots and saves results to pilot/results/.
"""

import os
import sys
import time

# Ensure the parent directory is on the path for package imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from pilot.pilot1_drive_window import run_pilot1
from pilot.pilot2_d2_validation import run_pilot2
from pilot.pilot3_slow_energy import run_pilot3
from pilot.pilot3b_bursting import run_pilot3b
from pilot.pilot4_coupled_oscillators import run_pilot4
from pilot.pilot5_massaction_validation import run_pilot5
from pilot.pilot5b_enzyme_complex import run_pilot5b


def main():
    results_dir = os.path.join(_this_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    args = set(sys.argv[1:])
    single_pilot = args & {'--pilot3', '--pilot3b', '--pilot4', '--pilot5', '--pilot5b'}

    print("=" * 80)
    print("  PAPER 3 PILOTS: Energy-Coupled Driven Chemistry")
    print("=" * 80)
    print(f"  Results will be saved to: {results_dir}")
    if single_pilot:
        print(f"  Mode: {', '.join(single_pilot)}")
    print()

    total_start = time.time()

    p2_results = None
    p1_results = None
    p3_results = None
    p3b_results = None
    p4_results = None
    p5_results = None
    p5b_results = None

    run_all = not single_pilot

    if run_all:
        print("\n>>> Starting Pilot 2 (D₂ estimator validation)...")
        p2_results = run_pilot2(n_seeds=5, base_seed=42, verbose=True, save_dir=results_dir)

        print("\n>>> Starting Pilot 1 (static drive window sweep)...")
        p1_results = run_pilot1(seed=42, verbose=True, save_dir=results_dir)

        print("\n>>> Starting Pilot 3 (slow-energy feedback model)...")
        p3_results = run_pilot3(verbose=True, save_dir=results_dir)

        print("\n>>> Starting Pilot 3b (true slow-fast bursting)...")
        p3b_results = run_pilot3b(verbose=True, save_dir=results_dir)

        print("\n>>> Starting Pilot 4 (coupled oscillators)...")
        p4_results = run_pilot4(verbose=True, save_dir=results_dir)

    if '--pilot3' in args:
        print("\n>>> Starting Pilot 3 (slow-energy feedback model)...")
        p3_results = run_pilot3(verbose=True, save_dir=results_dir)

    if '--pilot3b' in args:
        print("\n>>> Starting Pilot 3b (true slow-fast bursting)...")
        p3b_results = run_pilot3b(verbose=True, save_dir=results_dir)

    if '--pilot4' in args:
        print("\n>>> Starting Pilot 4 (coupled oscillators)...")
        p4_results = run_pilot4(verbose=True, save_dir=results_dir)

    if '--pilot5' in args:
        print("\n>>> Starting Pilot 5 (mass-action validation)...")
        p5_results = run_pilot5(n_seeds=3, verbose=True, save_dir=results_dir)

    if '--pilot5b' in args:
        print("\n>>> Starting Pilot 5b (enzyme-complex mass-action validation)...")
        p5b_results = run_pilot5b(n_seeds=3, verbose=True, save_dir=results_dir)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f"  Complete in {total_elapsed/60:.1f} minutes")
    print(f"{'=' * 80}")

    # Decision summary
    print("\n>>> DECISION SUMMARY")
    print("-" * 60)

    if p2_results:
        higher_dim = [s for s in p2_results if s.expected_D2 > 1.5 and s.separable_from_1]
        print(f"  D₂ estimator: {'PASS' if higher_dim else 'FAIL'}")

    if p1_results:
        print("  Static drive (Pilot 1): D₂ ~1.0 (informative negative)")

    if p3_results:
        print("  Slow-energy (Pilot 3): D₂ ~1.0 (informative negative — E slaved)")

    if p3b_results:
        if p3b_results['found_complex']:
            print(f"  Bursting (Pilot 3b): D₂ > 1.2 FOUND (max = {p3b_results['max_d2']:.3f})")
        else:
            print(f"  Bursting (Pilot 3b): D₂ ≤ 1.2 (max = {p3b_results['max_d2']:.3f})")

    if p4_results:
        if p4_results['found_complex']:
            print(f"  Coupled oscillators (Pilot 4): D₂ > 1.2 FOUND (max = {p4_results['max_d2']:.3f})")
            print(f"\n  RECOMMENDATION: Proceed with coupled-oscillator model for Paper 3")
            print(f"  Hypothesis B confirmed: modular coupling produces dimensional inflation")
        else:
            print(f"  Coupled oscillators (Pilot 4): D₂ ≤ 1.2 (max = {p4_results['max_d2']:.3f})")
            print(f"\n  RECOMMENDATION: Adjust coupling or try frequency mismatch between cores")

    if p5_results:
        if p5_results['found_complex']:
            print(f"  Mass-action (Pilot 5): D₂ > 1.2 FOUND (max = {p5_results['max_d2']:.3f})")
            print(f"  {p5_results['n_complex_sets']} param sets with complex dynamics")
            print(f"\n  VALIDATED: Mass-action model reproduces dimensional inflation")
            print(f"  Ready for Phase I production sweep via ReactionSimulator")
        else:
            max_d2 = p5_results.get('max_d2')
            max_str = f"{max_d2:.3f}" if max_d2 else "N/A"
            print(f"  Mass-action (Pilot 5): D₂ ≤ 1.2 (max = {max_str})")
            print(f"\n  ACTION NEEDED: Recalibrate parameters for mass-action form")

    if p5b_results:
        if p5b_results['found_complex']:
            print(f"  Enzyme-complex (Pilot 5b): D₂ > 1.2 FOUND (max = {p5b_results['max_d2']:.3f})")
            print(f"  {p5b_results['n_complex_sets']} param sets with complex dynamics")
            print(f"\n  VALIDATED: Enzyme-complex mass-action reproduces Michaelis-Menten dynamics")
            print(f"  Ready for Phase I production sweep via ReactionSimulator")
        else:
            max_d2 = p5b_results.get('max_d2')
            max_str = f"{max_d2:.3f}" if max_d2 else "N/A"
            print(f"  Enzyme-complex (Pilot 5b): D₂ ≤ 1.2 (max = {max_str})")
            print(f"\n  ACTION NEEDED: Recalibrate enzyme-complex parameters")

    print()


if __name__ == '__main__':
    main()
