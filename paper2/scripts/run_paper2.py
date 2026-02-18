"""
Paper 2 Experiment Runner — Standalone Script

Usage:
    python scripts/run_paper2.py --n_trajectories 200 --seed 42 --output data/paper2_results.json

Runs Group A (random) and Group B (feedback-aligned) progressive autocatalysis
experiments and saves full results to JSON.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import os
import sys

import numpy as np

from dimensional_opening.experiments import run_experiment_paper2


def main():
    parser = argparse.ArgumentParser(
        description="Paper 2: Random vs Feedback-Aligned Autocatalysis"
    )
    parser.add_argument(
        '--n_trajectories', type=int, default=200,
        help='Number of trajectories per group (default: 200)')
    parser.add_argument(
        '--n_steps', type=int, default=5,
        help='Number of autocatalytic additions (default: 5)')
    parser.add_argument(
        '--max_candidates', type=int, default=50,
        help='Max candidates per aligned step (default: 50)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)')
    parser.add_argument(
        '--output', type=str, default='data/paper2_results.json',
        help='Output JSON path (default: data/paper2_results.json)')
    parser.add_argument(
        '--checkpoint_dir', type=str, default=None,
        help='Directory for checkpoint files (default: None)')
    parser.add_argument(
        '--checkpoint_every', type=int, default=50,
        help='Checkpoint interval (default: 50)')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run experiment
    result = run_experiment_paper2(
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        max_candidates=args.max_candidates,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        verbose=True,
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Quick validation
    print(f"\nValidation:")
    print(f"  Group A valid at baseline: {result.group_a_valid_counts[0]}")
    print(f"  Group B valid at baseline: {result.group_b_valid_counts[0]}")
    print(f"  Output file size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
