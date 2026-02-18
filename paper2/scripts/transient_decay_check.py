"""
Phase 2, Step 2: Transient Decay Sanity Check

Run 10 random k=3 networks to t=200, compare filter outcome at t=100 vs t=200.
Must agree on >= 9/10.

Per research plan Section 2.5, item 5.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import json
import os

from dimensional_opening.network_generator import NetworkGenerator
from dimensional_opening.oscillation_filter import check_oscillation
from dimensional_opening.simulator import ReactionSimulator, DrivingMode

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_transient_decay_check(n_target=10, max_attempts=50, seed=42):
    """Run transient decay sanity check. Generates up to max_attempts networks
    to get n_target with successful integration."""
    agreements = 0
    results = []
    n_valid = 0

    print(f"Running transient decay check (target: {n_target} valid networks)...")

    for i in range(max_attempts):
        if n_valid >= n_target:
            break
        gen = NetworkGenerator(template='brusselator', seed=seed + i)
        net = gen.generate_test(n_autocatalytic=3, n_random=0)

        sim = ReactionSimulator()
        network = sim.build_network(net.reactions)

        try:
            result = sim.simulate(
                network,
                rate_constants=net.rate_constants,
                initial_concentrations=net.initial_concentrations,
                t_span=(0, 200),
                n_points=4000,
                driving_mode=DrivingMode.CHEMOSTAT,
                chemostat_species=dict(net.chemostat_species),
            )
        except Exception as e:
            print(f"  Attempt {i}: simulation failed ({e}), skipping")
            continue

        if not result.success or np.any(np.isnan(result.concentrations)):
            print(f"  Attempt {i}: integration failed (skipping)")
            continue

        # Check at t=100: use first 2000 points (t=0 to t=100)
        t_100 = result.time[:2000]
        c_100 = result.concentrations[:2000]
        osc_100 = check_oscillation(
            c_100, t_100,
            species_names=result.species_names,
            food_species=net.food_set,
        )

        # Check at t=200: use full trajectory (t=0 to t=200)
        osc_200 = check_oscillation(
            result.concentrations, result.time,
            species_names=result.species_names,
            food_species=net.food_set,
        )

        agree = osc_100.passes == osc_200.passes
        if agree:
            agreements += 1
        n_valid += 1

        print(f"  Valid {n_valid}: t=100 {'PASS' if osc_100.passes else 'FAIL'}, "
              f"t=200 {'PASS' if osc_200.passes else 'FAIL'} "
              f"{'(agree)' if agree else '(DISAGREE)'}")

        results.append({
            'network_id': i,
            'osc_100_passes': osc_100.passes,
            'osc_100_cv': float(osc_100.cv),
            'osc_100_sign_changes': int(osc_100.sign_changes),
            'osc_200_passes': osc_200.passes,
            'osc_200_cv': float(osc_200.cv),
            'osc_200_sign_changes': int(osc_200.sign_changes),
            'agree': agree,
        })

    agreement_rate = agreements / n_valid if n_valid > 0 else 0

    print(f"\nAgreement: {agreements}/{n_valid} "
          f"({100*agreement_rate:.0f}%)")
    passes = agreements >= 9 if n_valid >= 10 else agreement_rate >= 0.9
    print(f"Transient decay check: {'PASS' if passes else 'FAIL'} "
          f"(criterion: >= 9/10 agreement)")

    summary = {
        'n_target': n_target,
        'n_valid': n_valid,
        'agreements': agreements,
        'agreement_rate': agreement_rate,
        'passes': passes,
    }

    output = {'summary': summary, 'results': results}
    output_path = os.path.join(OUTPUT_DIR, 'transient_decay_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    return summary


if __name__ == '__main__':
    summary = run_transient_decay_check(n_target=10, seed=42)
