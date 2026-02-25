"""Tests for network generator module (template-based, v4)."""

import numpy as np
import pytest

from dimensional_opening.network_generator import (
    NetworkGenerator, GeneratedNetwork, AlignedProgressiveResult,
    BRUSSELATOR, OREGONATOR, TEMPLATES
)
from dimensional_opening.oscillation_filter import passes_oscillation_filter


class TestNetworkGenerator:
    """Tests for NetworkGenerator."""
    
    def test_init_default(self):
        gen = NetworkGenerator()
        assert gen.template_name == 'brusselator'
        assert gen.n_extra_species == 4
    
    def test_init_custom_template(self):
        gen = NetworkGenerator(template='oregonator')
        assert gen.template_name == 'oregonator'
    
    def test_invalid_template(self):
        with pytest.raises(ValueError):
            NetworkGenerator(template='nonexistent')
    
    def test_extra_species(self):
        gen = NetworkGenerator(n_extra_species=6)
        assert len(gen.extra_species) == 6
        assert gen.extra_species[0] == 'Z0'


class TestTemplates:
    """Tests for oscillator templates."""
    
    def test_brusselator_exists(self):
        assert 'brusselator' in TEMPLATES
        assert BRUSSELATOR.n_autocatalytic == 1
    
    def test_oregonator_exists(self):
        assert 'oregonator' in TEMPLATES
        assert OREGONATOR.n_autocatalytic == 1
    
    def test_template_has_required_fields(self):
        for name, template in TEMPLATES.items():
            assert len(template.reactions) > 0
            assert len(template.rate_constants) == len(template.reactions)
            assert len(template.initial_concentrations) > 0
            assert len(template.feed_species) > 0


class TestControlNetworks:
    """Tests for control network generation."""
    
    def test_generate_control_basic(self):
        gen = NetworkGenerator(seed=42)
        net = gen.generate_control(n_added=3)
        assert isinstance(net, GeneratedNetwork)
        # Base Brusselator has 4 reactions + 3 added
        assert net.n_reactions == 4 + 3
        # Should not add autocatalytic (only template's autocatalytic count)
        assert net.n_autocatalytic == BRUSSELATOR.n_autocatalytic
    
    def test_generate_control_reproducible(self):
        gen1 = NetworkGenerator(seed=42)
        net1 = gen1.generate_control(n_added=3)
        gen2 = NetworkGenerator(seed=42)
        net2 = gen2.generate_control(n_added=3)
        assert net1.reactions == net2.reactions
    
    def test_generate_batch_control(self):
        gen = NetworkGenerator(seed=42)
        networks = gen.generate_batch_control(n_networks=5, n_added=2)
        assert len(networks) == 5
        for net in networks:
            assert net.n_reactions == 4 + 2  # Brusselator + added


class TestTestNetworks:
    """Tests for test network generation."""
    
    def test_generate_test_basic(self):
        gen = NetworkGenerator(seed=42)
        net = gen.generate_test(n_autocatalytic=2, n_random=1)
        assert isinstance(net, GeneratedNetwork)
        # Base Brusselator has 4 reactions + 2 autocatalytic + 1 random
        assert net.n_reactions == 4 + 2 + 1
        # Template autocatalytic + added autocatalytic
        assert net.n_autocatalytic == BRUSSELATOR.n_autocatalytic + 2
    
    def test_generate_test_reproducible(self):
        gen1 = NetworkGenerator(seed=42)
        net1 = gen1.generate_test(n_autocatalytic=2, n_random=1)
        gen2 = NetworkGenerator(seed=42)
        net2 = gen2.generate_test(n_autocatalytic=2, n_random=1)
        assert net1.reactions == net2.reactions
    
    def test_generate_batch_test(self):
        gen = NetworkGenerator(seed=42)
        networks = gen.generate_batch_test(n_networks=5, n_autocatalytic=2, n_random=1)
        assert len(networks) == 5
        for net in networks:
            assert net.n_autocatalytic == BRUSSELATOR.n_autocatalytic + 2


class TestBaselineNetworks:
    """Tests for baseline network generation."""
    
    def test_generate_baseline(self):
        gen = NetworkGenerator(seed=42)
        net = gen.generate_baseline()
        # Should be exactly the template
        assert net.n_reactions == len(BRUSSELATOR.reactions)
        assert net.n_autocatalytic == BRUSSELATOR.n_autocatalytic
        assert net.n_added_reactions == 0


class TestProgressiveNetworks:
    """Tests for progressive network generation."""
    
    def test_generate_progressive_basic(self):
        gen = NetworkGenerator(seed=42)
        networks = gen.generate_progressive(n_steps=3)
        assert len(networks) == 4  # baseline + 3 steps
        
        # Check autocatalytic count increases
        for i, net in enumerate(networks):
            assert net.n_autocatalytic == BRUSSELATOR.n_autocatalytic + i
    
    def test_generate_progressive_reactions_accumulate(self):
        gen = NetworkGenerator(seed=42)
        networks = gen.generate_progressive(n_steps=3)
        for i in range(1, len(networks)):
            assert networks[i].n_reactions == networks[i-1].n_reactions + 1


class TestParameterVariation:
    """Tests for parameter variation."""
    
    def test_with_varied_parameters(self):
        gen = NetworkGenerator(seed=42)
        base = gen.generate_baseline()
        variants = gen.with_varied_parameters(base, rate_variation=0.2, n_variants=5)
        
        assert len(variants) == 5
        for v in variants:
            assert v.reactions == base.reactions
            # Rate constants should differ
            assert v.rate_constants != base.rate_constants


class TestAlignedAddition:
    """Tests for feedback-aligned addition generation."""

    def test_aligned_addition_returns_valid_or_none(self):
        gen = NetworkGenerator(template='brusselator', seed=42)
        base = gen.generate_baseline()
        result = gen.generate_aligned_addition(base, max_candidates=20)
        if result is not None:
            net, osc_result, n_tried = result
            assert isinstance(net, GeneratedNetwork)
            assert osc_result.passes is True
            assert n_tried >= 1
            assert net.n_reactions == base.n_reactions + 1

    def test_aligned_addition_preserves_oscillation(self):
        """Returned network must pass the oscillation filter."""
        gen = NetworkGenerator(template='brusselator', seed=42)
        base = gen.generate_baseline()
        result = gen.generate_aligned_addition(base, max_candidates=50)
        if result is not None:
            net, _, _ = result
            verify = passes_oscillation_filter(net)
            assert verify.passes is True

    def test_aligned_addition_max_candidates_respected(self):
        """With max_candidates=1, should return quickly (either success or None)."""
        gen = NetworkGenerator(template='brusselator', seed=42)
        base = gen.generate_baseline()
        result = gen.generate_aligned_addition(base, max_candidates=1)
        # Just verify it doesn't hang or crash
        assert result is None or len(result) == 3


class TestProgressiveAligned:
    """Tests for progressive feedback-aligned generation."""

    def test_progressive_aligned_returns_result(self):
        gen = NetworkGenerator(template='brusselator', seed=42)
        result = gen.generate_progressive_aligned(n_steps=2, max_candidates=30)
        assert isinstance(result, AlignedProgressiveResult)
        # Should have baseline + up to 2 steps
        assert len(result.networks) >= 1  # at least baseline
        assert len(result.networks) <= 3  # at most baseline + 2

    def test_progressive_aligned_networks_oscillate(self):
        """All networks in the progression should pass the oscillation filter."""
        gen = NetworkGenerator(template='brusselator', seed=42)
        result = gen.generate_progressive_aligned(n_steps=2, max_candidates=30)
        # Skip baseline (step 0) — filter_results starts at step 1
        for i, osc in enumerate(result.filter_results):
            assert osc.passes is True, f"Step {i+1} network failed oscillation filter"

    def test_progressive_aligned_acceptance_rates(self):
        gen = NetworkGenerator(template='brusselator', seed=42)
        result = gen.generate_progressive_aligned(n_steps=2, max_candidates=30)
        for rate in result.acceptance_rates:
            assert 0 < rate <= 1.0

    def test_progressive_aligned_early_termination(self):
        """With max_candidates=1, early termination is likely."""
        gen = NetworkGenerator(template='brusselator', seed=100)
        result = gen.generate_progressive_aligned(n_steps=5, max_candidates=1)
        # Either it terminates early or gets lucky on all 5
        assert isinstance(result.terminated_early, bool)
        if result.terminated_early:
            assert result.termination_step is not None
            assert len(result.networks) < 6  # fewer than baseline + 5

    def test_progressive_aligned_correct_counts(self):
        gen = NetworkGenerator(template='brusselator', seed=42)
        result = gen.generate_progressive_aligned(n_steps=3, max_candidates=50)
        n_completed = len(result.networks) - 1  # exclude baseline
        assert len(result.filter_results) == n_completed
        assert len(result.acceptance_rates) == n_completed


class TestTrackerIntegration:
    """Tests for integration with ActivationTracker."""
    
    def test_to_tracker_input(self):
        gen = NetworkGenerator(seed=42)
        net = gen.generate_control(n_added=2)
        input_dict = net.to_tracker_input()
        assert 'reactions' in input_dict
        assert 'rate_constants' in input_dict
        assert 'initial_concentrations' in input_dict
        assert 'chemostat_species' in input_dict
        assert 'network_id' in input_dict
    
    def test_tracker_accepts_input(self):
        from dimensional_opening import ActivationTracker
        gen = NetworkGenerator(seed=42)
        net = gen.generate_test(n_autocatalytic=1, n_random=1)
        
        tracker = ActivationTracker(t_span=(0, 100), n_points=5000, random_state=42)
        
        # Use CSTR mode
        input_dict = net.to_tracker_input()
        feed = input_dict.pop('chemostat_species')
        input_dict['cstr_dilution_rate'] = 0.1
        input_dict['cstr_feed_concentrations'] = feed
        
        result = tracker.analyze_network(**input_dict)
        assert result.network_id == net.network_id


# ===========================================================================
# Phase 2 additional tests — Edge cases for Paper 5
# ===========================================================================


class TestSpeciesNameCollisions:
    """Phase 2.5: Extra species names should not collide with template species.

    Extra species are auto-named Z0, Z1, Z2, etc. If a template already
    uses these names, reactions would silently conflate species.
    """

    def test_default_templates_no_z_prefix(self):
        """Default templates should NOT use Z-prefixed species names.

        This is a regression guard — if someone modifies a template to
        use species like Z0, the auto-naming scheme would collide.
        """
        for name, template in TEMPLATES.items():
            all_species = set()
            for rxn in template.reactions:
                # Extract species names from reaction
                for side in rxn.split(" -> "):
                    for term in side.split(" + "):
                        term = term.strip()
                        # Remove leading coefficient
                        sp = term.lstrip("0123456789")
                        if sp:
                            all_species.add(sp)

            # Also check initial concentrations and feed species
            all_species.update(template.initial_concentrations.keys())
            all_species.update(template.feed_species.keys())

            z_names = [s for s in all_species if s.startswith("Z") and
                       len(s) > 1 and s[1:].isdigit()]
            assert len(z_names) == 0, (
                f"Template '{name}' uses Z-prefixed species {z_names} "
                f"which would collide with auto-generated extra species"
            )

    def test_extra_species_are_distinct_from_template(self):
        """Generated extra species should not overlap with template species."""
        for name in TEMPLATES:
            gen = NetworkGenerator(template=name, n_extra_species=10)

            template_species = set(gen.template_species)
            extra = set(gen.extra_species)

            overlap = template_species & extra
            assert len(overlap) == 0, (
                f"Template '{name}': overlap {overlap} between template "
                f"species and auto-generated extra species"
            )


class TestDuplicateReactionDetection:
    """Phase 2.6: Duplicate reaction detection in the generator.

    The generator uses _format_reaction(sorted(reactants), sorted(products))
    to build an exclusion set. This tests that ordering doesn't fool it
    and that large batches don't accumulate duplicates.
    """

    def test_sorted_exclusion_catches_reordered_reactants(self):
        """Reactions A + B -> C and B + A -> C should be detected as duplicates.

        The generator normalizes by sorting reactants and products before
        comparison. Verify this works.
        """
        gen = NetworkGenerator(seed=42)

        # The internal method sorts before adding to exclude set
        rxn1 = gen._format_reaction(sorted(["B", "A"]), sorted(["C"]))
        rxn2 = gen._format_reaction(sorted(["A", "B"]), sorted(["C"]))
        assert rxn1 == rxn2, (
            f"Sorted format should be identical: '{rxn1}' vs '{rxn2}'"
        )

    def test_large_batch_no_duplicate_stoichiometry(self):
        """Generate 20 networks with 5 added reactions each.

        No single network should contain two reactions with identical
        net stoichiometry (after accounting for the template reactions).
        """
        gen = NetworkGenerator(seed=42)

        for i in range(20):
            gen_i = NetworkGenerator(seed=42 + i)
            net = gen_i.generate_control(n_added=5)

            # Build set of normalized reaction strings
            seen = set()
            for rxn in net.reactions:
                parts = rxn.split(" -> ")
                lhs = " + ".join(sorted(parts[0].split(" + ")))
                rhs = " + ".join(sorted(parts[1].split(" + ")))
                normalized = f"{lhs} -> {rhs}"

                assert normalized not in seen, (
                    f"Network seed={42+i}: duplicate reaction '{rxn}' "
                    f"(normalized: '{normalized}')"
                )
                seen.add(normalized)
