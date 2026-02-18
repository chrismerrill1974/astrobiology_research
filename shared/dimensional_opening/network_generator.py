"""
Network Generator for dimensional activation experiments.

Step 7 Part 2 (v4): Template-based generation using known oscillators.

Key insight: Random chemistry rarely oscillates. Instead, we use known
oscillator templates (Brusselator, Oregonator) and add reactions to test
whether additional autocatalysis increases η.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import warnings


@dataclass
class GeneratedNetwork:
    """A generated reaction network with metadata."""
    reactions: List[str]
    species: List[str]
    food_set: List[str]
    n_species: int
    n_reactions: int
    n_autocatalytic: int
    rate_constants: List[float]
    initial_concentrations: Dict[str, float]
    chemostat_species: Dict[str, float]
    network_id: str = ""
    is_autocatalytic: bool = False
    template: str = ""
    n_added_reactions: int = 0
    
    def to_tracker_input(self) -> Dict:
        """Convert to input format for ActivationTracker."""
        return {
            'reactions': self.reactions,
            'rate_constants': self.rate_constants,
            'initial_concentrations': self.initial_concentrations,
            'chemostat_species': self.chemostat_species,
            'network_id': self.network_id,
        }


@dataclass
class AlignedProgressiveResult:
    """Result of progressive feedback-aligned generation."""
    networks: List[GeneratedNetwork]
    filter_results: list              # List[OscillationResult]
    acceptance_rates: List[float]     # candidates_tried at each step
    terminated_early: bool
    termination_step: Optional[int]


@dataclass
class OscillatorTemplate:
    """A known oscillator template."""
    name: str
    reactions: List[str]
    rate_constants: List[float]
    initial_concentrations: Dict[str, float]
    feed_species: Dict[str, float]  # For CSTR mode
    n_autocatalytic: int
    # Parameter ranges that maintain oscillation
    rate_range: Tuple[float, float] = (0.5, 2.0)
    
    def copy(self) -> 'OscillatorTemplate':
        """Deep copy the template."""
        return OscillatorTemplate(
            name=self.name,
            reactions=list(self.reactions),
            rate_constants=list(self.rate_constants),
            initial_concentrations=dict(self.initial_concentrations),
            feed_species=dict(self.feed_species),
            n_autocatalytic=self.n_autocatalytic,
            rate_range=self.rate_range,
        )


# Known oscillator templates
BRUSSELATOR = OscillatorTemplate(
    name="Brusselator",
    reactions=[
        "A -> X",
        "B + X -> Y + D",
        "X + X + Y -> X + X + X",  # Autocatalytic
        "X -> E",
    ],
    rate_constants=[1.0, 1.0, 1.0, 1.0],
    initial_concentrations={'A': 1.0, 'B': 3.0, 'X': 1.0, 'Y': 1.0, 'D': 0.0, 'E': 0.0},
    feed_species={'A': 1.0, 'B': 3.0},
    n_autocatalytic=1,
)

OREGONATOR = OscillatorTemplate(
    name="Oregonator",
    reactions=[
        "A + Y -> X + P",
        "X + Y -> 2P",
        "A + X -> 2X + 2Z",  # Autocatalytic
        "2X -> A + P",
        "B + Z -> Y",
    ],
    rate_constants=[1.0, 1.0, 1.0, 1.0, 1.0],
    initial_concentrations={'A': 1.0, 'B': 1.0, 'X': 0.1, 'Y': 0.1, 'Z': 0.1, 'P': 0.0},
    feed_species={'A': 1.0, 'B': 1.0},
    n_autocatalytic=1,
)

# Simplified autocatalytic oscillator
SIMPLE_AUTOCATALYST = OscillatorTemplate(
    name="SimpleAutocatalyst",
    reactions=[
        "F -> X",           # Food to X
        "X + S -> X + X",   # Autocatalytic: X makes more X
        "X -> W",           # X decay
        "F -> S",           # Food to substrate
        "S -> W",           # Substrate decay
    ],
    rate_constants=[0.5, 1.0, 0.3, 0.5, 0.1],
    initial_concentrations={'F': 2.0, 'X': 0.5, 'S': 1.0, 'W': 0.0},
    feed_species={'F': 2.0},
    n_autocatalytic=1,
)

TEMPLATES = {
    'brusselator': BRUSSELATOR,
    'oregonator': OREGONATOR,
    'simple': SIMPLE_AUTOCATALYST,
}


class NetworkGenerator:
    """
    Generate reaction networks based on oscillator templates.
    
    Strategy: Start with a known oscillator, then add reactions to test
    whether additional autocatalysis affects η.
    
    Parameters
    ----------
    template : str
        Base oscillator template ('brusselator', 'oregonator', 'simple')
    n_extra_species : int
        Additional species beyond template (for added reactions)
    rate_range : tuple
        (min, max) for rate constants of added reactions
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        template: str = 'brusselator',
        n_extra_species: int = 4,
        rate_range: Tuple[float, float] = (0.3, 1.5),
        seed: Optional[int] = None,
    ):
        if template not in TEMPLATES:
            raise ValueError(f"Unknown template: {template}. Choose from {list(TEMPLATES.keys())}")
        
        self.base_template = TEMPLATES[template].copy()
        self.template_name = template
        self.n_extra_species = n_extra_species
        self.rate_range = rate_range
        self.rng = np.random.default_rng(seed)
        self._network_counter = 0
        
        # Get species from template
        self.template_species = set()
        for rxn in self.base_template.reactions:
            self._extract_species(rxn, self.template_species)
        
        # Extra species for added reactions
        self.extra_species = [f"Z{i}" for i in range(n_extra_species)]
        self.all_species = list(self.template_species) + self.extra_species
        
        # Food species (those that are fed in CSTR)
        self.food_species = list(self.base_template.feed_species.keys())
    
    def _extract_species(self, reaction: str, species_set: Set[str]) -> None:
        """Extract species names from reaction string."""
        lhs, rhs = reaction.split(" -> ")
        for side in [lhs, rhs]:
            for term in side.split(" + "):
                term = term.strip()
                # Handle stoichiometry like "2X"
                if term and term[0].isdigit():
                    term = term[1:]
                if term:
                    species_set.add(term)
    
    def _sample_rate_constant(self) -> float:
        """Sample rate constant from log-uniform distribution."""
        log_min, log_max = np.log10(self.rate_range[0]), np.log10(self.rate_range[1])
        return 10 ** self.rng.uniform(log_min, log_max)
    
    def _format_reaction(self, reactants: List[str], products: List[str]) -> str:
        """Format reaction as string."""
        lhs = " + ".join(reactants)
        rhs = " + ".join(products)
        return f"{lhs} -> {rhs}"
    
    def _is_autocatalytic(self, reactants: List[str], products: List[str]) -> bool:
        """Check if reaction is autocatalytic."""
        reactant_counts = {}
        for r in reactants:
            reactant_counts[r] = reactant_counts.get(r, 0) + 1
        product_counts = {}
        for p in products:
            product_counts[p] = product_counts.get(p, 0) + 1
        
        for sp in product_counts:
            if product_counts[sp] > reactant_counts.get(sp, 0):
                if sp in reactant_counts:
                    return True
        return False
    
    def _generate_random_reaction(
        self, 
        exclude: Set[str],
        force_non_autocatalytic: bool = False,
        use_extra_species: bool = True,
    ) -> Tuple[List[str], List[str], float]:
        """Generate a random reaction."""
        # Choose species pool
        if use_extra_species:
            species_pool = self.all_species
        else:
            species_pool = list(self.template_species)
        
        for _ in range(100):
            n_r = self.rng.integers(1, 3)  # 1-2 reactants
            n_p = self.rng.integers(1, 3)  # 1-2 products
            
            reactants = list(self.rng.choice(species_pool, size=n_r, replace=True))
            products = list(self.rng.choice(species_pool, size=n_p, replace=True))
            
            # Don't create reactions that only involve food species
            all_involved = set(reactants + products)
            if all_involved.issubset(set(self.food_species)):
                continue
            
            rxn_str = self._format_reaction(sorted(reactants), sorted(products))
            if rxn_str in exclude:
                continue
            if force_non_autocatalytic and self._is_autocatalytic(reactants, products):
                continue
            
            rate = self._sample_rate_constant()
            return reactants, products, rate
        
        raise RuntimeError("Could not generate reaction")
    
    def _generate_autocatalytic_reaction(
        self,
        exclude: Set[str],
        use_extra_species: bool = True,
    ) -> Tuple[List[str], List[str], float]:
        """Generate an autocatalytic reaction (X + ... -> 2X + ...)."""
        # Choose catalyst from non-food species
        if use_extra_species:
            non_food = [s for s in self.all_species if s not in self.food_species]
        else:
            non_food = [s for s in self.template_species if s not in self.food_species]
        
        for _ in range(100):
            catalyst = self.rng.choice(non_food)
            
            # Reactants: catalyst + maybe food or other species
            reactants = [catalyst]
            if self.rng.random() < 0.7:
                reactants.append(self.rng.choice(self.food_species))
            
            # Products: 2 * catalyst + maybe byproduct
            products = [catalyst, catalyst]
            if self.rng.random() < 0.3:
                products.append(self.rng.choice(non_food))
            
            rxn_str = self._format_reaction(sorted(reactants), sorted(products))
            if rxn_str not in exclude:
                rate = self._sample_rate_constant()
                return reactants, products, rate
        
        raise RuntimeError("Could not generate autocatalytic reaction")
    
    def _build_network(
        self,
        added_reactions: List[str],
        added_rates: List[float],
        n_autocatalytic_added: int,
        network_id: str,
    ) -> GeneratedNetwork:
        """Build network from template + added reactions."""
        # Combine template and added reactions
        all_reactions = list(self.base_template.reactions) + added_reactions
        all_rates = list(self.base_template.rate_constants) + added_rates
        
        # Collect all species
        all_species = set()
        for rxn in all_reactions:
            self._extract_species(rxn, all_species)
        
        # Initial concentrations
        init_conc = dict(self.base_template.initial_concentrations)
        for sp in all_species:
            if sp not in init_conc:
                init_conc[sp] = 0.1  # Small initial amount for new species
        
        # Total autocatalytic count
        total_autocat = self.base_template.n_autocatalytic + n_autocatalytic_added
        
        return GeneratedNetwork(
            reactions=all_reactions,
            species=list(all_species),
            food_set=self.food_species,
            n_species=len(all_species),
            n_reactions=len(all_reactions),
            n_autocatalytic=total_autocat,
            rate_constants=all_rates,
            initial_concentrations=init_conc,
            chemostat_species=dict(self.base_template.feed_species),
            network_id=network_id,
            is_autocatalytic=total_autocat > 0,
            template=self.template_name,
            n_added_reactions=len(added_reactions),
        )
    
    def generate_control(
        self,
        n_added: int = 3,
        network_id: Optional[str] = None,
    ) -> GeneratedNetwork:
        """
        Generate control network: template + random NON-autocatalytic reactions.
        
        Parameters
        ----------
        n_added : int
            Number of random reactions to add
        network_id : str, optional
            Network identifier
        """
        if network_id is None:
            self._network_counter += 1
            network_id = f"control_{self._network_counter}"
        
        # Get existing reactions to avoid duplicates
        exclude = set(self.base_template.reactions)
        
        added_reactions = []
        added_rates = []
        
        for _ in range(n_added):
            reactants, products, rate = self._generate_random_reaction(
                exclude, force_non_autocatalytic=True
            )
            rxn = self._format_reaction(reactants, products)
            added_reactions.append(rxn)
            added_rates.append(rate)
            exclude.add(self._format_reaction(sorted(reactants), sorted(products)))
        
        return self._build_network(added_reactions, added_rates, 0, network_id)
    
    def generate_test(
        self,
        n_autocatalytic: int = 2,
        n_random: int = 1,
        network_id: Optional[str] = None,
    ) -> GeneratedNetwork:
        """
        Generate test network: template + autocatalytic reactions (+ optional random).
        
        Parameters
        ----------
        n_autocatalytic : int
            Number of autocatalytic reactions to add
        n_random : int
            Number of random (non-autocatalytic) reactions to add
        network_id : str, optional
            Network identifier
        """
        if network_id is None:
            self._network_counter += 1
            network_id = f"test_{self._network_counter}"
        
        exclude = set(self.base_template.reactions)
        
        added_reactions = []
        added_rates = []
        
        # Add autocatalytic reactions
        for _ in range(n_autocatalytic):
            reactants, products, rate = self._generate_autocatalytic_reaction(exclude)
            rxn = self._format_reaction(reactants, products)
            added_reactions.append(rxn)
            added_rates.append(rate)
            exclude.add(self._format_reaction(sorted(reactants), sorted(products)))
        
        # Add random filler reactions
        for _ in range(n_random):
            reactants, products, rate = self._generate_random_reaction(
                exclude, force_non_autocatalytic=True
            )
            rxn = self._format_reaction(reactants, products)
            added_reactions.append(rxn)
            added_rates.append(rate)
            exclude.add(self._format_reaction(sorted(reactants), sorted(products)))
        
        return self._build_network(added_reactions, added_rates, n_autocatalytic, network_id)
    
    def generate_baseline(self, network_id: Optional[str] = None) -> GeneratedNetwork:
        """Generate baseline network: just the template, no additions."""
        if network_id is None:
            self._network_counter += 1
            network_id = f"baseline_{self._network_counter}"
        
        return self._build_network([], [], 0, network_id)
    
    def generate_progressive(
        self,
        n_steps: int = 5,
        base_id: Optional[str] = None,
    ) -> List[GeneratedNetwork]:
        """
        Generate progression: baseline → increasing autocatalysis.
        
        Returns list of networks with 0, 1, 2, ... n_steps autocatalytic additions.
        """
        if base_id is None:
            self._network_counter += 1
            base_id = f"prog_{self._network_counter}"
        
        networks = []
        exclude = set(self.base_template.reactions)
        
        added_reactions = []
        added_rates = []
        n_autocat_added = 0
        
        # Step 0: baseline
        networks.append(self._build_network([], [], 0, f"{base_id}_0auto"))
        
        # Steps 1 to n_steps: add one autocatalytic reaction each
        for step in range(1, n_steps + 1):
            reactants, products, rate = self._generate_autocatalytic_reaction(exclude)
            rxn = self._format_reaction(reactants, products)
            added_reactions.append(rxn)
            added_rates.append(rate)
            exclude.add(self._format_reaction(sorted(reactants), sorted(products)))
            n_autocat_added += 1
            
            net = self._build_network(
                list(added_reactions), list(added_rates), 
                n_autocat_added, f"{base_id}_{step}auto"
            )
            networks.append(net)
        
        return networks
    
    def generate_batch_control(
        self, n_networks: int = 20, n_added: int = 3, verbose: bool = False
    ) -> List[GeneratedNetwork]:
        """Generate batch of control networks."""
        networks = []
        for i in range(n_networks):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{n_networks} control networks...")
            net = self.generate_control(n_added, f"control_{i}")
            networks.append(net)
        return networks
    
    def generate_batch_test(
        self, n_networks: int = 20, n_autocatalytic: int = 2, n_random: int = 1,
        verbose: bool = False
    ) -> List[GeneratedNetwork]:
        """Generate batch of test networks."""
        networks = []
        for i in range(n_networks):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{n_networks} test networks...")
            net = self.generate_test(n_autocatalytic, n_random, f"test_{i}")
            networks.append(net)
        return networks
    
    def generate_aligned_addition(
        self,
        base_net: GeneratedNetwork,
        max_candidates: int = 50,
    ) -> Optional[Tuple[GeneratedNetwork, 'OscillationResult']]:
        """
        Generate a single feedback-aligned autocatalytic addition.

        Tries up to max_candidates random autocatalytic reactions. Each
        candidate is added to base_net and tested with the oscillation
        filter. The first candidate that passes is returned.

        Parameters
        ----------
        base_net : GeneratedNetwork
            Network to augment.
        max_candidates : int
            Maximum candidates to try before giving up.

        Returns
        -------
        (augmented_network, filter_result, n_candidates_tried) or None
            None if no candidate passed within max_candidates tries.
        """
        from .oscillation_filter import passes_oscillation_filter

        exclude = set(base_net.reactions)

        for attempt in range(max_candidates):
            try:
                reactants, products, rate = self._generate_autocatalytic_reaction(
                    exclude
                )
            except RuntimeError:
                continue

            rxn = self._format_reaction(reactants, products)
            exclude.add(self._format_reaction(sorted(reactants), sorted(products)))

            # Build candidate network
            candidate = self._build_network(
                added_reactions=list(base_net.reactions[len(self.base_template.reactions):]) + [rxn],
                added_rates=list(base_net.rate_constants[len(self.base_template.rate_constants):]) + [rate],
                n_autocatalytic_added=base_net.n_autocatalytic - self.base_template.n_autocatalytic + 1,
                network_id=f"{base_net.network_id}_aligned",
            )

            osc_result = passes_oscillation_filter(candidate)
            if osc_result.passes:
                return candidate, osc_result, attempt + 1

        return None

    def generate_progressive_aligned(
        self,
        n_steps: int = 5,
        max_candidates: int = 50,
        base_id: Optional[str] = None,
    ) -> 'AlignedProgressiveResult':
        """
        Generate progression with feedback-aligned additions.

        At each step, adds one autocatalytic reaction that preserves
        oscillation. If no valid candidate is found within max_candidates
        tries, the trajectory terminates early.

        Parameters
        ----------
        n_steps : int
            Target number of additions.
        max_candidates : int
            Max candidates per step.
        base_id : str, optional
            Base identifier for networks.

        Returns
        -------
        AlignedProgressiveResult
            Networks, filter results, acceptance rates, and termination info.
        """
        if base_id is None:
            self._network_counter += 1
            base_id = f"aligned_{self._network_counter}"

        baseline = self.generate_baseline(network_id=f"{base_id}_0auto")
        networks = [baseline]
        filter_results = []
        acceptance_rates = []
        terminated_early = False
        termination_step = None

        current_net = baseline

        for step in range(1, n_steps + 1):
            result = self.generate_aligned_addition(
                current_net, max_candidates=max_candidates
            )

            if result is None:
                terminated_early = True
                termination_step = step
                break

            augmented_net, osc_result, n_tried = result
            augmented_net.network_id = f"{base_id}_{step}auto"
            networks.append(augmented_net)
            filter_results.append(osc_result)
            acceptance_rates.append(1.0 / n_tried)  # fraction accepted

            current_net = augmented_net

        return AlignedProgressiveResult(
            networks=networks,
            filter_results=filter_results,
            acceptance_rates=acceptance_rates,
            terminated_early=terminated_early,
            termination_step=termination_step,
        )

    def with_varied_parameters(
        self,
        base_network: GeneratedNetwork,
        rate_variation: float = 0.3,
        n_variants: int = 10,
    ) -> List[GeneratedNetwork]:
        """
        Create variants of a network with perturbed rate constants.
        
        Parameters
        ----------
        base_network : GeneratedNetwork
            Network to vary
        rate_variation : float
            Fractional variation (0.3 = ±30%)
        n_variants : int
            Number of variants to generate
        """
        variants = []
        
        for i in range(n_variants):
            # Perturb rate constants
            new_rates = []
            for k in base_network.rate_constants:
                factor = 1 + self.rng.uniform(-rate_variation, rate_variation)
                new_rates.append(k * factor)
            
            variant = GeneratedNetwork(
                reactions=list(base_network.reactions),
                species=list(base_network.species),
                food_set=list(base_network.food_set),
                n_species=base_network.n_species,
                n_reactions=base_network.n_reactions,
                n_autocatalytic=base_network.n_autocatalytic,
                rate_constants=new_rates,
                initial_concentrations=dict(base_network.initial_concentrations),
                chemostat_species=dict(base_network.chemostat_species),
                network_id=f"{base_network.network_id}_var{i}",
                is_autocatalytic=base_network.is_autocatalytic,
                template=base_network.template,
                n_added_reactions=base_network.n_added_reactions,
            )
            variants.append(variant)
        
        return variants


# Backwards compatibility aliases
def generate_batch_random(n_networks: int = 20, seed: int = 42, **kwargs) -> List[GeneratedNetwork]:
    """Backwards compatible: generates control networks."""
    gen = NetworkGenerator(seed=seed)
    return gen.generate_batch_control(n_networks, **kwargs)

def generate_batch_autocatalytic(n_networks: int = 20, seed: int = 42, **kwargs) -> List[GeneratedNetwork]:
    """Backwards compatible: generates test networks."""
    gen = NetworkGenerator(seed=seed)
    return gen.generate_batch_test(n_networks, **kwargs)
