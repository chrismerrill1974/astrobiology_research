"""
Energy-coupled Brusselator model for Paper 3.

Extends the standard Brusselator with an energy-coupling layer:
- Two chemostatted energy carriers: Ehi, Elo
- Drive strength J controls Ehi concentration (Elo = max(0.01, 1 - J))
- One driven reaction: X + Ehi -> Y + Elo (rate = k_drive * [X] * [Ehi])
- At J=0: Ehi=0, driven reaction vanishes → recovers standard Brusselator
- At high J: strong energy flux creates new X↔Y coupling channel
"""

import sys
import os

# Add astrobiology2 to path so we can import dimensional_opening
_astro2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'astrobiology2')
if os.path.abspath(_astro2_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(_astro2_path))

from dimensional_opening.network_generator import GeneratedNetwork


def make_driven_brusselator(J: float, k_drive: float = 1.0) -> GeneratedNetwork:
    """
    Create a Brusselator with energy-coupled driving at strength J.

    Parameters
    ----------
    J : float
        Drive strength. J=0 recovers the standard Brusselator.
        Ehi is chemostatted at J, Elo at max(0.01, 1 - J).
    k_drive : float
        Rate constant for the driven reaction X + Ehi -> Y + Elo.

    Returns
    -------
    GeneratedNetwork
        Network ready for simulation via ActivationTracker.
    """
    # Standard Brusselator reactions
    reactions = [
        "A -> X",
        "B + X -> Y + D",
        "X + X + Y -> X + X + X",  # Autocatalytic
        "X -> E",
    ]
    rate_constants = [1.0, 1.0, 1.0, 1.0]

    # Chemostat species: A, B always; Ehi, Elo when J > 0
    chemostat = {'A': 1.0, 'B': 3.0}
    init_conc = {
        'A': 1.0, 'B': 3.0, 'X': 1.0, 'Y': 1.0,
        'D': 0.0, 'E': 0.0,
    }

    if J > 0:
        # Add energy carriers and driven reaction
        reactions.append("X + Ehi -> Y + Elo")
        rate_constants.append(k_drive)

        ehi_conc = J
        elo_conc = max(0.01, 1.0 - J)

        chemostat['Ehi'] = ehi_conc
        chemostat['Elo'] = elo_conc
        init_conc['Ehi'] = ehi_conc
        init_conc['Elo'] = elo_conc

    # Collect all species
    all_species = set()
    for rxn in reactions:
        lhs, rhs = rxn.split(" -> ")
        for side in [lhs, rhs]:
            for term in side.split(" + "):
                term = term.strip()
                if term and term[0].isdigit():
                    term = term[1:]
                if term:
                    all_species.add(term)

    return GeneratedNetwork(
        reactions=reactions,
        species=sorted(all_species),
        food_set=list(chemostat.keys()),
        n_species=len(all_species),
        n_reactions=len(reactions),
        n_autocatalytic=1,
        rate_constants=rate_constants,
        initial_concentrations=init_conc,
        chemostat_species=chemostat,
        network_id=f"driven_J{J:.2f}",
        is_autocatalytic=True,
        template="driven_brusselator",
        n_added_reactions=1 if J > 0 else 0,
    )


def add_random_autocatalytic_reaction(
    net: GeneratedNetwork,
    rng,
    max_attempts: int = 50,
) -> GeneratedNetwork:
    """
    Add a random autocatalytic reaction to an existing network.

    Parameters
    ----------
    net : GeneratedNetwork
        Base network.
    rng : numpy random Generator
        Random state.
    max_attempts : int
        Max attempts to find a valid reaction.

    Returns
    -------
    GeneratedNetwork
        New network with one additional autocatalytic reaction.
    """
    non_food = [s for s in net.species if s not in net.food_set]
    food = net.food_set

    for _ in range(max_attempts):
        catalyst = rng.choice(non_food)
        reactants = [catalyst]
        if rng.random() < 0.7 and food:
            reactants.append(rng.choice(food))
        products = [catalyst, catalyst]

        lhs = " + ".join(sorted(reactants))
        rhs = " + ".join(sorted(products))
        rxn_str = f"{lhs} -> {rhs}"

        if rxn_str not in net.reactions:
            log_rate = rng.uniform(-0.5, 0.2)  # rate in ~[0.3, 1.5]
            rate = 10 ** log_rate

            new_species = set(net.species)
            new_init = dict(net.initial_concentrations)
            for s in reactants + products:
                if s not in new_species:
                    new_species.add(s)
                    new_init[s] = 0.1

            return GeneratedNetwork(
                reactions=net.reactions + [rxn_str],
                species=sorted(new_species),
                food_set=list(net.food_set),
                n_species=len(new_species),
                n_reactions=net.n_reactions + 1,
                n_autocatalytic=net.n_autocatalytic + 1,
                rate_constants=list(net.rate_constants) + [rate],
                initial_concentrations=new_init,
                chemostat_species=dict(net.chemostat_species),
                network_id=net.network_id + f"_+ac",
                is_autocatalytic=True,
                template=net.template,
                n_added_reactions=net.n_added_reactions + 1,
            )

    raise RuntimeError("Could not generate autocatalytic reaction")
