"""
Paper 5 Topology Library: Topological Buffering of Transient Dynamical Complexity.

Defines 8 topology classes (T0 baseline + T1a/T1b random controls + T2-T6 motifs),
all built on top of the Paper 3/4 enzyme-complex coupled Brusselator.

Embedding protocol (RESEARCH_PLAN §2.2):
  1. All new species couple through the shared energy pool E
  2. All new reactions are mass-action
  3. New species initial conditions are O(1)
  4. D2 projection includes all dynamic species

Usage:
  python3 paper5_topology_library.py                    # structural census
  python3 paper5_topology_library.py --sanity-check     # 10 integrations per topology
"""

import sys
import os
import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_research_dir = os.path.join(_this_dir, '..', 'astrobiology_research')
_shared_path = os.path.join(_research_dir, 'shared')
_paper3_scripts = os.path.join(_research_dir, 'paper3', 'scripts')

for p in [_shared_path, _paper3_scripts]:
    if os.path.abspath(p) not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

from pilot5b_enzyme_complex import EnzymeComplexParams, make_enzyme_complex_network
from dimensional_opening.simulator import ReactionSimulator, DrivingMode
from dimensional_opening.correlation_dimension import CorrelationDimension
from dimensional_opening.network_generator import GeneratedNetwork


# ══════════════════════════════════════════════════════════════════════
# Structural descriptors
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TopologyDescriptors:
    """Structural properties of an augmented reaction network."""
    name: str
    group: str                      # "A" (baseline/control) or "B" (candidate motif)
    n_species: int                  # total dynamic species
    n_reactions: int                # total reactions
    n_added_species: int            # species added beyond T0
    n_added_reactions: int          # reactions added beyond T0
    cyclomatic_number: int          # M - N + C (edges - nodes + components)
    weakly_reversible: bool         # every reaction on a directed cycle?
    deficiency: int                 # n_complexes - n_linkage_classes - rank(S)
    d2_species: List[str]           # species included in D2 projection

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_reaction_graph(reactions: List[str]) -> Tuple[set, List[Tuple[str, str]]]:
    """
    Parse reaction strings into a directed graph of species interactions.
    Returns (nodes, edges) where edges are (reactant_species, product_species).
    """
    nodes = set()
    edges = []
    for rxn in reactions:
        parts = rxn.split('->')
        if len(parts) != 2:
            continue
        lhs_str, rhs_str = parts[0].strip(), parts[1].strip()

        def parse_species(s):
            species = set()
            for token in s.split('+'):
                token = token.strip()
                if not token:
                    continue
                # Strip leading coefficients (e.g., "3X1" -> "X1")
                i = 0
                while i < len(token) and (token[i].isdigit() or token[i] == '.'):
                    i += 1
                sp = token[i:].strip()
                if sp:
                    species.add(sp)
            return species

        lhs = parse_species(lhs_str)
        rhs = parse_species(rhs_str)
        nodes.update(lhs)
        nodes.update(rhs)
        # Directed edges from each reactant species to each product species
        for r in lhs:
            for p in rhs:
                if r != p:
                    edges.append((r, p))
    return nodes, edges


def compute_cyclomatic_number(reactions: List[str]) -> int:
    """
    Cyclomatic number mu = M - N + C.
    M = unique directed edges, N = nodes, C = weakly connected components.
    """
    nodes, edges = _parse_reaction_graph(reactions)
    unique_edges = set(edges)
    M = len(unique_edges)
    N = len(nodes)

    # Weakly connected components via union-find
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b) in unique_edges:
        union(a, b)

    C = len(set(find(n) for n in nodes)) if nodes else 0
    return M - N + C


def check_weak_reversibility(reactions: List[str]) -> bool:
    """
    A network is weakly reversible if every reaction complex lies on
    a directed cycle in the reaction graph (complex graph).

    We build the complex graph and check strong connectivity of each
    linkage class.
    """
    complexes, complex_edges = _build_complex_graph(reactions)
    if not complexes:
        return True

    # Build adjacency list
    adj = {c: set() for c in complexes}
    for (src, dst) in complex_edges:
        adj[src].add(dst)

    # Find strongly connected components (Tarjan's or Kosaraju's)
    # If every linkage class is a single SCC, the network is weakly reversible
    visited = set()
    order = []

    def dfs1(node):
        stack = [(node, False)]
        while stack:
            n, processed = stack.pop()
            if processed:
                order.append(n)
                continue
            if n in visited:
                continue
            visited.add(n)
            stack.append((n, True))
            for nb in adj.get(n, []):
                if nb not in visited:
                    stack.append((nb, False))

    for c in complexes:
        if c not in visited:
            dfs1(c)

    # Reverse graph
    radj = {c: set() for c in complexes}
    for (src, dst) in complex_edges:
        radj[dst].add(src)

    visited2 = set()
    sccs = []

    def dfs2(node, comp):
        stack = [node]
        while stack:
            n = stack.pop()
            if n in visited2:
                continue
            visited2.add(n)
            comp.append(n)
            for nb in radj.get(n, []):
                if nb not in visited2:
                    stack.append(nb)

    for c in reversed(order):
        if c not in visited2:
            comp = []
            dfs2(c, comp)
            sccs.append(set(comp))

    # Check: every complex in a linkage class must be in the same SCC
    # Linkage class = weakly connected component of the complex graph
    # Build undirected adjacency
    uadj = {c: set() for c in complexes}
    for (src, dst) in complex_edges:
        uadj[src].add(dst)
        uadj[dst].add(src)

    visited3 = set()

    def get_linkage_class(start):
        cc = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited3:
                continue
            visited3.add(n)
            cc.add(n)
            for nb in uadj[n]:
                if nb not in visited3:
                    stack.append(nb)
        return cc

    linkage_classes = []
    for c in complexes:
        if c not in visited3:
            linkage_classes.append(get_linkage_class(c))

    # For weak reversibility, each linkage class must be a subset of one SCC
    scc_map = {}
    for i, scc in enumerate(sccs):
        for c in scc:
            scc_map[c] = i

    for lc in linkage_classes:
        scc_ids = set(scc_map[c] for c in lc)
        if len(scc_ids) > 1:
            return False
    return True


def _build_complex_graph(reactions: List[str]) -> Tuple[set, List[Tuple[str, str]]]:
    """
    Build the complex graph from reaction strings.
    A complex is a multiset of species (e.g., "2X1 + Y1").
    Returns (set of complex strings, list of (source_complex, product_complex) edges).
    """
    complexes = set()
    edges = []
    for rxn in reactions:
        parts = rxn.split('->')
        if len(parts) != 2:
            continue
        lhs = _normalize_complex(parts[0].strip())
        rhs = _normalize_complex(parts[1].strip())
        complexes.add(lhs)
        complexes.add(rhs)
        edges.append((lhs, rhs))
    return complexes, edges


def _normalize_complex(s: str) -> str:
    """Normalize a complex string for comparison (sorted species)."""
    terms = []
    for token in s.split('+'):
        token = token.strip()
        if token:
            terms.append(token)
    return ' + '.join(sorted(terms))


def compute_deficiency(reactions: List[str]) -> int:
    """
    Deficiency = n_complexes - n_linkage_classes - rank(stoichiometric_matrix).

    Computed on the full reaction network (not a subgraph).
    """
    complexes, complex_edges = _build_complex_graph(reactions)
    n_complexes = len(complexes)

    # Linkage classes (weakly connected components of complex graph)
    uadj = {c: set() for c in complexes}
    for (src, dst) in complex_edges:
        uadj[src].add(dst)
        uadj[dst].add(src)

    visited = set()
    n_linkage_classes = 0
    for c in complexes:
        if c not in visited:
            n_linkage_classes += 1
            stack = [c]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                for nb in uadj[n]:
                    if nb not in visited:
                        stack.append(nb)

    # Stoichiometric matrix
    # Collect all species
    all_species = set()
    for c in complexes:
        for token in c.split('+'):
            token = token.strip()
            if not token:
                continue
            i = 0
            while i < len(token) and (token[i].isdigit() or token[i] == '.'):
                i += 1
            coeff = int(token[:i]) if i > 0 else 1
            sp = token[i:].strip()
            if sp:
                all_species.add(sp)

    species_list = sorted(all_species)
    sp_idx = {sp: i for i, sp in enumerate(species_list)}

    def parse_complex_to_vec(c_str):
        vec = np.zeros(len(species_list))
        for token in c_str.split('+'):
            token = token.strip()
            if not token:
                continue
            i = 0
            while i < len(token) and (token[i].isdigit() or token[i] == '.'):
                i += 1
            coeff = int(token[:i]) if i > 0 else 1
            sp = token[i:].strip()
            if sp and sp in sp_idx:
                vec[sp_idx[sp]] += coeff
        return vec

    # Build stoichiometric matrix: each column = product_vec - reactant_vec
    S = np.zeros((len(species_list), len(complex_edges)))
    for j, (src, dst) in enumerate(complex_edges):
        S[:, j] = parse_complex_to_vec(dst) - parse_complex_to_vec(src)

    rank_S = np.linalg.matrix_rank(S)
    deficiency = n_complexes - n_linkage_classes - rank_S
    return max(0, deficiency)  # should be non-negative by definition


# ══════════════════════════════════════════════════════════════════════
# Topology definitions
# ══════════════════════════════════════════════════════════════════════

# Baseline parameters from Paper 4
BASELINE_GAMMA = 0.00223
BASELINE_J = 4.388
BASELINE_KCAT = 0.278

# Default rate constants for motif reactions (O(1) scale, matching base system)
MOTIF_RATE_DEFAULT = 1.0

# Species that are part of D2 projection in the base system
BASE_D2_SPECIES = ['X1', 'Y1', 'X2', 'Y2', 'E']


def make_t0_baseline(p: EnzymeComplexParams, seed: int = 42) -> GeneratedNetwork:
    """T0: Unmodified coupled Brusselator baseline. 5 D2 species, 14 reactions."""
    return make_enzyme_complex_network(p, seed=seed)


def _augment_network(
    base_net: GeneratedNetwork,
    extra_reactions: List[str],
    extra_rate_constants: List[float],
    extra_species_ics: Dict[str, float],
    topology_id: str,
) -> GeneratedNetwork:
    """
    Add reactions/species to a base network.
    New species are appended to the species list.
    Chemostat species unchanged. All new species are dynamic.
    """
    new_reactions = list(base_net.reactions) + extra_reactions
    new_rates = list(base_net.rate_constants) + extra_rate_constants

    # Discover new species from extra reactions
    new_species_set = set()
    for rxn in extra_reactions:
        for side in rxn.split('->'):
            for token in side.split('+'):
                token = token.strip()
                i = 0
                while i < len(token) and (token[i].isdigit() or token[i] == '.'):
                    i += 1
                sp = token[i:].strip()
                if sp and sp not in base_net.species:
                    new_species_set.add(sp)

    new_species = list(base_net.species)
    for sp in sorted(new_species_set):
        if sp not in new_species:
            new_species.append(sp)

    new_ics = dict(base_net.initial_concentrations)
    for sp, val in extra_species_ics.items():
        new_ics[sp] = val
    # Ensure all new species have ICs
    for sp in new_species_set:
        if sp not in new_ics:
            new_ics[sp] = 1.0

    return GeneratedNetwork(
        reactions=new_reactions,
        species=new_species,
        food_set=list(base_net.food_set),
        n_species=len(new_species),
        n_reactions=len(new_reactions),
        n_autocatalytic=base_net.n_autocatalytic,
        rate_constants=new_rates,
        initial_concentrations=new_ics,
        chemostat_species=dict(base_net.chemostat_species),
        network_id=topology_id,
        is_autocatalytic=True,
        template=f"CoupledBrusselator_EC_{topology_id}",
        n_added_reactions=len(extra_reactions),
    )


def make_t2_simple_3cycle(
    p: EnzymeComplexParams, seed: int = 42,
    motif_rates: Optional[List[float]] = None,
) -> GeneratedNetwork:
    """
    T2: Simple 3-cycle.  A_m -> B_m -> C_m -> A_m
    All three species couple to E (energy pool consumption).
    3 added species, 4 added reactions (3 cycle + 1 E-coupling).

    Embedding: C_m catalytically consumes E (same interface as base gating).
      C_m + E -> C_m + Ew   (rate: motif_rates[3])
    This creates a slow-variable pathway through the motif cycle.
    """
    base = make_enzyme_complex_network(p, seed=seed)
    rates = motif_rates if motif_rates else [MOTIF_RATE_DEFAULT] * 4

    extra_reactions = [
        "Am -> Bm",                 # cycle step 1
        "Bm -> Cm",                 # cycle step 2
        "Cm -> Am",                 # cycle step 3 (closure)
        "Cm + E -> Cm + Ew",        # energy coupling: Cm drains E
    ]
    extra_rates = rates[:4]
    extra_ics = {'Am': 1.0, 'Bm': 1.0, 'Cm': 1.0}

    return _augment_network(base, extra_reactions, extra_rates, extra_ics, "T2_3cycle")


def make_t3_parallel_dual_path(
    p: EnzymeComplexParams, seed: int = 42,
    motif_rates: Optional[List[float]] = None,
) -> GeneratedNetwork:
    """
    T3: Parallel dual-path cycle.
      Am -> Bm -> Dm -> Am   (path 1)
      Am -> Cm -> Dm -> Am   (path 2)
    Plus energy coupling: Dm + E -> Dm + Ew

    4 added species, 6 added reactions (5 cycle + 1 E-coupling).
    Two redundant routes through the same loop.
    """
    base = make_enzyme_complex_network(p, seed=seed)
    rates = motif_rates if motif_rates else [MOTIF_RATE_DEFAULT] * 6

    extra_reactions = [
        "Am -> Bm",                 # path 1 step 1
        "Bm -> Dm",                 # path 1 step 2
        "Am -> Cm",                 # path 2 step 1
        "Cm -> Dm",                 # path 2 step 2
        "Dm -> Am",                 # shared closure
        "Dm + E -> Dm + Ew",        # energy coupling
    ]
    extra_rates = rates[:6]
    extra_ics = {'Am': 1.0, 'Bm': 1.0, 'Cm': 1.0, 'Dm': 1.0}

    return _augment_network(base, extra_reactions, extra_rates, extra_ics, "T3_dual_path")


def make_t4_branch_rejoin(
    p: EnzymeComplexParams, seed: int = 42,
    motif_rates: Optional[List[float]] = None,
) -> GeneratedNetwork:
    """
    T4: Branch-and-rejoin.
      Am -> Bm -> Dm
      Am -> Cm -> Dm
      Dm -> Am           (closure)
    Plus energy coupling: Dm + E -> Dm + Ew

    4 added species, 6 added reactions.
    Same graph as T3 but with explicit branch-and-rejoin framing.
    Note: T4 is topologically identical to T3. We keep it because
    the rate constants can differ (different path weights).
    We give path 1 faster rates to break the symmetry.
    """
    base = make_enzyme_complex_network(p, seed=seed)
    rates = motif_rates if motif_rates else [
        1.5,   # Am -> Bm  (fast path)
        1.5,   # Bm -> Dm
        0.5,   # Am -> Cm  (slow path)
        0.5,   # Cm -> Dm
        1.0,   # Dm -> Am
        1.0,   # E coupling
    ]

    extra_reactions = [
        "Am -> Bm",
        "Bm -> Dm",
        "Am -> Cm",
        "Cm -> Dm",
        "Dm -> Am",
        "Dm + E -> Dm + Ew",
    ]
    extra_rates = rates[:6]
    extra_ics = {'Am': 1.0, 'Bm': 1.0, 'Cm': 1.0, 'Dm': 1.0}

    return _augment_network(base, extra_reactions, extra_rates, extra_ics, "T4_branch_rejoin")


def make_t5_feedforward_return(
    p: EnzymeComplexParams, seed: int = 42,
    motif_rates: Optional[List[float]] = None,
) -> GeneratedNetwork:
    """
    T5: Feed-forward with return.
      Am -> Bm -> Cm
      Am -> Cm           (short-circuit / feed-forward)
      Cm -> Am           (return / closure)
    Plus energy coupling: Cm + E -> Cm + Ew

    3 added species, 5 added reactions.
    Tests whether a short-circuit path stabilises or destabilises.
    """
    base = make_enzyme_complex_network(p, seed=seed)
    rates = motif_rates if motif_rates else [MOTIF_RATE_DEFAULT] * 5

    extra_reactions = [
        "Am -> Bm",                 # long path step 1
        "Bm -> Cm",                 # long path step 2
        "Am -> Cm",                 # feed-forward short-circuit
        "Cm -> Am",                 # return
        "Cm + E -> Cm + Ew",        # energy coupling
    ]
    extra_rates = rates[:5]
    extra_ics = {'Am': 1.0, 'Bm': 1.0, 'Cm': 1.0}

    return _augment_network(base, extra_reactions, extra_rates, extra_ics, "T5_feedforward")


def make_t6_nested_dual_loop(
    p: EnzymeComplexParams, seed: int = 42,
    motif_rates: Optional[List[float]] = None,
) -> GeneratedNetwork:
    """
    T6: Nested dual loop.
      Am -> Bm -> Cm -> Am   (outer cycle)
      Bm -> Dm -> Bm         (inner cycle nested at Bm)
    Plus energy coupling: Cm + E -> Cm + Ew

    4 added species, 6 added reactions.
    Local redundancy nested inside a larger cycle.
    """
    base = make_enzyme_complex_network(p, seed=seed)
    rates = motif_rates if motif_rates else [MOTIF_RATE_DEFAULT] * 6

    extra_reactions = [
        "Am -> Bm",                 # outer step 1
        "Bm -> Cm",                 # outer step 2
        "Cm -> Am",                 # outer closure
        "Bm -> Dm",                 # inner step 1
        "Dm -> Bm",                 # inner closure
        "Cm + E -> Cm + Ew",        # energy coupling
    ]
    extra_rates = rates[:6]
    extra_ics = {'Am': 1.0, 'Bm': 1.0, 'Cm': 1.0, 'Dm': 1.0}

    return _augment_network(base, extra_reactions, extra_rates, extra_ics, "T6_nested_loop")


def make_t1_random_control(
    p: EnzymeComplexParams, seed: int = 42,
    n_added_species: int = 4,
    n_added_reactions: int = 6,
    rng_seed: int = 0,
) -> GeneratedNetwork:
    """
    T1: Random-wiring control.
    Same number of added species and reactions as the largest Group B motif
    (T3/T4/T6: 4 species, 6 reactions), but with random connectivity.

    Random wiring protocol:
      - n_added_species new species (Rm0, Rm1, ...)
      - (n_added_reactions - 1) random directed edges among {new species + E}
      - 1 energy-coupling reaction: random_species + E -> random_species + Ew
    """
    base = make_enzyme_complex_network(p, seed=seed)
    rng = np.random.RandomState(rng_seed)

    new_sp_names = [f"Rm{i}" for i in range(n_added_species)]
    # Pool of species the random wiring can connect: new species only
    # (coupling to base system is only through E, per embedding protocol)
    pool = list(new_sp_names)

    extra_reactions = []
    extra_rates = []

    # Generate (n_added_reactions - 1) random conversion reactions among pool
    for _ in range(n_added_reactions - 1):
        src = rng.choice(pool)
        dst = rng.choice(pool)
        # Avoid self-loops
        while dst == src:
            dst = rng.choice(pool)
        extra_reactions.append(f"{src} -> {dst}")
        extra_rates.append(MOTIF_RATE_DEFAULT)

    # One energy-coupling reaction
    coupler = rng.choice(pool)
    extra_reactions.append(f"{coupler} + E -> {coupler} + Ew")
    extra_rates.append(MOTIF_RATE_DEFAULT)

    extra_ics = {sp: 1.0 for sp in new_sp_names}

    tag = f"T1_random_s{rng_seed}"
    return _augment_network(base, extra_reactions, extra_rates, extra_ics, tag)


# ══════════════════════════════════════════════════════════════════════
# Topology registry
# ══════════════════════════════════════════════════════════════════════

TOPOLOGY_BUILDERS = {
    'T0': make_t0_baseline,
    'T1a': lambda p, seed=42: make_t1_random_control(p, seed=seed, rng_seed=1001),
    'T1b': lambda p, seed=42: make_t1_random_control(p, seed=seed, rng_seed=2002),
    'T2': make_t2_simple_3cycle,
    'T3': make_t3_parallel_dual_path,
    'T4': make_t4_branch_rejoin,
    'T5': make_t5_feedforward_return,
    'T6': make_t6_nested_dual_loop,
}

TOPOLOGY_GROUPS = {
    'T0': 'A', 'T1a': 'A', 'T1b': 'A',
    'T2': 'B', 'T3': 'B', 'T4': 'B', 'T5': 'B', 'T6': 'B',
}

TOPOLOGY_DESCRIPTIONS = {
    'T0': 'Baseline coupled Brusselator (unmodified)',
    'T1a': 'Random-wiring control (seed 1001)',
    'T1b': 'Random-wiring control (seed 2002)',
    'T2': 'Simple 3-cycle (Am -> Bm -> Cm -> Am)',
    'T3': 'Parallel dual-path cycle (two routes through Am-Dm)',
    'T4': 'Branch-and-rejoin (asymmetric path weights)',
    'T5': 'Feed-forward with return (Am -> Cm short-circuit)',
    'T6': 'Nested dual loop (outer 3-cycle + inner 2-cycle at Bm)',
}


def get_d2_species(topology_name: str, net: GeneratedNetwork) -> List[str]:
    """
    Return the species to include in the D2 projection.
    Base system: [X1, Y1, X2, Y2, E].
    Augmented: base + all motif species (Am, Bm, Cm, Dm, Rm*).
    """
    base = list(BASE_D2_SPECIES)
    # Add any motif species (not chemostat, not waste)
    waste = {'D1', 'W1', 'D2', 'W2', 'Ew'}
    chemostat = set(net.chemostat_species.keys())
    gate = {'G', 'GE'}  # fast auxiliary, excluded from D2 per Paper 3/4
    exclude = waste | chemostat | gate | set(base)

    for sp in net.species:
        if sp not in exclude:
            base.append(sp)
    return base


def build_all_topologies(
    p: Optional[EnzymeComplexParams] = None,
    seed: int = 42,
) -> Dict[str, GeneratedNetwork]:
    """Build all 8 topologies with the given parameters."""
    if p is None:
        p = EnzymeComplexParams(
            J=BASELINE_J, gamma=BASELINE_GAMMA, k_cat=BASELINE_KCAT,
            k_on=10.0, k_off=10.0, G_total=1.0,
            label="baseline",
        )
    nets = {}
    for name, builder in TOPOLOGY_BUILDERS.items():
        nets[name] = builder(p, seed=seed)
    return nets


def compute_all_descriptors(
    nets: Dict[str, GeneratedNetwork],
) -> Dict[str, TopologyDescriptors]:
    """Compute structural descriptors for all topologies."""
    t0 = nets['T0']
    n_base_species = len([sp for sp in t0.species
                          if sp not in set(t0.chemostat_species.keys())])
    n_base_reactions = len(t0.reactions)

    descriptors = {}
    for name, net in nets.items():
        d2_sp = get_d2_species(name, net)
        n_dynamic = len([sp for sp in net.species
                         if sp not in set(net.chemostat_species.keys())])

        desc = TopologyDescriptors(
            name=name,
            group=TOPOLOGY_GROUPS[name],
            n_species=n_dynamic,
            n_reactions=len(net.reactions),
            n_added_species=n_dynamic - n_base_species if name != 'T0' else 0,
            n_added_reactions=len(net.reactions) - n_base_reactions if name != 'T0' else 0,
            cyclomatic_number=compute_cyclomatic_number(net.reactions),
            weakly_reversible=check_weak_reversibility(net.reactions),
            deficiency=compute_deficiency(net.reactions),
            d2_species=d2_sp,
        )
        descriptors[name] = desc
    return descriptors


# ══════════════════════════════════════════════════════════════════════
# Sanity check integrations
# ══════════════════════════════════════════════════════════════════════

def run_sanity_check(
    n_seeds: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Phase 0 validation: run n_seeds integrations per topology at baseline
    parameters. Check for numerical stability and basic dynamics.
    """
    p = EnzymeComplexParams(
        J=BASELINE_J, gamma=BASELINE_GAMMA, k_cat=BASELINE_KCAT,
        k_on=10.0, k_off=10.0, G_total=1.0,
        label="sanity",
    )

    nets = build_all_topologies(p)
    results = {}

    for topo_name in TOPOLOGY_BUILDERS:
        net = nets[topo_name]
        d2_species = get_d2_species(topo_name, net)
        topo_results = []

        for s in range(n_seeds):
            seed = 42 + s * 137
            t0_time = time.time()

            # Rebuild with this seed (affects IC perturbation)
            net_s = TOPOLOGY_BUILDERS[topo_name](p, seed=seed)

            sim = ReactionSimulator()
            status = 'ok'
            d2_val = float('nan')
            tau = 0

            try:
                network_obj = sim.build_network(net_s.reactions)
                sim_result = sim.simulate(
                    network_obj,
                    rate_constants=net_s.rate_constants,
                    initial_concentrations=net_s.initial_concentrations,
                    t_span=(0, 20000),
                    n_points=40000,
                    driving_mode=DrivingMode.CHEMOSTAT,
                    chemostat_species=net_s.chemostat_species,
                    max_step=10.0,
                )

                if not sim_result.success:
                    status = 'solver_fail'
                else:
                    c = sim_result.concentrations
                    species_names = sim_result.species_names

                    # Check for NaN/Inf
                    if not np.all(np.isfinite(c)):
                        status = 'nan_inf'
                    # Check for pathological concentrations
                    elif np.any(c > 1e6):
                        status = 'pathological'
                    else:
                        # Compute D2 on the appropriate projection
                        sp_indices = [species_names.index(sp) for sp in d2_species
                                      if sp in species_names]

                        # Sliding-window tau computation (Paper 4 protocol)
                        t_start_idx = 5000   # t=2500 at 2pts/unit
                        window_pts = 5000    # 2500 time units at 2pts/unit
                        n_windows = 7
                        tau = 0

                        cd = CorrelationDimension()
                        for w in range(n_windows):
                            w_start = t_start_idx + w * window_pts
                            w_end = w_start + window_pts
                            if w_end > len(c):
                                break
                            traj = c[w_start:w_end, :][:, sp_indices]
                            try:
                                d2_result = cd.compute(traj)
                                if d2_result.D2 is not None and d2_result.D2 > 1.2:
                                    tau += 1
                                if w == 0:
                                    d2_val = float(d2_result.D2) if d2_result.D2 else float('nan')
                            except Exception:
                                pass

            except Exception as e:
                status = f'exception: {str(e)[:80]}'

            elapsed = time.time() - t0_time
            run_result = {
                'seed': seed,
                'status': status,
                'tau_exp': tau,
                'd2_window0': d2_val,
                'elapsed_s': round(elapsed, 1),
            }
            topo_results.append(run_result)

            if verbose:
                d2_str = f"{d2_val:.3f}" if not np.isnan(d2_val) else "N/A"
                print(f"  {topo_name} seed={seed}: status={status}, "
                      f"tau={tau}, D2_w0={d2_str}, {elapsed:.1f}s")

        results[topo_name] = topo_results

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def print_structural_census(descriptors: Dict[str, TopologyDescriptors]):
    """Pretty-print the structural census table."""
    print("\n" + "=" * 100)
    print("PAPER 5 — PHASE 0: TOPOLOGY STRUCTURAL CENSUS")
    print("=" * 100)

    print(f"\n{'Name':<8} {'Group':<8} {'Species':>7} {'Rxns':>5} "
          f"{'Added Sp':>8} {'Added Rx':>8} "
          f"{'Cyclo':>6} {'WR':>4} {'Def':>4} "
          f"{'D2 dim':>6}")
    print("-" * 100)

    for name in ['T0', 'T1a', 'T1b', 'T2', 'T3', 'T4', 'T5', 'T6']:
        d = descriptors[name]
        wr_str = "yes" if d.weakly_reversible else "no"
        print(f"{d.name:<8} {d.group:<8} {d.n_species:>7} {d.n_reactions:>5} "
              f"{d.n_added_species:>8} {d.n_added_reactions:>8} "
              f"{d.cyclomatic_number:>6} {wr_str:>4} {d.deficiency:>4} "
              f"{len(d.d2_species):>6}")

    print("-" * 100)


def print_sanity_results(results: Dict):
    """Pretty-print sanity check results."""
    print("\n" + "=" * 100)
    print("PAPER 5 — PHASE 0: SANITY CHECK RESULTS")
    print("=" * 100)

    print(f"\n{'Topology':<8} {'OK':>4} {'Fail':>5} {'Mean tau':>8} "
          f"{'Max tau':>7} {'Mean D2_w0':>10} {'Mean time(s)':>12}")
    print("-" * 80)

    all_ok = True
    for name in ['T0', 'T1a', 'T1b', 'T2', 'T3', 'T4', 'T5', 'T6']:
        runs = results[name]
        n_ok = sum(1 for r in runs if r['status'] == 'ok')
        n_fail = len(runs) - n_ok
        taus = [r['tau_exp'] for r in runs if r['status'] == 'ok']
        d2s = [r['d2_window0'] for r in runs
               if r['status'] == 'ok' and not np.isnan(r['d2_window0'])]
        times = [r['elapsed_s'] for r in runs]

        mean_tau = np.mean(taus) if taus else 0
        max_tau = max(taus) if taus else 0
        mean_d2 = np.mean(d2s) if d2s else float('nan')
        mean_time = np.mean(times)

        d2_str = f"{mean_d2:.3f}" if not np.isnan(mean_d2) else "N/A"
        print(f"{name:<8} {n_ok:>4} {n_fail:>5} {mean_tau:>8.2f} "
              f"{max_tau:>7} {d2_str:>10} {mean_time:>12.1f}")

        if n_fail > 0:
            all_ok = False
            for r in runs:
                if r['status'] != 'ok':
                    print(f"  ** FAIL seed={r['seed']}: {r['status']}")

    print("-" * 80)
    if all_ok:
        print("VERDICT: All topologies numerically stable at baseline. Phase 0 PASSED.")
    else:
        print("WARNING: Some topologies had failures. Investigate before Phase 1.")
    print("=" * 100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Paper 5 Topology Library")
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run 10 sanity-check integrations per topology')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of seeds for sanity check')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    # Always print structural census
    print("Building all topologies...")
    nets = build_all_topologies()
    descriptors = compute_all_descriptors(nets)
    print_structural_census(descriptors)

    # Print reaction lists
    print("\n\nREACTION LISTS PER TOPOLOGY")
    print("=" * 100)
    for name in ['T0', 'T1a', 'T1b', 'T2', 'T3', 'T4', 'T5', 'T6']:
        net = nets[name]
        d2_sp = get_d2_species(name, net)
        print(f"\n{name}: {TOPOLOGY_DESCRIPTIONS[name]}")
        print(f"  D2 projection: {d2_sp}")
        for i, (rxn, rate) in enumerate(zip(net.reactions, net.rate_constants)):
            marker = " [+]" if i >= 14 and name != 'T0' else ""
            print(f"  {i:>2}. {rxn:<55} rate={rate:.3f}{marker}")

    if args.sanity_check:
        print("\n\nRunning sanity-check integrations...")
        results = run_sanity_check(n_seeds=args.n_seeds, verbose=True)
        print_sanity_results(results)

        if args.save:
            output = {
                'descriptors': {k: v.to_dict() for k, v in descriptors.items()},
                'sanity_check': results,
            }
            save_path = args.save
            with open(save_path, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
