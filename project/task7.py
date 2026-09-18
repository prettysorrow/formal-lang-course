"""Task 7. Matrix algorithm for context-free path querying."""

from networkx import MultiDiGraph
from pyformlang.cfg import CFG
from scipy.sparse import csr_matrix, identity

from project.cfg import partition_productions
from project.rpq import build_transition_matrix, get_automaton_description
from project.task2 import graph_to_nfa
from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """Solve the context-free reachability problem with the boolean matrix algorithm."""
    normalized_cfg = cfg_to_weak_normal_form(cfg)
    productions = partition_productions(normalized_cfg)
    fa_desc = get_automaton_description(graph_to_nfa(graph, start_nodes, final_nodes))
    num_states = fa_desc.num_states

    # `M[X][u, v]` is True iff `X` derives the word on some path (u -> v).
    zero = csr_matrix((num_states, num_states), dtype=bool)
    derivations = {X: zero for X in normalized_cfg.variables}

    # (X -> epsilon)  ===>  M_X[v, v] = True for every v
    for X in productions.epsilon:
        derivations[X] = identity(num_states, dtype=bool, format="csr")

    # (X -> a) and (u -a-> v)  ===>  M_X[u, v] = True
    for a, heads in productions.terminal.items():
        matrix_of_a = build_transition_matrix(fa_desc, str(a))
        for X in heads:
            derivations[X] = (derivations[X] + matrix_of_a).astype(bool)

    # for each X -> YZ
    # (Y, u, w) and (Z, w, v)  ===>  (X, u, v)
    changed = True
    while changed:
        changed = False
        for X, Y, Z in productions.binary:
            # `(M_Y @ M_Z)[u, v] = True` iff exists `w` such as `M_Y[u, w] = True` and `M_Z[w, v] = True`
            # `M_X += ...` is logic OR of derivations from `X` and right hand side
            before = derivations[X].nnz
            derivations[X] += (derivations[Y] @ derivations[Z]).astype(bool)
            after = derivations[X].nnz
            if after != before:
                changed = True

    # `M_S[u, v] = True` where `S` is start symbol means the word is derived on path (u -> v).
    # Keep only pairs (u, v) with (u is start state) and (v is final state).
    S = normalized_cfg.start_symbol
    derived_from_start_symbol = zip(*derivations[S].nonzero())
    return {
        (fa_desc.states[u], fa_desc.states[v])
        for u, v in derived_from_start_symbol
        if fa_desc.states[u] in fa_desc.start_states
        and fa_desc.states[v] in fa_desc.final_states
    }
