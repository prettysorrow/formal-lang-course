"""Task 7. Matrix multiplication based CFPQ algorithm."""

from networkx import MultiDiGraph
from pyformlang.cfg import CFG
from scipy import sparse

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA
from project.task6 import cfg_to_weak_normal_form, partition_productions


def matrix_based_cfpq(
    cfg: CFG,
    graph: MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    ncfg = cfg_to_weak_normal_form(cfg)
    productions = partition_productions(ncfg)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    num_states = graph_fa.states_count

    # M_X[u, v] is True iff X derives the word on some path (u -> v).
    derivations = {
        X: sparse.csr_array((num_states, num_states), dtype=bool)
        for X in ncfg.variables
    }

    # (X -> epsilon)  ===>  M_X[v, v] is True for every v
    for X in productions.epsilon:
        derivations[X] = sparse.eye_array(num_states, dtype=bool, format="csr")

    # (X -> a) and (u -a-> v)  ===>  M_X[u, v] is True
    for a, heads in productions.terminal.items():
        transitions = graph_fa.transition_matrices.get(a)
        if transitions is None:
            continue
        for X in heads:
            derivations[X] = (derivations[X] + transitions).astype(bool)

    # For each rule (X -> YZ)
    # (M_Y[u, w] is True) and (M_Z[w, v] is True)  ===>  M_X[u, v] is True
    changed = True
    while changed:
        changed = False
        for X, Y, Z in productions.binary:
            before = derivations[X].nnz
            derivations[X] += (derivations[Y] @ derivations[Z]).astype(bool)
            after = derivations[X].nnz
            if after != before:
                changed = True

    # (M_S[u, v] is True) where (S is the start symbol) and (u is a start state) and (v is a final state)  ===>  (u, v)
    S = ncfg.start_symbol
    rows, columns = derivations[S].nonzero()
    return {
        (graph_fa.states[u], graph_fa.states[v])
        for u, v in zip(rows, columns)
        if u in graph_fa.start_indices and v in graph_fa.final_indices
    }
