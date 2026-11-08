"""Task 8. Tensor-product based CFPQ algorithm."""

from networkx import DiGraph
from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
from scipy import sparse

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    graph_size = graph_fa.states_count
    nonterminals = set(rsm.labels)

    relations = {
        label: sparse.csr_array((graph_size, graph_size), dtype=bool)
        for label in nonterminals
    }

    changed = True
    while changed:
        changed = False
        for label, box in rsm.boxes.items():
            fa = AdjacencyMatrixFA(box.dfa)
            fa_size = fa.states_count
            adj_size = graph_size * fa_size
            adjacency = sparse.csr_array((adj_size, adj_size), dtype=bool)
            for symbol, matrix in fa.transition_matrices.items():
                if symbol in nonterminals:
                    relation = relations[symbol]
                elif symbol in graph_fa.transition_matrices:
                    relation = graph_fa.transition_matrices[symbol]
                else:
                    continue

                adjacency += sparse.kron(relation, matrix, format="csr")

            closure = sparse.eye_array(adj_size, dtype=bool, format="csr")
            closure += adjacency
            previous_nnz = -1
            while closure.nnz != previous_nnz:
                previous_nnz = closure.nnz
                closure @= closure

            relation = sparse.csr_array((graph_size, graph_size), dtype=bool)
            for start in fa.start_indices:
                for final in fa.final_indices:
                    relation += closure[start::fa_size, final::fa_size]

            previous_nnz = relations[label].nnz
            relations[label] = (relations[label] + relation).astype(bool)
            if relations[label].nnz != previous_nnz:
                changed = True

    start_symbol = rsm.initial_label.value
    rows, cols = relations[start_symbol].nonzero()
    return {
        (graph_fa.states[u], graph_fa.states[v])
        for u, v in zip(rows, cols)
        if u in graph_fa.start_indices and v in graph_fa.final_indices
    }


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)
