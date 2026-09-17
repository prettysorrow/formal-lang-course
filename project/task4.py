"""Task 4. Regular path queries for multiple start vertices."""

from networkx import MultiDiGraph
from scipy.sparse import csr_matrix, kron

from project.rpq import (
    build_transition_matrix,
    get_automaton_description,
    transitions_by_label,
)
from project.task2 import graph_to_nfa, regex_to_dfa


def _build_product_adjacency(graph_desc, regex_desc):
    graph_transitions_by_label = transitions_by_label(graph_desc)
    regex_transitions_by_label = transitions_by_label(regex_desc)
    product_size = graph_desc.num_states * regex_desc.num_states
    product_matrices = [
        kron(
            build_transition_matrix(
                graph_transitions_by_label[label], graph_desc.num_states
            ),
            build_transition_matrix(
                regex_transitions_by_label[label], regex_desc.num_states
            ),
        )
        for label in set(graph_transitions_by_label) & set(regex_transitions_by_label)
    ]
    if not product_matrices:
        return csr_matrix((product_size, product_size), dtype=bool)
    return sum(product_matrices[1:], start=product_matrices[0])


def _build_start_reachability(
    graph_start_states,
    regex_start_states,
    graph_state_index,
    regex_state_index,
    n_regex,
):
    starts = [
        (i, graph_state_index[graph_start_state] * n_regex + regex_state_index[regex_start])
        for i, graph_start_state in enumerate(graph_start_states)
        for regex_start in regex_start_states
    ]
    rows, cols = zip(*starts) if starts else ([], [])
    return csr_matrix(
        ([True] * len(rows), (rows, cols)),
        shape=(len(graph_start_states), len(graph_state_index) * n_regex),
        dtype=bool,
    )


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    graph_desc = get_automaton_description(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_desc = get_automaton_description(regex_to_dfa(regex))

    product_adj = _build_product_adjacency(graph_desc, regex_desc)

    reachability = _build_start_reachability(
        graph_desc.start_states,
        regex_desc.start_states,
        graph_desc.state_index,
        regex_desc.state_index,
        regex_desc.num_states,
    )

    while True:
        next_reachability = reachability.maximum(reachability @ product_adj)
        if (next_reachability != reachability).nnz == 0:
            break
        reachability = next_reachability

    result = set()
    for start_idx, graph_start_state in enumerate(graph_desc.start_states):
        for graph_final_state in graph_desc.final_states:
            for regex_final_state in regex_desc.final_states:
                final_idx = (
                    graph_desc.state_index[graph_final_state] * regex_desc.num_states
                    + regex_desc.state_index[regex_final_state]
                )
                if reachability[start_idx, final_idx]:
                    result.add((graph_start_state, graph_final_state))

    return result