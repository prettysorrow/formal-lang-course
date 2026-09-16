"""Task 4. Regular path queries for multiple start vertices."""

from networkx import MultiDiGraph
from scipy.sparse import csr_matrix

from project.rpq import get_automaton_description, transitions_by_label
from project.task2 import graph_to_nfa, regex_to_dfa


def _build_product_adjacency(
    graph_edges_by_label,
    regex_edges_by_label,
    n_graph,
    n_regex,
):
    product_size = n_graph * n_regex
    rows, cols = [], []
    for label in set(graph_edges_by_label) & set(regex_edges_by_label):
        for g_from, g_to in graph_edges_by_label[label]:
            for r_from, r_to in regex_edges_by_label[label]:
                rows.append(g_from * n_regex + r_from)
                cols.append(g_to * n_regex + r_to)
    return csr_matrix(
        ([True] * len(rows), (rows, cols)),
        shape=(product_size, product_size),
        dtype=bool,
    )


def _build_start_reachability(
    graph_start_states,
    regex_start_states,
    graph_state_index,
    regex_state_index,
    n_regex,
):
    rows = [graph_start_state for graph_start_state, _ in enumerate(graph_start_states)]
    cols = [
        graph_state_index[graph_start_state] * n_regex
        + regex_state_index[regex_start_state]
        for graph_start_state in graph_start_states
        for regex_start_state in regex_start_states
    ]
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
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    regex_dfa = regex_to_dfa(regex)

    graph_desc = get_automaton_description(graph_nfa)
    regex_desc = get_automaton_description(regex_dfa)

    product_adj = _build_product_adjacency(
        transitions_by_label(graph_nfa, graph_desc.state_index),
        transitions_by_label(regex_dfa, regex_desc.state_index),
        len(graph_desc.states),
        len(regex_desc.states),
    )

    reachability = _build_start_reachability(
        graph_desc.start_states,
        regex_desc.start_states,
        graph_desc.state_index,
        regex_desc.state_index,
        len(regex_desc.states),
    )

    while True:
        next_reachability = (
            reachability + (reachability @ product_adj).astype(bool)
        ).astype(bool)
        if (next_reachability != reachability).nnz == 0:
            break
        reachability = next_reachability

    result = set()
    for start_idx, graph_start_state in enumerate(graph_desc.start_states):
        for graph_final_state in graph_desc.final_states:
            for regex_final_state in regex_desc.final_states:
                final_idx = (
                    graph_desc.state_index[graph_final_state]
                    * len(regex_desc.states)
                    + regex_desc.state_index[regex_final_state]
                )
                if reachability[start_idx, final_idx]:
                    result.add((graph_start_state, graph_final_state))

    return result