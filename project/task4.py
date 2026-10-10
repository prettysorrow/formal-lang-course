"""Task 4. Regular path queries for multiple start vertices."""

from networkx import MultiDiGraph
from scipy import sparse

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    graph_starts = sorted(graph_fa.start_indices)
    regex_transitions = _build_regex_transitions(graph_fa, regex_fa)

    front_shape = (regex_fa.states_count, graph_fa.states_count)
    fronts = _build_initial_fronts(graph_starts, regex_fa, front_shape)
    visited = [front.copy() for front in fronts]

    while any(front.nnz for front in fronts):
        for idx, front in enumerate(fronts):
            if not front.nnz:
                continue
            fronts[idx] = _build_next_front(
                front,
                visited[idx],
                graph_fa,
                regex_transitions,
                front_shape,
            )
            visited[idx] += fronts[idx]

    return _collect_reachable_pairs(visited, graph_fa, regex_fa, graph_starts)


def _build_regex_transitions(
    graph_fa: AdjacencyMatrixFA,
    regex_fa: AdjacencyMatrixFA,
) -> dict[str, sparse.csr_array]:
    symbols = set(graph_fa.transition_matrices) & set(regex_fa.transition_matrices)
    return {
        symbol: regex_fa.transition_matrices[symbol].transpose().tocsr()
        for symbol in symbols
    }


def _build_initial_fronts(
    graph_starts: list[int],
    regex_fa: AdjacencyMatrixFA,
    front_shape: tuple[int, int],
) -> list[sparse.csr_array]:
    (regex_start,) = regex_fa.start_indices
    return [
        sparse.csr_array(
            ([True], ([regex_start], [graph_start])),
            shape=front_shape,
            dtype=bool,
        )
        for graph_start in graph_starts
    ]


def _build_next_front(
    front: sparse.csr_array,
    visited: sparse.csr_array,
    graph_fa: AdjacencyMatrixFA,
    regex_transitions: dict[str, sparse.csr_array],
    front_shape: tuple[int, int],
) -> sparse.csr_array:
    reached = sparse.csr_array(front_shape, dtype=bool)
    for symbol, regex_transition in regex_transitions.items():
        reached += regex_transition @ front @ graph_fa.transition_matrices[symbol]
    return reached > visited


def _collect_reachable_pairs(
    visited: list[sparse.csr_array],
    graph_fa: AdjacencyMatrixFA,
    regex_fa: AdjacencyMatrixFA,
    graph_starts: list[int],
) -> set[tuple[int, int]]:
    result = set()
    for graph_start, visited_front in zip(graph_starts, visited):
        rows, cols = visited_front.nonzero()
        for regex_state, graph_state in zip(rows, cols):
            if regex_state not in regex_fa.final_indices:
                continue
            if graph_state not in graph_fa.final_indices:
                continue
            start_vertex = graph_fa.states[graph_start]
            final_vertex = graph_fa.states[graph_state]
            result.add((start_vertex, final_vertex))
    return result
