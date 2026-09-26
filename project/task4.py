"""Task 4. Regular path queries for multiple start vertices."""

from networkx import MultiDiGraph
import numpy
from scipy import sparse

from project.task3 import AdjacencyMatrixFA, intersect_automata
from project.task2 import graph_to_nfa, regex_to_dfa


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    intersection = intersect_automata(graph_fa, regex_fa)
    regex_size = regex_fa.states_count

    graph_starts = sorted(graph_fa.start_indices)
    reachability_matrix = _build_reachability_matrix(
        graph_starts, regex_fa, intersection
    )

    final_indices = sorted(intersection.final_indices)
    final_states = [graph_fa.states[index // regex_size] for index in final_indices]
    rows, columns = reachability_matrix[:, final_indices].nonzero()

    return {
        (graph_fa.states[graph_starts[row]], final_states[column])
        for row, column in zip(rows, columns)
    }


def _build_reachability_matrix(
    graph_starts: list[int],
    regex_fa: AdjacencyMatrixFA,
    intersection: AdjacencyMatrixFA,
) -> sparse.csr_array:
    regex_size = regex_fa.states_count
    regex_starts = sorted(regex_fa.start_indices)

    columns = [
        graph_start * regex_size + regex_start
        for graph_start in graph_starts
        for regex_start in regex_starts
    ]
    rows = numpy.repeat(numpy.arange(len(graph_starts)), len(regex_starts))
    reach = sparse.csr_array(
        (numpy.ones(len(columns), dtype=bool), (rows, columns)),
        shape=(len(graph_starts), intersection.states_count),
    )

    adjacency = intersection.reflexive_adjacency_matrix()
    while True:
        new_reach = (reach @ adjacency) > reach
        if new_reach.nnz == 0:
            break
        reach += new_reach
    return reach
