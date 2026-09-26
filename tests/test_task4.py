"""Tests for task 4."""

import networkx

from project.task3 import tensor_based_rpq
from project.task4 import ms_bfs_based_rpq


def _build_graph(edges) -> networkx.MultiDiGraph:
    graph = networkx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


# positive tests for ms_bfs_based_rpq


def test_ms_bfs_based_rpq_single_edge():
    graph = _build_graph([(0, 1, "b")])
    assert ms_bfs_based_rpq("b", graph, {0}, {1}) == {(0, 1)}


def test_ms_bfs_based_rpq_star():
    graph = _build_graph([(0, 1, "b")])
    assert ms_bfs_based_rpq("b*", graph, {0}, {0, 1}) == {(0, 0), (0, 1)}


def test_ms_bfs_based_rpq_concat():
    graph = _build_graph(
        [
            (0, 1, "p"),
            (1, 2, "q"),
            (0, 3, "p"),
            (3, 4, "r"),
            (2, 5, "p"),
            (4, 5, "p"),
            (5, 1, "p"),
        ]
    )
    answer = ms_bfs_based_rpq("p q", graph, {0}, {0, 1, 2, 3, 4, 5})

    assert answer == {(0, 2)}


def test_ms_bfs_based_rpq_concat_and_alt():
    graph = _build_graph(
        [
            (0, 1, "p"),
            (1, 2, "q"),
            (0, 3, "p"),
            (3, 4, "r"),
            (2, 5, "p"),
            (4, 5, "p"),
            (5, 1, "p"),
        ]
    )
    answer = ms_bfs_based_rpq("p (q | r)", graph, {0}, {0, 1, 2, 3, 4, 5})

    assert answer == {(0, 2), (0, 4)}


def test_ms_bfs_based_rpq_concat_and_alt_and_star():
    graph = _build_graph(
        [
            (0, 1, "p"),
            (1, 2, "q"),
            (0, 3, "p"),
            (3, 4, "r"),
            (2, 5, "p"),
            (4, 5, "p"),
            (5, 1, "p"),
        ]
    )
    answer = ms_bfs_based_rpq("(p | q | r)* p", graph, {0}, {0, 1, 2, 3, 4, 5})

    assert answer == {(0, 1), (0, 3), (0, 5)}


# tests comparing task 3 and task 4
def test_ms_bfs_based_rpq_matches_tensor_based_on_concat():
    graph = _build_graph(
        [(0, 1, "x"), (1, 2, "y"), (2, 3, "x"), (3, 0, "y"), (1, 3, "y")]
    )
    regex = "x y"
    nodes = set(graph.nodes)
    assert ms_bfs_based_rpq(regex, graph, nodes, nodes) == tensor_based_rpq(
        regex, graph, nodes, nodes
    )


def test_ms_bfs_based_rpq_matches_tensor_based_on_alt_and_star():
    graph = _build_graph([(0, 1, "x"), (1, 2, "y"), (2, 1, "x"), (1, 3, "y")])
    regex = "(x | y)*"
    start = {0, 2}
    final = {1, 2, 3}
    assert ms_bfs_based_rpq(regex, graph, start, final) == tensor_based_rpq(
        regex, graph, start, final
    )
