"""Tests for task 4."""

import pytest
import networkx as nx

from project.task4 import ms_bfs_based_rpq


def _build_graph(edges) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


# positive tests for ms_bfs_based_rpq


def test_ms_bfs_based_rpq_single_edge():
    graph = _build_graph([(0, 1, "b")])
    assert ms_bfs_based_rpq("b", graph, {0}, {1}) == {(0, 1)}


def test_ms_bfs_based_rpq_with_star():
    graph = _build_graph([(0, 1, "b")])
    assert ms_bfs_based_rpq("b*", graph, {0}, {0, 1}) == {(0, 0), (0, 1)}


def test_ms_bfs_based_rpq_matches_tensor_based():
    task3 = pytest.importorskip("project.task3")

    graph = _build_graph([(0, 1, "a"), (1, 2, "b"), (2, 0, "a")])
    regex = "(a | b)*"
    start = {0}
    final = {1, 2}
    assert ms_bfs_based_rpq(regex, graph, start, final) == task3.tensor_based_rpq(
        regex, graph, start, final
    )
