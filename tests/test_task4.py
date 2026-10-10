"""Tests for task 4."""

import networkx
import pytest
from pyformlang.regular_expression import MisformedRegexError

from project.task3 import tensor_based_rpq
from project.task4 import ms_bfs_based_rpq


def _build_graph(edges) -> networkx.MultiDiGraph:
    graph = networkx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


_single_edge = [(0, 1, "b")]

_cycle_edges = [(0, 1, "x"), (1, 2, "y"), (2, 3, "x"), (3, 0, "y"), (1, 3, "y")]

_branch_edges = [(0, 1, "x"), (1, 2, "y"), (2, 1, "x"), (1, 3, "y")]


# positive tests
@pytest.mark.parametrize(
    "edges, regex, start, final",
    [
        (_single_edge, "b", {0}, {1}),
        (_single_edge, "b*", {0}, {1}),
        (_single_edge, "b*", {0}, {0, 1}),
        (_cycle_edges, "x y", {0, 1, 2, 3}, {0, 1, 2, 3}),
        (_cycle_edges, "x y", {0}, {1, 2, 3}),
        (_cycle_edges, "(x | y)*", {0, 2}, {1, 2, 3}),
        (_cycle_edges, "(x | y)* y x", {0, 1}, {1, 3}),
        (_branch_edges, "(x | y)*", {0, 2}, {1, 2, 3}),
        (_branch_edges, "x y", {0}, {3}),
        (_branch_edges, "y+", {1}, {2, 3}),
    ],
    ids=[
        "single_edge",
        "star_all",
        "star_subset",
        "cycle_xy_all",
        "cycle_xy_one_start",
        "cycle_alt_and_star",
        "cycle_alt_and_star_and_concat",
        "branch_alt_and_star",
        "branch_concat",
        "branch_plus",
    ],
)
def test_ms_bfs_based_rpq_matches_tensor_based(edges, regex, start, final):
    graph = _build_graph(edges)
    assert ms_bfs_based_rpq(regex, graph, start, final) == tensor_based_rpq(
        regex, graph, start, final
    )


# negative tests
@pytest.mark.parametrize(
    "edges, regex, expected_error",
    [
        (None, "b", AttributeError),
        (_single_edge, None, AttributeError),
        (_single_edge, "(((", MisformedRegexError),
    ],
    ids=["none_regex", "none_graph", "malformed_regex"],
)
def test_ms_bfs_based_rpq_raises(edges, regex, expected_error):
    graph = _build_graph(edges) if edges is not None else None

    with pytest.raises(expected_error):
        ms_bfs_based_rpq(regex, graph, {0}, {1})
