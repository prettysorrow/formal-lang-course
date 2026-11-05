"""Tests for task 7."""

import networkx
import pytest
from pyformlang.cfg import CFG

from project.task6 import hellings_based_cfpq
from project.task7 import matrix_based_cfpq


def _build_graph(edges) -> networkx.MultiDiGraph:
    graph = networkx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


@pytest.mark.parametrize(
    "grammar, edges, start_nodes, final_nodes",
    [
        pytest.param("S -> a", [(0, 1, "a")], {0}, {1}, id="single_terminal"),
        pytest.param(
            "S -> a S | a",
            [(0, 1, "a"), (1, 2, "a"), (2, 3, "a")],
            {0},
            {1, 2, 3},
            id="alternative",
        ),
        pytest.param(
            "S -> a S | $",
            [(0, 1, "a"), (1, 2, "a")],
            {0},
            {0, 1, 2},
            id="epsilon",
        ),
        pytest.param(
            "S -> A | B | C\nA -> a\nB -> b\nC -> c",
            [(0, 1, "a"), (0, 2, "b"), (0, 3, "c")],
            {0},
            {1, 2, 3},
            id="multiple_productions",
        ),
    ],
)
def test_matrix_based_cfpq(grammar, edges, start_nodes, final_nodes):
    graph = _build_graph(edges)
    cfg = CFG.from_text(grammar)
    matrix_ans = matrix_based_cfpq(cfg, graph, start_nodes, final_nodes)
    hellings_ans = hellings_based_cfpq(cfg, graph, start_nodes, final_nodes)
    assert matrix_ans == hellings_ans
