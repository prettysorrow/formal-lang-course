"""Tests for task 8."""

import networkx
import pytest
from pyformlang.cfg import CFG

from project.task7 import matrix_based_cfpq
from project.task8 import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


def _build_graph(edges) -> networkx.MultiDiGraph:
    graph = networkx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


@pytest.mark.parametrize(
    "cfg, ebnf, edges, starts, finals",
    [
        pytest.param(
            "S -> a",
            "S -> a",
            [(0, 1, "a")],
            {0},
            {1},
            id="single_terminal",
        ),
        pytest.param(
            "S -> a S | a",
            "S -> a S | a",
            [(0, 1, "a"), (1, 2, "a"), (2, 3, "a")],
            {0},
            {1, 2, 3},
            id="alternative",
        ),
        pytest.param(
            "S -> a S | $",
            "S -> a*",
            [(0, 1, "a"), (1, 2, "a")],
            {0},
            {0, 1, 2},
            id="epsilon",
        ),
        pytest.param(
            "S -> A | B | C\nA -> a\nB -> b\nC -> c",
            "S -> a | b | c",
            [(0, 1, "a"), (0, 2, "b"), (0, 3, "c")],
            {0},
            {1, 2, 3},
            id="multiple_productions",
        ),
        pytest.param(
            "S -> S a | $",
            "S -> a*",
            [(0, 1, "a"), (1, 2, "a"), (2, 0, "a")],
            {0, 1},
            {0, 1, 2},
            id="cycle",
        ),
        pytest.param(
            "S -> a S b | $",
            "S -> a S b | $",
            [(0, 1, "a"), (1, 2, "a"), (2, 3, "b"), (3, 4, "b")],
            {0},
            {0, 2, 4},
            id="nested",
        ),
    ],
)
def test_tensor_based_cfpq_matches_matrix(cfg, ebnf, edges, starts, finals):
    graph = _build_graph(edges)
    cfg = CFG.from_text(cfg)

    matrix_ans = matrix_based_cfpq(cfg, graph, starts, finals)
    tensor_cfg_ans = tensor_based_cfpq(cfg_to_rsm(cfg), graph, starts, finals)
    tensor_ebnf_ans = tensor_based_cfpq(ebnf_to_rsm(ebnf), graph, starts, finals)

    assert tensor_cfg_ans == matrix_ans
    assert tensor_ebnf_ans == matrix_ans
