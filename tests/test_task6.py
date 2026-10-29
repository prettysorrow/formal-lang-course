"""Tests for task 6."""

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Terminal, Variable

from project.task6 import cfg_to_weak_normal_form, hellings_based_cfpq


def _build_graph(edges) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


# positive tests


@pytest.mark.parametrize(
    "grammar",
    [
        pytest.param("S -> a", id="single_terminal"),
        pytest.param("S -> a S | a", id="alternative"),
        pytest.param("S -> a S | $", id="epsilon"),
        pytest.param(
            "S -> A | B | C\nA -> a\nB -> b\nC -> c", id="multiple_productions"
        ),
    ],
)
def test_cfg_to_weak_normal_form_positive(grammar):
    normal = cfg_to_weak_normal_form(CFG.from_text(grammar))
    for production in normal.productions:
        body = production.body
        is_epsilon = len(body) == 0
        is_terminal = len(body) == 1 and isinstance(body[0], Terminal)
        is_vars = (
            len(body) == 2
            and isinstance(body[0], Variable)
            and isinstance(body[1], Variable)
        )
        assert is_epsilon or is_terminal or is_vars


@pytest.mark.parametrize(
    "grammar, edges, start_nodes, final_nodes, expected",
    [
        pytest.param("S -> a", [(0, 1, "a")], {0}, {1}, {(0, 1)}, id="single_terminal"),
        pytest.param(
            "S -> a S | a",
            [(0, 1, "a"), (1, 2, "a"), (2, 3, "a")],
            {0},
            {1, 2, 3},
            {(0, 1), (0, 2), (0, 3)},
            id="alternative",
        ),
        pytest.param(
            "S -> a S | $",
            [(0, 1, "a"), (1, 2, "a")],
            {0},
            {0, 1, 2},
            {(0, 0), (0, 1), (0, 2)},
            id="epsilon",
        ),
        pytest.param(
            "S -> A | B | C\nA -> a\nB -> b\nC -> c",
            [(0, 1, "a"), (0, 2, "b"), (0, 3, "c")],
            {0},
            {1, 2, 3},
            {(0, 1), (0, 2), (0, 3)},
            id="multiple_productions",
        ),
    ],
)
def test_hellings_based_cfpq_positive(
    grammar, edges, start_nodes, final_nodes, expected
):
    cfg = CFG.from_text(grammar)
    graph = _build_graph(edges)
    result = hellings_based_cfpq(cfg, graph, start_nodes, final_nodes)
    assert result == expected


# negative tests


@pytest.mark.parametrize(
    "cfg, expected_error",
    [
        pytest.param(None, AttributeError, id="none"),
        pytest.param("S -> a", AttributeError, id="string"),
        pytest.param(42, AttributeError, id="int"),
    ],
)
def test_cfg_to_weak_normal_form_negative(cfg, expected_error):
    with pytest.raises(expected_error):
        cfg_to_weak_normal_form(cfg)


_valid_cfg = CFG.from_text("S -> a")
_valid_graph = _build_graph([(0, 1, "a")])


@pytest.mark.parametrize(
    "cfg, graph, expected_error",
    [
        pytest.param(None, _valid_graph, AttributeError, id="none_cfg"),
        pytest.param(_valid_cfg, None, AttributeError, id="none_graph"),
        pytest.param("S -> a", _valid_graph, AttributeError, id="string_cfg"),
        pytest.param(_valid_cfg, "1 -> 2", AttributeError, id="string_graph"),
    ],
)
def test_hellings_based_cfpq_negative(cfg, graph, expected_error):
    with pytest.raises(expected_error):
        hellings_based_cfpq(cfg, graph, {0}, {1})
