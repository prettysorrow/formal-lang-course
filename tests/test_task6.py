"""Tests for task 6."""

import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable

from project.task6 import cfg_to_weak_normal_form, hellings_based_cfpq


def _build_graph(edges) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


def test_cfg_to_weak_normal_form_shape():
    normal = cfg_to_weak_normal_form(CFG.from_text("S -> a b c | a"))
    for production in normal.productions:
        body = production.body
        assert len(body) <= 2
        assert not (
            len(body) == 2 and any(isinstance(symbol, Terminal) for symbol in body)
        )
        assert not (len(body) == 1 and isinstance(body[0], Variable))


def test_cfg_to_weak_normal_form_keeps_epsilon():
    normal = cfg_to_weak_normal_form(CFG.from_text("S -> $"))
    assert any(len(production.body) == 0 for production in normal.productions)


def test_hellings_terminal_production():
    graph = _build_graph([(0, 1, "b")])
    assert hellings_based_cfpq(CFG.from_text("S -> b"), graph, {0}, {1}) == {(0, 1)}


def test_hellings_nullable_start():
    graph = _build_graph([(0, 1, "b")])
    assert hellings_based_cfpq(CFG.from_text("S -> S b | $"), graph, {0}, {0, 1}) == {
        (0, 0),
        (0, 1),
    }


def test_hellings_nested_path():
    graph = _build_graph([(0, 1, "a"), (1, 2, "b")])
    assert hellings_based_cfpq(CFG.from_text("S -> a S b | $"), graph, {0}, {2}) == {
        (0, 2)
    }
