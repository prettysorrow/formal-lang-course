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


def test_hellings_default_start_final_all_nodes():
    graph = _build_graph([(0, 1, "a")])
    assert hellings_based_cfpq(CFG.from_text("S -> a"), graph) == {(0, 1)}


def test_hellings_empty_start_nodes_default_to_all():
    graph = _build_graph([(0, 1, "a")])
    assert hellings_based_cfpq(CFG.from_text("S -> a"), graph, set(), {1}) == {(0, 1)}


def test_hellings_start_subset_of_nodes():
    graph = _build_graph([(1, 2, "a"), (2, 3, "a")])
    assert hellings_based_cfpq(CFG.from_text("S -> a S | a"), graph, {1}, {3}) == {
        (1, 3)
    }


def test_hellings_self_loop():
    graph = _build_graph([(0, 0, "a")])
    assert hellings_based_cfpq(CFG.from_text("S -> a"), graph, {0}, {0}) == {(0, 0)}


def test_hellings_long_chain():
    graph = _build_graph([(0, 1, "a"), (1, 2, "b"), (2, 3, "c")])
    assert hellings_based_cfpq(CFG.from_text("S -> a b c"), graph, {0}, {3}) == {(0, 3)}


def test_hellings_diamond_same_result():
    graph = _build_graph([(0, 1, "a"), (0, 2, "a"), (1, 3, "b"), (2, 3, "b")])
    assert hellings_based_cfpq(CFG.from_text("S -> a b"), graph, {0}, {3}) == {(0, 3)}


def test_hellings_sibling_nonterminals():
    graph = _build_graph([(0, 1, "a"), (1, 2, "b")])
    assert hellings_based_cfpq(
        CFG.from_text("S -> A B | A | B\nA -> a\nB -> b"), graph, {0}, {2}
    ) == {(0, 2)}


def test_hellings_left_recursion():
    graph = _build_graph([(0, 1, "b"), (1, 2, "b"), (2, 3, "b")])
    assert hellings_based_cfpq(CFG.from_text("S -> S b | b"), graph, {0}, {3}) == {
        (0, 3)
    }


def test_hellings_epsilon_on_isolated_node():
    graph = nx.MultiDiGraph()
    graph.add_node(0)
    assert hellings_based_cfpq(CFG.from_text("S -> S b | $"), graph, {0}, {0}) == {
        (0, 0)
    }


def test_cfg_to_weak_normal_form_preserves_language():
    original = CFG.from_text("S -> a S b | a b")
    normal = cfg_to_weak_normal_form(original)
    assert normal.contains("ab")
    assert normal.contains("aaabbb")
    assert not normal.contains("aab")
