"""Tests for task 7."""

import networkx as nx
from pyformlang.cfg import CFG

from project.task6 import hellings_based_cfpq
from project.task7 import matrix_based_cfpq


def _build_graph(edges) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


def test_matrix_terminal_production():
    graph = _build_graph([(0, 1, "b")])
    assert matrix_based_cfpq(CFG.from_text("S -> b"), graph, {0}, {1}) == {(0, 1)}


def test_matrix_nullable_start():
    graph = _build_graph([(0, 1, "b")])
    assert matrix_based_cfpq(CFG.from_text("S -> S b | $"), graph, {0}, {0, 1}) == {
        (0, 0),
        (0, 1),
    }


def test_matrix_nested_path():
    graph = _build_graph([(0, 1, "a"), (1, 2, "b")])
    assert matrix_based_cfpq(CFG.from_text("S -> a S b | $"), graph, {0}, {2}) == {
        (0, 2)
    }


def test_matrix_default_start_final_all_nodes():
    graph = _build_graph([(0, 1, "a")])
    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph) == {(0, 1)}


def test_matrix_empty_start_nodes_default_to_all():
    graph = _build_graph([(0, 1, "a")])
    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph, set(), {1}) == {(0, 1)}


def test_matrix_matches_hellings():
    graph = _build_graph([(0, 1, "a"), (1, 2, "b"), (2, 0, "a"), (0, 2, "b")])
    cfg = CFG.from_text("S -> A S B | A B | B\nA -> a\nB -> b")
    assert matrix_based_cfpq(cfg, graph) == hellings_based_cfpq(cfg, graph)


def test_matrix_epsilon_on_isolated_node():
    graph = nx.MultiDiGraph()
    graph.add_node(0)
    assert matrix_based_cfpq(CFG.from_text("S -> S b | $"), graph, {0}, {0}) == {(0, 0)}


def test_matrix_left_recursion():
    graph = _build_graph([(0, 1, "b"), (1, 2, "b"), (2, 3, "b")])
    assert matrix_based_cfpq(CFG.from_text("S -> S b | b"), graph, {0}, {3}) == {(0, 3)}


def test_matrix_single_edge():
    graph = _build_graph([(0, 1, "a")])
    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph) == {(0, 1)}


def test_matrix_no_matching_edge():
    graph = _build_graph([(0, 1, "a")])
    assert matrix_based_cfpq(CFG.from_text("S -> b"), graph) == set()


def test_matrix_concatenation():
    graph = _build_graph([(0, 1, "a"), (1, 2, "b")])
    assert matrix_based_cfpq(CFG.from_text("S -> A B\nA -> a\nB -> b"), graph) == {
        (0, 2)
    }


def test_matrix_self_loop():
    graph = _build_graph([(0, 0, "a")])
    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph, {0}, {0}) == {(0, 0)}


def test_matrix_empty_graph():
    graph = nx.MultiDiGraph()
    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph) == set()
