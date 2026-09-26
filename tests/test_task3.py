"""Tests for task 3."""

import networkx as nx

from project.task2 import regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata, tensor_based_rpq

# testing utilities


def _build_graph(edges) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


# tests for AdjacencyMatrixFA


def test_adjacency_matrix_fa_from_dfa_accepts_language():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a b|b a"))

    assert automaton.accepts(["a", "b"])
    assert automaton.accepts(["b", "a"])
    assert not automaton.accepts(["a", "a"])


def test_adjacency_matrix_fa_is_empty_false():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a*"))

    assert not automaton.is_empty()


def test_adjacency_matrix_fa_is_empty_true():
    automaton = AdjacencyMatrixFA(regex_to_dfa(""))

    assert automaton.is_empty()


# tests for intersect_automata


def test_intersect_automata_accepts_common_words():
    dfa1 = AdjacencyMatrixFA(regex_to_dfa("a b c"))
    dfa2 = AdjacencyMatrixFA(regex_to_dfa("a (b | c) c"))
    intersection = intersect_automata(dfa1, dfa2)

    assert isinstance(intersection, AdjacencyMatrixFA)
    assert intersection.accepts(["a", "b", "c"])
    assert not intersection.accepts(["a", "c", "c"])
    assert not intersection.accepts(["a", "b"])


# tests for tensor_based_rpq


def test_tensor_based_rpq_single_edge():
    graph = _build_graph([(0, 1, "b")])
    assert tensor_based_rpq("b", graph, {0}, {1}) == {(0, 1)}


def test_tensor_based_rpq_with_star():
    graph = _build_graph([(0, 1, "b")])
    assert tensor_based_rpq("b*", graph, {0}, {0, 1}) == {(0, 0), (0, 1)}


def test_tensor_based_rpq_point_graph():
    graph = nx.MultiDiGraph()
    graph.add_node(1)
    assert tensor_based_rpq("a*", graph, {1}, {1}) == {(1, 1)}
