"""Tests for task 3."""

import networkx
import pytest
from pyformlang.regular_expression import MisformedRegexError

from project.task2 import regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata, tensor_based_rpq

# testing utilities


def _build_graph(edges) -> networkx.MultiDiGraph:
    graph = networkx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


# tests for accepts


def test_adjacency_matrix_fa_from_dfa_accepts_alt():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a b|b a"))

    assert automaton.accepts(["a", "b"])
    assert automaton.accepts(["b", "a"])
    assert not automaton.accepts(["a", "a"])


def test_adjacency_matrix_fa_from_dfa_accepts_star():
    automaton = AdjacencyMatrixFA(regex_to_dfa("(a b)*"))

    assert automaton.accepts([])
    assert automaton.accepts(["a", "b", "a", "b"])
    assert not automaton.accepts(["b"])
    assert not automaton.accepts(["b", "a"])
    assert not automaton.accepts(["a", "b", "a"])


# tests for is_empty


def test_adjacency_matrix_fa_is_empty_false():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a*"))

    assert not automaton.is_empty()


def test_adjacency_matrix_fa_is_empty_true():
    automaton = AdjacencyMatrixFA(regex_to_dfa(""))

    assert automaton.is_empty()


# tests for intersect_automata


def test_intersect_automata_alt():
    dfa1 = AdjacencyMatrixFA(regex_to_dfa("a b c"))
    dfa2 = AdjacencyMatrixFA(regex_to_dfa("a (b | c) c"))
    intersection = intersect_automata(dfa1, dfa2)

    assert isinstance(intersection, AdjacencyMatrixFA)
    assert intersection.accepts(["a", "b", "c"])
    assert not intersection.accepts(["a", "c", "c"])
    assert not intersection.accepts(["a", "b"])


def test_intersect_automata_star():
    dfa1 = AdjacencyMatrixFA(regex_to_dfa("a* b*"))
    dfa2 = AdjacencyMatrixFA(regex_to_dfa("(a | b)*"))
    intersection = intersect_automata(dfa1, dfa2)

    assert intersection.accepts([])
    assert intersection.accepts(["a", "b"])
    assert intersection.accepts(["a", "a", "b", "b"])
    assert not intersection.accepts(["a", "b", "a"])


# tests for tensor_based_rpq


def test_tensor_based_rpq_concat():
    graph = _build_graph(
        [(0, 1, "a"), (0, 2, "a"), (1, 3, "b"), (2, 4, "c"), (3, 0, "a"), (4, 0, "a")]
    )
    answer = tensor_based_rpq("a b", graph, {0}, {0, 1, 2, 3, 4})

    assert answer == {(0, 3)}


def test_tensor_based_rpq_concat_and_alt():
    graph = _build_graph(
        [(0, 1, "a"), (0, 2, "a"), (1, 3, "b"), (2, 4, "c"), (3, 0, "a"), (4, 0, "a")]
    )
    answer = tensor_based_rpq("a (b | c)", graph, {0}, {0, 1, 2, 3, 4})

    assert answer == {(0, 3), (0, 4)}


def test_tensor_based_rpq_concat_and_alt_and_star():
    graph = _build_graph(
        [(0, 1, "a"), (0, 2, "a"), (1, 3, "b"), (2, 4, "c"), (3, 0, "a"), (4, 0, "a")]
    )
    answer = tensor_based_rpq("(a | b | c)* a", graph, {0}, {0, 1, 2, 3, 4})

    assert answer == {(0, 0), (0, 1), (0, 2)}


# type assertions


def test_accepts_returns_bool():
    automaton = AdjacencyMatrixFA(regex_to_dfa("(a | b)* c"))

    assert type(automaton.accepts(["a"])) is bool


def test_is_empty_returns_bool():
    automaton = AdjacencyMatrixFA(regex_to_dfa("$ | (a b c) | d*"))

    assert type(automaton.is_empty()) is bool


def test_intersect_automata_returns_adjacency_matrix_fa():
    automaton = AdjacencyMatrixFA(regex_to_dfa("(a* b)*"))

    assert isinstance(intersect_automata(automaton, automaton), AdjacencyMatrixFA)


def test_tensor_based_rpq_returns_set_of_int_pairs():
    graph = _build_graph([(0, 1, "a"), (1, 2, "b"), (2, 3, "c")])
    answer = tensor_based_rpq("a b c", graph, {0}, {3})

    assert type(answer) is set
    for pair in answer:
        assert type(pair) is tuple
        for vertex in pair:
            assert type(vertex) is int


# negative tests


def test_intersect_automata_raises_on_none_automaton():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a"))

    with pytest.raises(AttributeError):
        intersect_automata(None, automaton)


def test_intersect_automata_raises_on_second_none_automaton():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a"))

    with pytest.raises(AttributeError):
        intersect_automata(automaton, None)


def test_intersect_automata_raises_on_automaton_that_is_not_adjacency_matrix_fa():
    with pytest.raises(AttributeError):
        intersect_automata(regex_to_dfa("a"), AdjacencyMatrixFA(regex_to_dfa("a")))


def test_tensor_based_rpq_raises_on_none_regex():
    graph = _build_graph([(0, 1, "b")])

    with pytest.raises(AttributeError):
        tensor_based_rpq(None, graph, {0}, {1})


def test_tensor_based_rpq_raises_on_none_graph():
    with pytest.raises(AttributeError):
        tensor_based_rpq("b", None, {0}, {1})


def test_tensor_based_rpq_raises_on_malformed_regex():
    graph = _build_graph([(0, 1, "b")])

    with pytest.raises(MisformedRegexError):
        tensor_based_rpq("(((", graph, {0}, {1})
