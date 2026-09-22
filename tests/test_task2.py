"""Tests for task 2."""

import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import MisformedRegexError

from project.task2 import edges_to_graph, graph_to_nfa, regex_to_dfa


# positive tests for regex_to_dfa


def test_regex_to_dfa_returns_dfa():
    dfa = regex_to_dfa("a|b")

    assert isinstance(dfa, DeterministicFiniteAutomaton)


def test_regex_to_dfa_empty_regex():
    dfa = regex_to_dfa("")

    assert dfa.is_empty()
    assert not dfa.accepts([])
    assert not dfa.accepts(["a"])


def test_regex_to_dfa_epsilon():
    dfa = regex_to_dfa("$")

    assert dfa.accepts([])
    assert not dfa.accepts(["a"])


def test_regex_to_dfa_alternation():
    dfa = regex_to_dfa("a|b")

    assert dfa.accepts(["a"])
    assert dfa.accepts(["b"])
    assert not dfa.accepts(["c"])
    assert not dfa.accepts(["a", "b"])


def test_regex_to_dfa_kleene_star():
    dfa = regex_to_dfa("a*")

    assert dfa.accepts([])
    assert dfa.accepts(["a"])
    assert dfa.accepts(["a", "a", "a"])
    assert not dfa.accepts(["b"])


def test_regex_to_dfa_alternation_of_concatenations():
    dfa = regex_to_dfa("a b|b a")

    assert isinstance(dfa, DeterministicFiniteAutomaton)
    assert dfa.is_deterministic()
    assert len(dfa.minimize().states) == len(dfa.states)
    assert dfa.accepts(["a", "b"])
    assert dfa.accepts(["b", "a"])
    assert not dfa.accepts(["a", "a"])
    assert not dfa.accepts(["b", "b"])
    assert not dfa.accepts(["a"])
    assert not dfa.accepts(["b"])


def test_regex_to_dfa_star_of_concatenation():
    dfa = regex_to_dfa("(a b)*")

    assert dfa.accepts([])
    assert dfa.accepts(["a", "b"])
    assert dfa.accepts(["a", "b", "a", "b"])
    assert not dfa.accepts(["a"])
    assert not dfa.accepts(["b", "a"])


def test_regex_to_dfa_alternation_and_star_and_concatenation_and_epsilon():
    dfa = regex_to_dfa("((a|b)* c)|$")

    assert dfa.accepts([])
    assert dfa.accepts(["c"])
    assert dfa.accepts(["a", "c"])
    assert dfa.accepts(["a", "b", "a", "c"])
    assert not dfa.accepts(["a"])
    assert not dfa.accepts(["c", "a"])
    assert not dfa.accepts(["a", "c", "b"])


# negative tests for regex_to_dfa


def test_regex_to_dfa_star_without_operand_raises():
    with pytest.raises(MisformedRegexError):
        regex_to_dfa("*a")


def test_regex_to_dfa_star_with_invalid_operand_raises():
    with pytest.raises(MisformedRegexError):
        regex_to_dfa("|*abcde")


def test_regex_to_dfa_empty_parentheses_raises():
    with pytest.raises(Exception):
        regex_to_dfa("()")


def test_regex_to_dfa_unclosed_parenthesis_raises():
    with pytest.raises(MisformedRegexError):
        regex_to_dfa("(a|b")


def test_regex_to_dfa_unbalanced_parentheses_raises():
    with pytest.raises(MisformedRegexError):
        regex_to_dfa("a(b(c)d)e)f")


# positive tests for graph_to_nfa


def test_graph_to_nfa_returns_nfa():
    graph = edges_to_graph([(1, "a", 2)])
    nfa = graph_to_nfa(graph, {1}, {2})

    assert isinstance(nfa, NondeterministicFiniteAutomaton)


def test_graph_to_nfa_with_given_start_and_final():
    graph = edges_to_graph([(1, "a", 2), (2, "b", 3), (1, "c", 3)])

    nfa = graph_to_nfa(graph, {1}, {3})

    assert nfa.accepts(["a", "b"])
    assert nfa.accepts(["c"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b"])
    assert not nfa.accepts(["b", "a"])


def test_graph_to_nfa_all_nodes_are_start_and_final_by_default():
    graph = edges_to_graph([(1, "a", 2), (2, "b", 3)])

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.accepts([])
    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
    assert nfa.accepts(["a", "b"])
    assert not nfa.accepts(["b", "a"])


def test_graph_to_nfa_multiple_start_and_final_states():
    graph = edges_to_graph([(1, "a", 2), (2, "a", 3), (3, "b", 4)])

    nfa = graph_to_nfa(graph, {1, 3}, {2, 4})

    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
    assert not nfa.accepts(["b", "b"])


def test_graph_to_nfa_parallel_edges():
    graph = edges_to_graph([(1, "a", 2), (1, "b", 2)])

    nfa = graph_to_nfa(graph, {1}, {2})

    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
    assert not nfa.accepts(["a", "b"])


def test_graph_to_nfa_accepts_empty_word_from_self_loop_state():
    graph = edges_to_graph([(1, "a", 1)])

    nfa = graph_to_nfa(graph, {1}, {1})

    assert nfa.accepts([])
    assert nfa.accepts(["a"])
    assert nfa.accepts(["a", "a"])


# negative tests for graph_to_nfa


def test_graph_to_nfa_rejects_non_graph():
    with pytest.raises(AttributeError):
        graph_to_nfa(None, set(), set())


def test_graph_to_nfa_edge_without_label_raises():
    graph = MultiDiGraph()
    graph.add_edge(1, 2)

    with pytest.raises(KeyError):
        graph_to_nfa(graph, set(), set())


# cross-check tests comparing regex_to_dfa and graph_to_nfa


def test_regex_to_dfa_matches_graph_to_nfa_on_path():
    regex_fa = regex_to_dfa("a b")
    graph_fa = graph_to_nfa(edges_to_graph([(1, "a", 2), (2, "b", 3)]), {1}, {3})

    for word in ([], ["a"], ["b"], ["a", "b"], ["a", "b", "a"], ["b", "a"]):
        assert regex_fa.accepts(word) == graph_fa.accepts(word)


def test_regex_to_dfa_matches_graph_to_nfa_on_star_loop():
    regex_fa = regex_to_dfa("(a b)*")
    graph_fa = graph_to_nfa(edges_to_graph([(1, "a", 2), (2, "b", 1)]), {1}, {1})

    for word in (
        [],
        ["a"],
        ["b"],
        ["a", "b"],
        ["a", "b", "a", "b"],
        ["a", "b", "a", "b", "a"],
        ["a", "a"],
        ["b", "b"],
    ):
        assert regex_fa.accepts(word) == graph_fa.accepts(word)


# positive tests for edges_to_graph


def test_edges_to_graph_returns_multidigraph():
    graph = edges_to_graph([(1, "a", 2)])

    assert isinstance(graph, MultiDiGraph)


def test_edges_to_graph_with_empty_edges_is_empty():
    graph = edges_to_graph([])

    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_edges_to_graph_with_edge_is_not_empty():
    graph = edges_to_graph([(1, "a", 2)])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge(1, 2)


def test_edges_to_graph_keeps_parallel_edges():
    graph = edges_to_graph([(1, "a", 2), (1, "b", 2)])

    assert graph.number_of_edges(1, 2) == 2


def test_edges_to_graph_adds_edges_with_labels():
    graph = edges_to_graph([(1, "a", 2), (2, "b", 3)])

    assert graph.get_edge_data(1, 2, 0)["label"] == "a"
    assert graph.get_edge_data(2, 3, 0)["label"] == "b"


# negative tests for edges_to_graph


def test_edges_to_graph_rejects_non_iterable_edges():
    with pytest.raises(TypeError):
        edges_to_graph(None)


def test_edges_to_graph_rejects_edge_with_wrong_arity():
    with pytest.raises(ValueError):
        edges_to_graph([(1, 2)])
