"""Task 2. Constructing a DFA from a Regular Expression and a NFA from a Graph."""

from collections.abc import Iterable
from typing import Any

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Build a minimal deterministic finite automaton accepting the language of ``regex``.

    Args:
        regex: Regular expression in the pyformlang format.

    Returns:
        Minimal deterministic finite automaton equivalent to the ``regex``.
    """
    regex_pyformlang = Regex(regex)
    nfa = regex_pyformlang.to_epsilon_nfa()
    dfa = nfa.to_deterministic()
    minimal_dfa = dfa.minimize()
    return minimal_dfa


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: set[int],
    final_states: set[int],
) -> NondeterministicFiniteAutomaton:
    """Build a nondeterministic finite automaton from a ``MultiDiGraph``.

    Args:
        graph: Graph from which to build the NFA.
        start_states:
            Start states of the NFA.
            If empty, all graph nodes are taken as start states.
        final_states:
            Final states of the NFA.
            If empty, all graph nodes are taken as final states.

    Returns:
        Nondeterministic finite automaton built from the ``graph``.
    """
    start_states = start_states or set(graph.nodes)
    final_states = final_states or set(graph.nodes)
    nfa = NondeterministicFiniteAutomaton()
    for state in start_states:
        nfa.add_start_state(state)
    for state in final_states:
        nfa.add_final_state(state)
    for source, target, weight in graph.edges(data=True):
        symbol_by = Symbol(weight["label"])
        nfa.add_transition(source, symbol_by, target)
    return nfa


def edges_to_graph(
    edges: Iterable[tuple[Any, str, Any]],
) -> MultiDiGraph:
    """Build a ``MultiDiGraph`` from labeled edges.

    Args:
        edges:
            An iterable of ``(source, label, target)`` triples.
            Each triple describes one directed edge of the resulting graph.

    Returns:
        A directed multigraph whose edges are the ones given in ``edges``.
    """
    graph = MultiDiGraph()
    for source, label, target in edges:
        graph.add_edge(source, target, label=label)
    return graph
