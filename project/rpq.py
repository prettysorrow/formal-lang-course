"""Shared helpers for regular path querying (tasks 3 and 4)."""

from collections import defaultdict
from typing import NamedTuple

from pyformlang.finite_automaton import Epsilon
from scipy.sparse import csr_matrix


class AutomatonDescription(NamedTuple):
    """States of an automaton and the associated indices and start/final states."""

    automaton: object
    states: list
    state_index: dict
    start_states: set
    final_states: set
    num_states: int


def get_automaton_description(automaton) -> AutomatonDescription:
    """Extract states, state-to-index map, start and final states from an automaton."""
    states = sorted({state.value for state in automaton.states}, key=str)
    state_index = {state: index for index, state in enumerate(states)}
    start_states = {state.value for state in automaton.start_states}
    final_states = {state.value for state in automaton.final_states}
    return AutomatonDescription(
        automaton, states, state_index, start_states, final_states, len(states)
    )


def transitions_by_label(description: AutomatonDescription):
    """Group numeric transitions as ``{symbol: [(source, target), ...]}``."""
    edges_by_label = defaultdict(list)
    automaton = description.automaton
    state_index = description.state_index
    for from_state, transitions in automaton._transition_function._transitions.items():
        source = state_index[from_state.value]
        for symbol, to_states in transitions.items():
            if isinstance(symbol, Epsilon):
                continue
            if not isinstance(to_states, set):
                to_states = {to_states}
            for to_state in to_states:
                target = state_index[to_state.value]
                edges_by_label[str(symbol)].append((source, target))
    return edges_by_label


def build_transition_matrix(
    description: AutomatonDescription, symbol: str | None = None
):
    """Build a boolean ``csr_matrix`` for the automaton's transitions.

    ``symbol`` optionally restricts the matrix to the edges with that label.
    """
    edges = [
        edge
        for label, label_edges in transitions_by_label(description).items()
        if symbol is None or label == symbol
        for edge in label_edges
    ]
    num_states = description.num_states
    if not edges:
        return csr_matrix((num_states, num_states), dtype=bool)
    rows = [source for source, _ in edges]
    cols = [target for _, target in edges]
    return csr_matrix(
        ([True] * len(edges), (rows, cols)),
        shape=(num_states, num_states),
        dtype=bool,
    )
