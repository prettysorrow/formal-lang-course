"""Shared helpers for regular path querying (tasks 3 and 4)."""

from collections import defaultdict

from pyformlang.finite_automaton import Epsilon


def state_info(automaton):
    """Extract states, state-to-index map, start and final states from an automaton."""
    states = sorted({state.value for state in automaton.states}, key=str)
    state_index = {state: index for index, state in enumerate(states)}
    start_states = {state.value for state in automaton.start_states}
    final_states = {state.value for state in automaton.final_states}
    return states, state_index, start_states, final_states


def transitions_by_label(automaton, state_index):
    """Group numeric transitions as ``{symbol: [(source, target), ...]}``."""
    edges_by_label = defaultdict(list)
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