"""Task 9. GLL-based CFPQ algorithm."""

from collections import defaultdict

from networkx import DiGraph
from pyformlang.rsa import RecursiveAutomaton

from project.task2 import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    graph_fa = graph_to_nfa(graph, start_nodes, final_nodes)

    transitions = defaultdict(lambda: defaultdict(list))
    for source, target, label in graph_fa.to_networkx().edges(data="label"):
        transitions[label][source].append(target)

    start_states = {state.value for state in graph_fa.start_states}
    final_states = {state.value for state in graph_fa.final_states}

    boxes = {}
    for label, box in rsm.boxes.items():
        box_start_states = {state.value for state in box.start_state}
        box_final_states = {state.value for state in box.final_states}
        box_transitions = defaultdict(list)
        for source, target, symbol in box.dfa.to_networkx().edges(data="label"):
            box_transitions[source].append((symbol, target))
        boxes[label.value] = (
            box_start_states,
            box_final_states,
            box_transitions,
        )

    start_symbol = rsm.initial_label.value

    gss_edges = defaultdict(lambda: defaultdict(set))
    popped = defaultdict(lambda: defaultdict(set))

    visited = set()
    stack = []

    def add_descriptor(label, source, state, position):
        descriptor = (label, source, state, position)
        if descriptor not in visited:
            visited.add(descriptor)
            stack.append(descriptor)

    for start in start_states:
        for state in boxes[start_symbol][0]:
            add_descriptor(start_symbol, start, state, start)

    while stack:
        label, source, state, position = stack.pop()
        _, box_final_states, box_transitions = boxes[label]

        if state in box_final_states:
            popped[label][source].add(position)
            for return_state, parent_label, parent_source in gss_edges[label][source]:
                add_descriptor(parent_label, parent_source, return_state, position)

        for symbol, target_state in box_transitions[state]:
            if symbol == "epsilon":
                add_descriptor(label, source, target_state, position)
            elif symbol in boxes:
                gss_edges[symbol][position].add((target_state, label, source))
                for end in popped[symbol][position]:
                    add_descriptor(label, source, target_state, end)
                for callee_state in boxes[symbol][0]:
                    add_descriptor(symbol, position, callee_state, position)
            else:
                for target in transitions[symbol][position]:
                    add_descriptor(label, source, target_state, target)

    return {
        (source, target)
        for label, sources in popped.items()
        if label == start_symbol
        for source, targets in sources.items()
        if source in start_states
        for target in targets
        if target in final_states
    }
