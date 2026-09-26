"""Task 3. All-pairs RPQ algorithm."""

from networkx import MultiDiGraph
from project.task2 import graph_to_nfa, regex_to_dfa

from collections import defaultdict
from typing import Iterable

import numpy
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy import sparse


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton = None):
        if automaton is None:
            automaton = NondeterministicFiniteAutomaton()

        self.alphabet = automaton.symbols

        self.states = [state.value for state in automaton.states]
        self.states_count = len(self.states)

        states_index = {
            state_value: index for index, state_value in enumerate(self.states)
        }
        self.start_indices = {
            states_index[state.value] for state in automaton.start_states
        }
        self.final_indices = {
            states_index[state.value] for state in automaton.final_states
        }

        edges = defaultdict(lambda: ([], []))
        for source, target, label in automaton.to_networkx().edges(data="label"):
            if label is None:
                continue
            rows, columns = edges[label]
            rows.append(states_index[source])
            columns.append(states_index[target])

        self.transitions = {
            symbol: sparse.csr_array(
                ([True] * len(rows), (rows, columns)),
                shape=(self.states_count, self.states_count),
                dtype=bool,
            )
            for symbol, (rows, columns) in edges.items()
        }

    def accepts(self, word: Iterable[Symbol]) -> bool:
        states = numpy.zeros(self.states_count, dtype=bool)
        for start_index in self.start_indices:
            states[start_index] = True

        for symbol in word:
            transitions = self.transitions.get(symbol)
            if transitions is None:
                return False
            states = transitions.T @ states

        return any(states[final_index] for final_index in self.final_indices)

    def transitive_closure(self) -> sparse.csr_array:
        closure = sparse.eye_array(self.states_count, dtype=bool, format="csr")
        for transitions in self.transitions.values():
            closure = closure + transitions

        nzz = -1
        while closure.nnz != nzz:
            nzz = closure.nnz
            closure = closure @ closure
        return closure

    def is_empty(self) -> bool:
        closure = self.transitive_closure()
        return not any(
            closure[start_index, final_index]
            for start_index in self.start_indices
            for final_index in self.final_indices
        )


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersection = AdjacencyMatrixFA()
    intersection.states = [
        f"{automaton1.states[i]}:{automaton2.states[j]}"
        for i in range(automaton1.states_count)
        for j in range(automaton2.states_count)
    ]
    intersection.states_count = automaton1.states_count * automaton2.states_count
    intersection.alphabet = automaton1.alphabet & automaton2.alphabet
    intersection.transitions = {
        symbol: sparse.kron(
            automaton1.transitions[symbol],
            automaton2.transitions[symbol],
            format="csr",
        )
        for symbol in intersection.alphabet
        if symbol in automaton1.transitions and symbol in automaton2.transitions
    }
    intersection.start_indices = {
        i * automaton2.states_count + j
        for i in automaton1.start_indices
        for j in automaton2.start_indices
    }
    intersection.final_indices = {
        i * automaton2.states_count + j
        for i in automaton1.final_indices
        for j in automaton2.final_indices
    }
    return intersection


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    if not start_nodes or not final_nodes:
        return set()
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    intersection = intersect_automata(graph_fa, regex_fa)
    closure = intersection.transitive_closure()

    regex_size = regex_fa.states_count
    return {
        (
            graph_fa.states[product_start // regex_size],
            graph_fa.states[product_final // regex_size],
        )
        for product_start in intersection.start_indices
        for product_final in intersection.final_indices
        if closure[product_start, product_final]
    }
