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

        states_index = {state: index for index, state in enumerate(self.states)}
        edges = defaultdict(lambda: ([], []))
        for source, target, label in automaton.to_networkx().edges(data="label"):
            if label is None:
                continue
            rows, columns = edges[label]
            rows.append(states_index[source])
            columns.append(states_index[target])

        self.start_indices = {
            states_index[start_state.value] for start_state in automaton.start_states
        }
        self.final_indices = {
            states_index[final_state.value] for final_state in automaton.final_states
        }
        self.transition_matrices = {
            symbol: sparse.csr_array(
                ([True] * len(rows), (rows, columns)),
                shape=(self.states_count, self.states_count),
                dtype=bool,
            )
            for symbol, (rows, columns) in edges.items()
        }

    def accepts(self, word: Iterable[Symbol]) -> bool:
        reached_states = numpy.zeros(self.states_count, dtype=bool)
        for start_index in self.start_indices:
            reached_states[start_index] = True

        for symbol in word:
            transition_matrix = self.transition_matrices.get(symbol)
            if transition_matrix is None:
                return False
            reached_states = transition_matrix.T @ reached_states

        return any(reached_states[final_index] for final_index in self.final_indices)

    def is_empty(self) -> bool:
        closure = self.transitive_closure()
        return not any(
            closure[start_index, final_index]
            for start_index in self.start_indices
            for final_index in self.final_indices
        )

    def reflexive_adjacency_matrix(self) -> sparse.csr_array:
        adjacency = sparse.eye_array(self.states_count, dtype=bool, format="csr")
        for transition_matrix in self.transition_matrices.values():
            adjacency += transition_matrix
        return adjacency

    def transitive_closure(self) -> sparse.csr_array:
        closure = self.reflexive_adjacency_matrix()
        previous_nnz = -1
        while closure.nnz != previous_nnz:
            previous_nnz = closure.nnz
            closure @= closure
        return closure


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
    intersection.transition_matrices = {
        symbol: sparse.kron(
            automaton1.transition_matrices[symbol],
            automaton2.transition_matrices[symbol],
            format="csr",
        )
        for symbol in intersection.alphabet
        if symbol in automaton1.transition_matrices
        and symbol in automaton2.transition_matrices
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
            graph_fa.states[start_index // regex_size],
            graph_fa.states[final_index // regex_size],
        )
        for start_index in intersection.start_indices
        for final_index in intersection.final_indices
        if closure[start_index, final_index]
    }
