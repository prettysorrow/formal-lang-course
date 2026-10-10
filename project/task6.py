"""Task 6. Weak Chomsky Normal Form (wCNF) transformation, Helling's CFPQ Algorithm."""

from collections import defaultdict
from typing import NamedTuple

from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Epsilon, Production, Variable

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA


class Productions(NamedTuple):
    epsilon: set[Variable]  # X -> epsilon
    terminal: dict[str, set[Variable]]  # X -> a
    binary: list[tuple[Variable, Variable, Variable]]  # X -> YZ


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    cfg_normal_form = cfg.to_normal_form()
    cfg_start_symbol = cfg_normal_form.start_symbol
    cfg_productions = set(cfg_normal_form.productions)
    cfg_nullable_symbols = cfg.get_nullable_symbols()
    for symbol in cfg_nullable_symbols:
        production = Production(symbol, [Epsilon()])
        cfg_productions.add(production)
    return CFG(
        start_symbol=cfg_start_symbol,
        productions=cfg_productions,
    )


def hellings_based_cfpq(
    cfg: CFG,
    graph: MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    ncfg = cfg_to_weak_normal_form(cfg)
    productions = partition_productions(ncfg)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    derived = set()  # { (X, u, v) | (X, u, v) is derived }
    frontier = set()  # derivations not yet used for the closure
    derived_from = defaultdict(set)  # (X, u) -> { v | (X, u, v) is derived }
    derived_to = defaultdict(set)  # (X, v) -> { u | (X, u, v) is derived }

    def add_derivation(X, u, v):
        if (X, u, v) in derived:
            return
        derived.add((X, u, v))
        frontier.add((X, u, v))
        derived_from[(X, u)].add(v)
        derived_to[(X, v)].add(u)

    # (X -> epsilon)  ===> (X, v, v) for every v
    for v in range(graph_fa.states_count):
        for X in productions.epsilon:
            add_derivation(X, v, v)

    # (X -> a) and (u -a-> v)  ===>  (X, u, v)
    for a, transitions in graph_fa.transition_matrices.items():
        heads = productions.terminal.get(a)
        if not heads:
            continue
        sources, targets = transitions.nonzero()
        for u, v in zip(sources, targets):
            for X in heads:
                add_derivation(X, u, v)

    # For every new derivation and every rule X -> YZ:
    # 1. (Y, u, v) and (Z, v, w)  ===>  (X, u, w)
    # 2. (Z, u, v) and (Y, w, u)  ===>  (X, w, v)
    while frontier:
        current = list(frontier)
        frontier.clear()
        for X, Y, Z in productions.binary:
            for head, u, v in current:
                if head == Y:
                    for w in list(derived_from[(Z, v)]):
                        add_derivation(X, u, w)
                if head == Z:
                    for w in list(derived_to[(Y, u)]):
                        add_derivation(X, w, v)

    # (S, u, v) where S is the start symbol and u/v are start/final states  ==>  (u, v)
    return {
        (graph_fa.states[u], graph_fa.states[v])
        for S, u, v in derived
        if S == ncfg.start_symbol
        and u in graph_fa.start_indices
        and v in graph_fa.final_indices
    }


def partition_productions(ncfg: CFG) -> Productions:
    epsilon: set[Variable] = set()
    terminal: dict[str, set[Variable]] = defaultdict(set)
    binary: list[tuple[Variable, Variable, Variable]] = []

    for production in ncfg.productions:
        body = production.body
        if len(body) == 0:
            # X -> epsilon
            epsilon.add(production.head)
        elif len(body) == 1:
            # X -> a
            terminal[body[0].value].add(production.head)
        else:
            # X -> YZ
            binary.append((production.head, body[0], body[1]))

    return Productions(epsilon, terminal, binary)
