"""Task 6. CFG to weak Chomsky normal form and Hellings' algorithm."""

from collections import defaultdict

from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Epsilon, Production, Variable

from project.cfg import partition_productions
from project.rpq import get_automaton_description, transitions_by_label
from project.task2 import graph_to_nfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    cfg_normal_form = cfg.to_normal_form()
    cfg_start_symbol = cfg_normal_form.start_symbol
    cfg_productions = set(cfg_normal_form.productions)
    cfg_nullable_symbols = cfg.get_nullable_symbols()
    for symbol in cfg_nullable_symbols:
        production = Production(Variable(symbol.value), [Epsilon()])
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
    """Solve the context-free reachability problem with Hellings' algorithm."""
    normalized_cfg = cfg_to_weak_normal_form(cfg)
    productions = partition_productions(normalized_cfg)
    fa_desc = get_automaton_description(graph_to_nfa(graph, start_nodes, final_nodes))

    # "derivation" (X, u, v) means: X derives the word on some path (u -> v)
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
    for v in range(fa_desc.num_states):
        for X in productions.epsilon:
            add_derivation(X, v, v)

    # (X -> a) and (u -a-> v)  ===>  (X, u, v)
    for a, edges in transitions_by_label(fa_desc).items():
        for X in productions.terminal.get(a, ()):
            for u, v in edges:
                add_derivation(X, u, v)

    # 1. (X -> Y Z) and (Y, u, v) and (Z, v, w)  ===>  (X, u, w)
    # 2. (X -> Y Z) and (Z, u, v) and (Y, w, u)  ===>  (X, w, v)
    while frontier:
        current = list(frontier)
        frontier.clear()
        for X, Y, Z in productions.binary:  # X -> Y Z
            for head, u, v in current:
                # 1.
                if head == Y:  # for (Y, u, v)
                    for w in derived_from[(Z, v)]:  # every (Z, v, w)
                        add_derivation(X, u, w)  # maps to (X, u, w)
                # 2.
                if head == Z:  # for (Z, u, v)
                    for w in derived_to[(Y, u)]:  # every (Y, w, u)
                        add_derivation(X, w, v)  # maps to (X, w, v)

    # (S, u, v) where (S is start symbol) and (u is start state) and (v is final state)  ==>  (u, v)
    return {
        (fa_desc.states[u], fa_desc.states[v])
        for S, u, v in derived
        if S == normalized_cfg.start_symbol
        and fa_desc.states[u] in fa_desc.start_states
        and fa_desc.states[v] in fa_desc.final_states
    }
