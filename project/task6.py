"""Task 6. CFG to weak Chomsky normal form and Hellings' algorithm."""

from collections import defaultdict

from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable

from project.rpq import get_automaton_description, transitions_by_label
from project.task2 import graph_to_nfa

class ShouldNotHappenException(Exception):
    pass

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

    terminal_productions = defaultdict(set)  # maps `a` to the set of `X` such that `X -> a` is a rule
    epsilon_productions = set()              # set of `X` such that `X -> epsilon` is a rule
    binary_productions = []                  # list of rules of the form `X -> YZ`

    for production in normalized_cfg.productions:
        body = production.body
        if len(body) == 0:
            # X -> epsilon
            X = production.head
            epsilon_productions.add(X)
        elif len(body) == 1 and isinstance(body[0], Terminal):
            # X -> a
            X = production.head
            a = body[0].value
            terminal_productions[a].add(X)
        elif len(body) == 2 and isinstance(body[0], Variable) and isinstance(body[1], Variable):
            # X -> YZ
            X = production.head
            Y = body[0]
            Z = body[1]
            binary_productions.append((X, Y, Z))
        else:
            raise ShouldNotHappenException("unexpected production form")

    fa = graph_to_nfa(graph, start_nodes, final_nodes)
    fa_desc = get_automaton_description(fa)
    start_indices = {
        fa_desc.state_index[node] for node in fa_desc.start_states
    }
    final_indices = {
        fa_desc.state_index[node] for node in fa_desc.final_states
    }

    derivations = set()
    by_left = defaultdict(set)
    by_right = defaultdict(set)

    def add_derivation(variable, left, right):
        if (variable, left, right) in derivations:
            return False
        derivations.add((variable, left, right))
        by_left[(variable, left)].add(right)
        by_right[(variable, right)].add(left)
        return True

    for label, edges in transitions_by_label(fa_desc).items():
        for head in terminal_productions.get(label, ()):
            for left, right in edges:
                add_derivation(head, left, right)

    for node in range(fa_desc.num_states):
        for head in epsilon_productions:
            add_derivation(head, node, node)

    new_derivations = set(derivations)
    while new_derivations:
        current = new_derivations
        new_derivations = set()
        for head, first, second in binary_productions:
            for variable, nt_left, nt_right in current:
                if variable == first:
                    for right in by_left[(second, nt_right)]:
                        if add_derivation(head, nt_left, right):
                            new_derivations.add((head, nt_left, right))
                if variable == second:
                    for left in by_right[(first, nt_left)]:
                        if add_derivation(head, left, nt_right):
                            new_derivations.add((head, left, nt_right))

    return {
        (fa_desc.states[left], fa_desc.states[right])
        for variable, left, right in derivations
        if variable == normalized_cfg.start_symbol
        and left in start_indices
        and right in final_indices
    }