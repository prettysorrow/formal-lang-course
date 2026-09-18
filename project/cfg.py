"""Shared helpers for context-free path querying (tasks 6 to 9)."""

from collections import defaultdict
from typing import NamedTuple

from pyformlang.cfg import CFG, Variable


class Productions(NamedTuple):
    """Production rules of a CFG grouped by kind.

    Attributes:
        terminal:   maps `a` to the set of `X` such that `X -> a` is a rule;
        epsilon:    set of `X` such that `X -> epsilon` is a rule;
        binary:     list of rules of the form `X -> YZ`.
    """

    terminal: dict[str, set[Variable]]
    epsilon: set[Variable]
    binary: list[tuple[Variable, Variable, Variable]]


def partition_productions(cfg: CFG) -> Productions:
    """Split the productions of ``cfg`` into terminal, epsilon and binary buckets."""
    terminal: dict[str, set[Variable]] = defaultdict(set)
    epsilon: set[Variable] = set()
    binary: list[tuple[Variable, Variable, Variable]] = []

    for production in cfg.productions:
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

    return Productions(terminal, epsilon, binary)
