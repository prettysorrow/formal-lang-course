"""Task 1. Initialization of the working environment."""

import pathlib
from typing import NamedTuple

import cfpq_data
import pydot
from networkx import MultiDiGraph


class GraphDescription(NamedTuple):
    """Description of a graph with labeled edges."""

    nodes_count: int
    edges_count: int
    labels: set[str]


def download_graph_description_from_cfpq_data(graph_name: str) -> GraphDescription:
    """Download a graph from the cfpq-data dataset and return its description.

    Args:
        graph_name: Graph name from the cfpq-data dataset.

    Returns:
        Graph description: nodes count, edges count, labels.
    """
    path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(path)
    nodes_count = graph.number_of_nodes()
    edges_count = graph.number_of_edges()
    labels = {weight["label"] for _, _, weight in graph.edges(data=True)}
    return GraphDescription(nodes_count, edges_count, labels)


def store_two_cycles_graph_to_dot_file(
    first_cycle_size: int,
    second_cycle_size: int,
    labels: tuple[str, str],
    path: str | pathlib.Path,
) -> None:
    """Build a graph of two cycles and save it to ``path`` in DOT format.

    Args:
        first_cycle_size: Number of nodes in the first cycle.
        second_cycle_size: Number of nodes in the second cycle.
        labels: Edge labels for the two cycles.
        path: Path to the output DOT file.
    """
    graph = cfpq_data.labeled_two_cycles_graph(
        first_cycle_size, second_cycle_size, labels=labels
    )
    dot = _convert_graph_to_dot(graph)
    pathlib.Path(path).write_text(dot.to_string())


def load_graph_description_from_dot_file(path: str | pathlib.Path) -> GraphDescription:
    """Load a graph from a DOT file and return its description.

    Args:
        path: Path to the DOT file.

    Returns:
        Graph description: nodes count, edges count, labels.
    """
    graph = pydot.graph_from_dot_file(str(path))[0]
    nodes_count = len(graph.get_nodes())
    edges_count = len(graph.get_edges())
    labels = {edge.get("label") for edge in graph.get_edges()}
    return GraphDescription(nodes_count, edges_count, labels)


def _convert_graph_to_dot(graph: MultiDiGraph) -> pydot.Dot:
    """Convert a ``MultiDiGraph`` into a ``pydot.Dot``.

    Args:
        graph: Graph to convert.
    """
    dot = pydot.Dot(graph_type="digraph")
    for node in graph.nodes:
        dot.add_node(pydot.Node(str(node)))
    for source, target, weight in graph.edges(data=True):
        dot.add_edge(pydot.Edge(str(source), str(target), label=weight["label"]))
    return dot
