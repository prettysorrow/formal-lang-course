import networkx


def build_graph(edges) -> networkx.MultiDiGraph:
    graph = networkx.MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph
