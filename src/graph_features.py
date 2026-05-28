from typing import Dict

import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    """
    Construye un grafo no dirigido a partir de un DataFrame de aristas.

    El DataFrame debe tener:
    - from
    - to
    """
    graph = nx.from_pandas_edgelist(
        edges,
        source="from",
        target="to"
    )

    return graph


def get_graph_summary(graph: nx.Graph) -> dict:
    """
    Devuelve información básica del grafo.
    """
    summary = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "is_connected": nx.is_connected(graph)
    }

    if nx.is_connected(graph):
        summary["num_connected_components"] = 1
        summary["largest_component_size"] = graph.number_of_nodes()
    else:
        components = list(nx.connected_components(graph))
        largest_component = max(components, key=len)

        summary["num_connected_components"] = len(components)
        summary["largest_component_size"] = len(largest_component)

    return summary


def compute_communities(graph: nx.Graph) -> Dict[int, int]:
    """
    Detecta comunidades usando greedy modularity.

    Devuelve un diccionario:
    nodo -> comunidad
    """
    communities = greedy_modularity_communities(graph)

    community_dict = {}

    for community_id, community in enumerate(communities):
        for node in community:
            community_dict[node] = community_id

    return community_dict


def compute_graph_features(
    graph: nx.Graph,
    betweenness_k: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calcula las métricas relacionales principales para cada nodo.

    Métricas:
    - degree
    - degree_centrality
    - clustering
    - pagerank
    - closeness
    - betweenness aproximado
    - community
    """
    degree = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    clustering = nx.clustering(graph)
    pagerank = nx.pagerank(graph)
    closeness = nx.closeness_centrality(graph)

    betweenness = nx.betweenness_centrality(
        graph,
        k=min(betweenness_k, graph.number_of_nodes()),
        seed=random_state
    )

    communities = compute_communities(graph)

    features_df = pd.DataFrame({
        "new_id": list(graph.nodes()),
        "degree": [degree[node] for node in graph.nodes()],
        "degree_centrality": [degree_centrality[node] for node in graph.nodes()],
        "clustering": [clustering[node] for node in graph.nodes()],
        "pagerank": [pagerank[node] for node in graph.nodes()],
        "closeness": [closeness[node] for node in graph.nodes()],
        "betweenness": [betweenness[node] for node in graph.nodes()],
        "community": [communities[node] for node in graph.nodes()]
    })

    return features_df