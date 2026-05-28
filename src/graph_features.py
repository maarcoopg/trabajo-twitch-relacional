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
    if graph.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0,
            "is_connected": False,
            "num_connected_components": 0,
            "largest_component_size": 0,
        }

    is_connected = nx.is_connected(graph) if graph.number_of_nodes() > 0 else False

    summary = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "is_connected": is_connected,
    }

    if is_connected:
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
    if graph.number_of_nodes() == 0:
        return {}

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
    feature_columns = [
        "new_id",
        "degree",
        "degree_centrality",
        "clustering",
        "pagerank",
        "closeness",
        "betweenness",
        "community",
    ]

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=feature_columns)

    degree = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    clustering = nx.clustering(graph)
    pagerank = nx.pagerank(graph)
    closeness = nx.closeness_centrality(graph)

    betweenness_sample_size = min(betweenness_k, graph.number_of_nodes())
    if betweenness_sample_size >= graph.number_of_nodes():
        betweenness = nx.betweenness_centrality(graph)
    else:
        betweenness = nx.betweenness_centrality(
            graph,
            k=betweenness_sample_size,
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

    return features_df[feature_columns]