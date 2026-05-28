from pathlib import Path

import pandas as pd


def json_features_to_dataframe(features_json: dict) -> pd.DataFrame:
    """
    Convierte ES_features.json en un DataFrame sencillo.

    En vez de expandir todas las features del JSON en muchas columnas,
    se crea una variable:
    - num_features: número de características asociadas a cada usuario
    """
    features_df = pd.DataFrame({
        "new_id": [int(node_id) for node_id in features_json.keys()],
        "num_features": [len(values) if values is not None else 0 for values in features_json.values()]
    })

    return features_df


def build_final_dataset(
    graph_features: pd.DataFrame,
    target: pd.DataFrame,
    json_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Une las métricas relacionales del grafo con:
    - datos del usuario
    - variable objetivo mature
    - num_features procedente del JSON
    """
    df = target.copy()

    df = df.merge(
        graph_features,
        on="new_id",
        how="left"
    )

    df = df.merge(
        json_features,
        on="new_id",
        how="left"
    )

    selected_columns = [
        "new_id",
        "degree",
        "degree_centrality",
        "clustering",
        "pagerank",
        "closeness",
        "betweenness",
        "community",
        "num_features",
        "days",
        "views",
        "mature"
    ]

    df = df[selected_columns]

    numeric_columns = [
        "degree",
        "degree_centrality",
        "clustering",
        "pagerank",
        "closeness",
        "betweenness",
        "num_features",
        "days",
        "views",
    ]
    for column in numeric_columns:
        df[column] = df[column].fillna(0)

    df["community"] = df["community"].fillna(-1).astype(int)

    df = df.dropna(subset=["mature"])

    df["mature"] = df["mature"].astype(bool)

    return df


def save_processed_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda el dataset procesado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


def get_feature_target_split(df: pd.DataFrame):
    """
    Separa variables predictoras y variable objetivo.

    Variable objetivo:
    - mature

    Se elimina:
    - new_id, porque es solo identificador
    """
    X = df.drop(columns=["new_id", "mature"])
    y = df["mature"]

    return X, y