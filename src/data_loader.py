import json
from pathlib import Path

import pandas as pd


def load_edges(path: str | Path) -> pd.DataFrame:
    """
    Carga el archivo de aristas ES_edges.csv.

    Debe contener las columnas:
    - from
    - to
    """
    path = Path(path)
    edges = pd.read_csv(path)

    required_columns = {"from", "to"}
    if not required_columns.issubset(edges.columns):
        raise ValueError(f"El archivo de aristas debe contener las columnas {required_columns}")

    return edges


def load_target(path: str | Path) -> pd.DataFrame:
    """
    Carga el archivo ES_target.csv.

    Debe contener, al menos:
    - new_id
    - mature
    """
    path = Path(path)
    target = pd.read_csv(path)

    required_columns = {"new_id", "mature"}
    if not required_columns.issubset(target.columns):
        raise ValueError(f"El archivo target debe contener las columnas {required_columns}")

    return target


def load_json_features(path: str | Path) -> dict:
    """
    Carga el archivo ES_features.json.

    Devuelve un diccionario donde:
    - clave: id del nodo
    - valor: lista de features asociadas al nodo
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        features = json.load(file)

    return features


def load_processed_dataset(path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset ya procesado con las features relacionales.
    """
    path = Path(path)
    return pd.read_csv(path)