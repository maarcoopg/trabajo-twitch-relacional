from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_model(y_true, y_pred) -> dict:
    """
    Calcula las métricas principales de clasificación.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics


def get_classification_report(y_true, y_pred) -> str:
    """
    Devuelve el classification report en formato texto.
    """
    return classification_report(y_true, y_pred, zero_division=0)


def get_confusion_matrix(y_true, y_pred):
    """
    Devuelve la matriz de confusión.
    """
    return confusion_matrix(y_true, y_pred)


def save_metrics(metrics: dict | list[dict], output_path: str | Path) -> None:
    """
    Guarda una lista de métricas en un CSV.

    Cada elemento de metrics debe ser un diccionario.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(metrics, dict):
        df_metrics = pd.DataFrame([metrics])
    else:
        df_metrics = pd.DataFrame(metrics)

    df_metrics.to_csv(output_path, index=False)


def save_model(model, output_path: str | Path) -> None:
    """
    Guarda un modelo entrenado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)


def load_model(path: str | Path):
    """
    Carga un modelo entrenado.
    """
    return joblib.load(path)