from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_mature_distribution_plot(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda un gráfico con la distribución de la variable mature.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df["mature"].value_counts()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Distribución de la variable mature")
    plt.xlabel("Mature")
    plt.ylabel("Número de usuarios")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_correlation_matrix(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda una matriz de correlación de las variables numéricas.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix_plot(model, X_test, y_test, output_path: str | Path) -> None:
    """
    Guarda la matriz de confusión de un modelo.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_model_comparison_plot(results_df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda un gráfico comparando los modelos según F1-score.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = results_df.sort_values("f1", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["f1"])
    plt.title("Comparación de modelos según F1-score")
    plt.xlabel("Modelo")
    plt.ylabel("F1-score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()