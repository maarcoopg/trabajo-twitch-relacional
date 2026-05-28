from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_mature_distribution_plot(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda un gráfico con la distribución de la variable mature.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df["mature"].value_counts()

    plt.figure(figsize=(6, 4))
    ax = counts.plot(kind="bar", color=["#2a9d8f", "#e76f51"])
    ax.set_title("Distribución de la variable mature")
    ax.set_xlabel("Mature")
    ax.set_ylabel("Número de usuarios")
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
    sns.heatmap(corr, annot=False, cmap="viridis", linewidths=0.5)
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

    y_pred = model.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(confusion_matrix=confusion).plot(cmap="Blues")

    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_model_comparison_plot(results_df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda un gráfico comparando los modelos según todas las métricas.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_columns = [column for column in ["accuracy", "precision", "recall", "f1"] if column in results_df.columns]
    plot_df = results_df.set_index("model")[metric_columns].sort_values(by="f1", ascending=False)

    plt.figure(figsize=(10, 6))
    plot_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Comparación de modelos")
    plt.xlabel("Modelo")
    plt.ylabel("Puntuación")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Métrica")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()