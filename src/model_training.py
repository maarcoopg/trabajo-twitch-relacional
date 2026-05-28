from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Divide los datos en entrenamiento y test.
    Se usa stratify para mantener la proporción de mature/no mature.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def get_base_models(random_state: int = 42) -> dict:
    """
    Devuelve los modelos base que se van a comparar.
    """
    models = {
        "decision_tree": DecisionTreeClassifier(random_state=random_state),

        "random_forest": RandomForestClassifier(random_state=random_state),

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ]),

        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=random_state))
        ])
    }

    return models


def get_param_grids() -> dict:
    """
    Devuelve los grids de hiperparámetros para cada modelo.
    """
    param_grids = {
        "decision_tree": {
            "max_depth": [None, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced"]
        },

        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced"]
        },

        "knn": {
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]
        },

        "logistic_regression": {
            "logreg__C": [0.01, 0.1, 1, 10],
            "logreg__class_weight": [None, "balanced"]
        }
    }

    return param_grids


def train_model(model, X_train, y_train):
    """
    Entrena un modelo concreto.
    """
    model.fit(X_train, y_train)
    return model


def train_with_grid_search(
    model,
    param_grid: dict,
    X_train,
    y_train,
    scoring: str = "f1",
    cv: int = 5
):
    """
    Entrena un modelo usando GridSearchCV.
    """
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid