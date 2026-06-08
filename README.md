# Trabajo Twitch Relacional

Repositorio para el trabajo universitario de Aprendizaje Automatico Relacional sobre el dataset Twitch Social Networks (ES).

## Objetivo

El proyecto plantea una tarea de clasificacion binaria de nodos: predecir la variable `mature` de cada usuario de Twitch usando informacion relacional del grafo y atributos no relacionales disponibles en el dataset.

## Estructura de ejecucion

1. Ejecutar `notebooks/procesado/notebook_exploracion_twitch.ipynb`.
2. Ejecutar `notebooks/procesado/notebook_features_grafo_twitch.ipynb`.
3. Ejecutar los notebooks de modelos en `notebooks/decision_tree/`, `notebooks/random_forest/` y `notebooks/knn/`.
4. Ejecutar `notebooks/comparativa_final/notebook_comparativa_final_twitch.ipynb`.

## Salidas principales

- `data/processed/twitch_mature_features.csv`: dataset final procesado.
- `img/`: graficos de exploracion, matrices de confusion y comparativa de modelos.
- `results/`: metricas de modelos, validacion cruzada y tablas comparativas.
- `models/`: modelos entrenados guardados con `joblib`, incluido `mejor_modelo_twitch.joblib`.

## Resultados obtenidos

La metrica principal usada para comparar modelos es `f1`, ya que la clase positiva `mature=True` es minoritaria y la exactitud por si sola puede ser enganosa.

Mejores configuraciones ordenadas por `f1`:

| Modelo | Configuracion | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Decision Tree | `max_depth=3` | 0.570 | 0.378 | 0.728 | 0.497 |
| Decision Tree | `max_depth=5` | 0.622 | 0.402 | 0.607 | 0.484 |
| Random Forest | `n_estimators=200`, `max_depth=10` | 0.702 | 0.490 | 0.452 | 0.470 |
| Random Forest | `n_estimators=100`, `max_depth=10` | 0.701 | 0.488 | 0.449 | 0.467 |
| Decision Tree | `max_depth=10` | 0.602 | 0.368 | 0.504 | 0.425 |

El mejor modelo final segun `f1` es `DecisionTreeClassifier(max_depth=3)`, con `f1=0.4975`. Aunque Random Forest alcanza mayor accuracy en algunas configuraciones, el arbol de decision limitado a profundidad 3 obtiene mejor equilibrio entre precision y recall.

Resultados de validacion cruzada usando `f1`:

| Modelo | F1 medio | Desv. tipica |
| --- | ---: | ---: |
| Decision Tree | 0.3665 | 0.0172 |
| Random Forest | 0.2824 | 0.0121 |
| kNN | 0.3104 | 0.0189 |

Nota sobre `NaN` en los CSV: en `dt_grid_metrics.csv` y `rf_grid_metrics.csv`, una celda vacia en `max_depth` representa `max_depth=None`, es decir, arboles sin limite explicito de profundidad. Al leer el CSV con pandas, esas celdas vacias pueden mostrarse como `NaN`, pero no son un error de calculo.

## Requisitos cubiertos por el codigo

- Dataset relacional representado como grafo.
- Extraccion manual de metricas relacionales con NetworkX.
- Comparacion de propiedades relacionales, no relacionales y mixtas.
- Entrenamiento de al menos tres clasificadores: Decision Tree, Random Forest y kNN.
- Evaluacion con metricas de clasificacion, validacion cruzada y ajuste de hiperparametros.
- Seleccion y guardado de un modelo final cargable.
