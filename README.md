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

## Requisitos cubiertos por el codigo

- Dataset relacional representado como grafo.
- Extraccion manual de metricas relacionales con NetworkX.
- Comparacion de propiedades relacionales, no relacionales y mixtas.
- Entrenamiento de al menos tres clasificadores: Decision Tree, Random Forest y kNN.
- Evaluacion con metricas de clasificacion, validacion cruzada y ajuste de hiperparametros.
- Seleccion y guardado de un modelo final cargable.

La memoria en PDF con formato de articulo cientifico y la presentacion de defensa deben prepararse aparte para completar la entrega indicada en el enunciado.
