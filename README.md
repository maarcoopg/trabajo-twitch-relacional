# Trabajo Twitch Relacional

Repositorio para el trabajo universitario de Aprendizaje Automático Relacional sobre el dataset Twitch Social Networks (ES).

## Estructura de ejecución

1. Ejecutar `notebooks/procesado/notebook_exploracion_twitch.ipynb`.
2. Ejecutar `notebooks/procesado/notebook_features_grafo_twitch.ipynb`.
3. Ejecutar los notebooks de modelos en `notebooks/decision_tree/`, `notebooks/random_forest/` y `notebooks/knn/`.
4. Ejecutar `notebooks/comparativa_final/notebook_comparativa_final_twitch.ipynb`.

## Salidas

- `data/processed/twitch_mature_features.csv`: dataset final procesado.
- `img/`: gráficos generados en la exploración y comparación.
- `results/`: métricas de modelos y tablas comparativas.
- `models/`: modelos entrenados guardados con `joblib`.

## Notas

- Los notebooks usan importaciones desde `src/`.
- Las rutas son relativas al directorio del notebook.
- El objetivo es predecir `mature`.
