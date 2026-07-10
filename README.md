# AI Repository Discovery

Proyecto de descubrimiento y recomendación de repositorios públicos de inteligencia artificial.

## Objetivo

Explorar un catálogo de repositorios de IA mediante:

* recomendación híbrida basada en contenido, PCA, clúster, popularidad, actividad y crecimiento;
* análisis estructural del grafo de repositorios por topics compartidos.

## Arquitectura

* `src/recommender.py`: carga del catálogo, preparación de features, ranking híbrido y recomendación.
* `src/graph_service.py`: carga del grafo W12, centralidades y subgrafo interactivo con PyVis.
* `dashboard/app.py`: interfaz Streamlit con dos secciones: Recomendador y Grafo interactivo.

## Secciones del dashboard

### Recomendador

* selector de repositorio;
* slider de `top N`;
* ficha del repositorio seleccionado;
* tabla de recomendaciones;
* explicación de cada resultado;
* enlaces a GitHub.

### Grafo interactivo

* selector de repositorio;
* filtro de peso mínimo;
* cantidad máxima de vecinos;
* visualización PyVis;
* tabla de centralidades;
* comparación PageRank vs estrellas.

## Artefactos usados

* `artifacts/W10/repos_with_pca_kmeans_w10.csv`
* `notebooks/artifacts/W12/w12_edge_list.csv`
* `notebooks/artifacts/W12/w12_centrality_ranking.csv`
* `notebooks/artifacts/W12/w12_pagerank_vs_stars.csv`

## Ejecución

Ver `RUNBOOK.md`.

