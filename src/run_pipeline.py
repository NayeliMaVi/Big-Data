from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "repos_clean.csv"
W10_PATH = PROJECT_ROOT / "artifacts" / "W10" / "repos_with_pca_kmeans_w10.csv"
EDGE_LIST_PATH = PROJECT_ROOT / "notebooks" / "artifacts" / "W12" / "w12_edge_list.csv"
CENTRALITY_PATH = PROJECT_ROOT / "notebooks" / "artifacts" / "W12" / "w12_centrality_ranking.csv"
PAGERANK_STARS_PATH = PROJECT_ROOT / "notebooks" / "artifacts" / "W12" / "w12_pagerank_vs_stars.csv"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "pipeline" / "pipeline_report.json"

REQUIRED_CATALOG_COLUMNS = [
    "nombre",
    "descripcion",
    "lenguaje",
    "stars",
    "forks",
    "topics",
    "dias_sin_commit",
    "antiguedad_dias",
    "num_topics",
    "ratio_forks_stars",
    "stars_por_dia",
]


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}")


def _validate_required_files() -> None:
    required_paths = [
        CATALOG_PATH,
        W10_PATH,
        EDGE_LIST_PATH,
        CENTRALITY_PATH,
        PAGERANK_STARS_PATH,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos requeridos para la demo:\n- " + "\n- ".join(missing)
        )


def _validate_catalog() -> pd.DataFrame:
    from src.recommender import load_catalog

    catalog = load_catalog()
    if len(catalog) < 3000:
        raise ValueError(f"El catálogo tiene {len(catalog)} filas y se esperaban al menos 3000.")

    missing_columns = [column for column in REQUIRED_CATALOG_COLUMNS if column not in catalog.columns]
    if missing_columns:
        raise ValueError("Faltan columnas obligatorias en el catálogo: " + ", ".join(missing_columns))

    return catalog


def _prepare_recommender(catalog: pd.DataFrame):
    from src.recommender import prepare_recommender

    recommender = prepare_recommender(catalog)
    required_keys = [
        "catalog",
        "content_matrix",
        "content_similarity",
        "latent_matrix",
        "latent_similarity",
        "clusters",
        "popularity_score",
        "activity_score",
        "growth_score",
    ]
    missing_keys = [key for key in required_keys if key not in recommender]
    if missing_keys:
        raise ValueError("El recomendador no devolvió las claves esperadas: " + ", ".join(missing_keys))

    catalog_len = len(recommender["catalog"])
    if catalog_len != len(catalog):
        raise ValueError("El catálogo preparado no coincide con el catálogo de entrada.")

    return recommender


def _validate_hybrid_recommender(recommender) -> pd.DataFrame:
    from src.recommender import recommend_hybrid

    catalog = recommender["catalog"]
    query = "huggingface/transformers"
    if query not in set(catalog["nombre"].astype(str)):
        raise KeyError(f"No se encontró el repositorio de prueba: {query}")

    recommendations = recommend_hybrid(query, top_n=5, recommender=recommender)
    if len(recommendations) != 5:
        raise ValueError(f"Se esperaban 5 resultados y se obtuvieron {len(recommendations)}.")

    if query in set(recommendations["Repositorio recomendado"].astype(str)):
        raise ValueError("La recomendación híbrida incluyó el repositorio de consulta.")

    print(
        "Repos recomendados: "
        + ", ".join(recommendations["Repositorio recomendado"].astype(str).tolist())
    )
    return recommendations


def _validate_preferences_recommender(recommender) -> pd.DataFrame:
    from src.recommender import recommend_by_preferences

    results = recommend_by_preferences(
        language="Python",
        topics=["computer-vision"],
        min_stars=1000,
        max_days_without_commit=180,
        top_n=5,
        recommender=recommender,
    )
    print(f"Resultados por preferencias: {len(results)}")
    return results


def _validate_graph():
    from src.graph_service import (
        build_interactive_subgraph,
        load_centrality_ranking,
        load_graph_edges,
        load_pagerank_vs_stars,
    )

    edges = load_graph_edges()
    if len(edges) <= 20000:
        raise ValueError(f"El grafo tiene {len(edges)} aristas y se esperaban más de 20000.")

    centrality = load_centrality_ranking()
    pagerank_vs_stars = load_pagerank_vs_stars()

    if centrality.empty:
        raise ValueError("El ranking de centralidad está vacío.")
    if pagerank_vs_stars.empty:
        raise ValueError("La comparación PageRank vs estrellas está vacía.")

    graph_html = ""
    graph_name = "tensorflow/tensorflow"
    graph_payload = build_interactive_subgraph(
        selected_repository=graph_name,
        max_neighbors=8,
        min_weight=4,
    )
    graph_html = graph_payload.get("html", "")
    if not graph_html or not graph_html.strip():
        raise ValueError("El HTML del grafo interactivo está vacío.")

    return {
        "edges": len(edges),
        "centrality_rows": len(centrality),
        "pagerank_vs_stars_rows": len(pagerank_vs_stars),
        "subgraph_nodes": graph_payload.get("graph_stats", {}).get("subgrafo_nodos", 0),
        "subgraph_edges": graph_payload.get("graph_stats", {}).get("subgrafo_aristas", 0),
        "selected_repository": graph_name,
    }


def _write_report(payload: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_validated": [],
        "catalog_rows": None,
        "graph_edges": None,
        "recommender_status": "pending",
        "preferences_status": "pending",
        "graph_status": "pending",
        "errors": [],
    }

    try:
        _validate_required_files()
        report["files_validated"] = [
            str(CATALOG_PATH),
            str(W10_PATH),
            str(EDGE_LIST_PATH),
            str(CENTRALITY_PATH),
            str(PAGERANK_STARS_PATH),
        ]
        _ok("Archivos validados")

        catalog = _validate_catalog()
        report["catalog_rows"] = len(catalog)
        _ok("Dataset validado")

        recommender = _prepare_recommender(catalog)
        _ok("Recomendador preparado")

        hybrid_results = _validate_hybrid_recommender(recommender)
        report["recommender_status"] = {
            "status": "ok",
            "rows": int(len(hybrid_results)),
            "query": "huggingface/transformers",
            "top_names": hybrid_results["Repositorio recomendado"].astype(str).tolist(),
        }
        _ok("Recomendación híbrida validada")

        preferences_results = _validate_preferences_recommender(recommender)
        report["preferences_status"] = {
            "status": "ok",
            "rows": int(len(preferences_results)),
            "query": {
                "language": "Python",
                "topics": ["computer-vision"],
                "min_stars": 1000,
                "max_days_without_commit": 180,
            },
        }
        _ok("Búsqueda por preferencias validada")

        graph_status = _validate_graph()
        report["graph_edges"] = graph_status["edges"]
        report["graph_status"] = {
            "status": "ok",
            "selected_repository": graph_status["selected_repository"],
            "edges": graph_status["edges"],
            "centrality_rows": graph_status["centrality_rows"],
            "pagerank_vs_stars_rows": graph_status["pagerank_vs_stars_rows"],
            "subgraph_nodes": graph_status["subgraph_nodes"],
            "subgraph_edges": graph_status["subgraph_edges"],
        }
        _ok("Grafo validado")

        _write_report(report)
        _ok("Pipeline completado")
        return 0

    except Exception as error:
        report["errors"].append(str(error))
        try:
            _write_report(report)
        except Exception as report_error:
            _fail(f"No se pudo escribir el reporte: {report_error}")
        _fail(str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
