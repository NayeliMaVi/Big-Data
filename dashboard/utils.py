from pathlib import Path

import pandas as pd


# Ruta absoluta de la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPOSITORIES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "repos_clean.csv"
)

CENTRALITY_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "artifacts"
    / "W12"
    / "w12_centrality_ranking.csv"
)

PAGERANK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "artifacts"
    / "W12"
    / "w12_pagerank_vs_stars.csv"
)


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Lee un CSV y muestra un error claro si no existe."""

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido: {path}"
        )

    return pd.read_csv(path)


def load_repositories() -> pd.DataFrame:
    """Carga el catálogo principal de repositorios."""

    return read_csv_safely(REPOSITORIES_PATH)


def load_centrality() -> pd.DataFrame:
    """Carga el ranking de centralidad del grafo."""

    return read_csv_safely(CENTRALITY_PATH)


def load_pagerank_comparison() -> pd.DataFrame:
    """Carga la comparación entre PageRank y estrellas."""

    return read_csv_safely(PAGERANK_PATH)