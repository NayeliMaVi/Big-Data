from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import (
    MinMaxScaler,
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
    normalize,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
W10_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "W10" / "repos_with_pca_kmeans_w10.csv"
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "repos_clean.csv"

TOPIC_COUNT = 30
PCA_COMPONENTS = 10
KMEANS_CLUSTERS = 5

TFIDF_MAX_FEATURES = 1200
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_TOKEN_PATTERN = r"(?u)\b[\w\-\.\+/#]+\b"

HYBRID_WEIGHTS = {
    "content": 0.35,
    "pca_similarity": 0.20,
    "same_cluster": 0.15,
    "popularity": 0.12,
    "activity": 0.10,
    "growth": 0.08,
}


def parse_topics(value) -> List[str]:
    if isinstance(value, list):
        return [str(topic).strip().lower() for topic in value if str(topic).strip()]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, list):
        return [str(topic).strip().lower() for topic in parsed if str(topic).strip()]

    return [topic.strip().lower() for topic in text.split(",") if topic.strip()]


def _numeric_column(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype=float)

    return (
        pd.to_numeric(dataframe[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(float)
    )


def _resolve_stars_per_day_column(dataframe: pd.DataFrame) -> pd.Series:
    if "stars_por_dia" in dataframe.columns:
        return _numeric_column(dataframe, "stars_por_dia")
    if "stars_per_day" in dataframe.columns:
        return _numeric_column(dataframe, "stars_per_day")
    return dataframe["stars"] / dataframe["antiguedad_dias"].replace(0, 1)


def _prepare_base_catalog(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    required_columns = [
        "nombre",
        "descripcion",
        "lenguaje",
        "topics",
        "stars",
        "forks",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(missing_columns))

    dataframe["nombre"] = dataframe["nombre"].fillna("Repositorio sin nombre").astype(str).str.strip()
    dataframe["descripcion"] = dataframe["descripcion"].fillna("No description").astype(str).str.strip()
    dataframe["lenguaje"] = dataframe["lenguaje"].fillna("Unknown").astype(str).str.strip()
    dataframe["topics_list"] = dataframe["topics"].apply(parse_topics)
    dataframe["stars"] = _numeric_column(dataframe, "stars")
    dataframe["forks"] = _numeric_column(dataframe, "forks")
    dataframe["dias_sin_commit"] = _numeric_column(dataframe, "dias_sin_commit")
    dataframe["antiguedad_dias"] = _numeric_column(dataframe, "antiguedad_dias", default=1).replace(0, 1)

    if "num_topics" in dataframe.columns:
        dataframe["num_topics"] = _numeric_column(dataframe, "num_topics")
    else:
        dataframe["num_topics"] = dataframe["topics_list"].apply(len).astype(float)

    if "ratio_forks_stars" in dataframe.columns:
        dataframe["ratio_forks_stars"] = _numeric_column(dataframe, "ratio_forks_stars")
    else:
        dataframe["ratio_forks_stars"] = dataframe["forks"] / dataframe["stars"].replace(0, 1)

    dataframe["stars_por_dia"] = _resolve_stars_per_day_column(dataframe)

    dataframe = dataframe.drop_duplicates(subset=["nombre"], keep="first").reset_index(drop=True)
    return dataframe


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    if W10_ARTIFACT_PATH.exists():
        dataframe = pd.read_csv(W10_ARTIFACT_PATH)
    elif CATALOG_PATH.exists():
        dataframe = pd.read_csv(CATALOG_PATH)
    else:
        raise FileNotFoundError(
            "No se encontró ni artifacts/W10/repos_with_pca_kmeans_w10.csv ni data/processed/repos_clean.csv"
        )

    return _prepare_base_catalog(dataframe)


def _join_topics(topics: List[str]) -> str:
    return " ".join(topics) if isinstance(topics, list) else ""


def _build_content_text(dataframe: pd.DataFrame) -> pd.Series:
    topics_text = dataframe["topics_list"].apply(_join_topics)
    return (
        dataframe["descripcion"].fillna("").astype(str)
        + " "
        + ((topics_text.fillna("").astype(str) + " ") * 3)
        + dataframe["lenguaje"].fillna("Unknown").astype(str)
    )


def _build_top_topics(dataframe: pd.DataFrame, limit: int = TOPIC_COUNT) -> List[str]:
    all_topics = [topic for topics in dataframe["topics_list"] for topic in topics]
    topic_counts = pd.Series(all_topics).value_counts()
    return topic_counts.head(limit).index.tolist()


def _fit_language_encoder(dataframe: pd.DataFrame):
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    return encoder.fit_transform(dataframe[["lenguaje"]])


def _fit_topic_encoder(dataframe: pd.DataFrame, top_topics: List[str]) -> csr_matrix:
    filtered_topics = dataframe["topics_list"].apply(
        lambda topics: [topic for topic in topics if topic in top_topics]
    )
    try:
        encoder = MultiLabelBinarizer(classes=top_topics, sparse_output=True)
        topic_matrix = encoder.fit_transform(filtered_topics)
    except TypeError:
        encoder = MultiLabelBinarizer(classes=top_topics)
        topic_matrix = encoder.fit_transform(filtered_topics)
        topic_matrix = csr_matrix(topic_matrix)
    return csr_matrix(topic_matrix)


def _fit_pca_and_clusters(feature_matrix: csr_matrix, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    dense_features = feature_matrix.toarray()
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    latent_matrix = pca.fit_transform(dense_features)
    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=100)
    clusters = kmeans.fit_predict(latent_matrix)
    return latent_matrix, clusters


def prepare_recommender(catalog: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    dataframe = _prepare_base_catalog(catalog if catalog is not None else load_catalog())

    pca_columns = [f"pca_kmeans_{index}" for index in range(1, PCA_COMPONENTS + 1)]
    has_precomputed_pca = all(column in dataframe.columns for column in pca_columns)
    has_precomputed_clusters = "cluster_kmeans" in dataframe.columns

    dataframe["recommendation_text"] = _build_content_text(dataframe)

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        token_pattern=TFIDF_TOKEN_PATTERN,
        lowercase=True,
    )
    content_matrix = tfidf.fit_transform(dataframe["recommendation_text"])
    content_similarity = cosine_similarity(content_matrix).astype(np.float32)

    numeric_columns = [
        "stars",
        "forks",
        "dias_sin_commit",
        "antiguedad_dias",
        "num_topics",
        "ratio_forks_stars",
        "stars_por_dia",
    ]

    numeric_matrix = StandardScaler().fit_transform(dataframe[numeric_columns])
    numeric_matrix = csr_matrix(numeric_matrix)
    language_matrix = _fit_language_encoder(dataframe)
    top_topics = _build_top_topics(dataframe, TOPIC_COUNT)
    topic_matrix = _fit_topic_encoder(dataframe, top_topics)

    feature_matrix = hstack([numeric_matrix, language_matrix, topic_matrix], format="csr")

    if has_precomputed_pca:
        latent_matrix = dataframe[pca_columns].astype(float).to_numpy()
        if has_precomputed_clusters:
            clusters = dataframe["cluster_kmeans"].astype(int).to_numpy()
        else:
            _, clusters = _fit_pca_and_clusters(feature_matrix, dataframe)
    else:
        latent_matrix, clusters = _fit_pca_and_clusters(feature_matrix, dataframe)
        for index in range(PCA_COMPONENTS):
            dataframe[pca_columns[index]] = latent_matrix[:, index]

    latent_similarity = cosine_similarity(latent_matrix).astype(np.float32)

    popularity_features = pd.DataFrame(
        {
            "log_stars": np.log1p(dataframe["stars"]),
            "log_forks": np.log1p(dataframe["forks"]),
            "log_stars_por_dia": np.log1p(dataframe["stars_por_dia"].clip(lower=0)),
        }
    )
    popularity_scaled = MinMaxScaler().fit_transform(popularity_features)
    popularity_score = (
        0.60 * popularity_scaled[:, 0]
        + 0.25 * popularity_scaled[:, 1]
        + 0.15 * popularity_scaled[:, 2]
    )

    activity_raw = 1 / (1 + dataframe["dias_sin_commit"].to_numpy())
    activity_score = MinMaxScaler().fit_transform(activity_raw.reshape(-1, 1)).ravel()

    growth_raw = np.log1p(dataframe["stars_por_dia"].clip(lower=0).to_numpy()).reshape(-1, 1)
    growth_score = MinMaxScaler().fit_transform(growth_raw).ravel()

    cluster_names = {
        0: "Herramientas de IA modernas",
        1: "Proyectos establecidos y educativos",
        2: "Ecosistema LLM emergente",
        3: "Proyectos masivos e influyentes",
        4: "Repos virales recientes",
    }

    if not has_precomputed_clusters:
        dataframe["cluster_kmeans"] = clusters
    dataframe["cluster_nombre"] = dataframe["cluster_kmeans"].map(cluster_names)

    return {
        "catalog": dataframe,
        "content_matrix": content_matrix,
        "content_similarity": content_similarity,
        "latent_matrix": latent_matrix,
        "latent_similarity": latent_similarity,
        "clusters": dataframe["cluster_kmeans"].to_numpy(),
        "cluster_names": cluster_names,
        "popularity_score": popularity_score,
        "activity_score": activity_score,
        "growth_score": growth_score,
        "top_topics": top_topics,
        "tfidf": tfidf,
    }


def _get_recommender_state(recommender: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if recommender is None:
        return prepare_recommender()
    return recommender


def recommend_hybrid(
    repository_name: str,
    top_n: int = 10,
    recommender: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    state = _get_recommender_state(recommender)
    catalog = state["catalog"]

    matches = catalog.index[catalog["nombre"] == repository_name].tolist()
    if not matches:
        return pd.DataFrame()

    query_idx = matches[0]
    content_scores = state["content_similarity"][query_idx].copy()
    pca_scores = state["latent_similarity"][query_idx].copy()
    cluster_labels = state["clusters"]
    selected_cluster = cluster_labels[query_idx]
    same_cluster = (cluster_labels == selected_cluster).astype(float)
    popularity_score = state["popularity_score"]
    activity_score = state["activity_score"]
    growth_score = state["growth_score"]

    hybrid_scores = (
        HYBRID_WEIGHTS["content"] * content_scores
        + HYBRID_WEIGHTS["pca_similarity"] * pca_scores
        + HYBRID_WEIGHTS["same_cluster"] * same_cluster
        + HYBRID_WEIGHTS["popularity"] * popularity_score
        + HYBRID_WEIGHTS["activity"] * activity_score
        + HYBRID_WEIGHTS["growth"] * growth_score
    )
    hybrid_scores[query_idx] = -np.inf

    ranked_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    selected_topics = set(catalog.iloc[query_idx]["topics_list"])
    selected_language = catalog.iloc[query_idx]["lenguaje"]

    rows = []
    for position, candidate_idx in enumerate(ranked_indices, start=1):
        candidate = catalog.iloc[candidate_idx]
        candidate_topics = set(candidate["topics_list"])
        shared_topics = sorted(selected_topics.intersection(candidate_topics))
        candidate_cluster = int(cluster_labels[candidate_idx])
        candidate_same_cluster = int(candidate_cluster == selected_cluster)
        candidate_popularity = float(popularity_score[candidate_idx])
        candidate_activity = float(activity_score[candidate_idx])
        candidate_growth = float(growth_score[candidate_idx])

        reasons = []
        if shared_topics:
            reasons.append("Topics compartidos: " + ", ".join(shared_topics[:5]))
        if candidate_same_cluster:
            reasons.append("Mismo clúster")
        if candidate["lenguaje"] == selected_language:
            reasons.append("Mismo lenguaje principal")
        if not reasons:
            reasons.append("Alta similitud en el espacio de características")

        rows.append(
            {
                "Posición": position,
                "Repositorio recomendado": candidate["nombre"],
                "Score híbrido": round(float(hybrid_scores[candidate_idx]), 4),
                "Similitud de contenido": round(float(content_scores[candidate_idx]), 4),
                "Similitud PCA": round(float(pca_scores[candidate_idx]), 4),
                "Mismo clúster": candidate_same_cluster,
                "Lenguaje": candidate["lenguaje"],
                "Estrellas": int(candidate["stars"]),
                "Forks": int(candidate["forks"]),
                "Topics compartidos": ", ".join(shared_topics) if shared_topics else "",
                "Motivo de recomendación": ". ".join(reasons),
                "URL": candidate.get("url", ""),
                "Bonus clúster": float(HYBRID_WEIGHTS["same_cluster"] * candidate_same_cluster),
                "Popularidad": candidate_popularity,
                "Actividad": candidate_activity,
                "Crecimiento": candidate_growth,
            }
        )

    return pd.DataFrame(rows)


def recommend_by_preferences(
    language: Optional[str] = None,
    topics: Optional[List[str]] = None,
    min_stars: float = 0,
    max_days_without_commit: Optional[float] = None,
    cluster: Optional[int] = None,
    top_n: int = 10,
    recommender: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    state = _get_recommender_state(recommender)
    catalog = state["catalog"].copy()

    filtered = catalog.copy()

    if language:
        filtered = filtered[filtered["lenguaje"] == language]

    filtered = filtered[filtered["stars"] >= float(min_stars)]

    if max_days_without_commit is not None:
        filtered = filtered[filtered["dias_sin_commit"] <= float(max_days_without_commit)]

    if cluster is not None:
        filtered = filtered[filtered["cluster_kmeans"].astype(int) == int(cluster)]

    selected_topics = [topic for topic in (topics or []) if topic]
    topic_match_scores = []
    language_match_scores = []
    popularity_scores = []
    activity_scores = []
    growth_scores = []

    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "Posición",
                "Repositorio",
                "Score de preferencias",
                "Coincidencia de topics",
                "Coincidencia de lenguaje",
                "Popularidad",
                "Actividad",
                "Crecimiento",
                "Lenguaje",
                "Estrellas",
                "Forks",
                "Días sin commit",
                "Clúster",
                "Topics coincidentes",
                "Motivo",
                "URL",
            ]
        )

    if selected_topics:
        for repo_topics in filtered["topics_list"]:
            matches = len(set(selected_topics).intersection(set(repo_topics)))
            topic_match_scores.append(matches / len(selected_topics))
    else:
        topic_match_scores = [0.0] * len(filtered)

    if language:
        language_match_scores = [1.0] * len(filtered)
    else:
        language_match_scores = [0.0] * len(filtered)

    index_map = filtered.index.to_list()
    popularity_scores = state["popularity_score"][index_map].astype(float).tolist()
    activity_scores = state["activity_score"][index_map].astype(float).tolist()
    growth_scores = state["growth_score"][index_map].astype(float).tolist()

    preferences_score = (
        0.40 * np.array(topic_match_scores)
        + 0.20 * np.array(language_match_scores)
        + 0.15 * np.array(popularity_scores)
        + 0.15 * np.array(activity_scores)
        + 0.10 * np.array(growth_scores)
    )

    filtered = filtered.copy()
    filtered["topic_match_score"] = topic_match_scores
    filtered["language_match_score"] = language_match_scores
    filtered["popularidad"] = popularity_scores
    filtered["actividad"] = activity_scores
    filtered["crecimiento"] = growth_scores
    filtered["preferences_score"] = preferences_score

    filtered = filtered.sort_values(
        ["preferences_score", "stars", "forks"],
        ascending=[False, False, False],
    ).head(top_n)

    rows = []
    for position, (_, row) in enumerate(filtered.iterrows(), start=1):
        matched_topics = []
        if selected_topics:
            matched_topics = sorted(set(selected_topics).intersection(set(row["topics_list"])))

        reasons = []
        if selected_topics:
            reasons.append(
                f"Coincide {len(matched_topics)} de {len(selected_topics)} topics seleccionados"
            )
        if language:
            reasons.append("Coincide con el lenguaje seleccionado")
        if row["stars"] >= float(min_stars):
            reasons.append(f"Cumple el mínimo de {int(min_stars)} estrellas")
        if max_days_without_commit is not None:
            reasons.append(f"Se mantiene dentro de {int(max_days_without_commit)} días sin commit")
        if cluster is not None:
            reasons.append(f"Pertenece al clúster {int(cluster)}")
        if not reasons:
            reasons.append("Cumple los criterios de preferencias seleccionados")

        rows.append(
            {
                "Posición": position,
                "Repositorio": row["nombre"],
                "Score de preferencias": round(float(row["preferences_score"]), 4),
                "Coincidencia de topics": round(float(row["topic_match_score"]), 4),
                "Coincidencia de lenguaje": round(float(row["language_match_score"]), 4),
                "Popularidad": round(float(row["popularidad"]), 4),
                "Actividad": round(float(row["actividad"]), 4),
                "Crecimiento": round(float(row["crecimiento"]), 4),
                "Lenguaje": row["lenguaje"],
                "Estrellas": int(row["stars"]),
                "Forks": int(row["forks"]),
                "Días sin commit": int(row["dias_sin_commit"]),
                "Clúster": int(row.get("cluster_kmeans", -1)),
                "Topics coincidentes": ", ".join(matched_topics) if matched_topics else "",
                "Motivo": ". ".join(reasons),
                "URL": row.get("url", ""),
            }
        )

    return pd.DataFrame(rows)


def get_repository_row(repository_name: str, recommender: Optional[Dict[str, object]] = None) -> pd.Series:
    state = _get_recommender_state(recommender)
    catalog = state["catalog"]
    matches = catalog.index[catalog["nombre"] == repository_name].tolist()
    if not matches:
        raise KeyError(f"No se encontró el repositorio: {repository_name}")
    return catalog.iloc[matches[0]]


def list_repository_names(recommender: Optional[Dict[str, object]] = None) -> List[str]:
    state = _get_recommender_state(recommender)
    return sorted(state["catalog"]["nombre"].dropna().astype(str).unique().tolist())
