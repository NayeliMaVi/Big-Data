from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_service import (
    build_interactive_subgraph,
    load_centrality_ranking,
    load_pagerank_vs_stars,
)
from src.recommender import (
    get_repository_row,
    load_catalog,
    prepare_recommender,
    recommend_by_preferences,
    recommend_hybrid,
)


st.set_page_config(
    page_title="AI Repository Discovery",
    page_icon="🔎",
    layout="wide",
)


def container_border():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "N/A"


def fmt_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def score_label(score: float) -> str:
    if score >= 0.75:
        return "Alta afinidad"
    if score >= 0.65:
        return "Afinidad media"
    return "Afinidad moderada"


def format_repository_option(row: pd.Series) -> str:
    return f"{row['nombre']} — {row['lenguaje']} — {fmt_int(row['stars'])} estrellas"


st.title("AI Repository Discovery")
st.caption(
    "Dashboard para descubrir repositorios similares de inteligencia artificial y explorar su red temática."
)


@st.cache_data(show_spinner=False)
def cached_catalog() -> pd.DataFrame:
    return load_catalog()


@st.cache_resource(show_spinner=False)
def cached_recommender() -> Dict[str, object]:
    return prepare_recommender(cached_catalog())


@st.cache_data(show_spinner=False)
def cached_centrality() -> pd.DataFrame:
    return load_centrality_ranking()


@st.cache_data(show_spinner=False)
def cached_pagerank_vs_stars() -> pd.DataFrame:
    return load_pagerank_vs_stars()


catalog = cached_catalog()
recommender = cached_recommender()
centrality_df = cached_centrality()
pagerank_vs_stars_df = cached_pagerank_vs_stars()
repository_names = sorted(catalog["nombre"].dropna().astype(str).unique().tolist())

section = st.sidebar.radio("Navegación", ["Recomendador", "Grafo interactivo"])


def repository_filters(df: pd.DataFrame) -> pd.DataFrame:
    languages = ["Todos"] + sorted(df["lenguaje"].dropna().astype(str).unique().tolist())
    selected_language = st.selectbox(
        "Filtro por lenguaje",
        languages,
        help="Reduce el catálogo por lenguaje principal.",
    )

    topic_options = ["Todos"] + sorted({topic for topics in df["topics_list"] for topic in topics})
    selected_topic = st.selectbox(
        "Filtro por topic",
        topic_options,
        help="Opcional: limita el catálogo a repositorios con ese topic.",
    )

    name_query = st.text_input(
        "Buscar por nombre",
        placeholder="Escribe parte del nombre del repositorio",
        help="Busca por coincidencia parcial en el nombre.",
    )

    filtered = df.copy()
    if selected_language != "Todos":
        filtered = filtered[filtered["lenguaje"] == selected_language]
    if selected_topic != "Todos":
        filtered = filtered[filtered["topics_list"].apply(lambda topics: selected_topic in topics)]
    if name_query.strip():
        filtered = filtered[filtered["nombre"].str.contains(name_query.strip(), case=False, na=False)]

    return filtered.reset_index(drop=True)


def render_repository_card(selected_row: pd.Series, selected_cluster: int) -> None:
    with container_border():
        st.subheader(selected_row["nombre"])
        st.write(selected_row["descripcion"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lenguaje", selected_row["lenguaje"])
        col2.metric("Estrellas", fmt_int(selected_row["stars"]))
        col3.metric("Forks", fmt_int(selected_row["forks"]))
        col4.metric("Clúster", str(selected_cluster))

        col5, col6, col7 = st.columns(3)
        col5.metric("Actividad reciente", fmt_int(selected_row["dias_sin_commit"]))
        col6.metric("Stars por día", fmt_float(selected_row["stars_por_dia"], 3))
        col7.metric("Topics", fmt_int(selected_row["num_topics"]))

        topics_text = ", ".join(selected_row["topics_list"]) if selected_row["topics_list"] else "Sin topics disponibles"
        st.markdown(f"**Topics:** {topics_text}")

        if isinstance(selected_row.get("url", ""), str) and selected_row["url"].strip():
            st.markdown(f"[Abrir en GitHub]({selected_row['url']})")


def render_recommendation_card(row: pd.Series) -> None:
    score = float(row["Score híbrido"])
    with container_border():
        st.markdown(f"### {int(row['Posición'])}. {row['Repositorio recomendado']}")
        st.caption(score_label(score))
        st.progress(min(max(score, 0.0), 1.0))

        c1, c2, c3 = st.columns(3)
        c1.metric("Score híbrido", fmt_float(score, 4))
        c2.metric("Lenguaje", row["Lenguaje"])
        c3.metric("Clúster", str(row["Mismo clúster"]))

        st.write(f"**Estrellas:** {fmt_int(row['Estrellas'])} | **Forks:** {fmt_int(row['Forks'])}")
        if row.get("Topics compartidos"):
            st.write(f"**Topics compartidos:** {row['Topics compartidos']}")
        st.write(f"**Motivo:** {row['Motivo de recomendación']}")
        if isinstance(row.get("URL", ""), str) and row["URL"].strip():
            st.markdown(f"[Abrir en GitHub]({row['URL']})")


def render_signal_breakdown(row: pd.Series) -> None:
    content_value = float(
        row.get(
            "Similitud de contenido",
            row.get("Similitud contenido", 0.0),
        )
    )

    pca_value = float(
        row.get(
            "Similitud PCA",
            0.0,
        )
    )

    cluster_raw = row.get(
        "Bonus clúster",
        row.get(
            "Bonus de clúster",
            row.get(
                "Mismo clúster",
                0.0,
            ),
        ),
    )

    # Por si "Mismo clúster" viene como True/False
    if isinstance(cluster_raw, bool):
        cluster_value = 1.0 if cluster_raw else 0.0
    else:
        cluster_value = float(cluster_raw)

    popularity_value = float(
        row.get(
            "Popularidad",
            row.get("Score popularidad", 0.0),
        )
    )

    activity_value = float(
        row.get(
            "Actividad",
            row.get("Score actividad", 0.0),
        )
    )

    growth_value = float(
        row.get(
            "Crecimiento",
            row.get("Score crecimiento", 0.0),
        )
    )

    signals = [
        ("Similitud de contenido", content_value),
        ("Similitud PCA", pca_value),
        ("Bonus de clúster", cluster_value),
        ("Popularidad", popularity_value),
        ("Actividad", activity_value),
        ("Crecimiento", growth_value),
    ]

    with st.expander(
        f"Desglose de señales: {row['Repositorio recomendado']}"
    ):
        for label, raw_value in signals:
            visual_value = min(
                max(float(raw_value), 0.0),
                1.0,
            )

            st.caption(
                f"{label}: {fmt_float(raw_value, 4)}"
            )

            st.progress(visual_value)


def render_hybrid_recommender_tab() -> None:
    st.header("Descubre repositorios similares de inteligencia artificial")
    st.write(
        "Selecciona un repositorio del catálogo y el sistema comparará su contenido, topics, lenguaje, posición en PCA, clúster, popularidad, actividad y crecimiento para recomendar proyectos relacionados."
    )

    with container_border():
        step1, step2, step3 = st.columns(3)
        step1.markdown("**1. Selecciona un repositorio**")
        step1.caption("Filtra por lenguaje, topic y nombre para encontrar un punto de partida.")
        step2.markdown("**2. El sistema analiza sus características**")
        step2.caption("El ranking combina contenido, PCA, clúster y señales de actividad.")
        step3.markdown("**3. Explora las mejores recomendaciones**")
        step3.caption("Revisa las tarjetas, la tabla y el desglose de señales.")

    with st.expander("¿Cómo se calcula la recomendación?", expanded=False):
        st.markdown(
            "- 35 % similitud de contenido\n"
            "- 20 % similitud PCA\n"
            "- 15 % mismo clúster\n"
            "- 12 % popularidad\n"
            "- 10 % actividad\n"
            "- 8 % crecimiento"
        )
        st.caption(
            "Contenido: texto, topics y lenguaje. PCA: cercanía en el espacio latente. Clúster: mismo segmento. Popularidad, actividad y crecimiento añaden señales de relevancia y vigencia."
        )

    st.info(
        "Empieza seleccionando un lenguaje o escribe parte del nombre. Las recomendaciones se ordenan de mayor a menor afinidad."
    )

    filtered_catalog = repository_filters(catalog)
    if filtered_catalog.empty:
        st.warning("No hay repositorios con los filtros seleccionados.")
        return

    option_to_repo = {format_repository_option(row): row["nombre"] for _, row in filtered_catalog.iterrows()}
    selected_option = st.selectbox(
        "Repositorio de referencia",
        list(option_to_repo.keys()),
        help="Selecciona el repositorio que servirá como referencia para el ranking.",
    )
    selected_repository = option_to_repo[selected_option]
    top_n = st.slider("Número de recomendaciones", min_value=5, max_value=20, value=10, step=1)

    selected_row = get_repository_row(selected_repository, recommender)
    selected_index = catalog.index[catalog["nombre"] == selected_repository][0]
    selected_cluster = int(recommender["clusters"][selected_index])
    render_repository_card(selected_row, selected_cluster)

    recommendations = recommend_hybrid(selected_repository, top_n=top_n, recommender=recommender)

    st.subheader("Recomendaciones principales")
    if recommendations.empty:
        st.warning("No se encontraron recomendaciones.")
    else:
        top3 = recommendations.head(3)
        top_cols = st.columns(3)
        for idx, (_, row) in enumerate(top3.iterrows()):
            with top_cols[idx]:
                render_recommendation_card(row)

        st.subheader("Detalle del Top 5")
        for _, row in recommendations.head(5).iterrows():
            render_signal_breakdown(row)

        st.subheader("Tabla completa del Top N")
        st.dataframe(
            recommendations[
                [
                    "Posición",
                    "Repositorio recomendado",
                    "Score híbrido",
                    "Similitud de contenido",
                    "Similitud PCA",
                    "Mismo clúster",
                    "Lenguaje",
                    "Estrellas",
                    "Forks",
                    "Topics compartidos",
                    "Motivo de recomendación",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_preferences_tab() -> None:
    st.header("Encuentra repositorios según tus necesidades")
    st.write(
        "Configura lenguaje, topics, popularidad y actividad para descubrir los proyectos que mejor se ajustan a tus criterios."
    )

    with st.expander("¿Cómo se calcula esta búsqueda?", expanded=False):
        st.markdown(
            "Este modo es una búsqueda personalizada distinta al recomendador híbrido item-to-item. Filtra primero por tus criterios y luego ordena los resultados con un score de preferencias."
        )
        st.markdown(
            "- 40 % coincidencia de topics\n"
            "- 20 % coincidencia de lenguaje\n"
            "- 15 % popularidad\n"
            "- 15 % actividad\n"
            "- 10 % crecimiento"
        )

    languages = ["Cualquiera"] + sorted(catalog["lenguaje"].dropna().astype(str).unique().tolist())
    cluster_options = ["Cualquiera"] + [f"{int(cluster)}" for cluster in sorted(catalog["cluster_kmeans"].dropna().astype(int).unique().tolist())]
    topic_options = sorted({topic for topics in catalog["topics_list"] for topic in topics})

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        selected_language = st.selectbox(
            "Lenguaje principal",
            languages,
            help="Filtra por lenguaje principal o deja Cualquiera.",
        )
        selected_topics = st.multiselect(
            "Topics",
            topic_options,
            help="Selecciona uno o varios topics para buscar coincidencias.",
        )
        min_stars = st.number_input(
            "Estrellas mínimas",
            min_value=0,
            value=0,
            step=50,
            help="Descarta repositorios por debajo de este umbral.",
        )

    with control_col2:
        max_days_without_commit = st.slider(
            "Máximo de días sin commit",
            min_value=0,
            max_value=int(catalog["dias_sin_commit"].max()),
            value=min(180, int(catalog["dias_sin_commit"].max())),
            step=1,
            help="Descarta repositorios con demasiada inactividad.",
        )
        selected_cluster = st.selectbox(
            "Clúster opcional",
            cluster_options,
            help="Si eliges un clúster, la búsqueda se restringe a ese segmento.",
        )
        top_n = st.slider(
            "Cantidad de resultados",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Número de resultados devueltos por la búsqueda.",
        )

    language_value = None if selected_language == "Cualquiera" else selected_language
    cluster_value = None if selected_cluster == "Cualquiera" else int(selected_cluster)

    preferences_results = recommend_by_preferences(
        language=language_value,
        topics=selected_topics,
        min_stars=min_stars,
        max_days_without_commit=max_days_without_commit,
        cluster=cluster_value,
        top_n=top_n,
        recommender=recommender,
    )

    if preferences_results.empty:
        st.warning("No hay resultados con los filtros seleccionados.")
        return

    st.subheader("Top 3")
    top3 = preferences_results.head(3)
    top_cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        with top_cols[idx]:
            with container_border():
                st.markdown(f"### {int(row['Posición'])}. {row['Repositorio']}")
                st.caption(score_label(float(row["Score de preferencias"])))
                st.progress(min(max(float(row["Score de preferencias"]), 0.0), 1.0))
                st.metric("Score de preferencias", fmt_float(row["Score de preferencias"], 4))
                st.metric("Lenguaje", row["Lenguaje"])
                st.metric("Estrellas", fmt_int(row["Estrellas"]))
                if row.get("Topics coincidentes"):
                    st.write(f"**Topics coincidentes:** {row['Topics coincidentes']}")
                st.write(f"**Motivo:** {row['Motivo']}")
                if isinstance(row.get("URL", ""), str) and row["URL"].strip():
                    st.markdown(f"[Abrir en GitHub]({row['URL']})")

    st.subheader("Tabla completa")
    st.dataframe(
        preferences_results[
            [
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
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


if section == "Recomendador":
    tab_hybrid, tab_preferences = st.tabs(
        ["Similar a un repositorio", "Buscar por preferencias"]
    )

    with tab_hybrid:
        render_hybrid_recommender_tab()

    with tab_preferences:
        render_preferences_tab()


elif section == "Grafo interactivo":
    st.header("Explora la red temática de repositorios")
    st.write(
        "Cada nodo representa un repositorio. Una conexión indica que dos repositorios comparten topics. El grosor de la línea representa cuántos topics comparten."
    )

    with container_border():
        legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
        legend_col1.markdown("**Nodo central**")
        legend_col1.caption("Repositorio seleccionado.")
        legend_col2.markdown("**Nodos vecinos**")
        legend_col2.caption("Repositorios relacionados.")
        legend_col3.markdown("**Línea gruesa**")
        legend_col3.caption("Más topics compartidos.")
        legend_col4.markdown("**Color / tamaño**")
        legend_col4.caption("Color: nodo central vs vecinos. Tamaño: importancia estructural o estrellas según el artefacto.")

    control_col1, control_col2, control_col3 = st.columns([2.4, 1, 1])
    with control_col1:
        selected_repository = st.selectbox(
            "Repositorio central",
            repository_names,
            key="graph_repository",
            help="El nodo central del grafo.",
        )
    with control_col2:
        min_weight = st.slider(
            "Peso mínimo",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            help="Solo se muestran aristas con al menos este número de topics compartidos.",
        )
    with control_col3:
        max_neighbors = st.slider(
            "Vecinos máximos",
            min_value=5,
            max_value=30,
            value=12,
            step=1,
            help="Limita la cantidad de nodos vecinos visibles.",
        )

    graph_payload = build_interactive_subgraph(
        selected_repository=selected_repository,
        max_neighbors=max_neighbors,
        min_weight=min_weight,
    )

    selected_summary = graph_payload["selected_summary"]

    with container_border():
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("PageRank", fmt_float(selected_summary.get("pagerank", 0.0), 6))
        metric_col2.metric("Weighted degree", fmt_float(selected_summary.get("weighted_degree", 0.0), 3))
        betweenness_value = selected_summary.get("betweenness")
        metric_col3.metric("Betweenness", "N/A" if betweenness_value is None else fmt_float(betweenness_value, 6))
        metric_col4.metric("Estrellas", "N/A" if selected_summary.get("stars") is None else fmt_int(selected_summary["stars"]))

        explanation_col1, explanation_col2, explanation_col3, explanation_col4 = st.columns(4)
        explanation_col1.caption("PageRank: importancia estructural.")
        explanation_col2.caption("Weighted degree: intensidad total de conexiones.")
        explanation_col3.caption("Betweenness: capacidad de conectar grupos.")
        explanation_col4.caption("Estrellas: popularidad en GitHub.")

    if graph_payload["html"]:
        components.html(graph_payload["html"], height=760, scrolling=True)
    else:
        st.warning("No fue posible construir el subgrafo con los filtros seleccionados.")

    st.subheader("Tabla de vecinos")
    if graph_payload["neighbors_table"].empty:
        st.info("No hay vecinos con el filtro actual.")
    else:
        neighbor_table = graph_payload["neighbors_table"].copy()
        neighbor_table = neighbor_table.rename(
            columns={
                "Repositorio": "Repositorio vecino",
                "Peso": "Peso de arista",
            }
        )
        neighbor_table["Topics compartidos"] = "No disponible en el edge list"
        st.dataframe(
            neighbor_table[
                [
                    "Repositorio vecino",
                    "Peso de arista",
                    "Topics compartidos",
                    "PageRank",
                    "Weighted degree",
                    "Betweenness",
                    "Estrellas",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with container_border():
        st.subheader("Popularidad vs. importancia estructural")
        st.write(
            "Un repositorio con muchas estrellas no necesariamente es el más central en la red. PageRank y weighted degree capturan conectividad temática, no solo popularidad."
        )
        st.dataframe(pagerank_vs_stars_df, use_container_width=True, hide_index=True)

    with st.expander("Tabla de centralidades"):
        st.dataframe(
            centrality_df[["nombre", "weighted_degree", "pagerank", "betweenness", "stars"]],
            use_container_width=True,
            hide_index=True,
        )
