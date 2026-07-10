from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDGE_LIST_PATH = PROJECT_ROOT / "notebooks" / "artifacts" / "W12" / "w12_edge_list.csv"
CENTRALITY_PATH = PROJECT_ROOT / "notebooks" / "artifacts" / "W12" / "w12_centrality_ranking.csv"
PAGERANK_STARS_PATH = PROJECT_ROOT / "notebooks" / "artifacts" / "W12" / "w12_pagerank_vs_stars.csv"


def _require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")


@lru_cache(maxsize=1)
def load_graph_edges() -> pd.DataFrame:
    _require_path(EDGE_LIST_PATH)
    return pd.read_csv(EDGE_LIST_PATH)


@lru_cache(maxsize=1)
def load_centrality_ranking() -> pd.DataFrame:
    _require_path(CENTRALITY_PATH)
    return pd.read_csv(CENTRALITY_PATH)


@lru_cache(maxsize=1)
def load_pagerank_vs_stars() -> pd.DataFrame:
    _require_path(PAGERANK_STARS_PATH)
    return pd.read_csv(PAGERANK_STARS_PATH)


@lru_cache(maxsize=1)
def _load_full_graph() -> nx.Graph:
    edges = load_graph_edges()
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.src), str(row.dst), weight=float(row.weight))
    return graph


@lru_cache(maxsize=1)
def _load_metrics() -> Dict[str, object]:
    graph = _load_full_graph()
    pagerank = nx.pagerank(graph, weight="weight", alpha=0.85)
    weighted_degree = dict(graph.degree(weight="weight"))
    centrality = load_centrality_ranking().copy()
    pagerank_vs_stars = load_pagerank_vs_stars().copy()
    return {
        "graph": graph,
        "pagerank": pagerank,
        "weighted_degree": weighted_degree,
        "centrality": centrality,
        "pagerank_vs_stars": pagerank_vs_stars,
    }


def _select_neighbors(
    selected_repository: str,
    max_neighbors: int,
    min_weight: int,
) -> pd.DataFrame:
    edges = load_graph_edges()
    mask = (
        (edges["weight"] >= min_weight)
        & ((edges["src"] == selected_repository) | (edges["dst"] == selected_repository))
    )
    neighborhood = edges.loc[mask].copy()
    if neighborhood.empty:
        return neighborhood

    neighborhood["neighbor"] = np.where(
        neighborhood["src"] == selected_repository,
        neighborhood["dst"],
        neighborhood["src"],
    )
    neighborhood = neighborhood.sort_values(["weight", "neighbor"], ascending=[False, True])
    neighborhood = neighborhood.drop_duplicates(subset=["neighbor"], keep="first")
    return neighborhood.head(max_neighbors)


def _node_metrics(repository_name: str, local_betweenness: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    metrics = _load_metrics()
    pagerank = metrics["pagerank"].get(repository_name, 0.0)
    weighted_degree = metrics["weighted_degree"].get(repository_name, 0.0)
    stars = None
    betweenness = None

    centrality_df = metrics["centrality"]
    row = centrality_df.loc[centrality_df["nombre"] == repository_name]
    if not row.empty:
        stars = int(row.iloc[0]["stars"])
        if "betweenness" in row.columns:
            betweenness = float(row.iloc[0]["betweenness"])

    if local_betweenness and repository_name in local_betweenness:
        betweenness = float(local_betweenness[repository_name])

    return {
        "nombre": repository_name,
        "pagerank": float(pagerank),
        "weighted_degree": float(weighted_degree),
        "betweenness": float(betweenness) if betweenness is not None else None,
        "stars": stars,
    }


def build_interactive_subgraph(
    selected_repository: str,
    max_neighbors: int = 12,
    min_weight: int = 4,
) -> Dict[str, object]:
    metrics = _load_metrics()
    graph = metrics["graph"]
    centrality_df = metrics["centrality"]
    pagerank_vs_stars = metrics["pagerank_vs_stars"]

    if selected_repository not in graph:
        return {
            "html": "",
            "selected_summary": {},
            "neighbors_table": pd.DataFrame(),
            "selected_neighbors": pd.DataFrame(),
            "centrality_table": centrality_df,
            "pagerank_vs_stars": pagerank_vs_stars,
            "graph_stats": {"nodos": graph.number_of_nodes(), "aristas": graph.number_of_edges()},
        }

    selected_edges = _select_neighbors(selected_repository, max_neighbors=max_neighbors, min_weight=min_weight)
    neighbors = selected_edges["neighbor"].tolist() if not selected_edges.empty else []
    selected_nodes = [selected_repository] + neighbors
    subgraph = graph.subgraph(selected_nodes).copy()

    local_betweenness = {}
    if subgraph.number_of_nodes() >= 3 and subgraph.number_of_edges() > 0:
        distances = {(u, v): 1.0 / data["weight"] for u, v, data in subgraph.edges(data=True)}
        nx.set_edge_attributes(subgraph, distances, "distance")
        local_betweenness = nx.betweenness_centrality(subgraph, weight="distance", normalized=True)

    neighbor_table_rows = []
    for neighbor in neighbors:
        edge_row = selected_edges.loc[selected_edges["neighbor"] == neighbor].iloc[0]
        node_metrics = _node_metrics(neighbor, local_betweenness=local_betweenness)
        neighbor_table_rows.append(
            {
                "Repositorio": neighbor,
                "Peso": int(edge_row["weight"]),
                "PageRank": round(node_metrics["pagerank"], 6),
                "Weighted degree": round(node_metrics["weighted_degree"], 3),
                "Betweenness": None if node_metrics["betweenness"] is None else round(node_metrics["betweenness"], 6),
                "Estrellas": node_metrics["stars"],
            }
        )

    neighbors_table = pd.DataFrame(neighbor_table_rows)

    selected_summary = _node_metrics(selected_repository, local_betweenness=local_betweenness)
    selected_summary["peso_minimo"] = min_weight
    selected_summary["max_vecinos"] = max_neighbors

    network = Network(
        height="720px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
        directed=False,
        cdn_resources="in_line",
    )
    network.barnes_hut(
        gravity=-2200,
        central_gravity=0.25,
        spring_length=170,
        spring_strength=0.04,
        damping=0.85,
        overlap=0,
    )

    max_weight = max(
        [float(row["weight"]) for _, row in selected_edges.iterrows()],
        default=1.0,
    )
    max_pr = max(
        [_node_metrics(node, local_betweenness=local_betweenness)["pagerank"] for node in selected_nodes],
        default=1.0,
    )

    for node in selected_nodes:
        node_metrics = _node_metrics(node, local_betweenness=local_betweenness)
        if node == selected_repository:
            color = "#e74c3c"
            size = 38
        else:
            ratio = node_metrics["pagerank"] / max_pr if max_pr else 0
            color = "#2e86de"
            size = float(18 + 18 * ratio)

        stars = node_metrics["stars"] if node_metrics["stars"] is not None else 0
        betweenness_text = "N/A"
        if node_metrics["betweenness"] is not None:
            betweenness_text = f"{node_metrics['betweenness']:.6f}"
        tooltip = (
            f"<b>{node}</b><br>"
            f"PageRank: {node_metrics['pagerank']:.6f}<br>"
            f"Weighted degree: {node_metrics['weighted_degree']:.3f}<br>"
            f"Betweenness: {betweenness_text}<br>"
            f"Estrellas: {stars if stars is not None else 'N/A'}"
        )
        network.add_node(
            node,
            label=node,
            title=tooltip,
            size=float(size),
            color=color,
            shape="dot",
            borderWidth=3 if node == selected_repository else 1,
        )

    for _, row in selected_edges.iterrows():
        neighbor = row["neighbor"]
        weight = float(row["weight"])
        network.add_edge(
            selected_repository,
            neighbor,
            value=weight,
            width=max(1.0, 6.0 * weight / max_weight),
            title=f"Topics compartidos: {int(weight)}",
        )

    network.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "tooltipDelay": 100
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 500
            }
          },
          "nodes": {
            "font": {
              "size": 14,
              "face": "Arial"
            }
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          }
        }
        """
    )

    html = network.generate_html()

    return {
        "html": html,
        "selected_summary": selected_summary,
        "neighbors_table": neighbors_table,
        "selected_neighbors": selected_edges,
        "centrality_table": centrality_df,
        "pagerank_vs_stars": pagerank_vs_stars,
        "graph_stats": {
            "nodos": graph.number_of_nodes(),
            "aristas": graph.number_of_edges(),
            "subgrafo_nodos": subgraph.number_of_nodes(),
            "subgrafo_aristas": subgraph.number_of_edges(),
        },
    }
