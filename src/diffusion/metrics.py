"""
Diffusion metrics: cascade depth, breadth, structural virality,
time-to-spread, centrality, and community detection.
"""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cascade-level metrics
# ---------------------------------------------------------------------------

def cascade_depth(G: nx.DiGraph) -> int:
    """
    Return the maximum depth (longest path) of a cascade tree.

    Args:
        G: Directed cascade graph (DAG).

    Returns:
        Maximum path length from root to any leaf.
    """
    root = G.graph.get("root")
    if root is None or root not in G:
        return 0
    try:
        lengths = nx.single_source_shortest_path_length(G, root)
        return int(max(lengths.values())) if lengths else 0
    except Exception:
        return 0


def cascade_breadth(G: nx.DiGraph) -> int:
    """
    Return the maximum number of nodes at any single depth level.

    Args:
        G: Directed cascade graph.

    Returns:
        Maximum breadth (width) across all levels.
    """
    root = G.graph.get("root")
    if root is None or root not in G:
        return 1
    try:
        lengths = nx.single_source_shortest_path_length(G, root)
        level_counts: dict[int, int] = {}
        for _, depth in lengths.items():
            level_counts[depth] = level_counts.get(depth, 0) + 1
        return max(level_counts.values()) if level_counts else 1
    except Exception:
        return 1


def structural_virality(G: nx.DiGraph) -> float:
    """
    Compute structural virality as the average distance between all node pairs
    in the weakly connected cascade subgraph (Wiener index / n(n-1)).

    Ref: Goel et al. (2016) "The structural virality of online diffusion".

    Args:
        G: Directed cascade graph.

    Returns:
        Structural virality score (float). Returns 0.0 for trivial cascades.
    """
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    undirected = G.to_undirected()
    try:
        total = 0
        count = 0
        for node in undirected.nodes():
            lengths = nx.single_source_shortest_path_length(undirected, node)
            for _, d in lengths.items():
                total += d
                count += 1
        # Divide by n*(n-1) to get average pairwise distance
        return float(total / max(n * (n - 1), 1))
    except Exception:
        return 0.0


def time_to_spread(
    G: nx.DiGraph,
    percentile: float = 0.5,
) -> Optional[float]:
    """
    Compute time (in hours) from root post to when a given percentile
    of cascade nodes had been reached.

    Args:
        G: Cascade graph with 'timestamp' node attributes.
        percentile: Fraction of total nodes (0–1). 0.5 = half the cascade.

    Returns:
        Time in hours, or None if timestamps are unavailable.
    """
    timestamps = []
    root = G.graph.get("root")
    root_ts = None

    for node, data in G.nodes(data=True):
        ts = data.get("timestamp")
        if ts is not None:
            try:
                t = pd.Timestamp(ts)
                timestamps.append(t)
                if node == root:
                    root_ts = t
            except Exception:
                pass

    if root_ts is None or len(timestamps) < 2:
        return None

    # Sort timestamps and find when we hit percentile of cascade
    sorted_ts = sorted(timestamps)
    target_idx = max(1, int(len(sorted_ts) * percentile)) - 1
    target_ts = sorted_ts[target_idx]

    delta = (target_ts - root_ts).total_seconds() / 3600.0
    return max(delta, 0.0)


def compute_cascade_metrics(G: nx.DiGraph) -> dict[str, Any]:
    """
    Compute all cascade-level metrics for a single cascade graph.

    Args:
        G: Directed cascade graph.

    Returns:
        Dict of metric name → value.
    """
    return {
        "size": G.number_of_nodes(),
        "depth": cascade_depth(G),
        "breadth": cascade_breadth(G),
        "structural_virality": structural_virality(G),
        "time_to_first_spread_hrs": time_to_spread(G, percentile=0.1),
        "time_to_peak_spread_hrs": time_to_spread(G, percentile=0.9),
        "label": G.graph.get("label", "unknown"),
        "root": G.graph.get("root"),
    }


def compute_all_cascade_metrics(cascades: list[nx.DiGraph]) -> pd.DataFrame:
    """
    Compute metrics for a list of cascades and return as DataFrame.

    Args:
        cascades: List of directed cascade graphs.

    Returns:
        DataFrame with one row per cascade.
    """
    rows = [compute_cascade_metrics(G) for G in cascades]
    df = pd.DataFrame(rows)
    logger.info(f"Computed metrics for {len(df)} cascades")
    return df


# ---------------------------------------------------------------------------
# Graph-level metrics
# ---------------------------------------------------------------------------

def compute_centrality_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute node-level centrality metrics for the interaction graph.

    Args:
        G: Directed user interaction graph.

    Returns:
        DataFrame with columns [node, in_degree, out_degree,
        pagerank, betweenness_centrality, label].
    """
    logger.info("Computing centrality metrics…")

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes()}

    try:
        # Use approximation for large graphs
        if G.number_of_nodes() > 500:
            sample = list(G.nodes())[:500]
            betweenness = nx.betweenness_centrality_subset(
                G.to_undirected(), sources=sample, targets=sample
            )
        else:
            betweenness = nx.betweenness_centrality(G.to_undirected(), normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}

    rows = []
    for node in G.nodes():
        rows.append({
            "node": node,
            "in_degree": in_deg.get(node, 0),
            "out_degree": out_deg.get(node, 0),
            "pagerank": float(pagerank.get(node, 0.0)),
            "betweenness_centrality": float(betweenness.get(node, 0.0)),
            "label": G.nodes[node].get("label", "unknown"),
        })

    return pd.DataFrame(rows)


def detect_communities(G: nx.DiGraph) -> dict[str, int]:
    """
    Detect communities in the interaction graph using Louvain method.

    Args:
        G: Directed user interaction graph.

    Returns:
        Dict mapping node → community ID.
    """
    undirected = G.to_undirected()
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(undirected)
        n_communities = len(set(partition.values()))
        logger.info(f"Detected {n_communities} communities (Louvain)")
        return partition
    except ImportError:
        pass

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(undirected)
        partition = {}
        for cid, community in enumerate(communities):
            for node in community:
                partition[node] = cid
        logger.info(f"Detected {len(communities)} communities (greedy modularity)")
        return partition
    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        return {n: 0 for n in G.nodes()}