"""
Graph builder: constructs user interaction graphs and cascade graphs
from reshare/reply data using NetworkX.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_interaction_graph(
    df: pd.DataFrame,
    weight_by_engagement: bool = True,
) -> nx.DiGraph:
    """
    Build a directed user-to-user interaction graph.

    An edge u → v means user v reshared or replied to a post by user u.

    Args:
        df: Post DataFrame with 'user_id', 'reply_to', 'reshare_of',
            'post_id', 'engagement_count'.
        weight_by_engagement: If True, edge weights reflect total engagement.

    Returns:
        Directed NetworkX graph with node attribute 'label' (majority label)
        and edge attribute 'weight' (interaction count or engagement sum).
    """
    post_to_user: dict[str, str] = dict(zip(df["post_id"], df["user_id"]))
    post_to_label: dict[str, str] = dict(zip(df["post_id"], df["label"]))
    user_label: dict[str, list[str]] = {}

    G: nx.DiGraph = nx.DiGraph()

    for _, row in df.iterrows():
        uid = str(row["user_id"])
        lbl = str(row.get("label", "unknown"))
        G.add_node(uid)
        user_label.setdefault(uid, []).append(lbl)

        for ref_col in ["reshare_of", "reply_to"]:
            ref = row.get(ref_col)
            if pd.notna(ref) and ref in post_to_user:
                source_uid = post_to_user[ref]
                eng = int(row.get("engagement_count", 1)) or 1

                if G.has_edge(source_uid, uid):
                    G[source_uid][uid]["weight"] += eng if weight_by_engagement else 1
                    G[source_uid][uid]["count"] += 1
                else:
                    G.add_edge(
                        source_uid,
                        uid,
                        weight=eng if weight_by_engagement else 1,
                        count=1,
                    )

    # Set majority label on each node
    for uid, labels in user_label.items():
        majority = max(set(labels), key=labels.count)
        G.nodes[uid]["label"] = majority

    logger.info(
        f"Interaction graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


def build_cascade_graphs(
    df: pd.DataFrame,
    relation_col: str = "reshare_of",
) -> list[nx.DiGraph]:
    """
    Build a list of cascade trees — one per root post.

    A cascade starts at an original post (no reshare_of) and follows
    the reshare chain. Each node is a post_id.

    Args:
        df: Post DataFrame.
        relation_col: Column containing the ID of the parent post.

    Returns:
        List of directed NetworkX graphs (trees) representing cascades.
        Each node has attributes: post_id, user_id, timestamp, label.
    """
    post_info: dict[str, dict] = {
        row["post_id"]: {
            "user_id": row["user_id"],
            "timestamp": row.get("timestamp"),
            "label": row.get("label", "unknown"),
        }
        for _, row in df.iterrows()
    }

    # Build children map
    children: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        parent = row.get(relation_col)
        if pd.notna(parent) and parent in post_info:
            children.setdefault(str(parent), []).append(str(row["post_id"]))

    # Find root posts (no parent or parent not in dataset)
    all_posts = set(post_info.keys())
    has_parent = set()
    for _, row in df.iterrows():
        parent = row.get(relation_col)
        if pd.notna(parent) and parent in all_posts:
            has_parent.add(str(row["post_id"]))

    roots = all_posts - has_parent

    cascades: list[nx.DiGraph] = []
    for root in roots:
        G = nx.DiGraph()
        stack = [root]
        while stack:
            node = stack.pop()
            info = post_info.get(node, {})
            G.add_node(node, **info)
            for child in children.get(node, []):
                child_info = post_info.get(child, {})
                G.add_node(child, **child_info)
                G.add_edge(node, child)
                stack.append(child)

        if G.number_of_nodes() >= 1:
            G.graph["root"] = root
            G.graph["label"] = post_info.get(root, {}).get("label", "unknown")
            cascades.append(G)

    logger.info(f"Built {len(cascades)} cascades from {len(df)} posts")
    return cascades


def filter_cascades_by_size(
    cascades: list[nx.DiGraph],
    min_size: int = 2,
) -> list[nx.DiGraph]:
    """
    Keep only cascades with at least min_size posts.

    Args:
        cascades: List of cascade graphs.
        min_size: Minimum node count.

    Returns:
        Filtered list.
    """
    filtered = [c for c in cascades if c.number_of_nodes() >= min_size]
    logger.info(f"Cascades after size filter (≥{min_size}): {len(filtered)}")
    return filtered