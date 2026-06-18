"""
Statistical comparison of AI-generated vs human content diffusion patterns.

Computes group metrics, significance tests, and summary reports.
"""

from __future__ import annotations

from typing import Any

import networkx as nx  # FIX: was missing at module level; nx_density() used nx bare
import numpy as np
import pandas as pd
from scipy import stats

from src.diffusion.graph_builder import (
    build_cascade_graphs,
    build_interaction_graph,
    filter_cascades_by_size,
)
from src.diffusion.metrics import (
    compute_all_cascade_metrics,
    compute_centrality_metrics,
    detect_communities,
)
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_NUMERIC_CASCADE_METRICS = [
    "size", "depth", "breadth", "structural_virality",
    "time_to_first_spread_hrs", "time_to_peak_spread_hrs",
]


def compare_distributions(
    ai_values: np.ndarray,
    human_values: np.ndarray,
    metric_name: str,
) -> dict[str, Any]:
    """
    Compare two distributions using t-test and Mann–Whitney U.

    Args:
        ai_values: Metric values for AI-generated content.
        human_values: Metric values for human content.
        metric_name: Name for logging/reporting.

    Returns:
        Dict with descriptive stats and test results.
    """
    ai_clean = ai_values[~np.isnan(ai_values)]
    human_clean = human_values[~np.isnan(human_values)]

    result: dict[str, Any] = {
        "metric": metric_name,
        "ai_mean": float(np.mean(ai_clean)) if len(ai_clean) > 0 else None,
        "ai_median": float(np.median(ai_clean)) if len(ai_clean) > 0 else None,
        "ai_std": float(np.std(ai_clean)) if len(ai_clean) > 0 else None,
        "human_mean": float(np.mean(human_clean)) if len(human_clean) > 0 else None,
        "human_median": float(np.median(human_clean)) if len(human_clean) > 0 else None,
        "human_std": float(np.std(human_clean)) if len(human_clean) > 0 else None,
        "n_ai": len(ai_clean),
        "n_human": len(human_clean),
    }

    if len(ai_clean) >= 5 and len(human_clean) >= 5:
        # t-test
        t_stat, t_pval = stats.ttest_ind(ai_clean, human_clean, equal_var=False)
        result["ttest_stat"] = float(t_stat)
        result["ttest_pvalue"] = float(t_pval)

        # Mann-Whitney U
        u_stat, u_pval = stats.mannwhitneyu(
            ai_clean, human_clean, alternative="two-sided"
        )
        result["mannwhitney_stat"] = float(u_stat)
        result["mannwhitney_pvalue"] = float(u_pval)

        # Significance at p<0.05
        result["significant_ttest"] = bool(t_pval < 0.05)
        result["significant_mannwhitney"] = bool(u_pval < 0.05)
    else:
        result["ttest_stat"] = None
        result["ttest_pvalue"] = None
        result["mannwhitney_stat"] = None
        result["mannwhitney_pvalue"] = None
        result["significant_ttest"] = False
        result["significant_mannwhitney"] = False

    return result


def run_diffusion_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """
    Full diffusion analysis pipeline on a labeled post DataFrame.

    Steps:
        1. Build cascade graphs
        2. Compute per-cascade metrics
        3. Build user interaction graph
        4. Compute centrality and community detection
        5. Compare AI vs human distributions with significance tests

    Args:
        df: Preprocessed post DataFrame with 'label' column.

    Returns:
        Dict containing cascade metrics, comparison tables, graph stats.
    """
    cfg = get_config()
    min_cascade_size = cfg["diffusion"]["min_cascade_size"]

    logger.info("Starting diffusion analysis…")

    # ----------------------------------------------------------------
    # 1. Cascades
    # ----------------------------------------------------------------
    cascades = build_cascade_graphs(df)
    cascades = filter_cascades_by_size(cascades, min_size=min_cascade_size)
    cascade_metrics_df = compute_all_cascade_metrics(cascades)

    # ----------------------------------------------------------------
    # 2. Interaction graph
    # ----------------------------------------------------------------
    interaction_graph = build_interaction_graph(df)
    centrality_df = compute_centrality_metrics(interaction_graph)
    community_partition = detect_communities(interaction_graph)

    # Add community to centrality
    centrality_df["community"] = centrality_df["node"].map(community_partition)

    # ----------------------------------------------------------------
    # 3. Statistical comparison
    # ----------------------------------------------------------------
    ai_mask = cascade_metrics_df["label"] == "ai_generated"
    human_mask = cascade_metrics_df["label"] == "human"

    comparison_rows: list[dict] = []
    for metric in _NUMERIC_CASCADE_METRICS:
        if metric not in cascade_metrics_df.columns:
            continue
        ai_vals = cascade_metrics_df.loc[ai_mask, metric].values.astype(float)
        human_vals = cascade_metrics_df.loc[human_mask, metric].values.astype(float)
        row = compare_distributions(ai_vals, human_vals, metric)
        comparison_rows.append(row)
        logger.info(
            f"  {metric}: AI μ={row['ai_mean']:.3f} Human μ={row['human_mean']:.3f} "
            f"p(MW)={row['mannwhitney_pvalue']}"
        )

    comparison_df = pd.DataFrame(comparison_rows)

    # ----------------------------------------------------------------
    # 4. Summary
    # ----------------------------------------------------------------
    graph_summary = {
        "n_nodes": interaction_graph.number_of_nodes(),
        "n_edges": interaction_graph.number_of_edges(),
        "n_communities": len(set(community_partition.values())),
        "density": nx_density(interaction_graph),  # nx now available at module level
        "n_cascades": len(cascades),
        "n_cascades_ai": int(ai_mask.sum()),
        "n_cascades_human": int(human_mask.sum()),
    }

    return {
        "cascade_metrics": cascade_metrics_df,
        "comparison": comparison_df,
        "centrality": centrality_df,
        "community_partition": community_partition,
        "graph_summary": graph_summary,
        "interaction_graph": interaction_graph,
        "cascades": cascades,
    }


def nx_density(G) -> float:
    """Safe wrapper for nx.density. nx is imported at module level."""
    try:
        return float(nx.density(G))
    except Exception:
        return 0.0


def format_comparison_report(comparison_df: pd.DataFrame) -> str:
    """
    Format the comparison DataFrame as a readable text report.

    Args:
        comparison_df: Output of run_diffusion_analysis()['comparison'].

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 70,
        "  AI vs Human Diffusion Comparison",
        "=" * 70,
        f"{'Metric':<35} {'AI Mean':>10} {'Human Mean':>10} {'p(MW)':>10} {'Sig?':>6}",
        "-" * 70,
    ]
    for _, row in comparison_df.iterrows():
        sig = "✓" if row.get("significant_mannwhitney") else ""
        ai_mean = f"{row['ai_mean']:.3f}" if row["ai_mean"] is not None else "N/A"
        h_mean = f"{row['human_mean']:.3f}" if row["human_mean"] is not None else "N/A"
        p_val = f"{row['mannwhitney_pvalue']:.4f}" if row["mannwhitney_pvalue"] is not None else "N/A"
        lines.append(
            f"{row['metric']:<35} {ai_mean:>10} {h_mean:>10} {p_val:>10} {sig:>6}"
        )

    lines.append("=" * 70)
    lines.append("✓ = significant at p < 0.05 (Mann–Whitney U, two-sided)")
    return "\n".join(lines)