"""
Visualization functions for model evaluation and diffusion analysis.

Returns matplotlib Figure objects or Plotly figures so callers
(dashboard / report scripts) can handle rendering.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PALETTE = {"human": "#4C72B0", "ai_generated": "#DD8452", "unknown": "#8C8C8C"}


def _get_cmap(name: str, n: int):
    """
    Retrieve a colormap in a way that works across Matplotlib versions.

    matplotlib.cm.get_cmap() was deprecated in 3.7 and removed in 3.9.
    Use matplotlib.colormaps[name] (available from 3.5+) with a fallback
    for older installs.
    """
    try:
        # Preferred API: Matplotlib >= 3.5
        return matplotlib.colormaps[name].resampled(n)
    except AttributeError:
        # Fallback for Matplotlib < 3.5
        return cm.get_cmap(name, n)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Detection evaluation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm_data: list[list[int]],
    labels: list[str] = ("human", "ai_generated"),
    title: str = "Confusion Matrix",
    model_name: str = "",
) -> plt.Figure:
    """
    Render a labeled confusion matrix heatmap.

    Args:
        cm_data: 2×2 confusion matrix as list of lists.
        labels: Axis label names.
        title: Plot title.
        model_name: Optional model name appended to title.

    Returns:
        Matplotlib Figure.
    """
    import seaborn as sns

    cm_arr = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    full_title = f"{title} — {model_name}" if model_name else title
    ax.set_title(full_title)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curves",
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        roc_data: Dict mapping model_name → (fpr, tpr, auc_score).
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10.colors

    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        label = f"{name} (AUC={auc:.3f})" if auc is not None else name
        ax.plot(fpr, tpr, label=label, color=colors[i % len(colors)], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 20,
    title: str = "Feature Importance (mean |SHAP|)",
) -> plt.Figure:
    """
    Horizontal bar chart of top-k features by importance.

    Args:
        importance_df: DataFrame with columns ['feature', 'mean_abs_shap']
                       (or 'importance') sorted descending.
        top_k: Number of features to show.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    value_col = "mean_abs_shap" if "mean_abs_shap" in importance_df.columns else "importance"
    df = importance_df.head(top_k).copy()

    fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.35)))
    ax.barh(
        y=df["feature"][::-1],
        width=df[value_col][::-1],
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    fig.tight_layout()
    return fig


def plot_metric_comparison(
    metrics_dict: dict[str, dict],
    metric_keys: list[str] = ("accuracy", "f1", "roc_auc"),
) -> plt.Figure:
    """
    Grouped bar chart comparing metrics across models.

    Args:
        metrics_dict: Dict mapping model_name → metrics dict.
        metric_keys: Which metrics to plot.

    Returns:
        Matplotlib Figure.
    """
    models = list(metrics_dict.keys())
    n_metrics = len(metric_keys)
    n_models = len(models)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 2), 5))
    colors = plt.cm.tab10.colors

    for i, model in enumerate(models):
        values = [
            metrics_dict[model].get(k, 0.0) or 0.0
            for k in metric_keys
        ]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in metric_keys])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(fontsize=8)
    ax.axhline(y=1.0, linestyle="--", color="gray", linewidth=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Diffusion visualizations
# ---------------------------------------------------------------------------

def plot_cascade_metrics_boxplot(
    cascade_df: pd.DataFrame,
    metric: str = "depth",
) -> plt.Figure:
    """
    Side-by-side boxplots comparing a cascade metric for AI vs human content.

    Args:
        cascade_df: DataFrame with 'label' column and numeric metric column.
        metric: Name of the metric column to plot.

    Returns:
        Matplotlib Figure.
    """
    ai_data = cascade_df.loc[cascade_df["label"] == "ai_generated", metric].dropna()
    human_data = cascade_df.loc[cascade_df["label"] == "human", metric].dropna()

    fig, ax = plt.subplots(figsize=(5, 4))

    # FIX: Matplotlib 3.9 renamed the boxplot 'labels' kwarg to 'label'.
    # Try the new name first; fall back to the old name for older installs.
    _tick_labels = ["Human", "AI Generated"]
    try:
        bp = ax.boxplot(
            [human_data, ai_data],
            label=_tick_labels,
            patch_artist=True,
            notch=True,
        )
    except TypeError:
        bp = ax.boxplot(
            [human_data, ai_data],
            labels=_tick_labels,
            patch_artist=True,
            notch=True,
        )
    bp["boxes"][0].set_facecolor(_PALETTE["human"])
    bp["boxes"][1].set_facecolor(_PALETTE["ai_generated"])

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Cascade {metric.replace('_', ' ').title()}: AI vs Human")
    fig.tight_layout()
    return fig


def plot_label_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Pie chart of label distribution in the dataset.

    Args:
        df: Post DataFrame with 'label' column.

    Returns:
        Matplotlib Figure.
    """
    counts = df["label"].value_counts()
    colors = [_PALETTE.get(lbl, "#999") for lbl in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
    )
    ax.set_title("Dataset Label Distribution")
    fig.tight_layout()
    return fig


def plot_network_graph(
    G,
    community_partition: Optional[dict[str, int]] = None,
    max_nodes: int = 200,
    title: str = "User Interaction Graph",
) -> plt.Figure:
    """
    Draw the interaction graph with nodes colored by community or label.

    Args:
        G: NetworkX DiGraph.
        community_partition: Dict mapping node → community ID.
        max_nodes: Subsample if graph is large.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    import networkx as nx

    if G.number_of_nodes() > max_nodes:
        nodes = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes)

    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(G, seed=42, k=0.5)

    if community_partition:
        unique_comms = list(set(community_partition.values()))
        # FIX: cm.get_cmap() deprecated in Matplotlib 3.7+, removed in 3.9.
        # Use the _get_cmap() helper which prefers matplotlib.colormaps[].
        cmap = _get_cmap("tab20", len(unique_comms))
        color_map = {c: cmap(i) for i, c in enumerate(unique_comms)}
        node_colors = [
            color_map.get(community_partition.get(n, 0), "#999")
            for n in G.nodes()
        ]
    else:
        node_colors = [
            _PALETTE.get(G.nodes[n].get("label", "unknown"), "#999")
            for n in G.nodes()
        ]

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        with_labels=False,
        node_size=30,
        node_color=node_colors,
        edge_color="#ccc",
        arrows=False,
        alpha=0.8,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_significance_summary(comparison_df: pd.DataFrame) -> plt.Figure:
    """
    Dot plot showing effect sizes and significance for diffusion metrics.

    Args:
        comparison_df: DataFrame from run_diffusion_analysis()['comparison'].

    Returns:
        Matplotlib Figure.
    """
    df = comparison_df.copy()
    df = df.dropna(subset=["ai_mean", "human_mean"])
    df["effect"] = df["ai_mean"] - df["human_mean"]
    df["significant"] = df.get("significant_mannwhitney", False)

    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.5)))
    colors = ["#DD8452" if sig else "#8C8C8C" for sig in df["significant"]]
    ax.barh(df["metric"], df["effect"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("AI Mean − Human Mean (positive = AI higher)")
    ax.set_title("Diffusion Metric Effect Sizes (orange = significant)")
    fig.tight_layout()
    return fig