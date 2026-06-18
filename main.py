"""
Main CLI entry point.

Commands:
    generate  — Generate synthetic demo dataset
    train     — Train all detection models
    infer     — Run inference on a file
    diffusion — Run diffusion analysis on a dataset
    api       — Start FastAPI server
    dashboard — Start Streamlit dashboard
    report    — Generate evaluation report
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from src.utils.config import get_config, ensure_dirs
from src.utils.logger import setup_logger

cfg = get_config()
ensure_dirs()
setup_logger(log_dir="outputs/logs", level="INFO")


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """AI-Generated Social Content Detection and Diffusion Analysis."""
    pass


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--n-posts", default=2000, show_default=True, help="Number of posts to generate")
@click.option("--ai-fraction", default=0.35, show_default=True, help="Fraction of AI-generated posts")
@click.option("--n-users", default=300, show_default=True, help="Number of synthetic users")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default="data/synthetic_posts.csv", show_default=True)
def generate(n_posts: int, ai_fraction: float, n_users: int, seed: int, output: str):
    """Generate a synthetic labeled post dataset and edge list."""
    from src.data.synthetic_generator import generate_dataset, generate_edge_list

    click.echo(f"Generating {n_posts} posts ({ai_fraction:.0%} AI)…")
    df = generate_dataset(
        n_posts=n_posts,
        ai_fraction=ai_fraction,
        n_users=n_users,
        seed=seed,
        output_path=output,
    )

    edge_output = Path(output).parent / "synthetic_edges.csv"
    generate_edge_list(df, output_path=str(edge_output))
    click.secho(f"✓ Dataset saved: {output}", fg="green")
    click.secho(f"✓ Edge list saved: {edge_output}", fg="green")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data", default="data/synthetic_posts.csv", show_default=True, help="Path to labeled dataset")
@click.option("--use-embeddings", is_flag=True, help="Include transformer embeddings (slow)")
@click.option("--no-cv", is_flag=True, help="Skip cross-validation")
@click.option("--models-dir", default=None, help="Override models output directory")
def train(data: str, use_embeddings: bool, no_cv: bool, models_dir: str | None):
    """Train all detection models on labeled data."""
    import joblib
    import numpy as np
    from src.data.loader import load_any
    from src.data.preprocessor import preprocess
    from src.detection.models import encode_labels
    from src.detection.trainer import DetectionTrainer
    from src.features.feature_pipeline import FeaturePipeline
    from src.utils.config import get_path

    out_dir = Path(models_dir) if models_dir else get_path("models_dir")

    # Load and preprocess
    click.echo(f"Loading data: {data}")
    df = load_any(data)
    df = preprocess(df)

    # Filter labeled rows
    labeled_df = df[df["label"].isin(["human", "ai_generated"])].copy()
    click.echo(f"Labeled posts: {len(labeled_df):,}")

    # Feature extraction
    click.echo(f"Extracting features (embeddings={'yes' if use_embeddings else 'no'})…")
    pipeline = FeaturePipeline(use_embeddings=use_embeddings)
    X = pipeline.fit_transform(labeled_df)
    y = encode_labels(labeled_df["label"])

    # Save pipeline
    pipeline_path = out_dir / "feature_pipeline.joblib"
    pipeline.save(pipeline_path)

    # Train models
    click.echo("Training models…")
    trainer = DetectionTrainer()
    metrics = trainer.train_all(X, y, run_cv=not no_cv)
    trainer.save_artifacts(out_dir)

    # SHAP feature importance
    click.echo("Computing SHAP feature importance…")
    try:
        from src.detection.explainability import Explainer
        best_model = trainer.trained_models.get("ensemble") or list(trainer.trained_models.values())[0]
        explainer = Explainer(best_model, pipeline.get_feature_names())
        mask = y >= 0
        X_sample = X[mask][:min(200, len(X))]
        imp_df = explainer.feature_importance_global(X_sample)
        imp_path = get_path("outputs_dir") / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        click.secho(f"✓ Feature importance saved: {imp_path}", fg="green")
    except Exception as e:
        click.echo(f"SHAP skipped: {e}", err=True)

    # Summary
    click.secho("\n=== Training Complete ===", fg="green", bold=True)
    for model, m in metrics.items():
        click.echo(
            f"  {model:25s}  acc={m.get('accuracy', 0):.4f}  "
            f"f1={m.get('f1', 0):.4f}  "
            f"auc={m.get('roc_auc', 'N/A')}"
        )


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_path")
@click.option("--output", default=None, help="Output CSV path for predictions")
@click.option("--model", default="ensemble", show_default=True, help="Model to use for inference")
@click.option("--models-dir", default=None)
def infer(input_path: str, output: str | None, model: str, models_dir: str | None):
    """Run inference on a CSV/JSON/JSONL file."""
    from src.detection.inference import Predictor
    from src.utils.config import get_path

    out_dir = Path(models_dir) if models_dir else get_path("models_dir")
    out_path = output or f"outputs/predictions_{Path(input_path).stem}.csv"

    click.echo(f"Loading predictor ({model})…")
    predictor = Predictor.from_saved(model_name=model, models_dir=out_dir)

    click.echo(f"Scoring: {input_path}")
    results = predictor.predict_file(input_path, output_path=out_path)

    n_ai = (results["predicted_label"] == "ai_generated").sum()
    n_human = (results["predicted_label"] == "human").sum()
    click.secho(f"\n✓ Predictions saved: {out_path}", fg="green")
    click.echo(f"  Human: {n_human:,}  |  AI Generated: {n_ai:,}")


# ---------------------------------------------------------------------------
# diffusion
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("data_path")
@click.option("--output-dir", default="outputs", show_default=True)
def diffusion(data_path: str, output_dir: str):
    """Run diffusion analysis and save reports."""
    from src.data.loader import load_any
    from src.data.preprocessor import preprocess
    from src.diffusion.analysis import run_diffusion_analysis, format_comparison_report

    click.echo(f"Loading: {data_path}")
    df = load_any(data_path)
    df = preprocess(df)

    click.echo("Running diffusion analysis…")
    results = run_diffusion_analysis(df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save cascade metrics
    cascade_path = out / "cascade_metrics.csv"
    results["cascade_metrics"].to_csv(cascade_path, index=False)

    # Save comparison
    comparison_path = out / "diffusion_comparison.csv"
    results["comparison"].to_csv(comparison_path, index=False)

    # Save text report
    report_path = out / "diffusion_report.txt"
    report_path.write_text(format_comparison_report(results["comparison"]))

    # Save visualizations
    from src.visualization.plots import (
        plot_cascade_metrics_boxplot,
        plot_network_graph,
        plot_significance_summary,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ["depth", "breadth", "structural_virality"]:
        fig = plot_cascade_metrics_boxplot(results["cascade_metrics"], metric=metric)
        fig.savefig(out / f"cascade_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    net_fig = plot_network_graph(results["interaction_graph"], results["community_partition"])
    net_fig.savefig(out / "network_graph.png", dpi=150, bbox_inches="tight")
    plt.close(net_fig)

    sig_fig = plot_significance_summary(results["comparison"])
    sig_fig.savefig(out / "diffusion_significance.png", dpi=150, bbox_inches="tight")
    plt.close(sig_fig)

    click.secho("\n✓ Diffusion analysis complete!", fg="green")
    gs = results["graph_summary"]
    click.echo(f"  Nodes: {gs['n_nodes']}  Edges: {gs['n_edges']}  "
               f"Communities: {gs['n_communities']}  Cascades: {gs['n_cascades']}")
    click.echo(f"\n{format_comparison_report(results['comparison'])}")


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True)
def api(host: str, port: int, reload: bool):
    """Start the FastAPI REST API server."""
    import uvicorn
    click.echo(f"Starting API at http://{host}:{port}")
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--port", default=8501, show_default=True)
def dashboard(port: int):
    """Launch the Streamlit dashboard."""
    import subprocess
    click.echo(f"Starting dashboard at http://localhost:{port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/dashboard.py",
        f"--server.port={port}",
        "--server.headless=true",
    ])


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--output", default="outputs/evaluation_report.md", show_default=True)
def report(output: str):
    """Generate an evaluation report from saved metrics and plots."""
    import json
    from src.utils.config import get_path

    metrics_path = get_path("models_dir") / "metrics.json"
    if not metrics_path.exists():
        click.echo("No metrics found. Train models first.", err=True)
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    lines = [
        "# AI Content Detection — Evaluation Report\n",
        "## Detection Model Results\n",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|-------|----------|-----------|--------|----|---------|",
    ]
    for model, m in metrics.items():
        row = "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            model,
            m.get("accuracy", 0),
            m.get("precision", 0),
            m.get("recall", 0),
            m.get("f1", 0),
            f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A",
        )
        lines.append(row)

    lines += [
        "\n## Diffusion Analysis\n",
        "See `outputs/diffusion_report.txt` and `outputs/cascade_metrics.csv`.\n",
        "\n## Outputs\n",
        "- `outputs/predictions_*.csv` — batch inference results\n",
        "- `outputs/cascade_*.png` — cascade metric plots\n",
        "- `outputs/network_graph.png` — user interaction network\n",
        "- `outputs/feature_importance.csv` — SHAP feature attribution\n",
    ]

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    click.secho(f"✓ Report saved: {output}", fg="green")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()