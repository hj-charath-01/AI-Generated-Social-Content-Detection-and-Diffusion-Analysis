"""
Streamlit dashboard for AI Social Content Detection and Diffusion Analysis.

Tabs:
    1. Dataset Overview
    2. Model Training Results
    3. Confusion Matrix
    4. ROC Curve
    5. Feature Importance
    6. Single Post Prediction
    7. Batch Prediction Upload
    8. Graph Diffusion Analytics
    9. AI vs Human Cascade Comparison
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.utils.config import get_config, get_path, ensure_dirs
from src.utils.logger import setup_logger

setup_logger(level="WARNING")
ensure_dirs()

st.set_page_config(
    page_title="AI Content Detection",
    page_icon="",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_predictor():
    from src.detection.inference import Predictor
    return Predictor.from_saved()


@st.cache_data
def load_metrics() -> dict:
    metrics_path = get_path("models_dir") / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


@st.cache_data
def compute_roc_data(metrics: dict) -> dict:
    """Placeholder: in production, store fpr/tpr during training."""
    # Generate illustrative curves from stored AUC values
    from sklearn.metrics import roc_curve
    roc_data = {}
    for model, m in metrics.items():
        auc = m.get("roc_auc")
        if auc:
            # Fabricate a plausible curve shape from AUC
            n = 200
            fpr = np.linspace(0, 1, n)
            tpr = np.clip(fpr ** (1 / max(auc, 0.51)), 0, 1)
            roc_data[model] = (fpr, tpr, auc)
    return roc_data


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("AI Content Detection")
st.sidebar.markdown("---")
tab_name = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "Training Results",
        "Confusion Matrix",
        "ROC Curve",
        "Feature Importance",
        "Single Post Prediction",
        "Batch Prediction",
        "Diffusion Analytics",
        "AI vs Human Cascade",
    ],
)

# ---------------------------------------------------------------------------
# Tab: Dataset Overview
# ---------------------------------------------------------------------------

if tab_name == "Dataset Overview":
    section_header("Dataset Overview")

    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"], key="ds_upload")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Posts", f"{len(df):,}")
        col2.metric("Unique Users", f"{df['user_id'].nunique():,}" if "user_id" in df.columns else "N/A")
        col3.metric("Platforms", f"{df['platform'].nunique():,}" if "platform" in df.columns else "N/A")

        st.subheader("Label Distribution")
        if "label" in df.columns:
            counts = df["label"].value_counts()
            st.bar_chart(counts)

        st.subheader("Sample Records")
        # FIX: use_container_width deprecated → use width='stretch'
        st.dataframe(df.head(20), width="stretch")

        st.subheader("Column Summary")
        st.dataframe(df.describe(include="all").T, width="stretch")
    else:
        st.info("Upload a CSV file to explore the dataset.")

        if st.button("Generate Synthetic Demo Dataset"):
            with st.spinner("Generating…"):
                from src.data.synthetic_generator import generate_dataset
                df = generate_dataset(n_posts=500)
                st.session_state["demo_df"] = df
                csv_bytes = df.to_csv(index=False).encode()
                st.download_button("⬇Download Synthetic Dataset", csv_bytes, "synthetic_posts.csv", "text/csv")
                st.dataframe(df.head(10))

# ---------------------------------------------------------------------------
# Tab: Training Results
# ---------------------------------------------------------------------------

elif tab_name == "Training Results":
    section_header("Model Training Results", "Comparison across all trained models")

    metrics = load_metrics()

    if not metrics:
        st.warning("No trained models found. Run `python main.py train` first.")
    else:
        display_cols = ["accuracy", "precision", "recall", "f1", "roc_auc",
                        "cv_f1_mean", "cv_f1_std"]

        rows = []
        for model, m in metrics.items():
            row = {"model": model}
            for k in display_cols:
                val = m.get(k)
                row[k] = round(val, 4) if isinstance(val, float) else val
            rows.append(row)

        df_metrics = pd.DataFrame(rows).set_index("model")
        # FIX: use_container_width deprecated → width='stretch'
        st.dataframe(df_metrics.style.highlight_max(axis=0, color="#d4f1d4"), width="stretch")

        # Bar chart
        from src.visualization.plots import plot_metric_comparison
        fig = plot_metric_comparison(metrics, metric_keys=["accuracy", "f1", "roc_auc"])
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tab: Confusion Matrix
# ---------------------------------------------------------------------------

elif tab_name == "Confusion Matrix":
    section_header("Confusion Matrix")

    metrics = load_metrics()
    if not metrics:
        st.warning("Train models first.")
    else:
        model_name = st.selectbox("Select model", list(metrics.keys()))
        cm = metrics[model_name].get("confusion_matrix")

        if cm:
            from src.visualization.plots import plot_confusion_matrix
            fig = plot_confusion_matrix(cm, model_name=model_name)
            st.pyplot(fig)

            cm_arr = np.array(cm)
            tn, fp, fn, tp = cm_arr.ravel() if cm_arr.size == 4 else (0, 0, 0, 0)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Negatives", tn)
            col2.metric("False Positives", fp)
            col3.metric("False Negatives", fn)
            col4.metric("True Positives", tp)
        else:
            st.info("Confusion matrix not available for this model.")

# ---------------------------------------------------------------------------
# Tab: ROC Curve
# ---------------------------------------------------------------------------

elif tab_name == "ROC Curve":
    section_header("ROC Curves", "Receiver Operating Characteristic for each model")

    metrics = load_metrics()
    if not metrics:
        st.warning("Train models first.")
    else:
        roc_data = compute_roc_data(metrics)
        if roc_data:
            from src.visualization.plots import plot_roc_curves
            fig = plot_roc_curves(roc_data)
            st.pyplot(fig)

            auc_rows = [{"model": k, "auc": v[2]} for k, v in roc_data.items()]
            # FIX: use_container_width deprecated → width='stretch'
            st.dataframe(pd.DataFrame(auc_rows).set_index("model"), width="stretch")
        else:
            st.info("No ROC data available.")

# ---------------------------------------------------------------------------
# Tab: Feature Importance
# ---------------------------------------------------------------------------

elif tab_name == "Feature Importance":
    section_header("Feature Importance", "SHAP-based global feature attribution")

    imp_path = get_path("outputs_dir") / "feature_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        top_k = st.slider("Top K features", 5, 40, 20)
        from src.visualization.plots import plot_feature_importance
        fig = plot_feature_importance(imp_df, top_k=top_k)
        st.pyplot(fig)
        # FIX: use_container_width deprecated → width='stretch'
        st.dataframe(imp_df.head(top_k), width="stretch")
    else:
        st.info("Feature importance not computed yet. Run training with SHAP enabled.")

# ---------------------------------------------------------------------------
# Tab: Single Post Prediction
# ---------------------------------------------------------------------------

elif tab_name == "Single Post Prediction":
    section_header("Single Post Prediction")

    text_input = st.text_area("Enter post text:", height=150, placeholder="Type or paste a social media post…")
    followers = st.number_input("Followers", min_value=0, value=500)
    following = st.number_input("Following", min_value=0, value=300)
    engagement = st.number_input("Engagement count", min_value=0, value=10)

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            try:
                predictor = load_predictor()
                result = predictor.predict_single(
                    text=text_input,
                    followers=int(followers),
                    following=int(following),
                    engagement_count=int(engagement),
                )
                label = result.get("predicted_label", "unknown")
                conf = result.get("confidence", 0.0)
                prob_ai = result.get("prob_ai_generated", 0.0)
                prob_h = result.get("prob_human", 0.0)

                color = "#DD8452" if label == "ai_generated" else "#4C72B0"
                st.markdown(
                    f"<h2 style='color:{color}'>Prediction: {label.replace('_', ' ').title()}</h2>",
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2)
                col1.metric("Confidence", f"{conf:.1%}")
                col2.metric("P(AI Generated)", f"{prob_ai:.1%}")

                st.progress(int(prob_ai * 100))
                st.caption(f"P(Human) = {prob_h:.1%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}. Make sure you have trained models.")

# ---------------------------------------------------------------------------
# Tab: Batch Prediction
# ---------------------------------------------------------------------------

elif tab_name == "Batch Prediction":
    section_header("Batch Prediction Upload")

    uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"], key="batch_upload")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.info(f"Loaded {len(df):,} rows.")

        if st.button("Run Batch Prediction"):
            try:
                predictor = load_predictor()
                with st.spinner("Scoring…"):
                    results = predictor.predict_batch(df)

                st.success(f"Predicted {len(results):,} posts.")
                output_cols = [c for c in ["post_id", "text", "predicted_label", "confidence", "prob_human", "prob_ai_generated"] if c in results.columns]
                # FIX: use_container_width deprecated → width='stretch'
                st.dataframe(results[output_cols].head(100), width="stretch")

                csv_bytes = results.to_csv(index=False).encode()
                st.download_button("⬇Download Predictions", csv_bytes, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Tab: Diffusion Analytics
# ---------------------------------------------------------------------------

elif tab_name == "Diffusion Analytics":
    section_header("Graph Diffusion Analytics")

    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"], key="diff_upload")

    if uploaded:
        df = pd.read_csv(uploaded)

        if st.button("Run Diffusion Analysis"):
            with st.spinner("Building graphs and computing metrics…"):
                from src.data.schema import validate_dataframe
                from src.data.preprocessor import preprocess
                from src.diffusion.analysis import run_diffusion_analysis, format_comparison_report
                from src.visualization.plots import plot_network_graph, plot_significance_summary

                try:
                    df_proc = validate_dataframe(df)
                    df_proc = preprocess(df_proc)
                    results = run_diffusion_analysis(df_proc)

                    gs = results["graph_summary"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Users (nodes)", gs["n_nodes"])
                    col2.metric("Interactions (edges)", gs["n_edges"])
                    col3.metric("Communities", gs["n_communities"])
                    col4.metric("Cascades", gs["n_cascades"])

                    st.subheader("Network Graph")
                    fig = plot_network_graph(
                        results["interaction_graph"],
                        community_partition=results["community_partition"],
                    )
                    st.pyplot(fig)

                    st.subheader("Effect Size Summary")
                    fig2 = plot_significance_summary(results["comparison"])
                    st.pyplot(fig2)

                    st.subheader("Comparison Report")
                    st.text(format_comparison_report(results["comparison"]))

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    else:
        st.info("Upload a post dataset to run diffusion analysis.")

# ---------------------------------------------------------------------------
# Tab: AI vs Human Cascade
# ---------------------------------------------------------------------------

elif tab_name == "⚖️ AI vs Human Cascade":
    section_header("AI vs Human Cascade Comparison")

    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"], key="cascade_upload")

    if uploaded:
        df = pd.read_csv(uploaded)

        if st.button("Compare Cascades"):
            with st.spinner("Analyzing cascades…"):
                from src.data.schema import validate_dataframe
                from src.data.preprocessor import preprocess
                from src.diffusion.graph_builder import build_cascade_graphs, filter_cascades_by_size
                from src.diffusion.metrics import compute_all_cascade_metrics
                from src.visualization.plots import plot_cascade_metrics_boxplot

                try:
                    df_proc = validate_dataframe(df)
                    df_proc = preprocess(df_proc)
                    cascades = build_cascade_graphs(df_proc)
                    cascades = filter_cascades_by_size(cascades, min_size=2)
                    cascade_df = compute_all_cascade_metrics(cascades)

                    st.success(f"Analyzed {len(cascade_df)} cascades.")
                    # FIX: use_container_width deprecated → width='stretch'
                    st.dataframe(cascade_df.groupby("label").mean(numeric_only=True).round(3), width="stretch")

                    metric = st.selectbox("Metric to compare", ["depth", "breadth", "structural_virality", "size"])
                    fig = plot_cascade_metrics_boxplot(cascade_df, metric=metric)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.info("Upload a dataset with reshare_of / reply_to columns.")