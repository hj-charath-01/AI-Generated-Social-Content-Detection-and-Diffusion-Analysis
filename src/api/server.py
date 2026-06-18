"""
FastAPI REST API server for detection and diffusion analysis.

Endpoints:
    GET  /health              — liveness check
    POST /predict             — single-post prediction
    POST /predict_batch       — batch CSV upload
    POST /analyze_diffusion   — diffusion analysis on uploaded dataset
    GET  /metrics             — retrieve saved model metrics
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.config import get_config, get_path
from src.utils.logger import get_logger, setup_logger

setup_logger(level="INFO")
logger = get_logger(__name__)

app = FastAPI(
    title="AI Social Content Detection API",
    version="1.0.0",
    description="Detect AI-generated social media posts and analyze diffusion patterns.",
)

# ---------------------------------------------------------------------------
# Global state — loaded lazily on first request
# ---------------------------------------------------------------------------

_predictor: Optional[Any] = None
_feature_pipeline: Optional[Any] = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from src.detection.inference import Predictor
        try:
            _predictor = Predictor.from_saved()
            logger.info("Predictor loaded successfully")
        except Exception as e:
            logger.error(f"Could not load predictor: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Model not available. Train first: {e}",
            )
    return _predictor


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PostRequest(BaseModel):
    text: str = Field(..., description="Raw post text")
    user_id: str = Field(default="unknown", description="Author user ID")
    post_id: str = Field(default="single", description="Post identifier")
    followers: int = Field(default=0, ge=0)
    following: int = Field(default=0, ge=0)
    engagement_count: int = Field(default=0, ge=0)


class PredictionResponse(BaseModel):
    post_id: str
    predicted_label: str
    confidence: float
    prob_human: float
    prob_ai_generated: float


class DiffusionRequest(BaseModel):
    """Accepts a JSON list of post records."""
    posts: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "service": "AI Content Detection API"}


@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
def predict(request: PostRequest) -> PredictionResponse:
    """
    Predict whether a single post is human or AI-generated.

    Returns predicted label, confidence, and class probabilities.
    """
    predictor = _get_predictor()
    result = predictor.predict_single(
        text=request.text,
        user_id=request.user_id,
        post_id=request.post_id,
        followers=request.followers,
        following=request.following,
        engagement_count=request.engagement_count,
    )
    return PredictionResponse(
        post_id=str(result.get("post_id", request.post_id)),
        predicted_label=str(result.get("predicted_label", "unknown")),
        confidence=float(result.get("confidence", 0.0)),
        prob_human=float(result.get("prob_human", 0.0)),
        prob_ai_generated=float(result.get("prob_ai_generated", 0.0)),
    )


@app.post("/predict_batch", tags=["Detection"])
async def predict_batch(file: UploadFile = File(...)) -> JSONResponse:
    """
    Batch prediction on an uploaded CSV file.

    Accepts a CSV with at minimum columns: post_id, user_id, text.
    Returns JSON array of predictions.
    """
    predictor = _get_predictor()

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    results = predictor.predict_batch(df)

    output_cols = ["post_id", "predicted_label", "confidence", "prob_human", "prob_ai_generated"]
    available = [c for c in output_cols if c in results.columns]
    return JSONResponse(content=results[available].to_dict(orient="records"))


@app.post("/analyze_diffusion", tags=["Diffusion"])
async def analyze_diffusion(file: UploadFile = File(...)) -> JSONResponse:
    """
    Run diffusion analysis on an uploaded post dataset.

    Accepts a CSV with the standard post schema including reshare_of / reply_to.
    Returns cascade statistics and AI vs human comparison.
    """
    from src.data.loader import load_csv
    from src.data.preprocessor import preprocess
    from src.diffusion.analysis import run_diffusion_analysis, format_comparison_report

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        from src.data.schema import validate_dataframe
        df = validate_dataframe(df)
        df = preprocess(df)
        results = run_diffusion_analysis(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Serialize non-trivial objects
    cascade_metrics = results["cascade_metrics"].to_dict(orient="records")
    comparison = results["comparison"].fillna("null").to_dict(orient="records")
    graph_summary = results["graph_summary"]
    report_text = format_comparison_report(results["comparison"])

    return JSONResponse(content={
        "graph_summary": graph_summary,
        "comparison": comparison,
        "cascade_metrics_sample": cascade_metrics[:50],  # cap for readability
        "report": report_text,
    })


@app.get("/metrics", tags=["Detection"])
def get_metrics() -> JSONResponse:
    """
    Return saved model evaluation metrics.
    """
    metrics_path = get_path("models_dir") / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Train models first.")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return JSONResponse(content=metrics)