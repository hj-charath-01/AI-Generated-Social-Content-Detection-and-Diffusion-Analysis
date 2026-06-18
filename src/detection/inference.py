"""
Inference engine: single-post and batch prediction with confidence scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.preprocessor import preprocess
from src.data.schema import validate_dataframe
from src.detection.models import decode_labels, LABEL_TO_INT
from src.detection.trainer import DetectionTrainer
from src.features.feature_pipeline import FeaturePipeline
from src.utils.config import get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Loads a saved feature pipeline + model and runs inference.

    Example:
        predictor = Predictor.from_saved()
        result = predictor.predict_single("This is a test post.")
        results_df = predictor.predict_batch(df)
    """

    def __init__(self, pipeline: FeaturePipeline, model: Any, model_name: str = "ensemble"):
        self.pipeline = pipeline
        self.model = model
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_saved(
        cls,
        model_name: str = "ensemble",
        models_dir: Optional[str | Path] = None,
    ) -> "Predictor":
        """
        Load the feature pipeline and model from saved artifacts.

        Args:
            model_name: Which model to load (e.g. 'ensemble', 'random_forest').
            models_dir: Directory containing saved artifacts.

        Returns:
            Initialized Predictor instance.
        """
        if models_dir is None:
            models_dir = get_path("models_dir")
        models_dir = Path(models_dir)

        pipeline_path = models_dir / "feature_pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Feature pipeline not found: {pipeline_path}")

        pipeline = FeaturePipeline.load(pipeline_path)
        model = DetectionTrainer.load_model(model_name, models_dir)
        return cls(pipeline=pipeline, model=model, model_name=model_name)

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict_single(
        self,
        text: str,
        user_id: str = "unknown",
        post_id: str = "single",
        followers: int = 0,
        following: int = 0,
        engagement_count: int = 0,
    ) -> dict[str, Any]:
        """
        Predict whether a single post is human or AI-generated.

        Args:
            text: Raw post text.
            user_id: Author user ID.
            post_id: Post identifier.
            followers: Author's follower count.
            following: Author's following count.
            engagement_count: Engagement (likes/shares) on the post.

        Returns:
            Dict with 'label', 'confidence', 'probabilities'.
        """
        row = {
            "post_id": post_id,
            "user_id": user_id,
            "text": text,
            "followers": followers,
            "following": following,
            "engagement_count": engagement_count,
        }
        df = pd.DataFrame([row])
        result_df = self.predict_batch(df)
        return result_df.iloc[0].to_dict()

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict labels and confidence for a batch of posts.

        Args:
            df: DataFrame with at least 'post_id', 'user_id', 'text'.

        Returns:
            Original DataFrame with added columns:
                predicted_label, confidence, prob_human, prob_ai_generated.
        """
        df = validate_dataframe(df.copy())
        df = preprocess(df)

        if len(df) == 0:
            logger.warning("No valid posts after preprocessing")
            return df

        X = self.pipeline.transform(df)
        predictions = self.model.predict(X)
        labels = decode_labels(predictions)

        try:
            probs = self.model.predict_proba(X)
            prob_human = probs[:, 0]
            prob_ai = probs[:, 1]
            confidence = probs.max(axis=1)
        except Exception:
            prob_human = np.where(predictions == 0, 1.0, 0.0)
            prob_ai = np.where(predictions == 1, 1.0, 0.0)
            confidence = np.ones(len(predictions))

        result = df.copy()
        result["predicted_label"] = labels
        result["confidence"] = confidence.round(4)
        result["prob_human"] = prob_human.round(4)
        result["prob_ai_generated"] = prob_ai.round(4)

        return result

    def predict_file(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Load a file, run batch prediction, optionally save results.

        Args:
            input_path: Path to CSV/JSON/JSONL file.
            output_path: If provided, saves predictions here.

        Returns:
            Predictions DataFrame.
        """
        from src.data.loader import load_any

        df = load_any(input_path)
        result = self.predict_batch(df)

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out, index=False)
            logger.info(f"Predictions saved to {out}")

        return result