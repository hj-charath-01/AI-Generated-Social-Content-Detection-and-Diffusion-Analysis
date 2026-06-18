"""
Unified feature pipeline: combines statistical text features,
transformer embeddings, and metadata features into one matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.text_features import extract_statistical_features, TransformerEmbedder
from src.features.metadata_features import extract_metadata_features
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """
    End-to-end feature pipeline.

    Extracts:
        - Statistical text features (always)
        - Metadata / behavioral features (always)
        - Transformer embeddings (optional, controlled by use_embeddings flag)

    Applies StandardScaler fitted on training data.
    """

    def __init__(
        self,
        use_embeddings: bool = False,
        transformer_model: Optional[str] = None,
        max_length: int = 256,
        embedding_batch_size: int = 32,
    ):
        cfg = get_config()
        self.use_embeddings = use_embeddings
        self.transformer_model = transformer_model or cfg["features"]["text"]["transformer_model"]
        self.max_length = max_length
        self.embedding_batch_size = embedding_batch_size

        self.scaler = StandardScaler()
        self._fitted = False
        self._feature_names: list[str] = []
        self._embedder: Optional[TransformerEmbedder] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the scaler on df and return the scaled feature matrix.

        Args:
            df: Preprocessed post DataFrame (must have 'clean_text').

        Returns:
            Scaled feature matrix of shape (n_samples, n_features).
        """
        feature_df = self._extract_all(df)
        self._feature_names = list(feature_df.columns)
        X = feature_df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        logger.info(f"FeaturePipeline fit_transform: {X_scaled.shape}")
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform df using the already-fitted scaler.

        Args:
            df: Preprocessed post DataFrame.

        Returns:
            Scaled feature matrix.
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline must be fit before calling transform.")
        feature_df = self._extract_all(df)
        # Align columns in case of mismatch
        feature_df = feature_df.reindex(columns=self._feature_names, fill_value=0.0)
        X = feature_df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X)

    def get_feature_names(self) -> list[str]:
        """Return list of feature names in column order."""
        return self._feature_names.copy()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save pipeline to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Don't pickle the heavy transformer model
        embedder = self._embedder
        self._embedder = None
        joblib.dump(self, path)
        self._embedder = embedder
        logger.info(f"FeaturePipeline saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Load a saved pipeline from disk."""
        path = Path(path)
        pipeline: FeaturePipeline = joblib.load(path)
        logger.info(f"FeaturePipeline loaded from {path}")
        return pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature extractors and concatenate results."""
        parts: list[pd.DataFrame] = []

        # 1. Statistical text features
        stat_feats = extract_statistical_features(df)
        parts.append(stat_feats)

        # 2. Metadata features
        meta_feats = extract_metadata_features(df)
        parts.append(meta_feats)

        # 3. Transformer embeddings (optional)
        if self.use_embeddings:
            if self._embedder is None:
                self._embedder = TransformerEmbedder(
                    model_name=self.transformer_model,
                    max_length=self.max_length,
                )
            emb_feats = self._embedder.encode_dataframe(
                df, batch_size=self.embedding_batch_size
            )
            parts.append(emb_feats)

        combined = pd.concat(parts, axis=1)
        return combined