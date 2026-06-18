"""
Explainability module.

Uses SHAP to attribute predictions to input features for tabular models.
Returns top-k contributing features per prediction.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Explainer:
    """
    Wraps SHAP explainers for trained detection models.

    Supports:
        - TreeExplainer for tree-based models (RF, XGB, LGBM, ensemble)
        - LinearExplainer for Logistic Regression
    """

    def __init__(self, model: Any, feature_names: list[str]):
        """
        Initialize the explainer.

        Args:
            model: Fitted sklearn-compatible classifier.
            feature_names: Ordered list of feature names matching X columns.
        """
        self.model = model
        self.feature_names = feature_names
        self._shap_explainer: Optional[Any] = None

    def _build_explainer(self, X_background: np.ndarray) -> None:
        """Lazy-build the SHAP explainer using a background sample."""
        try:
            import shap

            model_class = type(self.model).__name__.lower()

            if any(k in model_class for k in ("forest", "xgb", "lgbm", "boosting", "voting")):
                # For ensemble / tree models use TreeExplainer where possible
                try:
                    # VotingClassifier needs a workaround
                    if "voting" in model_class:
                        self._shap_explainer = shap.KernelExplainer(
                            self.model.predict_proba,
                            shap.sample(X_background, min(100, len(X_background))),
                        )
                    else:
                        self._shap_explainer = shap.TreeExplainer(self.model)
                except Exception:
                    self._shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        shap.sample(X_background, min(100, len(X_background))),
                    )
            elif "logistic" in model_class or "linear" in model_class:
                self._shap_explainer = shap.LinearExplainer(
                    self.model,
                    X_background,
                    feature_perturbation="interventional",
                )
            else:
                self._shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(X_background, min(100, len(X_background))),
                )

            logger.info(f"SHAP explainer built: {type(self._shap_explainer).__name__}")

        except ImportError:
            logger.warning("shap not installed — explainability unavailable")

    def explain(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Explain predictions for one or more samples.

        Args:
            X: Feature matrix for samples to explain (n_samples, n_features).
            X_background: Background data for SHAP (uses X if None).
            top_k: Number of top features to return per sample.

        Returns:
            List of explanation dicts, one per sample:
                {
                    'top_features': [{'feature': str, 'shap_value': float}, ...],
                    'base_value': float,
                }
        """
        if X_background is None:
            X_background = X

        if self._shap_explainer is None:
            self._build_explainer(X_background)

        if self._shap_explainer is None:
            # SHAP unavailable — return empty explanations
            return [{"top_features": [], "base_value": 0.0} for _ in range(len(X))]

        try:
            import shap

            shap_values = self._shap_explainer.shap_values(X)

            # For binary classifiers, shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                # Use class 1 (ai_generated) values
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            if sv.ndim == 1:
                sv = sv.reshape(1, -1)

            explanations: list[dict] = []
            for i in range(len(X)):
                row_sv = sv[i]
                sorted_idx = np.argsort(np.abs(row_sv))[::-1][:top_k]
                top_features = [
                    {
                        "feature": self.feature_names[j],
                        "shap_value": float(row_sv[j]),
                        "abs_shap": float(abs(row_sv[j])),
                    }
                    for j in sorted_idx
                ]
                base = float(self._shap_explainer.expected_value[1])  if isinstance(
                    self._shap_explainer.expected_value, (list, np.ndarray)
                ) else float(self._shap_explainer.expected_value)

                explanations.append({
                    "top_features": top_features,
                    "base_value": base,
                })

            return explanations

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return [{"top_features": [], "base_value": 0.0} for _ in range(len(X))]

    def explain_single(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            x: 1-D or 2-D (1, n_features) feature vector.
            X_background: Background data.
            top_k: Number of top features.

        Returns:
            Explanation dict.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        results = self.explain(x, X_background=X_background, top_k=top_k)
        return results[0]

    def feature_importance_global(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute global feature importance as mean absolute SHAP values.

        Args:
            X: Feature matrix.
            X_background: Background data.

        Returns:
            DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
        """
        explanations = self.explain(X, X_background=X_background, top_k=len(self.feature_names))

        feature_sums: dict[str, float] = {f: 0.0 for f in self.feature_names}
        for exp in explanations:
            for entry in exp.get("top_features", []):
                feature_sums[entry["feature"]] += entry["abs_shap"]

        n = max(len(explanations), 1)
        rows = [
            {"feature": f, "mean_abs_shap": v / n}
            for f, v in feature_sums.items()
        ]
        df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df