"""
Training pipeline for detection models.

Handles train/val/test splits, cross-validation, class imbalance,
model saving, and returns evaluation metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from src.detection.models import (
    make_logistic_regression,
    make_random_forest,
    make_xgboost,
    make_lightgbm,
    make_ensemble,
    encode_labels,
    decode_labels,
    LABEL_TO_INT,
)
from src.features.feature_pipeline import FeaturePipeline
from src.utils.config import get_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Compute standard classification metrics."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["human", "ai_generated"]),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            metrics["roc_auc"] = None
    return metrics


class DetectionTrainer:
    """
    Trains and evaluates detection models on pre-extracted feature matrices.

    Usage:
        trainer = DetectionTrainer()
        results = trainer.train_all(X, y, df)
        trainer.save_artifacts(models_dir)
    """

    def __init__(self):
        cfg = get_config()
        self.seed = cfg["project"]["seed"]
        self.test_size = cfg["data"]["test_size"]
        self.val_size = cfg["data"]["val_size"]
        self.cv_folds = cfg["training"]["cv_folds"]

        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.trained_models: dict[str, Any] = {}
        self.metrics: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stratified train / val / test split.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.seed
        )
        val_fraction = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_fraction,
            stratify=y_train_val,
            random_state=self.seed,
        )
        logger.info(f"Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray, model_name: str) -> dict[str, float]:
        """
        Run stratified k-fold cross-validation.

        Args:
            model: Unfitted sklearn-compatible classifier.
            X: Feature matrix.
            y: Integer label array.
            model_name: Name for logging.

        Returns:
            Dict of mean CV metrics.
        """
        import copy

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        fold_f1s: list[float] = []
        fold_aucs: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            m = copy.deepcopy(model)
            m.fit(X[train_idx], y[train_idx])
            y_pred = m.predict(X[val_idx])
            f1 = float(f1_score(y[val_idx], y_pred, zero_division=0))
            fold_f1s.append(f1)
            try:
                y_prob = m.predict_proba(X[val_idx])
                auc = float(roc_auc_score(y[val_idx], y_prob[:, 1]))
                fold_aucs.append(auc)
            except Exception:
                pass

        cv_results = {
            "cv_f1_mean": float(np.mean(fold_f1s)),
            "cv_f1_std": float(np.std(fold_f1s)),
        }
        if fold_aucs:
            cv_results["cv_auc_mean"] = float(np.mean(fold_aucs))
            cv_results["cv_auc_std"] = float(np.std(fold_aucs))

        logger.info(
            f"CV ({model_name}): F1={cv_results['cv_f1_mean']:.4f} ± {cv_results['cv_f1_std']:.4f}"
        )
        return cv_results

    # ------------------------------------------------------------------
    # Train individual model
    # ------------------------------------------------------------------

    def train_model(
        self,
        model: Any,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_cv: bool = True,
        X_full: Optional[np.ndarray] = None,
        y_full: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """
        Fit a model, evaluate on test set, optionally run CV.

        Args:
            model: Unfitted classifier.
            model_name: Display name.
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            run_cv: Whether to run cross-validation.
            X_full: Full feature matrix for CV (uses X_train if None).
            y_full: Full label array for CV.

        Returns:
            Metrics dict including cv results if run_cv=True.
        """
        logger.info(f"Training {model_name}…")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None

        metrics = _compute_metrics(y_test, y_pred, y_prob)
        logger.info(
            f"{model_name} → acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics.get('roc_auc', 'N/A')}"
        )

        if run_cv:
            cv_X = X_full if X_full is not None else X_train
            cv_y = y_full if y_full is not None else y_train
            cv_results = self.cross_validate(model, cv_X, cv_y, model_name)
            metrics.update(cv_results)

        self.trained_models[model_name] = model
        self.metrics[model_name] = metrics
        return metrics

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------

    def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        run_cv: bool = True,
    ) -> dict[str, dict]:
        """
        Train all model variants on the feature matrix.

        Args:
            X: Full feature matrix (n_samples, n_features).
            y: Integer label array.
            run_cv: Whether to run cross-validation for each model.

        Returns:
            Dict mapping model name → metrics dict.
        """
        # Filter out unlabeled rows
        mask = y >= 0
        X, y = X[mask], y[mask]

        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y)

        models_to_train = {
            "logistic_regression": make_logistic_regression(),
            "random_forest": make_random_forest(),
            "xgboost": make_xgboost(),
            "ensemble": make_ensemble(),
        }

        for name, model in models_to_train.items():
            self.train_model(
                model=model,
                model_name=name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                run_cv=run_cv,
                X_full=X,
                y_full=y,
            )

        return self.metrics

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_artifacts(self, models_dir: Optional[str | Path] = None) -> None:
        """
        Save all trained models and metrics to disk.

        Args:
            models_dir: Directory to save artifacts. Defaults to config path.
        """
        if models_dir is None:
            models_dir = get_path("models_dir")
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.trained_models.items():
            out = models_dir / f"{name}.joblib"
            joblib.dump(model, out)
            logger.info(f"Saved {name} → {out}")

        metrics_path = models_dir / "metrics.json"
        import json
        with open(metrics_path, "w") as f:
            # Remove non-serializable keys
            clean = {
                k: {mk: mv for mk, mv in v.items() if mk != "classification_report"}
                for k, v in self.metrics.items()
            }
            json.dump(clean, f, indent=2)
        logger.info(f"Saved metrics → {metrics_path}")

    @staticmethod
    def load_model(name: str, models_dir: Optional[str | Path] = None) -> Any:
        """
        Load a single saved model.

        Args:
            name: Model name (e.g. 'ensemble', 'random_forest').
            models_dir: Directory where models are saved.

        Returns:
            Loaded sklearn model.
        """
        if models_dir is None:
            models_dir = get_path("models_dir")
        path = Path(models_dir) / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        model = joblib.load(path)
        logger.info(f"Loaded {name} from {path}")
        return model