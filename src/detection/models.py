"""
Detection model definitions.

Provides factory functions for each model type and a stacking ensemble
that combines tabular and transformer predictions.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tabular baseline models
# ---------------------------------------------------------------------------

def make_logistic_regression(class_weight: str = "balanced", **kwargs: Any) -> LogisticRegression:
    """
    Build a Logistic Regression classifier.

    Args:
        class_weight: How to weight classes. 'balanced' handles imbalance.
        **kwargs: Additional arguments forwarded to LogisticRegression.

    Returns:
        Configured LogisticRegression instance.
    """
    cfg = get_config()["models"]["logistic_regression"]
    params = {
        "max_iter": cfg["max_iter"],
        "C": cfg["C"],
        "class_weight": class_weight,
        "solver": "lbfgs",
        "n_jobs": -1,
    }
    params.update(kwargs)
    logger.info(f"Creating LogisticRegression: {params}")
    return LogisticRegression(**params)


def make_random_forest(class_weight: str = "balanced", **kwargs: Any) -> RandomForestClassifier:
    """
    Build a Random Forest classifier.

    Args:
        class_weight: Class weighting strategy.
        **kwargs: Additional arguments.

    Returns:
        Configured RandomForestClassifier instance.
    """
    cfg = get_config()["models"]["random_forest"]
    params = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "class_weight": class_weight,
        "random_state": get_config()["project"]["seed"],
        "n_jobs": -1,
    }
    params.update(kwargs)
    logger.info(f"Creating RandomForest: {params}")
    return RandomForestClassifier(**params)


def make_xgboost(**kwargs: Any):
    """
    Build an XGBoost classifier.

    Falls back to a RandomForest if xgboost is not installed.

    Args:
        **kwargs: Additional arguments forwarded to XGBClassifier.

    Returns:
        XGBClassifier or RandomForestClassifier instance.
    """
    try:
        from xgboost import XGBClassifier
        cfg = get_config()["models"]["xgboost"]
        params = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": get_config()["project"]["seed"],
            "n_jobs": -1,
        }
        params.update(kwargs)
        logger.info(f"Creating XGBClassifier: {params}")
        return XGBClassifier(**params)
    except ImportError:
        logger.warning("xgboost not installed — using RandomForest as fallback")
        return make_random_forest(**kwargs)


def make_lightgbm(**kwargs: Any):
    """
    Build a LightGBM classifier.

    Falls back to XGBoost → RandomForest if not installed.

    Args:
        **kwargs: Additional arguments forwarded to LGBMClassifier.

    Returns:
        LGBMClassifier instance or fallback.
    """
    try:
        from lightgbm import LGBMClassifier
        params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "class_weight": "balanced",
            "random_state": get_config()["project"]["seed"],
            "n_jobs": -1,
            "verbose": -1,
        }
        params.update(kwargs)
        logger.info(f"Creating LGBMClassifier: {params}")
        return LGBMClassifier(**params)
    except ImportError:
        logger.warning("lightgbm not installed — falling back to XGBoost")
        return make_xgboost(**kwargs)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def make_ensemble(
    lr_weight: float = 1.0,
    rf_weight: float = 2.0,
    xgb_weight: float = 2.0,
) -> VotingClassifier:
    """
    Build a soft-voting ensemble of LR + RF + XGB.

    Args:
        lr_weight: Voting weight for Logistic Regression.
        rf_weight: Voting weight for Random Forest.
        xgb_weight: Voting weight for XGBoost.

    Returns:
        VotingClassifier instance.
    """
    lr = make_logistic_regression()
    rf = make_random_forest()
    xgb = make_xgboost()

    clf = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        voting="soft",
        weights=[lr_weight, rf_weight, xgb_weight],
        n_jobs=-1,
    )
    logger.info("Created soft-voting ensemble: LR + RF + XGB")
    return clf


# ---------------------------------------------------------------------------
# Label encoding helper
# ---------------------------------------------------------------------------

LABEL_TO_INT: dict[str, int] = {"human": 0, "ai_generated": 1}
INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}


def encode_labels(labels: "pd.Series | list[str]") -> np.ndarray:  # type: ignore[name-defined]
    """Convert string labels to integer array (0=human, 1=ai_generated)."""
    import pandas as pd
    s = pd.Series(labels)
    return s.map(LABEL_TO_INT).fillna(-1).astype(int).values


def decode_labels(ints: np.ndarray) -> list[str]:
    """Convert integer predictions to string labels."""
    return [INT_TO_LABEL.get(i, "unknown") for i in ints]