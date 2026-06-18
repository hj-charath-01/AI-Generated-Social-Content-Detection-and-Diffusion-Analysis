"""
Behavioral and metadata feature extraction.

Computes per-post and per-user features from account metadata and
temporal posting patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-row account-level metadata features.

    Args:
        df: DataFrame with 'followers', 'following', 'engagement_count'.

    Returns:
        DataFrame of features (same index as df).
    """
    feat: dict[str, pd.Series] = {}

    followers = df["followers"].clip(lower=0).astype(float)
    following = df["following"].clip(lower=0).astype(float)
    engagement = df["engagement_count"].clip(lower=0).astype(float)

    feat["followers"] = followers
    feat["following"] = following
    feat["engagement_count"] = engagement

    # Follower-to-following ratio (capped to avoid inf)
    feat["ff_ratio"] = (followers / (following + 1)).clip(upper=1000)

    # Engagement rate relative to followers
    feat["engagement_rate"] = (engagement / (followers + 1)).clip(upper=100)

    # Log-scaled versions for skewed distributions
    feat["log_followers"] = np.log1p(followers)
    feat["log_following"] = np.log1p(following)
    feat["log_engagement"] = np.log1p(engagement)

    return pd.DataFrame(feat, index=df.index)


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal posting-behavior features aggregated per user,
    then mapped back to each post.

    Requires a 'timestamp' column (datetime with UTC timezone).

    Args:
        df: DataFrame with 'user_id' and 'timestamp'.

    Returns:
        DataFrame of temporal features (same index as df).
    """
    feat_rows: list[dict] = []

    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        logger.warning("No valid timestamps — returning zero temporal features")
        zero_feat = {
            "user_post_count": 1.0,
            "inter_post_time_mean": 0.0,
            "inter_post_time_var": 0.0,
            "inter_post_time_cv": 0.0,
            "temporal_regularity": 0.0,
            "burst_score": 0.0,
            "hour_of_day": 0.0,
            "is_weekend": 0.0,
        }
        return pd.DataFrame([zero_feat] * len(df), index=df.index)

    # Build user posting stats
    user_stats: dict[str, dict] = {}
    for uid, grp in df.dropna(subset=["timestamp"]).groupby("user_id"):
        times = pd.to_datetime(grp["timestamp"], errors="coerce", utc=True).sort_values()
        times = times.dropna()
        n = len(times)
        if n > 1:
            deltas = times.diff().dt.total_seconds().dropna()
            mean_delta = float(deltas.mean())
            var_delta = float(deltas.var()) if n > 2 else 0.0
            cv = var_delta**0.5 / (mean_delta + 1e-9)
            # Temporal regularity: 1 - cv (higher → more regular)
            regularity = float(np.clip(1.0 - cv, 0, 1))
            # Burst score: fraction of consecutive pairs under 60s apart
            burst_score = float((deltas < 60).mean())
        else:
            mean_delta = 0.0
            var_delta = 0.0
            cv = 0.0
            regularity = 0.5
            burst_score = 0.0

        user_stats[uid] = {
            "user_post_count": float(n),
            "inter_post_time_mean": mean_delta,
            "inter_post_time_var": var_delta,
            "inter_post_time_cv": cv,
            "temporal_regularity": regularity,
            "burst_score": burst_score,
        }

    # Map back to each row
    for _, row in df.iterrows():
        uid = row["user_id"]
        stats = user_stats.get(uid, {
            "user_post_count": 1.0,
            "inter_post_time_mean": 0.0,
            "inter_post_time_var": 0.0,
            "inter_post_time_cv": 0.0,
            "temporal_regularity": 0.5,
            "burst_score": 0.0,
        })
        ts = row.get("timestamp")
        if pd.notna(ts):
            hour = pd.Timestamp(ts).hour
            dow = pd.Timestamp(ts).dayofweek
        else:
            hour = 0
            dow = 0

        feat_rows.append({
            **stats,
            "hour_of_day": float(hour),
            "is_weekend": float(dow >= 5),
        })

    return pd.DataFrame(feat_rows, index=df.index)


def compute_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reply and reshare behavior ratios per post.

    Args:
        df: DataFrame with 'reply_to', 'reshare_of', 'user_id'.

    Returns:
        DataFrame of network behavior features.
    """
    feat: dict[str, pd.Series] = {}

    # Reply and reshare counts per user
    if "reply_to" in df.columns:
        user_reply_count = df.dropna(subset=["reply_to"]).groupby("user_id")["reply_to"].count()
    else:
        user_reply_count = pd.Series(dtype=float)

    if "reshare_of" in df.columns:
        user_reshare_count = df.dropna(subset=["reshare_of"]).groupby("user_id")["reshare_of"].count()
    else:
        user_reshare_count = pd.Series(dtype=float)

    user_post_count = df.groupby("user_id")["post_id"].count()

    reply_ratio = (user_reply_count / user_post_count).fillna(0.0).clip(0, 1)
    reshare_ratio = (user_reshare_count / user_post_count).fillna(0.0).clip(0, 1)

    feat["reply_ratio"] = df["user_id"].map(reply_ratio).fillna(0.0)
    feat["reshare_ratio"] = df["user_id"].map(reshare_ratio).fillna(0.0)
    feat["is_reply"] = df["reply_to"].notna().astype(float) if "reply_to" in df.columns else 0.0
    feat["is_reshare"] = df["reshare_of"].notna().astype(float) if "reshare_of" in df.columns else 0.0

    return pd.DataFrame(feat, index=df.index)


def extract_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all metadata feature families into one DataFrame.

    Args:
        df: Preprocessed post DataFrame.

    Returns:
        Combined metadata feature DataFrame.
    """
    logger.info("Extracting metadata features…")

    account = compute_account_features(df)
    temporal = compute_temporal_features(df)
    network = compute_network_features(df)

    combined = pd.concat([account, temporal, network], axis=1)
    logger.info(f"  → {combined.shape[1]} metadata features")
    return combined