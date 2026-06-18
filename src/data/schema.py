"""
Data schemas and type definitions for social media posts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


class Label(str, Enum):
    """Ground-truth label for a post."""
    HUMAN = "human"
    AI_GENERATED = "ai_generated"
    UNKNOWN = "unknown"


# Required and optional columns for the unified post DataFrame schema
REQUIRED_COLUMNS: list[str] = ["post_id", "user_id", "text"]

OPTIONAL_COLUMNS: dict[str, object] = {
    "timestamp": None,
    "platform": "unknown",
    "reply_to": None,
    "reshare_of": None,
    "engagement_count": 0,
    "followers": 0,
    "following": 0,
    "label": Label.UNKNOWN.value,
}

ALL_COLUMNS: list[str] = REQUIRED_COLUMNS + list(OPTIONAL_COLUMNS.keys())


@dataclass
class SocialPost:
    """Single social media post record."""

    post_id: str
    user_id: str
    text: str
    timestamp: Optional[datetime] = None
    platform: str = "unknown"
    reply_to: Optional[str] = None
    reshare_of: Optional[str] = None
    engagement_count: int = 0
    followers: int = 0
    following: int = 0
    label: str = Label.UNKNOWN.value
    # Internal enrichment fields (populated during feature extraction)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize post to dictionary (excluding extra)."""
        return {
            "post_id": self.post_id,
            "user_id": self.user_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "platform": self.platform,
            "reply_to": self.reply_to,
            "reshare_of": self.reshare_of,
            "engagement_count": self.engagement_count,
            "followers": self.followers,
            "following": self.following,
            "label": self.label,
        }


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a DataFrame conforms to the post schema.

    Adds missing optional columns with defaults and casts types.

    Args:
        df: Raw input DataFrame.

    Returns:
        Validated and normalized DataFrame.

    Raises:
        ValueError: If required columns are absent.
    """
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"DataFrame missing required columns: {missing_required}")

    for col, default in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = default

    # Cast types
    df["post_id"] = df["post_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["engagement_count"] = pd.to_numeric(df["engagement_count"], errors="coerce").fillna(0).astype(int)
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0).astype(int)
    df["following"] = pd.to_numeric(df["following"], errors="coerce").fillna(0).astype(int)

    # Parse timestamps
    if df["timestamp"].dtype == object:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Normalize label
    valid_labels = {l.value for l in Label}
    df["label"] = df["label"].fillna(Label.UNKNOWN.value)
    df["label"] = df["label"].apply(lambda x: x if x in valid_labels else Label.UNKNOWN.value)

    return df