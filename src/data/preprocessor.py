"""
Text and metadata preprocessing pipeline for social posts.
"""

from __future__ import annotations

import re
import string
from typing import Optional

import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Regex patterns compiled once
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    normalize_mentions: bool = True,
    normalize_hashtags: bool = True,
    remove_punctuation: bool = False,
) -> str:
    """
    Apply text normalization steps.

    Args:
        text: Raw post text.
        lowercase: Convert to lowercase.
        remove_urls: Replace URLs with <URL> token.
        normalize_mentions: Replace @mentions with <MENTION> token.
        normalize_hashtags: Replace #hashtags with <HASHTAG> token.
        remove_punctuation: Strip punctuation characters.

    Returns:
        Normalized text string.
    """
    if not isinstance(text, str):
        return ""

    if remove_urls:
        text = _URL_RE.sub("<URL>", text)
    if normalize_mentions:
        text = _MENTION_RE.sub("<MENTION>", text)
    if normalize_hashtags:
        text = _HASHTAG_RE.sub("<HASHTAG>", text)
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = _PUNCT_RE.sub(" ", text)

    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def filter_by_language(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """
    Filter rows to a target language using langdetect.

    Falls back to keeping all rows if langdetect is not available.

    Args:
        df: DataFrame with a 'text' column.
        lang: BCP-47 language code to keep.

    Returns:
        Filtered DataFrame.
    """
    try:
        from langdetect import detect, LangDetectException

        def _detect(text: str) -> str:
            try:
                return detect(text) if len(text) > 20 else lang
            except LangDetectException:
                return lang

        before = len(df)
        detected = df["clean_text"].apply(_detect)
        df = df[detected == lang].copy()
        logger.info(f"Language filter ({lang}): {before:,} → {len(df):,} rows")
    except ImportError:
        logger.warning("langdetect not installed — skipping language filtering")

    return df


def remove_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate posts.

    Args:
        df: Input DataFrame.
        subset: Columns to consider. Defaults to ['post_id', 'text'].

    Returns:
        Deduplicated DataFrame.
    """
    subset = subset or ["post_id", "text"]
    before = len(df)
    df = df.drop_duplicates(subset=subset).copy()
    logger.info(f"Deduplication: {before:,} → {len(df):,} rows")
    return df


def filter_by_length(
    df: pd.DataFrame,
    min_length: int = 10,
    max_length: int = 2048,
) -> pd.DataFrame:
    """
    Drop rows where clean_text length is outside [min_length, max_length].

    Args:
        df: DataFrame with 'clean_text' column.
        min_length: Minimum character count.
        max_length: Maximum character count.

    Returns:
        Filtered DataFrame.
    """
    before = len(df)
    lengths = df["clean_text"].str.len()
    df = df[(lengths >= min_length) & (lengths <= max_length)].copy()
    logger.info(f"Length filter [{min_length}, {max_length}]: {before:,} → {len(df):,} rows")
    return df


def preprocess(
    df: pd.DataFrame,
    language: Optional[str] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on a post DataFrame.

    Steps:
        1. Normalize text → store in 'clean_text'
        2. Remove duplicates
        3. Filter by text length
        4. Filter by language (optional)
        5. Drop rows with null required fields

    Args:
        df: Validated input DataFrame.
        language: Language code to keep. Reads from config if None.
        min_length: Min text length. Reads from config if None.
        max_length: Max text length. Reads from config if None.

    Returns:
        Preprocessed DataFrame with a new 'clean_text' column.
    """
    cfg = get_config()["data"]

    language = language or cfg.get("language_filter")
    min_length = min_length or cfg.get("min_text_length", 10)
    max_length = max_length or cfg.get("max_text_length", 2048)

    logger.info(f"Preprocessing {len(df):,} posts…")

    # 1. Normalize text
    df = df.copy()
    df["clean_text"] = df["text"].apply(normalize_text)

    # 2. Drop duplicates
    df = remove_duplicates(df)

    # 3. Length filter
    df = filter_by_length(df, min_length=min_length, max_length=max_length)

    # 4. Language filter
    if language:
        df = filter_by_language(df, lang=language)

    # 5. Drop nulls in critical columns
    before = len(df)
    df = df.dropna(subset=["post_id", "user_id", "clean_text"]).copy()
    logger.info(f"Null drop: {before:,} → {len(df):,} rows")

    df = df.reset_index(drop=True)
    logger.info(f"Preprocessing complete: {len(df):,} posts remaining")
    return df