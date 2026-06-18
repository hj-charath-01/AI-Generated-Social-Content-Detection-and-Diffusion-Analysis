"""
Data loaders for CSV, JSON, and JSONL social post datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import validate_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Load posts from a CSV file.

    Args:
        path: Path to .csv file.
        **kwargs: Extra arguments forwarded to pd.read_csv.

    Returns:
        Validated DataFrame of posts.
    """
    path = Path(path)
    logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, **kwargs)
    logger.info(f"  Loaded {len(df):,} rows")
    return validate_dataframe(df)


def load_json(path: str | Path) -> pd.DataFrame:
    """
    Load posts from a JSON file (list of objects or records-oriented).

    Args:
        path: Path to .json file.

    Returns:
        Validated DataFrame of posts.
    """
    path = Path(path)
    logger.info(f"Loading JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Support {data: [...]} envelope
        records = data.get("data") or data.get("posts") or list(data.values())[0]
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    logger.info(f"  Loaded {len(df):,} rows")
    return validate_dataframe(df)


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """
    Load posts from a JSONL (newline-delimited JSON) file.

    Args:
        path: Path to .jsonl file.

    Returns:
        Validated DataFrame of posts.
    """
    path = Path(path)
    logger.info(f"Loading JSONL: {path}")
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    logger.info(f"  Loaded {len(df):,} rows")
    return validate_dataframe(df)


def load_edge_list(path: str | Path) -> pd.DataFrame:
    """
    Load a network edge list (source_id, target_id, optional weight/type).

    Args:
        path: Path to CSV file with at least columns [source, target].

    Returns:
        DataFrame with columns [source, target, ...].
    """
    path = Path(path)
    logger.info(f"Loading edge list: {path}")
    df = pd.read_csv(path)
    required = {"source", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"Edge list must have columns: {required}. Got: {list(df.columns)}")
    return df


def load_any(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Auto-detect file format and load posts.

    Args:
        path: Path to data file.
        **kwargs: Extra args for CSV loader.

    Returns:
        Validated DataFrame.

    Raises:
        ValueError: If file extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()

    loaders = {
        ".csv": lambda: load_csv(path, **kwargs),
        ".json": lambda: load_json(path),
        ".jsonl": lambda: load_jsonl(path),
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported file format: '{ext}'. Use: {list(loaders)}")

    return loaders[ext]()