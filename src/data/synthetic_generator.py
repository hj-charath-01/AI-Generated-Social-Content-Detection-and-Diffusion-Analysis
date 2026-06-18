"""
Synthetic dataset generator for demo / testing purposes.

Generates realistic-looking human and AI-generated post records with
plausible metadata and an optional interaction graph.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Template content pools
# ---------------------------------------------------------------------------

_HUMAN_TEMPLATES = [
    "just had the best {food} at {place}! totally worth it 😊",
    "can't believe it's already {month}. where did the year go??",
    "my {pet} is being so annoying today lmao",
    "does anyone else feel like {topic} is totally overrated?",
    "honestly {feeling} about {event} rn. mixed feelings for sure",
    "unpopular opinion: {opinion}. fight me",
    "throwback to that time when {memory}. good times fr",
    "ok why does everyone suddenly care about {trend}? been saying this for years",
    "you guys i just found out {discovery}. my whole life is a lie",
    "woke up at {time} for no reason and now i can't sleep. hate this",
]

_AI_TEMPLATES = [
    "The intersection of {topic_a} and {topic_b} presents significant opportunities "
    "for innovation. Organizations that prioritize {strategy} will be well-positioned "
    "to capitalize on emerging trends in the coming quarters.",
    "As we navigate the evolving landscape of {domain}, it becomes increasingly "
    "important to consider the multifaceted implications of {concept}. "
    "A comprehensive framework for analysis should incorporate both {factor_a} "
    "and {factor_b} to yield meaningful insights.",
    "Key considerations for {topic}: 1) {point_a}. 2) {point_b}. 3) {point_c}. "
    "By addressing these dimensions, stakeholders can achieve sustainable outcomes "
    "aligned with organizational objectives.",
    "The data clearly indicates that {trend} is reshaping {industry}. "
    "Forward-thinking leaders must adapt their strategies to remain competitive. "
    "Leveraging {technology} will be paramount in this transition.",
    "I'm excited to share insights on {topic}. The evidence strongly suggests that "
    "{claim}. This has profound implications for {stakeholder_group} worldwide. "
    "Let's discuss how we can leverage this for positive impact.",
]

_FILL_WORDS: dict[str, list[str]] = {
    "food": ["ramen", "sushi", "tacos", "pasta", "pizza", "biryani"],
    "place": ["this tiny spot downtown", "a random food truck", "my friend's place"],
    "month": ["January", "March", "August", "November"],
    "pet": ["cat", "dog", "hamster", "fish"],
    "topic": ["crypto", "AI", "remote work", "veganism", "meditation"],
    "feeling": ["conflicted", "excited", "nervous", "whatever"],
    "event": ["the announcement", "what happened yesterday", "that whole situation"],
    "opinion": ["coffee is overrated", "open offices are terrible", "mornings are fine"],
    "memory": ["we got completely lost", "we stayed up all night talking"],
    "trend": ["that new app", "this song everyone's playing"],
    "discovery": ["this has been happening for years", "I misunderstood everything"],
    "time": ["3am", "4:30am", "5am"],
    "topic_a": ["artificial intelligence", "sustainability", "digital transformation"],
    "topic_b": ["supply chain optimization", "customer experience", "talent management"],
    "strategy": ["data-driven decision-making", "agile methodologies", "stakeholder alignment"],
    "domain": ["healthcare", "finance", "education", "retail"],
    "concept": ["automation", "personalization", "scalability", "resilience"],
    "factor_a": ["quantitative metrics", "qualitative feedback", "market signals"],
    "factor_b": ["regulatory frameworks", "technological constraints", "behavioral patterns"],
    "point_a": ["Prioritize scalability from day one"],
    "point_b": ["Build cross-functional alignment early"],
    "point_c": ["Measure outcomes with clear KPIs"],
    "industry": ["healthcare", "finance", "retail", "manufacturing"],
    "technology": ["machine learning", "cloud infrastructure", "automation tools"],
    "trend": ["digital transformation", "remote collaboration", "AI adoption"],
    "claim": ["adoption rates have accelerated significantly", "outcomes improve with structured frameworks"],
    "stakeholder_group": ["organizations", "communities", "practitioners"],
}


def _fill_template(template: str) -> str:
    """Fill template placeholders with random words."""
    result = template
    for key, values in _FILL_WORDS.items():
        placeholder = f"{{{key}}}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values))
    return result


def _generate_user_ids(n_users: int) -> list[str]:
    return [f"user_{i:05d}" for i in range(n_users)]


def _random_timestamp(
    start: datetime,
    end: datetime,
    rng: np.random.Generator,
) -> datetime:
    delta = (end - start).total_seconds()
    offset = rng.uniform(0, delta)
    return start + timedelta(seconds=offset)


def generate_dataset(
    n_posts: int = 1000,
    ai_fraction: float = 0.35,
    n_users: int = 200,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic social media dataset.

    Args:
        n_posts: Total number of posts to generate.
        ai_fraction: Fraction of posts labeled as AI-generated.
        n_users: Number of unique users.
        seed: Random seed for reproducibility.
        output_path: If provided, saves the dataset to this CSV path.

    Returns:
        DataFrame conforming to the project schema.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n_ai = int(n_posts * ai_fraction)
    n_human = n_posts - n_ai

    users = _generate_user_ids(n_users)
    platforms = ["twitter", "reddit", "facebook", "mastodon"]

    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2024, 6, 30, tzinfo=timezone.utc)

    records: list[dict] = []
    post_ids: list[str] = []

    # Generate human posts
    for i in range(n_human):
        template = random.choice(_HUMAN_TEMPLATES)
        text = _fill_template(template)
        post_id = f"post_{uuid.uuid4().hex[:8]}"
        post_ids.append(post_id)
        records.append({
            "post_id": post_id,
            "user_id": rng.choice(users),
            "text": text,
            "timestamp": _random_timestamp(start_dt, end_dt, rng).isoformat(),
            "platform": rng.choice(platforms),
            "reply_to": None,
            "reshare_of": None,
            "engagement_count": int(rng.negative_binomial(1, 0.1)),
            "followers": int(rng.lognormal(5, 1.5)),
            "following": int(rng.lognormal(4.5, 1.2)),
            "label": "human",
        })

    # Generate AI posts
    for i in range(n_ai):
        template = random.choice(_AI_TEMPLATES)
        text = _fill_template(template)
        post_id = f"post_{uuid.uuid4().hex[:8]}"
        post_ids.append(post_id)
        records.append({
            "post_id": post_id,
            "user_id": rng.choice(users),
            "text": text,
            "timestamp": _random_timestamp(start_dt, end_dt, rng).isoformat(),
            "platform": rng.choice(platforms),
            "reply_to": None,
            "reshare_of": None,
            "engagement_count": int(rng.negative_binomial(2, 0.05)),
            "followers": int(rng.lognormal(7, 1.0)),   # AI accounts tend to have more followers
            "following": int(rng.lognormal(3.5, 0.8)),
            "label": "ai_generated",
        })

    # Add some reshare relations (for diffusion analysis)
    n_reshares = int(n_posts * 0.25)
    for _ in range(n_reshares):
        idx = rng.integers(0, len(records))
        records[idx]["reshare_of"] = rng.choice(post_ids)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    logger.info(
        f"Generated dataset: {len(df)} posts "
        f"({n_human} human, {n_ai} AI) across {n_users} users"
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved synthetic dataset to {out}")

    return df


def generate_edge_list(df: pd.DataFrame, output_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Derive a network edge list from reshare and reply relationships.

    Args:
        df: Post DataFrame with 'post_id', 'user_id', 'reshare_of', 'reply_to'.
        output_path: Optional CSV output path.

    Returns:
        DataFrame with columns [source, target, relation_type].
    """
    edges: list[dict] = []

    # Build post_id → user_id lookup
    id_to_user = df.set_index("post_id")["user_id"].to_dict()

    for _, row in df.iterrows():
        if pd.notna(row.get("reshare_of")) and row["reshare_of"] in id_to_user:
            edges.append({
                "source": id_to_user[row["reshare_of"]],
                "target": row["user_id"],
                "relation_type": "reshare",
                "post_id": row["post_id"],
                "timestamp": row.get("timestamp"),
            })
        if pd.notna(row.get("reply_to")) and row["reply_to"] in id_to_user:
            edges.append({
                "source": id_to_user[row["reply_to"]],
                "target": row["user_id"],
                "relation_type": "reply",
                "post_id": row["post_id"],
                "timestamp": row.get("timestamp"),
            })

    edge_df = pd.DataFrame(edges) if edges else pd.DataFrame(columns=["source", "target", "relation_type", "post_id", "timestamp"])

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        edge_df.to_csv(out, index=False)
        logger.info(f"Saved edge list ({len(edge_df)} edges) to {out}")

    return edge_df