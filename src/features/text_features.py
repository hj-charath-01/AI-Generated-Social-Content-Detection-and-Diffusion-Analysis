"""
Text feature extraction: statistical, linguistic, and embedding features.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Statistical / linguistic features
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer."""
    return text.split()


def _sentences(text: str) -> list[str]:
    """Naive sentence splitter on . ! ?"""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def compute_statistical_features(text: str) -> dict[str, float]:
    """
    Compute a suite of statistical text features for a single post.

    Args:
        text: Cleaned post text.

    Returns:
        Dictionary of feature names to float values.
    """
    tokens = _tokenize(text)
    sentences = _sentences(text)
    chars = len(text)
    n_tokens = len(tokens)
    n_unique = len(set(tokens))
    n_sentences = max(len(sentences), 1)

    # Token / char counts
    token_lengths = [len(t) for t in tokens] if tokens else [0]
    sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]

    # Type-token ratio
    ttr = n_unique / n_tokens if n_tokens > 0 else 0.0

    # Repetition rate (1 - ttr)
    repetition_rate = 1.0 - ttr

    # Punctuation distribution
    punct_chars = [c for c in text if c in string.punctuation]
    punct_ratio = len(punct_chars) / chars if chars > 0 else 0.0

    # Lexical diversity (MATTR approximation with window = min(n_tokens, 50))
    window = min(n_tokens, 50)
    if window > 0:
        mattr = len(set(tokens[:window])) / window
    else:
        mattr = 0.0

    # Token entropy
    if n_tokens > 0:
        freq = Counter(tokens)
        probs = np.array(list(freq.values()), dtype=float) / n_tokens
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
    else:
        entropy = 0.0

    # Burstiness: coefficient of variation of sentence lengths
    if len(sentence_lengths) > 1:
        mean_sl = np.mean(sentence_lengths)
        std_sl = np.std(sentence_lengths)
        burstiness = std_sl / (mean_sl + 1e-9)
    else:
        burstiness = 0.0

    # Capital ratio
    capital_ratio = sum(1 for c in text if c.isupper()) / chars if chars > 0 else 0.0

    # URL / mention / hashtag counts
    url_count = len(re.findall(r"<URL>", text))
    mention_count = len(re.findall(r"<MENTION>", text))
    hashtag_count = len(re.findall(r"<HASHTAG>", text))

    return {
        "char_count": float(chars),
        "token_count": float(n_tokens),
        "unique_token_count": float(n_unique),
        "sentence_count": float(n_sentences),
        "avg_token_length": float(np.mean(token_lengths)),
        "avg_sentence_length": float(np.mean(sentence_lengths)),
        "sentence_length_variance": float(np.var(sentence_lengths)),
        "type_token_ratio": ttr,
        "repetition_rate": repetition_rate,
        "punct_ratio": punct_ratio,
        "lexical_diversity_mattr": mattr,
        "token_entropy": entropy,
        "burstiness": burstiness,
        "capital_ratio": capital_ratio,
        "url_count": float(url_count),
        "mention_count": float(mention_count),
        "hashtag_count": float(hashtag_count),
    }


def extract_statistical_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Apply compute_statistical_features to every row of a DataFrame.

    Args:
        df: DataFrame with a text column.
        text_col: Name of the column containing cleaned text.

    Returns:
        DataFrame of features (same index as df).
    """
    logger.info(f"Extracting statistical text features from '{text_col}'…")
    feature_rows = df[text_col].apply(compute_statistical_features)
    feat_df = pd.DataFrame(list(feature_rows), index=df.index)
    logger.info(f"  → {feat_df.shape[1]} statistical features")
    return feat_df


# ---------------------------------------------------------------------------
# Transformer embedding features
# ---------------------------------------------------------------------------

class TransformerEmbedder:
    """
    Encode texts using a Hugging Face sentence-transformer encoder.

    Uses mean-pooling of last hidden states.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 256):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model identifier.
            max_length: Maximum token length for truncation.
        """
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                logger.info(f"Loading transformer: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                self._device = "cpu"  # CPU-only
                self._model.to(self._device)
                logger.info(f"  Loaded on {self._device} (CPU-only mode)")
            except Exception as e:
                logger.error(f"Failed to load transformer: {e}")
                raise

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts to embedding vectors.

        Args:
            texts: List of text strings.
            batch_size: Batch size for inference.

        Returns:
            Array of shape (n_texts, hidden_size).
        """
        import torch

        self._load()
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)

            # Mean-pool last hidden state with attention mask
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(pooled.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Encode all texts in a DataFrame column.

        Args:
            df: DataFrame with text column.
            text_col: Name of the column.
            batch_size: Batch size.

        Returns:
            DataFrame of embedding features (columns: emb_0, emb_1, …).
        """
        logger.info(f"Encoding {len(df)} texts with {self.model_name}…")
        texts = df[text_col].tolist()
        embeddings = self.encode(texts, batch_size=batch_size)
        cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, index=df.index, columns=cols)
        logger.info(f"  → embedding shape: {embeddings.shape}")
        return emb_df