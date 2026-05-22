"""Adaptive labeling acquisition function.

Diversity sampling over item-embedding clusters.
Candidates from under-sampled clusters are scored higher,
so the K revealed labels span the test distribution broadly.

No network calls — encoder is loaded from the pre-fetched HF cache.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# Load from local HF cache — same repo declared in models.txt
_encoder_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
_ACQ_ENCODER  = SentenceTransformer(_encoder_path)

_HERE          = os.path.dirname(os.path.abspath(__file__))
_CENTROID_PATH = os.path.join(_HERE, "centroids.npy")
_CENTROIDS: np.ndarray | None = (
    np.load(_CENTROID_PATH) if os.path.exists(_CENTROID_PATH) else None
)

# Per-round state (resets each container start = each round)
_seen_clusters: Counter       = Counter()
_emb_cache:     dict[str, np.ndarray] = {}


def _to_text(inp: dict) -> str:
    """Item text with benchmark+condition prefix — matches train_ncf.py."""
    return f"[{inp['benchmark']} / {inp['condition']}] {inp['item_content']}"


def acquisition_function(input: dict) -> float:
    """Return labeling priority score — higher means more desired.

    Returns 0.0 (triggering random fallback behaviour) only if centroids.npy
    was not shipped. Never raises or returns NaN/inf.
    """
    if _CENTROIDS is None:
        return 0.0

    key = json.dumps(
        {k: input[k] for k in ("benchmark", "condition", "item_content")},
        sort_keys=True,
    )
    if key not in _emb_cache:
        _emb_cache[key] = _ACQ_ENCODER.encode(
            _to_text(input), normalize_embeddings=True
        )
    x = _emb_cache[key]

    c = int(np.argmin(np.linalg.norm(_CENTROIDS - x, axis=1)))
    _seen_clusters[c] += 1
    # First candidate in a cluster gets -1, second gets -2, etc.
    # Higher score = more desired, so first-in-cluster wins.
    return float(-_seen_clusters[c])