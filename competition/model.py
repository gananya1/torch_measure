"""predict() is called once per hidden (model_id, item_id) pair.
All setup runs at module scope — once when the container starts.
No outbound network calls are made; the encoder is loaded from the
pre-fetched HF cache declared in models.txt.
"""
from __future__ import annotations

import math
import os
import pickle

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from ncf import NCF

# Module-level init — runs once when the container starts.
_HERE = os.path.dirname(os.path.abspath(__file__))

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # must match train_ncf.py and models.txt
EMBED_DIM    = 384                   # 384 for MiniLM, 1024 for bge-large
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Load encoder from the local HF cache (pre-fetched by the platform via models.txt).
# snapshot_download returns the local cache path without making a network call
# when the repo is already present.
_encoder_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
_encoder      = SentenceTransformer(_encoder_path)

_model = NCF(encoder=_encoder, embedding_dim=EMBED_DIM, device=DEVICE)
_model.load_head(os.path.join(_HERE, "ncf_head.pt"))
_model.net.eval()

with open(os.path.join(_HERE, "subject_cache.pkl"), "rb") as _f:
    _SUBJECT_CACHE: dict[str, np.ndarray] = pickle.load(_f)

# Platt scaling state — fit once per round from labeled examples.
# Each round is a fresh container, so these reset automatically between rounds.
_platt_a: float     = 1.0
_platt_b: float     = 0.0
_platt_fitted: bool = False


# Internal helpers
def _encode_item(inp: dict) -> torch.Tensor:
    """Encode item text with benchmark+condition prefix, matching training."""
    text = f"[{inp['benchmark']} / {inp['condition']}] {inp['item_content']}"
    return _encoder.encode(text, convert_to_tensor=True, device=DEVICE)


def _raw_logit(inp: dict) -> float:
    """Return the NCF's raw logit (before sigmoid) for one input dict."""
    s_key = inp["subject_content"]
    if s_key in _SUBJECT_CACHE:
        u = torch.tensor(_SUBJECT_CACHE[s_key], dtype=torch.float32, device=DEVICE)
    else:
        # Subject not seen at training time — encode on the fly.
        u = _encoder.encode(s_key, convert_to_tensor=True, device=DEVICE)

    v = _encode_item(inp)

    with torch.no_grad():
        x = torch.cat([u, v], dim=-1).unsqueeze(0)
        return float(_model.net(x).squeeze(-1).item())


def _fit_platt(labeled: list[dict]) -> None:
    """Fit a two-parameter Platt scaler on the labeled examples in-place."""
    from scipy.optimize import minimize
    from scipy.special import expit

    global _platt_a, _platt_b

    logits = np.array([_raw_logit(ex) for ex in labeled])
    ys     = np.array([float(ex["label"]) for ex in labeled])

    def neg_ll(params: np.ndarray) -> float:
        a, b = params
        p = expit(a * logits + b).clip(1e-7, 1 - 1e-7)
        return -float(np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p)))

    res = minimize(neg_ll, x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"maxiter": 500})
    _platt_a, _platt_b = float(res.x[0]), float(res.x[1])


# Required predict
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject passes item) as a float in [0, 1]."""
    global _platt_fitted

    # Fit Platt scaling once per round on the revealed labeled examples.
    # labeled is the same list on every call within a round, so we guard
    # with _platt_fitted to avoid refitting thousands of times.
    if labeled and not _platt_fitted:
        _fit_platt(labeled)
        _platt_fitted = True

    logit = _raw_logit(input)
    prob  = 1.0 / (1.0 + math.exp(-(_platt_a * logit + _platt_b)))
    return float(np.clip(prob, 1e-7, 1 - 1e-7))