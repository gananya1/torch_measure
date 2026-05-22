"""Fit k-means centroids for the diversity acquisition function.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

ENCODER_NAME = "all-MiniLM-L6-v2"   # must match model.py and labeling.py
N_CLUSTERS   = 64
OUTPUT       = "competition/centroids.npy"


def main() -> None:
    snap     = snapshot_download(repo_id="aims-foundations/measurement-db", repo_type="dataset")
    items_df = pd.read_parquet(os.path.join(snap, "items.parquet"))

    SKIP = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    trial_files = [
        f for f in glob.glob(os.path.join(snap, "*.parquet"))
        if not os.path.basename(f).endswith("_traces.parquet")
        and os.path.basename(f) not in SKIP
    ]
    rows = pd.concat(
        [pd.read_parquet(f, columns=["item_id", "benchmark_id", "test_condition"])
         for f in trial_files],
        ignore_index=True,
    ).drop_duplicates("item_id")
    rows = rows.merge(items_df[["item_id", "content"]], on="item_id", how="inner")

    # Build item texts the same way train_ncf.py does
    texts = [
        f"[{row['benchmark_id']} / {row['test_condition'] or 'none'}] {row['content'] or ''}"
        for _, row in rows.iterrows()
    ]

    encoder = SentenceTransformer(ENCODER_NAME)
    X = encoder.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(X)
    np.save(OUTPUT, km.cluster_centers_)
    print(f"Saved {N_CLUSTERS} centroids → {OUTPUT}")


if __name__ == "__main__":
    main()