"""Offline training script — run once to produce ncf_head.pt and subject_cache.pkl.

Usage (from the torch_measure repo root):
    python -m competition.train_ncf
    python -m competition.train_ncf --encoder BAAI/bge-large-en-v1.5 --embed-dim 1024 --epochs 15
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from torch_measure.models import NCF


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="all-MiniLM-L6-v2",
                   help="SentenceTransformer model name. "
                        "all-MiniLM-L6-v2=384-dim, BAAI/bge-large-en-v1.5=1024-dim.")
    p.add_argument("--embed-dim",    type=int,   default=384)
    p.add_argument("--hidden-dim",   type=int,   default=256)
    p.add_argument("--n-layers",     type=int,   default=3)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embeddings-checkpoint", default="competition/ncf_embeddings.pt",
                   help="Cache for encoded embeddings — reused on re-runs, not shipped in ZIP.")
    p.add_argument("--output",               default="competition/ncf_head.pt")
    p.add_argument("--subject-cache-output", default="competition/subject_cache.pkl")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Render subject_content exactly as the hosted runtime does (from README).
# The training strings must match test-time strings or cache lookups will miss.
# ---------------------------------------------------------------------------
def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    for key, label in (
        ("provider",     "Organization"),
        ("params",       "Parameters"),
        ("release_date", "Released"),
        ("family",       "Family"),
    ):
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os.makedirs("competition", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download the full dataset snapshot (HF cache, skipped on re-run)
    # ------------------------------------------------------------------
    print("Downloading measurement-db snapshot …")
    snap = snapshot_download(
        repo_id="aims-foundations/measurement-db",
        repo_type="dataset",
    )

    subjects_df  = pd.read_parquet(os.path.join(snap, "subjects.parquet"))
    items_df     = pd.read_parquet(os.path.join(snap, "items.parquet"))
    benchmarks_df = pd.read_parquet(os.path.join(snap, "benchmarks.parquet"))

    # Build lookup dicts matching the README's to_training_example() pattern
    subjects_by_id  = {r["subject_id"]: r for r in subjects_df.to_dict("records")}
    items_by_id     = {r["item_id"]:    r for r in items_df.to_dict("records")}
    benchmarks_by_id = {r["benchmark_id"]: r for r in benchmarks_df.to_dict("records")}

    SKIP = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    trial_files = [
        f for f in glob.glob(os.path.join(snap, "*.parquet"))
        if not os.path.basename(f).endswith("_traces.parquet")
        and os.path.basename(f) not in SKIP
    ]
    print(f"Found {len(trial_files)} response tables.")

    # Load with test_condition — needed for deduplication and item text
    trials = pd.concat(
        [pd.read_parquet(f, columns=["subject_id", "item_id", "benchmark_id",
                                      "trial", "test_condition", "response"])
         for f in trial_files],
        ignore_index=True,
    ).dropna(subset=["response"])

    # ------------------------------------------------------------------
    # 2. Deduplicate exactly as the README specifies:
    #    keep only the smallest trial per (subject_id, item_id, test_condition)
    # ------------------------------------------------------------------
    trials = (trials
              .sort_values("trial")
              .groupby(["subject_id", "item_id", "test_condition"], as_index=False)
              .first())

    # Keep only binary labels — hidden scoring uses binary labels
    trials = trials[trials["response"].isin([0.0, 1.0])].copy()
    print(f"Total binary training rows after dedup: {len(trials):,}")

    # ------------------------------------------------------------------
    # 3. Build text fields that match what predict() receives at test time
    # ------------------------------------------------------------------
    def to_training_example(row: pd.Series) -> dict:
        item      = items_by_id.get(row["item_id"], {})
        subject   = subjects_by_id.get(row["subject_id"], {})
        benchmark = benchmarks_by_id.get(row["benchmark_id"], {})
        # Use public benchmark_id (not human-readable "name") to match runtime
        benchmark_id = benchmark.get("benchmark_id") or row["benchmark_id"]
        condition    = row["test_condition"] or "none"
        return {
            "benchmark":       benchmark_id,
            "condition":       condition,
            "subject_content": render_subject_content(subject, row["subject_id"]),
            # Include benchmark+condition in item text so the embedding sees task context
            "item_content":    f"[{benchmark_id} / {condition}] "
                               + str(item.get("content") or ""),
            "label":           row["response"],
        }

    examples = [to_training_example(row) for _, row in trials.iterrows()]
    subject_texts = [ex["subject_content"] for ex in examples]
    item_texts    = [ex["item_content"]    for ex in examples]
    labels        = torch.tensor([ex["label"] for ex in examples], dtype=torch.float32)

    # ------------------------------------------------------------------
    # 4. Build the NCF model and encode
    # ------------------------------------------------------------------
    encoder = SentenceTransformer(args.encoder)
    model = NCF(
        encoder=encoder,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        device=args.device,
    )

    emb_path = args.embeddings_checkpoint
    if os.path.exists(emb_path):
        print(f"Loading cached embeddings from {emb_path} …")
        U, V = model.load_embeddings(emb_path)
    else:
        print("Encoding subjects and items (this takes a while) …")
        U, V = model.encode_batch(subject_texts, item_texts)
        torch.save({"subject_embeddings": U, "item_embeddings": V}, emb_path)
        print(f"Saved embeddings → {emb_path}")

    # ------------------------------------------------------------------
    # 5. Build subject_content → embedding lookup for predict()'s cache.
    #    Deduplicate: one vector per unique subject_content string.
    # ------------------------------------------------------------------
    seen: dict[str, int] = {}
    for i, s in enumerate(subject_texts):
        if s not in seen:
            seen[s] = i
    subject_cache = {s: U[i].cpu().numpy() for s, i in seen.items()}

    with open(args.subject_cache_output, "wb") as f:
        pickle.dump(subject_cache, f)
    print(f"Saved subject cache ({len(subject_cache)} entries) → {args.subject_cache_output}")

    # ------------------------------------------------------------------
    # 6. Train the MLP head
    # ------------------------------------------------------------------
    X       = torch.cat([U, V], dim=-1).to(args.device)
    labels  = labels.to(args.device)
    dataset = TensorDataset(X, labels)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    model.net.train()
    print("Training …")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model.net(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        print(f"  Epoch {epoch + 1}/{args.epochs}  loss={total_loss / len(dataset):.4f}")

    torch.save(model.net.state_dict(), args.output)
    print(f"Saved NCF head → {args.output}")


if __name__ == "__main__":
    main()