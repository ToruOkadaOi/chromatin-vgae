#!/usr/bin/env python3
"""
Compares two latent embedding matrices,
computes similarity metrics (cosine, Euclidean, L1)

Usage:
  python scripts/compare_embeddings.py \
      --emb1 results/emb.npy \
      --emb2 results/emb_eedi.npy \
      --label1 CTRL --label2 EEDi \
      --prefix results/chr21
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cityblock, cosine, euclidean


def compute_metrics(emb1, emb2):
    """Compute cosine similarity, cosine distance, L2, and L1 per row."""
    if emb1.ndim != 2 or emb1.shape != emb2.shape or min(emb1.shape) == 0:
        raise ValueError("Embeddings must be nonempty matrices of matching shape")
    if not np.isfinite(emb1).all() or not np.isfinite(emb2).all():
        raise ValueError("Embeddings must be finite")
    if (np.linalg.norm(emb1, axis=1) == 0).any() or (
        np.linalg.norm(emb2, axis=1) == 0
    ).any():
        raise ValueError("Cosine similarity is undefined for zero embedding vectors")
    cos_sims, cos_dists, l2_dists, l1_dists = [], [], [], []
    for a, b in zip(emb1, emb2):
        cos_sim = 1 - cosine(a, b)
        cos_dist = 1 - cos_sim
        l2 = euclidean(a, b)
        l1 = cityblock(a, b)
        cos_sims.append(cos_sim)
        cos_dists.append(cos_dist)
        l2_dists.append(l2)
        l1_dists.append(l1)
    return (
        np.array(cos_sims),
        np.array(cos_dists),
        np.array(l2_dists),
        np.array(l1_dists),
    )


def main():
    p = argparse.ArgumentParser(description="Compare two embedding matrices")
    p.add_argument("--emb1", required=True, help="Path to first embedding .npy")
    p.add_argument("--emb2", required=True, help="Path to second embedding .npy")
    p.add_argument("--label1", default="A", help="Label for first embedding")
    p.add_argument("--label2", default="B", help="Label for second embedding")
    p.add_argument(
        "--prefix", default="results/compare", help="Prefix for output files"
    )
    p.add_argument("--no-plot", action="store_true", help="Skip generating the plot")
    args = p.parse_args()

    # Load
    emb1 = np.load(args.emb1)
    emb2 = np.load(args.emb2)
    if emb1.shape != emb2.shape:
        raise ValueError(f"Shape mismatch: {emb1.shape} vs {emb2.shape}")

    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)
    n_bins, n_dim = emb1.shape
    print(f"Loaded embeddings: {n_bins} bins × {n_dim} dims")

    # Compute metrics
    cos_sims, cos_dists, l2_dists, l1_dists = compute_metrics(emb1, emb2)

    df = pd.DataFrame(
        {
            "bin_id": np.arange(n_bins),
            "cosine_similarity": cos_sims,
            "cosine_distance": cos_dists,
            "euclidean": l2_dists,
            "manhattan": l1_dists,
        }
    )
    csv_path = f"{args.prefix}_delta.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics → {csv_path}")

    # Plot
    if not args.no_plot:
        plt.figure(figsize=(12, 4))
        plt.plot(df["bin_id"], df["cosine_distance"], lw=0.8, color="steelblue")
        plt.title(f"Δ-Embedding ({args.label1} vs {args.label2})")
        plt.xlabel("Bin index")
        plt.ylabel("Cosine distance (1 – similarity)")
        plt.tight_layout()
        fig_path = f"{args.prefix}_delta.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved plot → {fig_path}")


if __name__ == "__main__":
    main()
