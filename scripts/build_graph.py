#!/usr/bin/env python3
"""Build a weighted chromatin contact graph from mcool and bigWig files."""

import argparse
import warnings

import cooler
import numpy as np
import pyBigWig
import torch
from torch_geometric.data import Data

if __package__:
    from .model import ensure_parent, prepare_graph
else:
    from model import ensure_parent, prepare_graph


def bin_bigwig(bw_path, chrom, bins):
    """Average bigWig signal over actual cooler bin boundaries."""
    bw = pyBigWig.open(bw_path)
    try:
        if chrom not in bw.chroms():
            raise ValueError(f"{chrom} not found in {bw_path}")
        chrom_len = bw.chroms(chrom)
        vals = []
        for s, e in bins:
            s, e = max(0, int(s)), min(chrom_len, int(e))
            if s >= e:
                vals.append(0.0)
                continue
            v = bw.stats(chrom, s, e, type="mean", exact=True)[0]
            vals.append(0.0 if v is None or np.isnan(v) else v)
        return np.array(vals)
    finally:
        bw.close()


def build_graph(mcool_path, chrom, res, bigwigs, out_path, max_dist=5_000_000):
    """Save a loop-free undirected Data object with log1p contact weights."""
    if res <= 0 or max_dist < 0 or not bigwigs:
        raise ValueError(
            "Require positive resolution, nonnegative max_dist, and bigWigs"
        )
    print(f"Processing {chrom} at {res} bp resolution...")
    c = cooler.Cooler(f"{mcool_path}::resolutions/{res}")
    bins_df = c.bins().fetch(chrom)
    balanced = "weight" in bins_df.columns
    if not balanced:
        warnings.warn(
            "No balancing weights found; using raw contact counts", stacklevel=2
        )
    pixels = c.matrix(balance=balanced, as_pixels=True, join=True).fetch(chrom)
    pixels = pixels.loc[
        (pixels["chrom1"] == pixels["chrom2"])
        & ((pixels["start2"] - pixels["start1"]).abs() <= max_dist)
    ]
    start_to_bin = dict(zip(bins_df["start"], range(len(bins_df))))
    valid = pixels["start1"].isin(start_to_bin) & pixels["start2"].isin(start_to_bin)
    pixels = pixels.loc[valid]
    w = pixels["balanced" if balanced else "count"].to_numpy()
    # Masked balancing bins must not reappear as raw-count or zero-weight positives.
    valid_weights = np.isfinite(w) & (w > 0)
    pixels, w = pixels.loc[valid_weights], w[valid_weights]
    edge_index = torch.from_numpy(
        np.stack(
            [
                pixels["start1"].map(start_to_bin).to_numpy(dtype=np.int64),
                pixels["start2"].map(start_to_bin).to_numpy(dtype=np.int64),
            ]
        )
    )
    edge_weight = torch.tensor(np.log1p(w), dtype=torch.float)
    bins = list(zip(bins_df["start"], bins_df["end"]))
    node_feats = [bin_bigwig(bw, chrom, bins) for bw in bigwigs]
    data = prepare_graph(
        Data(
            x=torch.tensor(np.stack(node_feats, axis=1), dtype=torch.float),
            edge_index=edge_index,
            edge_weight=edge_weight,
            chrom=chrom,
            resolution=res,
            bin_start=torch.tensor(bins_df["start"].to_numpy(), dtype=torch.long),
            bin_end=torch.tensor(bins_df["end"].to_numpy(), dtype=torch.long),
            feature_names=[str(bw) for bw in bigwigs],
            contact_weight_source="balanced" if balanced else "raw_count",
        )
    )
    ensure_parent(out_path)
    torch.save(data, out_path)
    print(
        f"Saved {chrom}: {data.num_nodes} nodes, {data.num_edges} directed edges -> {out_path}"
    )
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcool", required=True, help="Path to .mcool file")
    p.add_argument("--chrom", required=True, help="Chromosome name (e.g., chr21)")
    p.add_argument("--res", type=int, default=10000, help="Resolution (bp)")
    p.add_argument(
        "--bigwigs", nargs="+", required=True, help="Ordered bigWig feature files"
    )
    p.add_argument("--out", required=True, help="Output .pt file path")
    p.add_argument(
        "--max_dist", type=int, default=5_000_000, help="Max genomic distance for edges"
    )
    args = p.parse_args()
    build_graph(args.mcool, args.chrom, args.res, args.bigwigs, args.out, args.max_dist)
