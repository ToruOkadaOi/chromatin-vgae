#!/usr/bin/env python3
"""Encode a weighted graph using a trained VGAE checkpoint."""

import argparse

import numpy as np

if __package__:
    from .model import encode_graph, ensure_parent, load_graph, load_model
else:
    from model import encode_graph, ensure_parent, load_graph, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--graph", required=True, help="Trusted local Data .pt file")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = load_graph(args.graph)
    model = load_model(args.model, data.x.size(1))
    z = encode_graph(model, data)
    ensure_parent(args.out)
    np.save(args.out, z.cpu().numpy())
    print(f"Saved embeddings -> {args.out} shape={tuple(z.shape)}")


if __name__ == "__main__":
    main()
