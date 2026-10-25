#!/usr/bin/env python3
"""Train a weighted VGAE and evaluate held-out contact presence.

Outputs: model.pt (state_dict), emb.npy (full-graph posterior means), and
metrics.json (held-out validation/test AUROC and AP).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn.models import VGAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

if __package__:
    from .model import Encoder, encode_graph, load_graph, seed_everything
else:
    from model import Encoder, encode_graph, load_graph, seed_everything


def split_graph(data):
    """Keep held-out labels separate from weighted message-passing edges."""
    splits = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        split_labels=True,
    )(data)
    for name, subset in zip(("training", "validation", "test"), splits):
        if subset.pos_edge_label_index.size(1) == 0:
            raise ValueError(f"No {name} positive contacts; use a larger graph")
        if name != "training" and (
            "neg_edge_label_index" not in subset
            or subset.neg_edge_label_index.size(1) == 0
        ):
            raise ValueError(f"No {name} negative contacts; graph is too dense")
    return splits


def sample_training_negatives(full_graph, num_samples):
    """Exclude every known contact, including validation and test positives."""
    neg = negative_sampling(
        full_graph.edge_index,
        num_nodes=full_graph.num_nodes,
        num_neg_samples=num_samples,
        method="sparse",
    )
    if neg.size(1) == 0:
        raise ValueError("No negative contacts available for training")
    return neg


@torch.no_grad()
def eval_linkpred(model, subset):
    """Score held-out labels using only this split's message-passing graph."""
    z = encode_graph(model, subset)
    pos = subset.pos_edge_label_index
    neg = subset.neg_edge_label_index
    # Logits preserve ranking without sigmoid saturation at large magnitudes.
    y_true = np.concatenate([np.ones(pos.size(1)), np.zeros(neg.size(1))])
    y_pred = (
        torch.cat(
            [
                model.decoder(z, pos, sigmoid=False),
                model.decoder(z, neg, sigmoid=False),
            ]
        )
        .cpu()
        .numpy()
    )
    return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True, help="Trusted local Data .pt file")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    if min(args.epochs, args.hidden, args.latent) < 1 or args.lr <= 0:
        ap.error("epochs, hidden, latent, and lr must be positive")
    if not 0 <= args.dropout < 1:
        ap.error("dropout must be in [0, 1)")

    seed_everything(args.seed)
    data = load_graph(args.graph)
    train_data, val_data, test_data = split_graph(data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = VGAE(Encoder(data.x.size(1), args.hidden, args.latent, args.dropout))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    best_val_ap = None
    best_state = None
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index, train_data.edge_weight)
        pos = train_data.pos_edge_label_index
        neg = sample_training_negatives(data, pos.size(1))
        loss = model.recon_loss(z, pos, neg) + model.kl_loss() / data.num_nodes
        if not torch.isfinite(loss):
            raise ValueError(
                "Non-finite training loss; check feature and weight scales"
            )
        loss.backward()
        optimizer.step()

        val_auc, val_ap = eval_linkpred(model, val_data)
        if val_auc > best_val_auc:
            best_val_auc, best_val_ap, best_epoch = val_auc, val_ap, epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{epoch:03d}/{args.epochs}] loss={loss.item():.4f} | val AUC={val_auc:.4f} AP={val_ap:.4f}"
            )

    model.load_state_dict(best_state)
    test_auc, test_ap = eval_linkpred(model, test_data)
    # Export full-graph embeddings only after held-out evaluation is complete.
    z_final = encode_graph(model, data)
    torch.save(model.state_dict(), outdir / "model.pt")
    np.save(outdir / "emb.npy", z_final.cpu().numpy())
    metrics = {
        "val_auc": float(best_val_auc),
        "val_ap": float(best_val_ap),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "latent": args.latent,
        "dropout": args.dropout,
        "lr": args.lr,
        "seed": args.seed,
        "weighted_encoder": True,
        "reconstruction": "binary_contact_presence",
        "embedding_graph": "full_graph",
    }
    with (outdir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
    print(f"Saved model and full-graph embeddings -> {outdir}")
    print(f"Metrics: AUC(test)={test_auc:.4f}, AP(test)={test_ap:.4f}")


if __name__ == "__main__":
    main()
