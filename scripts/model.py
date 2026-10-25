"""Shared weighted VGAE encoder, graph loading, and preprocessing."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import VGAE
from torch_geometric.utils import remove_self_loops, to_undirected


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_graph(data):
    """Validate weights and construct a loop-free, weighted undirected graph.

    Mean reduction preserves weights when input already contains both directions.
    Zero-weight contacts are absent edges, not positive reconstruction labels.
    """
    if not isinstance(data, Data):
        raise TypeError("Expected a PyTorch Geometric Data object")
    data = data.clone().cpu()
    if data.x is None or data.x.ndim != 2 or min(data.x.shape) == 0:
        raise ValueError("Node features must be a nonempty [nodes, features] matrix")
    data.x = data.x.float()
    if not torch.isfinite(data.x).all():
        raise ValueError("Node features must be finite")
    ei = data.edge_index
    if ei is None or ei.ndim != 2 or ei.size(0) != 2 or ei.dtype != torch.long:
        raise ValueError("edge_index must be a [2, edges] int64 tensor")
    if ei.numel() and (ei.min() < 0 or ei.max() >= data.x.size(0)):
        raise ValueError("Edge indices must refer to existing nodes")
    w = data.edge_weight
    if w is None or w.ndim != 1 or w.numel() != ei.size(1):
        raise ValueError(
            "A matching edge_weight vector is required for weighted graphs"
        )
    w = w.float()
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("Edge weights must be finite and nonnegative")
    ei, w = remove_self_loops(ei, w)
    positive = w > 0
    ei, w = ei[:, positive], w[positive]
    if ei.size(1) == 0:
        raise ValueError("Graph must contain positive-weight, non-self contacts")
    data.edge_index, data.edge_weight = to_undirected(
        ei, w, num_nodes=data.x.size(0), reduce="mean"
    )
    return data


def load_graph(path):
    """Load a trusted local Data pickle (PyTorch >=2.6 defaults to weights-only)."""
    return prepare_graph(torch.load(path, map_location="cpu", weights_only=False))


class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden, latent, dropout=0.2):
        super().__init__()
        self.gc1 = GCNConv(in_dim, hidden)
        self.gc_mu = GCNConv(hidden, latent)
        self.gc_log = GCNConv(hidden, latent)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.gc1(x, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return (
            self.gc_mu(h, edge_index, edge_weight),
            self.gc_log(h, edge_index, edge_weight),
        )


def load_model(path, in_dim):
    state = torch.load(path, map_location="cpu", weights_only=True)
    first = state["encoder.gc1.lin.weight"]
    if first.size(1) != in_dim:
        raise ValueError(
            f"Feature count mismatch: model expects {first.size(1)}, graph has {in_dim}"
        )
    hidden = first.size(0)
    latent = state["encoder.gc_mu.lin.weight"].size(0)
    model = VGAE(Encoder(in_dim, hidden, latent))
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def encode_graph(model, data):
    model.eval()
    return model.encode(data.x, data.edge_index, data.edge_weight)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
