"""Regression coverage for weighted graphs and held-out link prediction."""

import json
import os
import subprocess
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
import pyBigWig
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn.models import VGAE

from scripts.build_graph import bin_bigwig, build_graph
from scripts.compare_embeddings import compute_metrics
from scripts.model import (
    Encoder,
    encode_graph,
    load_graph,
    prepare_graph,
    seed_everything,
)
from scripts.train_vgae import eval_linkpred, sample_training_negatives, split_graph

REPO = Path(__file__).resolve().parents[1]


def edges(ei):
    return set(map(tuple, ei.t().tolist()))


def undirected_pairs(ei):
    return {tuple(sorted(pair)) for pair in edges(ei)}


@pytest.fixture
def graph():
    n = 30
    pairs = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 6))]
    return prepare_graph(
        Data(
            x=torch.rand(n, 2, generator=torch.Generator().manual_seed(7)) + 0.1,
            edge_index=torch.tensor(pairs, dtype=torch.long).t().contiguous(),
            edge_weight=torch.tensor([float((i + j) % 7 + 1) for i, j in pairs]),
        )
    )


def test_preprocessing_preserves_weights_and_is_idempotent():
    data = Data(
        x=torch.ones(4, 1),
        edge_index=torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 2, 3]]),
        edge_weight=torch.tensor([2.0, 2.0, 5.0, 9.0, 0.0]),
    )
    clean = prepare_graph(data)
    assert dict(
        zip(map(tuple, clean.edge_index.t().tolist()), clean.edge_weight.tolist())
    ) == {
        (0, 1): 2.0,
        (1, 0): 2.0,
        (1, 2): 5.0,
        (2, 1): 5.0,
    }
    again = prepare_graph(clean)
    assert torch.equal(clean.edge_index, again.edge_index)
    assert torch.equal(clean.edge_weight, again.edge_weight)
    assert clean.num_nodes == 4  # Preserve isolated bins.
    assert data.num_edges == 5  # Do not mutate the caller's graph.


@pytest.mark.parametrize(
    "weights",
    [None, torch.ones(1), torch.tensor([float("nan"), 1.0]), torch.tensor([-1.0, 1.0])],
)
def test_invalid_weights_are_rejected(weights):
    data = Data(
        x=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_weight=weights,
    )
    with pytest.raises(ValueError, match="weight"):
        prepare_graph(data)


def test_contact_strengths_affect_every_encoder_layer(graph):
    model = VGAE(Encoder(2, 4, 2, dropout=0))
    with torch.no_grad():
        for layer in (model.encoder.gc1, model.encoder.gc_mu, model.encoder.gc_log):
            layer.lin.weight.fill_(0.2)
            layer.bias.zero_()
    received = []
    hooks = [
        layer.register_forward_pre_hook(
            lambda module, args: received.append(args[2].clone())
        )
        for layer in (model.encoder.gc1, model.encoder.gc_mu, model.encoder.gc_log)
    ]
    before = encode_graph(model, graph)
    for hook in hooks:
        hook.remove()
    assert len(received) == 3
    assert all(torch.equal(w, graph.edge_weight) for w in received)
    changed = graph.clone()
    pair = ((changed.edge_index[0] == 0) & (changed.edge_index[1] == 1)) | (
        (changed.edge_index[0] == 1) & (changed.edge_index[1] == 0)
    )
    changed.edge_weight[pair] *= 20
    assert not torch.allclose(before, encode_graph(model, changed))


def test_splits_have_heldout_labels_true_negatives_and_aligned_weights(graph):
    seed_everything(42)
    train, val, test = split_graph(graph)
    full = edges(graph.edge_index)
    positives = [undirected_pairs(s.pos_edge_label_index) for s in (train, val, test)]
    assert set.union(*positives) == undirected_pairs(graph.edge_index)
    assert all(
        positives[i].isdisjoint(positives[j]) for i, j in ((0, 1), (0, 2), (1, 2))
    )
    assert undirected_pairs(train.edge_index) == positives[0]
    assert undirected_pairs(val.edge_index) == positives[0]
    assert undirected_pairs(test.edge_index) == positives[0] | positives[1]
    weight_lookup = dict(
        zip(map(tuple, graph.edge_index.t().tolist()), graph.edge_weight.tolist())
    )
    for subset in (train, val, test):
        assert subset.edge_weight.numel() == subset.num_edges
        for pair, w in zip(subset.edge_index.t().tolist(), subset.edge_weight.tolist()):
            assert w == weight_lookup[tuple(pair)]
    for subset in (val, test):
        assert edges(subset.pos_edge_label_index).isdisjoint(edges(subset.edge_index))
        assert edges(subset.neg_edge_label_index).isdisjoint(full)
    for _ in range(5):
        assert edges(sample_training_negatives(graph, 100)).isdisjoint(full)


def test_seed_controls_splits_and_dynamic_negatives(graph):
    def sample():
        seed_everything(42)
        train, val, test = split_graph(graph)
        return (
            train.edge_index,
            val.neg_edge_label_index,
            test.neg_edge_label_index,
            sample_training_negatives(graph, 100),
        )

    assert all(torch.equal(a, b) for a, b in zip(sample(), sample()))


def test_evaluation_uses_split_adjacency_weights_and_labels(graph):
    seed_everything(42)
    _, val, _ = split_graph(graph)
    model = VGAE(Encoder(2, 4, 2))
    encoded, decoded = [], []
    h1 = model.encoder.register_forward_pre_hook(
        lambda module, args: encoded.append(args)
    )
    h2 = model.decoder.register_forward_pre_hook(
        lambda module, args: decoded.append(args[1])
    )
    auc, ap = eval_linkpred(model, val)
    h1.remove()
    h2.remove()
    assert len(encoded) == 1
    assert torch.equal(encoded[0][1], val.edge_index)
    assert torch.equal(encoded[0][2], val.edge_weight)
    assert len(decoded) == 2
    assert torch.equal(decoded[0], val.pos_edge_label_index)
    assert torch.equal(decoded[1], val.neg_edge_label_index)
    assert 0 <= auc <= 1 and 0 <= ap <= 1


@pytest.mark.parametrize("dense", [False, True])
def test_unusable_splits_fail_clearly(dense):
    n = 6 if dense else 3
    pairs = (
        [(i, j) for i in range(n) for j in range(i + 1, n)]
        if dense
        else [(0, 1), (1, 2)]
    )
    data = prepare_graph(
        Data(
            x=torch.ones(n, 1),
            edge_index=torch.tensor(pairs).t(),
            edge_weight=torch.ones(len(pairs)),
        )
    )
    with pytest.raises(ValueError, match="No .* contacts"):
        split_graph(data)


@pytest.fixture
def genomic_inputs(tmp_path):
    bins = pd.DataFrame(
        {
            "chrom": ["chr1"] * 30,
            "start": np.arange(30) * 1000,
            "end": np.arange(1, 31) * 1000,
            "weight": np.ones(30),
        }
    )
    bins.loc[29, "weight"] = np.nan
    pairs = [
        (i, j, (i + j) % 7 + 1) for i in range(30) for j in range(i, min(30, i + 6))
    ]
    pixels = pd.DataFrame(pairs, columns=["bin1_id", "bin2_id", "count"])
    bw_path = tmp_path / "signal.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", 30000)])
    bw.addEntries(
        ["chr1"] * 30,
        bins.start.tolist(),
        ends=bins.end.tolist(),
        values=[float(i % 7 + 1) for i in range(30)],
    )
    bw.close()
    return bins, pixels, bw_path


@pytest.mark.parametrize("balanced", [True, False])
def test_build_graph_balanced_and_raw_count_fallback(
    tmp_path, genomic_inputs, balanced
):
    bins, pixels, bw = genomic_inputs
    if not balanced:
        bins = bins.drop(columns="weight")
    mcool = tmp_path / "contacts.mcool"
    cooler.create_cooler(f"{mcool}::resolutions/1000", bins, pixels)
    out = tmp_path / "new" / "nested" / "graph.pt"
    if balanced:
        data = build_graph(str(mcool), "chr1", 1000, [str(bw)], out)
    else:
        with pytest.warns(UserWarning, match="raw contact counts"):
            data = build_graph(str(mcool), "chr1", 1000, [str(bw)], out)
    loaded = load_graph(out)
    assert torch.equal(loaded.edge_index, data.edge_index)
    assert torch.equal(loaded.edge_weight, data.edge_weight)
    assert data.is_undirected() and not data.has_self_loops()
    assert data.contact_weight_source == ("balanced" if balanced else "raw_count")
    assert data.x.shape == (30, 1)
    assert data.bin_start.tolist() == bins.start.tolist()
    assert torch.isfinite(data.edge_weight).all()
    if balanced:
        assert 29 not in data.edge_index
    lookup = dict(
        zip(map(tuple, data.edge_index.t().tolist()), data.edge_weight.tolist())
    )
    assert lookup[(0, 1)] == pytest.approx(np.log1p(2))


def test_bigwig_uses_given_bin_boundaries(genomic_inputs):
    _, _, bw = genomic_inputs
    assert bin_bigwig(str(bw), "chr1", [(0, 1500)]).tolist() == pytest.approx([4 / 3])
    with pytest.raises(ValueError, match="not found"):
        bin_bigwig(str(bw), "missing", [(0, 1000)])


def run_cli(tmp_path, *args):
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(tmp_path / "mpl"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / args[0]), *map(str, args[1:])],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_cli_training_is_reproducible_and_inference_matches_export(tmp_path, graph):
    graph_path = tmp_path / "graph.pt"
    torch.save(graph, graph_path)
    for name in ("first", "second"):
        run_cli(
            tmp_path,
            "train_vgae.py",
            "--graph",
            graph_path,
            "--epochs",
            "2",
            "--hidden",
            "8",
            "--latent",
            "4",
            "--outdir",
            tmp_path / name,
        )
    a = np.load(tmp_path / "first/emb.npy")
    b = np.load(tmp_path / "second/emb.npy")
    np.testing.assert_array_equal(a, b)
    first_metrics = json.loads((tmp_path / "first/metrics.json").read_text())
    assert first_metrics == json.loads((tmp_path / "second/metrics.json").read_text())
    assert first_metrics["weighted_encoder"] is True
    assert first_metrics["embedding_graph"] == "full_graph"
    encoded = tmp_path / "new" / "emb.npy"
    run_cli(
        tmp_path,
        "encode_graph.py",
        "--model",
        tmp_path / "first/model.pt",
        "--graph",
        graph_path,
        "--out",
        encoded,
    )
    np.testing.assert_array_equal(a, np.load(encoded))
    run_cli(
        tmp_path,
        "compare_embeddings.py",
        "--emb1",
        tmp_path / "first/emb.npy",
        "--emb2",
        encoded,
        "--prefix",
        "comparison",
    )
    metrics = pd.read_csv(tmp_path / "comparison_delta.csv")
    np.testing.assert_allclose(metrics.cosine_distance, 0, atol=1e-6)
    np.testing.assert_allclose(metrics.euclidean, 0)
    assert (tmp_path / "comparison_delta.png").is_file()


@pytest.mark.parametrize(
    "other", [np.ones((1, 2)), np.ones(2), np.zeros((2, 2)), np.full((2, 2), np.nan)]
)
def test_comparison_rejects_invalid_arrays(other):
    with pytest.raises(ValueError):
        compute_metrics(np.ones((2, 2)), other)
