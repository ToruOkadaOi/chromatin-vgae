# Graph learning for genome architecture

Train a variational graph autoencoder on weighted chromatin contacts and bigWig
node features. Contact strengths affect message passing in every GCN layer; the
inner-product decoder predicts **contact presence**, not contact strength.

## Setup

```bash
conda env create -f env.yml
conda activate chromatin_gnn_aman
```

## Workflow

```bash
# Build control and treatment graphs using matching chromosome, resolution,
# genome assembly, and feature order. Output directories are created automatically.
python scripts/build_graph.py --mcool control.mcool --chrom chr21 --res 10000 \
  --bigwigs control_CTCF.bw control_H3K27me3.bw --out data/chr21_control.pt
python scripts/build_graph.py --mcool treatment.mcool --chrom chr21 --res 10000 \
  --bigwigs treatment_CTCF.bw treatment_H3K27me3.bw --out data/chr21_treatment.pt

# Train and evaluate on held-out contacts.
python scripts/train_vgae.py --graph data/chr21_control.pt --epochs 100 --outdir results

# Encode treatment with the SAME checkpoint to retain a shared latent basis.
python scripts/encode_graph.py --model results/model.pt \
  --graph data/chr21_treatment.pt --out results/emb_treatment.npy

# Compare matching genomic bins.
python scripts/compare_embeddings.py --emb1 results/emb.npy --emb2 results/emb_treatment.npy \
  --label1 Control --label2 Treatment --prefix results/chr21
```

Graph `.pt` files contain pickled PyTorch Geometric objects. Load only trusted
files; graph loading explicitly uses `weights_only=False`. Model checkpoints
contain tensor state dictionaries and load with `weights_only=True`.

## Weights, evaluation, and outputs

- Edge weights are `log1p(balanced contact)` when cooler balancing weights exist.
  Otherwise the builder warns and uses `log1p(raw count)`. These scales differ;
  use the same balancing convention across conditions.
- Nonfinite/nonpositive contacts and input self-loops are removed. GCN layers add
  their own self-loops. Missing bigWig signal becomes zero.
- Shared preprocessing symmetrizes edges and averages duplicate/reverse weights,
  preserving weights when an input graph is already undirected. Weights stay
  aligned with edges through splitting, training, and inference.
- Ten percent of unique undirected contacts are held out for validation and ten
  percent for test. Validation encodes training contacts; test encodes training
  plus validation contacts. Each scores its own held-out positive/negative labels.
  Negatives exclude all known contacts. Graphs need enough edges and nonedges to
  populate both held-out sets.
- The best validation-AUC checkpoint is saved to `model.pt`. `metrics.json` records
  validation/test AUC and AP, the seed, model dimensions, and the binary objective.
  Python, NumPy, and PyTorch RNGs are seeded; exact reproducibility across different
  platforms or library versions is not guaranteed.
- `emb.npy` contains posterior means encoded from the **full graph after held-out
  evaluation**, suitable for comparison with another full graph. Encoding the same
  input graph with `encode_graph.py` reproduces these embeddings. Full-graph export
  embeddings must not be reused to compute held-out link-prediction metrics.
- Comparison outputs are `<prefix>_delta.csv` and `<prefix>_delta.png`; use
  `--no-plot` to skip the plot. A basename prefix such as `--prefix chr21` works.
  Arrays must be finite matrices of matching shape with nonzero row norms.
  Equal shape alone cannot verify matching loci or feature order: maintain the same
  bin coordinates and feature semantics across conditions. Independently trained
  checkpoints do not provide aligned latent coordinates.

Older checkpoints were trained with unweighted message passing and incorrect
held-out evaluation. Retrain them to use this weighted workflow.

## Tests

```bash
python -m pytest -q
```

The suite includes synthetic mcool/bigWig input, weighted preprocessing and split
checks, held-out evaluation checks, and reproducible CLI training/inference.
