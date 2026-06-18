# Clinical cross-validation

Reproduce paper metrics for ImmuVis and competitor models on clinical outcome prediction.

## Quick start (paper reproduction)

```bash
cd clinical/
python generate.py --config configs/generate/603.yaml
python evaluate.py --config configs/evaluate/603.yaml

python evaluate.py --config configs/evaluate/paper.yaml
```

## Configuration (evaluation stage)

Models, panels, preprocessing variants, and CV parameters are set in a YAML config:

```yaml
clinical_data:
  path: /raid_encrypted/.../train_meta_table.csv

models:
  - name: ImmuVis-603-beta-768-ma1
    format: immuvis           # "immuvis" or "virtues"
    path: /raid_encrypted/.../embeddings
    panels: [danenberg]
    img_path_col: img_path
    batch_agg: mean           # how to aggregate batch (H,W) dims
    patch_agg: mean           # how to aggregate patches per image
    variants: [scaled]        # subset of preprocessing variants to run

datasets:
  danenberg:
    features: [DeathBreast, ERBB2_pos, ERStatus, Grade, PAM50]
```

## Input data format

### `virtues` — used by Virtues, Eva, and other competitor models

Pre-split train/test embeddings stored per panel directory:

```
{embeddings_path}/{panel}/
  train_metadata.csv
  test_metadata.csv
  train_embeddings.npy      # shape (n_train, embedding_dim)
  test_embeddings.npy       # shape (n_test, embedding_dim)
```

Rows are concatenated, grouped by image filename (basename of `image_paths` column), and averaged to per-image embeddings.

### `immuvis` — used by ImmuVis model family

Batch-based patch embeddings stored as flat files:

```
{embeddings_path}/
  {model}_train_image_patches_embeddings_batch_{N}.npy    # shape (batch, dim, H, W)
  {model}_train_image_patches_metadata_batch_{N}.csv
  {model}_test_image_patches_embeddings_batch_{N}.npy
  {model}_test_image_patches_metadata_batch_{N}.csv
```

Each embedding batch is aggregated over spatial dimensions (H, W) via `batch_agg`, then patches belonging to the same image are combined via `patch_agg`. Image names are normalised to `.tiff` extension for matching against clinical metadata.

### Clinical metadata

Shared across all models:

```
train_meta_table.csv
```

Columns: `img_path`, `dataset`, `feature`, `feature_value`. Rows are indexed by `img_path` and filtered by dataset panel. Each `feature` column gets its own cross-validation run.

## Preprocessing variants

| Key | Pipeline |
|---|---|
| `standard` | LogisticRegression |
| `scaled` | StandardScaler → LogisticRegression |
| `pca50` | PCA(0.5) → LogisticRegression |
| `pca75` | PCA(0.75) → LogisticRegression |
| `pca99` | PCA(0.99) → LogisticRegression |
| `pca50w` | PCA(0.5, whiten) → LogisticRegression |
| `pca75w` | PCA(0.75, whiten) → LogisticRegression |
| `pca99w` | PCA(0.99, whiten) → LogisticRegression |

Rare classes (<5% frequency) are filtered out before each CV run.

## Output

All CSVs in `output_dir/` (default `clinical/results/`):

- `clinical_cv_results.csv` — disaggregated per (model, panel, feature, variant, fold) with accuracy, F1-macro, and ROC AUC
- `clinical_summary.csv` — aggregated mean metrics per (model, variant, panel, feature)

A pivot table (mean F1-macro) is also printed to stdout after all models finish.

## Stage 1: Embedding generation

```bash
cd clinical/
python generate.py --config configs/generate/<name>.yaml
```

Configs in `configs/generate/` use a `generation` list with per-entry `family` and `name` fields. Shared defaults (datasets, splits, paths, inference params) can be placed at the top level and are merged into each generation entry.

Available model families are registered in `clinical/models/registry.yaml`.
