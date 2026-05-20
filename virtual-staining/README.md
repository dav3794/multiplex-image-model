# Virtual staining evaluation

Reproduce paper metrics for five models on IMC virtual-staining task.

## Quick start (paper reproduction)

```bash
python3 evaluate_all.py --config configs/paper.yaml --table
python3 evaluate_all.py plot --results results/per_point_results.csv
```

## Input data format

### `immuvis_npz` — used by ImmuVis models

Each file is named `recn-{image_id}.npz` and contains:

| Key | Shape | Content |
|---|---|---|
| `recon` | `(C, H, W)` | Model predictions, one channel per marker |
| `target` | `(C, H, W)` | Ground truth, same channel ordering |
| `marker_names` | `(C,)` | Array of marker name strings |
| `metadata` | scalar (JSON string) | `{"image_path": "...", "dataset_name": "hn"}` |

Image ID is extracted from `metadata["image_path"]` by taking the filename stem (last `/`-separated component, minus extension).

Predictions/ground truth are indexed by `[channel, :, :]` — the `marker_names` array is searched for the target marker name to find the channel index.

### `virtues_npy` — used by Virtues and Eva models

Directory of single-channel `.npy` files named `{image_id}_{marker}_recon.npy`, each shape `(H, W)`.

Marker names in filenames use underscores and short forms; they are normalised on load:

| Filename marker | Canonical name |
|---|---|
| `Carbonic` | `Carbonic Anhydrase` |
| `PARP` | `cl.PARP` |
| `H3` or `Histone` | `Histone H3` |
| `Carbonic_Anhydrase` | `Carbonic Anhydrase` (underscores → spaces) |

No ground truth is stored in these files. Ground truth is always sourced from a separate `immuvis_npz` model (configured via `ground_truth.source_model`).

## Output

All CSVs in `output_dir/` (default `results/`):

- `per_point_results.csv` — per (model, image, marker) MSE + Pearson
- `per_marker_pearson.csv` — per-marker aggregated Pearson (sufficient statistics)
- `summary.csv` — per-(model, marker) mean ± std
- `boxplot.pdf` — grouped box plot via `plot` subcommand
