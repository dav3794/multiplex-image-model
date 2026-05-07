# Cell Phenotyping Experiments

Minimal instructions for running the cell phenotyping experiments used in the paper.

## Setup

Clone the repository and create the environment:

```bash
cd celltyping

uv sync

uv pip install \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu124torch2.6-cp312-cp312-linux_x86_64.whl
```

---

## Configuration

Prepare dataset-specific config files based on the examples:

* `./gold_standard/gs_config.yaml`
* `./danenberg/dberg_config.yaml`
* `./cords/cords_config.yaml`

---

## Generate Marker Embeddings

Generate ESM-2 embeddings for marker sequences (required by VirTues).

> Note: this step requires **100GB+ RAM**.

```bash
python3 -m core.utils.get_virtues_marker_embeddings \
  --csv "/path/to/channels/dataset_name/markers.csv" \
  --fastas-dir "/path/to/dataset_name/dataset/fastas" \
  --embeddings-dir "/path/to/dataset_name/dataset/marker_embeddings"
```

---

## Dataset Subsampling

For CORDS and Danenberg, subsample the datasets to reproduce the paper experiments.

### Danenberg

```bash
python -m danenberg.subsample \
  --input_path "/path/to/danenberg/SingleCells.csv" \
  --output_path "/path/to/danenberg/danenberg_sc_meta_300k.csv" \
  --target_size 300000
```

### CORDS

```bash
python -m cords.subsample \
  --input_path "/path/to/cords/cords_full.csv" \
  --output_path "/path/to/cords/cords_300k.csv" \
  --target_size 300000
```

---

## Data Preparation

Run once per dataset:

```bash
python3 -m gold_standard.data
python3 -m danenberg.data
python3 -m cords.data
```

---

## Run Experiments

Run embedding inference for a selected model from `./core/models/registry.yaml`.

Example:

```bash
python3 -m core.embeddings.inference \
  --config "./path/to/dataset/config.yaml" \
  --model "model_name" \
  --scheme "patch" \
  --batch_size 32
```

Then run cross-validation:

```bash
python3 -m core.crossval \
  --config "./path/to/dataset/config.yaml" \
  --run {model_name}_{scheme}
```
