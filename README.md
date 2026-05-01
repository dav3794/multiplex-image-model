![ImmuVis logo](./imgs/immuvis_logo.png)

# ImmuVis

ImmuVis is a masked autoencoder for multiplex imaging data. It combines a marker-agnostic encoder, a hyperkernel-based multiplex encoder, and a channel-aware decoder to reconstruct masked multiplex images.

## Installation

Preferred setup:

```bash
uv sync
```

This installs the project dependencies from `pyproject.toml` and the lockfile.

Alternative editable install:

```bash
pip install -e .
```

After installation you can run commands with `uv run ...` or with the Python environment created by your editable install.

## Model

The model is a masked autoencoder for multiplex microscopy-style inputs where each sample contains several channels corresponding to biological markers.

The architecture has three main parts:

- A marker-agnostic encoder that processes each channel independently with a shared backbone.
- A hyperkernel module that transforms the channels into a shared pan-marker representation, with a consequtive pan-marker encoder.
- A marker-agnostic decoder that learns reconstructs each masked channel.

The default architecture is defined in `train_masked_config.yaml` and can be customized through the `encoder` and `decoder` sections.

### Encoder

The encoder is configured with:

- `ma_layers_blocks` and `ma_embedding_dims` for the marker-agnostic stage.
- `pm_layers_blocks` and `pm_embedding_dims` for the pan-marker stage.
- `hyperkernel`, which defines the channel-conditioned projection (`kernel_size`, `stride`, `padding`, `use_bias`).
- `encoder_type`, which can be a string registry name such as `convnext`, `resnet`, `swin`, or `vit`, or a dict with `type` and `module_parameters`.


### Decoder

The decoder uses:

- `decoded_embed_dim` for the hidden decoder width.
- `num_blocks` for the number of decoder blocks.
- `hyperkernel` for channel-aware decoding.
- `num_outputs`, which defaults to `2` and determines the number of output values per pixel and channel.
- `block_type`, which defaults to `convnext` and can be changed through the registry system.

### Output

The model predicts two values per output pixel and channel:

- `mi`, interpreted as the reconstructed mean after a sigmoid.
- `logvar`, interpreted as the predicted uncertainty and clamped during training for stability.

## Data preparation

The dataset loader expects a split-by-panel directory structure like the following:

```text
.
└── data
    ├── test
    │   ├── dataset1
    │   │   └── imgs
    │   ├── dataset2
    │   │   └── imgs
    │   └── dataset3
    │       └── imgs
    └── train
        ├── dataset1
        │   └── imgs
        ├── dataset2
        │   └── imgs
        └── dataset3
            └── imgs
```

`train` and `test` are the data splits, and each `dataset...` directory is a panel. Store the image files for each panel in the corresponding `imgs` directory.

The loader supports:

- `.tiff` files through `tifffile`
- `.npy` files through `numpy`

The file extension is controlled by `data_config.file_extension` in the training config.

Then update `configs/all_panels_config.yaml`:

- Set `paths.train` and `paths.test` to the full paths of your split directories.
- List the panel subdirectories you want to use under `datasets`.
- For each panel, provide the ordered marker names under `markers`; these must match the consecutive image channels in the corresponding files.
- Use `clip_limits` when you want panel-specific clipping during scaling.
- Use `marker_stats` when you want per-dataset normalization statistics loaded from CSV.

The dataset loader validates that:

- `paths` contains the requested split name.
- `datasets` exists and contains the panel subdirectories.
- `markers` contains the channel names for every listed panel.

The tokenizer config in `configs/all_markers_tokenizer.yaml` must include all marker names used by the panel config.

## Configuration

The main training config is `train_masked_config.yaml`. It is split into these groups:

- `encoder` and `decoder`: architecture configuration for `MultiplexAutoencoder`.
- `panel_config`: path to `configs/all_panels_config.yaml` or an inline panel config dict.
- `tokenizer_config`: path to `configs/all_markers_tokenizer.yaml` or an inline tokenizer config dict.
- `input_image_size`: final spatial size after preprocessing and transforms.
- `data_config`: preprocessing, denoising, scaling, normalization, file extension, and kwargs.
- `device`, `lr`, `final_lr`, `weight_decay`, `epochs`, `gradient_accumulation_steps`, and `frac_warmup_steps`: optimizer and schedule settings.
- `min_channels_frac`, `fully_masked_channels_max_frac`, `spatial_masking_ratio`, and `mask_patch_size`: masking strategy during training and validation.
- `checkpoints_dir`, `from_checkpoint`, and `save_checkpoint_freq`: checkpoint handling.
- `comet_project`, `comet_workspace`, `comet_api_key`, `tags`, and `run_name`: Comet.ml experiment metadata.

## Training

Training uses Comet.ml logging. Create a free account at https://www.comet.com and set the following fields in `train_masked_config.yaml`:

- `comet_project`
- `comet_workspace`
- `comet_api_key`

Run training with:

```bash
uv run python train_masked_model.py train_masked_config.yaml
```

or, if you used the editable install:

```bash
python3 train_masked_model.py train_masked_config.yaml
```

During training the script:

- builds the model from the encoder/decoder config,
- loads the panel and tokenizer config,
- applies channel masking and spatial masking,
- optimizes with AdamW and a warmup plus cosine schedule,
- logs metrics and validation reconstructions to Comet.ml,
- saves periodic checkpoints and the latest checkpoint.

The default checkpoint directory is `checkpoints`. The model writes:

- periodic checkpoints every `save_checkpoint_freq` epochs,
- `last_checkpoint-<run_name>.pth` after each epoch,
- `final_model-<run_name>.pth` after training finishes.

To resume training, set `from_checkpoint` to a checkpoint path or to `last`.

## Loading a Checkpoint

Use `MultiplexAutoencoder.load_from_checkpoint(...)` to restore a saved model. If the checkpoint includes `model_config`, the model can be rebuilt directly from the checkpoint contents.

## Notes

- The repo uses registry-based architecture resolution, so new encoder or block types can be added without changing the training script.
- If you change marker ordering in the panel config, update the tokenizer and any downstream analysis to match.

