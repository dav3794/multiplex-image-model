import os

import numpy as np
import pandas as pd
import torch


def generate_embeddings_patch(
    dataset,
    model,
    prepare_fn,
    get_channels,
    compute_fn,
    annotations_path: str,
    out_dir: str,
    device,
    batch_size=128,
):
    annotations = pd.read_csv(annotations_path, index_col=0)
    crop_index = dataset.crop_index
    all_tids = dataset.tissue_index["tissue_id"].unique()

    for tid in all_tids:
        current_crop_index = crop_index[crop_index["tissue_id"] == tid]
        cell_ids = current_crop_index["crop_id"].values
        current_ann = annotations[annotations["tissue_id"] == tid].copy()

        valid_cell_ids = np.intersect1d(cell_ids, current_ann.index)
        print(
            f"{tid}: annotations has {len(current_ann)} cells, "
            f"inference processed {len(cell_ids)} cells, "
            f"intersection = {len(valid_cell_ids)} cells "
            f"({len(cell_ids) - len(valid_cell_ids)} cells dropped)"
        )

        if len(cell_ids) == 0:
            print(f"[WARNING] {tid}: No crops found, skipping")
            continue

        crops_list = []
        for idx in range(len(current_crop_index)):
            row = current_crop_index.iloc[idx]
            crop_id = row["crop_id"]

            # preprocess=False, because we handle it in universal prepare_fn
            crop_raw = dataset.get_crop(tid, crop_id, preprocess=False)

            if crop_raw.shape[-2:] != (32, 32):
                print(
                    f"[WARNING] {tid} crop_id={crop_id}: "
                    f"expected (32,32), got {tuple(crop_raw.shape)}"
                )
                raise ValueError

            crop_prepared, _ = prepare_fn(crop_raw, None, tid, dataset)

            crops_list.append(crop_prepared)

        channels = get_channels(tid, dataset)
        if hasattr(channels, "to"):
            channels = channels.to(device)
        marker_indices = channels
        marker_list = [marker_indices] * len(crops_list)

        all_tokens = []
        all_uncertainties = []
        for start in range(0, len(crops_list), batch_size):
            end = start + batch_size
            batch_crops = crops_list[start:end]
            batch_markers = marker_list[start:end]

            _, cell_tokens, uncertainty, _ = compute_fn(
                model,
                batch_crops,
                batch_markers,
                None,
                device,
                batch_size=batch_size,
            )
            all_tokens.append(cell_tokens.cpu())
            if uncertainty is not None:
                all_uncertainties.append(uncertainty.cpu())

        cell_tokens = torch.cat(all_tokens, dim=0)
        has_uncertainty = len(all_uncertainties) > 0
        if has_uncertainty:
            cell_uncertainties = torch.cat(all_uncertainties, dim=0)

        if cell_tokens.shape[0] != len(crops_list):
            print(
                "[WARNING] embeddings != crops",
                cell_tokens.shape[0],
                len(crops_list),
            )

        keep_mask = np.isin(cell_ids, valid_cell_ids)
        aligned_embeddings = cell_tokens[keep_mask]
        ordered_valid_cell_ids = cell_ids[keep_mask]
        aligned_labels = current_ann.loc[
            ordered_valid_cell_ids, "cell_category_regen"
        ].values

        if has_uncertainty:
            aligned_uncertainties = cell_uncertainties[keep_mask]

        if aligned_embeddings.shape[0] != len(aligned_labels):
            print(
                f"[WARNING] {tid}: Number of embeddings ({aligned_embeddings.shape[0]}) "
                f"does not match number of labels ({len(aligned_labels)})"
            )

        np.save(
            os.path.join(out_dir, f"{tid}_embeddings.npy"),
            aligned_embeddings.cpu().numpy(),
        )
        np.save(os.path.join(out_dir, f"{tid}_labels.npy"), aligned_labels)
        np.save(os.path.join(out_dir, f"{tid}_cell_ids.npy"), ordered_valid_cell_ids)
        if has_uncertainty:
            np.save(
                os.path.join(out_dir, f"{tid}_uncertainty.npy"),
                aligned_uncertainties.numpy(),
            )
