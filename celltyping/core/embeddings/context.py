import os
import warnings
import numpy as np
import pandas as pd


def generate_embeddings_context(
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
    all_tids = dataset.tissue_index["tissue_id"].unique()

    for tid in all_tids:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = dataset.get_tissue(tid, preprocess=False)
            mask = dataset.get_segmentation_mask(tid)

            x, mask = prepare_fn(x, mask, tid, dataset)
            channels = get_channels(tid, dataset)

            cell_ids, cell_tokens, _, _ = compute_fn(
                model, x, channels, mask, device, batch_size
            )
            cell_ids = np.asarray(cell_ids)

        current_ann = annotations[annotations["tissue_id"] == tid].copy()
        valid_cell_ids = np.intersect1d(cell_ids, current_ann.index)
        print(
            f"{tid}: annotations has {len(current_ann)} cells, "
            f"inference processed {len(cell_ids)} cells, "
            f"intersection = {len(valid_cell_ids)} cells "
            f"({len(cell_ids) - len(valid_cell_ids)} cells dropped)"
        )  # "context" processes all cells from the mask, but not all of them have an annotation

        keep_mask = np.isin(cell_ids, valid_cell_ids)
        aligned_embeddings = cell_tokens[keep_mask]
        ordered_valid_cell_ids = cell_ids[keep_mask]
        aligned_labels = current_ann.loc[
            ordered_valid_cell_ids, "cell_category_regen"
        ].values

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
