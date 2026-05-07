import numpy as np


def _get_uniform_crops(img, stride, crop_size=128):
    crops = []

    h, w = img.shape[-2:]
    indices_set = set()
    for i in range(0, h - crop_size + 1, stride):
        for j in range(0, w - crop_size + 1, stride):
            indices_set.add((i, j))

    last_i = h - crop_size
    for j in range(0, w - crop_size + 1, stride):
        indices_set.add((last_i, j))
    last_j = w - crop_size
    for i in range(0, h - crop_size + 1, stride):
        indices_set.add((i, last_j))
    indices_set.add((last_i, last_j))
    indices = sorted(list(indices_set))
    for i, j in indices:
        crop = img[:, i : i + crop_size, j : j + crop_size]
        crops.append(crop)
    # turn the indices back to the original coordinates
    for i in range(len(indices)):
        indices[i] = (indices[i][0], indices[i][1])
    return crops, indices


def _assign_patch_tokens_to_cells(crop_tokens, crop_mask, patch_size):
    cell_tokens = {}
    weights = {}
    for i in range(crop_tokens.shape[0]):
        for j in range(crop_tokens.shape[1]):
            patch_mask = crop_mask[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            unique, counts = np.unique(patch_mask, return_counts=True)
            patch_cell_coverage = dict(zip(unique, counts))
            for cell_id, overlap_pixels in patch_cell_coverage.items():
                if cell_id == 0:
                    continue
                if cell_id not in weights:
                    weights[cell_id] = []
                weights[cell_id].append(overlap_pixels)
                if cell_id not in cell_tokens:
                    cell_tokens[cell_id] = []
                cell_tokens[cell_id].append(crop_tokens[i, j, :])
    return cell_tokens, weights
