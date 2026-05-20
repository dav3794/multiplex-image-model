"""Metric computation: MSE, Pearson correlation."""

import numpy as np


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(((x.flatten() - y.flatten()) ** 2).mean())


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    r = np.corrcoef(x.flatten(), y.flatten())[0, 1]
    return float(r)


def center_crop(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    cx, cy = img.shape[0] // 2, img.shape[1] // 2
    sx, sy = crop_size[0] // 2, crop_size[1] // 2
    return img[cx - sx : cx + sx, cy - sy : cy + sy]


def accumulate(stats: dict, marker: str, pred: np.ndarray, gt: np.ndarray):
    p = pred.flatten()
    g = gt.flatten()
    counts = stats.setdefault("count", {})
    counts[marker] = counts.get(marker, 0) + 1
    for key, val in [
        ("sum_pred", p.mean()),
        ("sum_gt", g.mean()),
        ("sum_pred_sq", (p**2).mean()),
        ("sum_gt_sq", (g**2).mean()),
        ("sum_pred_gt", (p * g).mean()),
    ]:
        stats.setdefault(key, {}).setdefault(marker, []).append(val)


def finalize_pearson(stats: dict) -> dict[str, float]:
    result = {}
    keys = ["sum_pred", "sum_gt", "sum_pred_sq", "sum_gt_sq", "sum_pred_gt"]
    for marker in stats.get("sum_pred", {}):
        mean_pred = np.mean(stats["sum_pred"][marker])
        mean_gt = np.mean(stats["sum_gt"][marker])
        mean_cross = np.mean(stats["sum_pred_gt"][marker])
        sd_pred = (np.mean(stats["sum_pred_sq"][marker]) - mean_pred**2) ** 0.5
        sd_gt = (np.mean(stats["sum_gt_sq"][marker]) - mean_gt**2) ** 0.5
        denom = sd_pred * sd_gt
        result[marker] = float((mean_cross - mean_gt * mean_pred) / denom) if denom > 0 else 0.0
    return result
