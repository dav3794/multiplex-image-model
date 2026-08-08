"""Redundancy vs SCALE-INDEPENDENT reconstruction quality in LOO.

Separates 'easy because redundant' from 'easy because bright'. Instead of raw MSE we use
per-(image, marker) NMSE = MSE / Var(target) (= 1 - R^2), plus Pearson(recon, target) —
both invariant to the marker's dynamic range. Computed from the saved LOO NPZ reconstructions
(recon, target), aggregated per marker with the median (robust), then correlated with K_C
redundancy scores.
"""

import glob

import numpy as np
from scipy.stats import pearsonr, spearmanr

KC_NPZ = "/home/mzmyslowski/marcin_multiplex/logs/marker_covariance_ImVs-34_hn.npz"
RECON_DIR = "/raid_encrypted/immucan/recons/immuvis-gp/immuvis_last_checkpoint-ImVs-34_loo"
VAR_FLOOR = 1e-6  # skip (image, marker) where the target is essentially flat


def main() -> None:
    kc = np.load(KC_NPZ, allow_pickle=True)
    names = list(kc["marker_names"])
    k = kc["k_c"].copy()
    np.fill_diagonal(k, np.nan)

    files = sorted(glob.glob(RECON_DIR + "/*.npz"))
    ref = [m for m in names if m in list(np.load(files[0], allow_pickle=True)["marker_names"])]

    nmse: dict[str, list[float]] = {m: [] for m in ref}
    pear: dict[str, list[float]] = {m: [] for m in ref}
    for f in files:
        d = np.load(f, allow_pickle=True)
        fn = list(d["marker_names"])
        recon = d["recon"].astype(np.float64)
        target = d["target"].astype(np.float64)
        for m in ref:
            c = fn.index(m)
            t = target[c].ravel()
            r = recon[c].ravel()
            v = t.var()
            if v < VAR_FLOOR:
                continue
            nmse[m].append(((r - t) ** 2).mean() / v)
            if r.std() > 1e-8:
                pear[m].append(np.corrcoef(r, t)[0, 1])

    rows = []
    for i, m in enumerate(names):
        if m not in ref or not nmse[m]:
            continue
        row = k[i]
        rows.append(
            {
                "marker": m,
                "kc_max": np.nanmax(row),
                "kc_mean": np.nanmean(row),
                "kc_nn05": int(np.nansum(row > 0.5)),
                "nmse": float(np.median(nmse[m])),          # scale-free (1 - R^2)
                "r2": float(1 - np.median(nmse[m])),
                "pearson": float(np.median(pear[m])) if pear[m] else np.nan,
            }
        )

    markers = [r["marker"] for r in rows]
    arr = {kk: np.array([r[kk] for r in rows]) for kk in rows[0] if kk != "marker"}
    print(f"markers: {len(rows)}")
    print("\nK_C redundancy vs SCALE-INDEPENDENT quality:")
    print(f"  {'score':<9} {'vs':<9} {'Spearman':>9} {'Pearson':>9}   {'wanted':>7}")
    for score in ["kc_max", "kc_mean", "kc_nn05"]:
        for tgt, want in [("nmse", "-"), ("r2", "+"), ("pearson", "+")]:
            rho = spearmanr(arr[score], arr[tgt]).correlation
            rp = pearsonr(arr[score], arr[tgt])[0]
            print(f"  {score:<9} {tgt:<9} {rho:>+9.3f} {rp:>+9.3f}   {want:>7}")

    order = np.argsort(-arr["kc_max"])
    print("\nBy kc_max (best twin) — high twin should mean low NMSE / high R^2 if effect is real:")
    print(f"  {'marker':<16} {'kc_max':>7} {'nmse':>7} {'r2':>7} {'pearson':>8}")
    for j in list(order[:8]) + list(order[-8:]):
        print(f"  {markers[j]:<16} {arr['kc_max'][j]:>7.3f} {arr['nmse'][j]:>7.3f} {arr['r2'][j]:>7.3f} {arr['pearson'][j]:>8.3f}")


if __name__ == "__main__":
    main()
