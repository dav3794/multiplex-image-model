"""Does the learned K_C predict which markers have correlated leave-one-out errors?

K_C models the cross-marker covariance of pixel residuals. Direct test: from the saved
LOO reconstructions, compute the empirical cross-marker correlation of residual maps
(recon - target), averaged over images, and correlate it against K_C — both full and
with the dominant shared component removed. A permutation test over marker labels gives
a null for the residual (structure-specific) comparison.
"""

import glob

import numpy as np

KC_NPZ = "/home/mzmyslowski/marcin_multiplex/logs/marker_covariance_ImVs-34_hn.npz"
RECON_DIR = "/raid_encrypted/immucan/recons/immuvis-gp/immuvis_last_checkpoint-ImVs-34_loo"


def residualize(m: np.ndarray) -> np.ndarray:
    """Remove the leading (shared) eigen-component of a symmetric matrix."""
    w, v = np.linalg.eigh(m)
    return m - w[-1] * np.outer(v[:, -1], v[:, -1])


def empirical_error_correlation(files: list[str], ref_names: list[str]) -> np.ndarray:
    """Average per-image cross-marker correlation of residual maps (recon - target)."""
    c = len(ref_names)
    acc = np.zeros((c, c))
    cnt = np.zeros((c, c))
    for f in files:
        d = np.load(f, allow_pickle=True)
        names = list(d["marker_names"])
        idx = [names.index(m) for m in ref_names]
        resid = (d["recon"].astype(np.float64) - d["target"].astype(np.float64))[idx]
        resid = resid.reshape(c, -1)
        corr = np.corrcoef(resid)  # nan where a residual map is constant
        good = np.isfinite(corr)
        acc[good] += corr[good]
        cnt[good] += 1
    return acc / np.maximum(cnt, 1)


def main() -> None:
    kc = np.load(KC_NPZ, allow_pickle=True)
    kc_names = list(kc["marker_names"])
    k_full = kc["k_c"]

    files = sorted(glob.glob(RECON_DIR + "/*.npz"))
    ref_names = [m for m in kc_names if m in list(np.load(files[0], allow_pickle=True)["marker_names"])]
    print(f"images: {len(files)} | markers aligned: {len(ref_names)}")

    err = empirical_error_correlation(files, ref_names)

    # Align K_C to the same marker order.
    ik = [kc_names.index(m) for m in ref_names]
    kc_a = k_full[np.ix_(ik, ik)]
    off = ~np.eye(len(ref_names), dtype=bool)

    r_full = np.corrcoef(kc_a[off], err[off])[0, 1]
    kc_r, err_r = residualize(kc_a), residualize(err)
    r_resid = np.corrcoef(kc_r[off], err_r[off])[0, 1]

    # Permutation null for the residual comparison: shuffle marker labels of err.
    rng_orders = [np.roll(np.arange(len(ref_names)), s) for s in range(1, len(ref_names))]
    perm = []
    for o in rng_orders:
        er = err_r[np.ix_(o, o)]
        perm.append(np.corrcoef(kc_r[off], er[off])[0, 1])
    perm = np.array(perm)
    p_val = (np.sum(np.abs(perm) >= abs(r_resid)) + 1) / (len(perm) + 1)

    print(f"\ncorr(K_C full,     error-corr full)     = {r_full:+.3f}")
    print(f"corr(K_C residual, error-corr residual) = {r_resid:+.3f}   (perm p = {p_val:.3f}, null |r| max {np.abs(perm).max():.3f})")

    # Interpretable pairs: strongest residual K_C pairs and whether errors track them.
    names = ref_names
    pairs = [(names[i], names[j], kc_r[i, j], err_r[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    pairs.sort(key=lambda p: p[2], reverse=True)
    print("\nTop 12 K_C-grouped pairs -> their empirical LOO error correlation:")
    print(f"  {'pair':<24} {'K_C_resid':>10} {'err_resid':>10}")
    for a, b, kv, ev in pairs[:12]:
        print(f"  {a+' - '+b:<24} {kv:>+10.3f} {ev:>+10.3f}")
    print("\nBottom 6 (K_C anti-grouped) -> error correlation:")
    for a, b, kv, ev in pairs[-6:]:
        print(f"  {a+' - '+b:<24} {kv:>+10.3f} {ev:>+10.3f}")


if __name__ == "__main__":
    main()
