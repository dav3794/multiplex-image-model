"""Do markers with more K_C neighbours reconstruct better in LOO? (redundancy -> lower MSE)

K_C measures marker similarity. Hypothesis: a marker that is similar to others in the panel
is easy to impute from them when masked -> lower LOO error. We score each marker's redundancy
from K_C (mean and max similarity to the rest of the panel) and correlate it with per-marker
mean MSE from the LOO CSV. Pearson (scale-invariant recon quality) is reported alongside MSE
to guard against the marker-intensity confound.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

KC_NPZ = "/home/mzmyslowski/marcin_multiplex/logs/marker_covariance_ImVs-34_hn.npz"
CSV = "/raid_encrypted/immucan/results/with_reconstructs/immuvis_last_checkpoint-ImVs-34_loo.csv"


def main() -> None:
    kc = np.load(KC_NPZ, allow_pickle=True)
    names = list(kc["marker_names"])
    k = kc["k_c"].copy()
    np.fill_diagonal(k, np.nan)  # ignore self

    df = pd.read_csv(CSV)
    per_marker = df.groupby("marker").agg(mse=("mse", "mean"), pearson=("pearson", "mean"), n=("mse", "size"))

    rows = []
    for i, m in enumerate(names):
        if m not in per_marker.index:
            continue
        row = k[i]
        rows.append(
            {
                "marker": m,
                "kc_mean": np.nanmean(row),          # overall similarity to panel
                "kc_max": np.nanmax(row),            # best single "twin"
                "kc_nn05": int(np.nansum(row > 0.5)),  # count of strong neighbours
                "mse": per_marker.loc[m, "mse"],
                "pearson": per_marker.loc[m, "pearson"],
            }
        )
    t = pd.DataFrame(rows)
    print(f"markers matched: {len(t)}")

    print("\nCorrelation of K_C redundancy score vs per-marker LOO metric:")
    print(f"  {'score':<10} {'vs':<8} {'Spearman':>10} {'Pearson':>10}")
    for score in ["kc_mean", "kc_max", "kc_nn05"]:
        for target, sign in [("mse", "(want -)"), ("pearson", "(want +)")]:
            rho = spearmanr(t[score], t[target]).correlation
            r = pearsonr(t[score], t[target])[0]
            print(f"  {score:<10} {target:<8} {rho:>+10.3f} {r:>+10.3f}   {sign}")

    print("\nMost redundant markers (high kc_mean) — do they reconstruct better?")
    print(t.sort_values("kc_mean", ascending=False)[["marker", "kc_mean", "kc_max", "kc_nn05", "mse", "pearson"]].head(8).to_string(index=False))
    print("\nLeast redundant markers (low kc_mean):")
    print(t.sort_values("kc_mean")[["marker", "kc_mean", "kc_max", "kc_nn05", "mse", "pearson"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
