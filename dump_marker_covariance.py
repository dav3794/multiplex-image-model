"""Inspect the learned marker covariance K_C of a Kronecker-marker GP model.

K_C is the C×C correlation across markers that distinguishes the marker-covariance
model from the plain GP model (which implicitly assumes K_C = I, markers independent).
It is image-independent: the Hyperkernel marker embeddings are an nn.Embedding lookup
(immuvis.py), projected and row-normalised, so

    K_C = normalize(embedding_projection(E)) @ normalize(...).T + marker_jitter·I

(see gp_covariance.py:549-555). This script loads the checkpoint, rebuilds K_C over the
full marker vocabulary, and reports whether it carries real off-diagonal structure or is
essentially identity — the direct test of "did the model learn anything interesting?".

Runs on CPU in seconds; no dataset needed. Run on szary where the checkpoint lives.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from multiplex_model.modules.immuvis import MultiplexAutoencoder
from multiplex_model.utils.configuration import DecoderConfig, EncoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump and visualise the learned marker covariance K_C.")
    parser.add_argument(
        "--checkpoint",
        default="/raid_encrypted/immucan/models/last_checkpoint-ImVs-34.pth",
        help="Checkpoint with model_state_dict AND gp_covariance_state_dict.",
    )
    parser.add_argument(
        "--model-config",
        default="/raid_encrypted/immucan/models/config.last_checkpoint-ImVs-34.yaml",
        help="Model config YAML (encoder/decoder + marker_jitter).",
    )
    parser.add_argument(
        "--tokenizer-config",
        default="/home/mzmyslowski/marcin_multiplex/configs/all_markers_tokenizer.yaml",
    )
    parser.add_argument(
        "--panel-config",
        default="/home/mzmyslowski/marcin_multiplex/configs/all_panels_config.yaml",
    )
    parser.add_argument(
        "--panel",
        default=None,
        help="Restrict K_C to one dataset's markers (e.g. 'hn'). Markers only ever share a "
        "K_C within their own panel during training, so the full vocabulary is not meaningful.",
    )
    parser.add_argument(
        "--out",
        default="/home/mzmyslowski/marcin_multiplex/logs/marker_covariance_ImVs-34.png",
    )
    parser.add_argument("--top-pairs", type=int, default=15, help="How many strongest marker pairs to print.")
    return parser.parse_args()


def build_marker_covariance(
    hyperkernel_weights: torch.Tensor,
    projection_weight: torch.Tensor,
    projection_bias: torch.Tensor,
) -> np.ndarray:
    """Reproduce the row-normalised marker embeddings that feed K_C (gp_covariance.py:549-555)."""
    e = hyperkernel_weights @ projection_weight.T + projection_bias  # [C, D]
    e = torch.nn.functional.normalize(e, p=2, dim=1)
    return e.numpy()


def participation_ratio(eig: np.ndarray) -> float:
    """(Σλ)²/Σλ² — an effective dimensionality; low when one component dominates."""
    eig = eig[eig > 0]
    return float(eig.sum() ** 2 / (eig**2).sum())


def residual_correlation(e: np.ndarray) -> np.ndarray:
    """Correlation of embeddings after removing the shared component.

    K_C is dominated by a mean 'everything co-varies' direction; the marker-specific
    structure lives in the residual. This is the biologically informative view.
    """
    r = e - e.mean(axis=0, keepdims=True)
    r = r / np.linalg.norm(r, axis=1, keepdims=True)
    corr: np.ndarray = r @ r.T
    return corr


def signed_pairs(k: np.ndarray, names: list[str], n: int) -> tuple[list, list]:
    c = k.shape[0]
    pairs = [(names[i], names[j], float(k[i, j])) for i in range(c) for j in range(i + 1, c)]
    pairs.sort(key=lambda p: p[2])
    return pairs[-n:][::-1], pairs[:n]


def main() -> None:
    args = parse_args()
    yaml = YAML(typ="safe")

    with open(args.tokenizer_config, "r") as f:
        tokenizer = yaml.load(f)
    inv_tokenizer = {v: k for k, v in tokenizer.items()}
    model_num_channels = len(tokenizer)  # nn.Embedding row count — must match checkpoint

    if args.panel:
        with open(args.panel_config, "r") as f:
            panel_markers = yaml.load(f)["markers"][args.panel]
        names = [m for m in panel_markers if m in tokenizer]
        channel_ids = [tokenizer[m] for m in names]
        print(f"Panel '{args.panel}': {len(names)}/{len(panel_markers)} markers in tokenizer")
    else:
        channel_ids = sorted(tokenizer.values())
        names = [inv_tokenizer[i] for i in channel_ids]
    num_channels = len(names)

    with open(args.model_config, "r") as f:
        model_config = yaml.load(f)
    marker_jitter = model_config.get("marker_jitter", 1e-2)

    model = MultiplexAutoencoder(
        num_channels=model_num_channels,
        encoder_config=EncoderConfig(**model_config["encoder"]).model_dump(),
        decoder_config=DecoderConfig(**model_config["decoder"]).model_dump(),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "gp_covariance_state_dict" not in checkpoint:
        raise KeyError(
            f"{args.checkpoint} has no 'gp_covariance_state_dict'. The embedding_projection "
            "weights are not in this checkpoint, so K_C cannot be reconstructed — the analysis "
            "would use random projection weights and be meaningless."
        )
    gp_state = checkpoint["gp_covariance_state_dict"]
    projection_weight = gp_state["embedding_projection.weight"]
    projection_bias = gp_state["embedding_projection.bias"]

    with torch.no_grad():
        ids = torch.tensor(channel_ids, dtype=torch.long)
        embeddings = model.encoder.hyperkernel.hyperkernel_weights(ids)  # [C, model_dim]
        e = build_marker_covariance(embeddings, projection_weight, projection_bias)

    k = e @ e.T + marker_jitter * np.eye(num_channels)
    k_resid = residual_correlation(e)
    eig = np.linalg.eigvalsh(k)[::-1]

    # K_C is dominated by a shared 'everything co-varies' component; the marker-specific
    # structure is the residual. Report both so the shared component is not mistaken for collapse.
    print(f"Markers (C):                     {num_channels}")
    print(f"Leading eigenvector of K_C:      {eig[0] / eig.sum():.1%} of total  (shared component)")
    print(f"||mean of embeddings||:          {np.linalg.norm(e.mean(0)):.3f}  (1.0 = all markers identical)")
    print(f"Residual effective dimensions:   {participation_ratio(np.linalg.eigvalsh(k_resid)):.1f} / {num_channels}")
    top_pos, top_neg = signed_pairs(k_resid, names, args.top_pairs)
    print(f"\nTop {args.top_pairs} co-grouped marker pairs (residual, shared component removed):")
    for a, b, v in top_pos:
        print(f"  {v:+.3f}  {a} — {b}")
    print(f"\nTop {args.top_pairs} anti-grouped marker pairs (residual):")
    for a, b, v in top_neg:
        print(f"  {v:+.3f}  {a} — {b}")

    order = leaves_list(linkage(squareform(np.clip(1.0 - k_resid, 0.0, 2.0), checks=False), method="average"))
    off = ~np.eye(num_channels, dtype=bool)
    panel_tag = f" — {args.panel}" if args.panel else ""

    fig, axes = plt.subplots(1, 2, figsize=(21, 9))
    for ax, mat, title in ((axes[0], k, "K_C (full)"), (axes[1], k_resid, "residual (shared component removed)")):
        mat = mat[np.ix_(order, order)]
        labels = [names[i] for i in order]
        vmax = float(np.abs(mat[off]).max())
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(num_channels), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(num_channels), labels, fontsize=6)
        ax.set_title(f"{title}{panel_tag}\n(markers clustered by residual)", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    npz_out = args.out.rsplit(".", 1)[0] + ".npz"
    np.savez(npz_out, k_c=k, k_residual=k_resid, marker_names=np.array(names), channel_ids=np.array(channel_ids))
    print(f"\nSaved figure: {args.out}")
    print(f"Saved matrix: {npz_out}")


if __name__ == "__main__":
    main()
