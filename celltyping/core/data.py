from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter


def get_mask_id(mask, row_px, col_px, window_size=3):
    r = int(np.round(row_px))
    c = int(np.round(col_px))

    half = window_size // 2
    r_start = max(0, r - half)
    r_end = min(mask.shape[0], r + half + 1)
    c_start = max(0, c - half)
    c_end = min(mask.shape[1], c + half + 1)

    values = mask[r_start:r_end, c_start:c_end].flatten()

    # filter out background
    values = values[values > 0]

    if len(values) == 0:
        return -1

    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def extract_padded_patch(img, mask, row_px, col_px, cell_id, patch_size=32):
    """
    img:  np.ndarray of shape (C, Y, X)  - multiplex image
    mask: np.ndarray of shape (Y, X)     - instance segmentation mask
    row_px, col_px: int                  - cell center in pixel coordinates
    cell_id: int                         - ID of the cell in the mask
    patch_size: int                      - spatial size of square patch

    returns:
        patch:      (C, patch_size, patch_size)
        cell_mask:  (patch_size, patch_size) binary mask for this cell
    """
    H, W = mask.shape
    half = patch_size // 2

    row_px = int(np.round(row_px))
    col_px = int(np.round(col_px))

    row_px = int(np.clip(row_px, 0, H - 1))
    col_px = int(np.clip(col_px, 0, W - 1))

    r1 = max(0, row_px - half)
    r2 = min(H, row_px + half)
    c1 = max(0, col_px - half)
    c2 = min(W, col_px + half)
    img_patch = img[:, r1:r2, c1:c2]
    mask_patch = mask[r1:r2, c1:c2]

    pad_top = half - (row_px - r1)
    pad_left = half - (col_px - c1)
    pad_bottom = patch_size - (pad_top + img_patch.shape[1])
    pad_right = patch_size - (pad_left + img_patch.shape[2])
    pad_bottom = max(pad_bottom, 0)
    pad_right = max(pad_right, 0)

    img_patch = np.pad(
        img_patch,
        pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    cell_mask = (mask_patch == cell_id).astype(np.uint8)

    cell_mask = np.pad(
        cell_mask,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    img_patch = img_patch * cell_mask

    return img_patch, cell_mask


class BaseDataset(ABC):
    """
    Dataset agnostic class for preparing data.
    """

    def __init__(self, config: dict):
        self.config = config
        self.out_dir: Path = Path(config["processed_dir"])
        self.patch_size = config.get("processed", {}).get("patch_size", 32)
        self.modality = config.get("processed", {}).get("modality", "imc")

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Must return DF with columns: tissue_id, Pos_X, Pos_Y, celltypes."""
        pass

    @abstractmethod
    def load_channels_meta(self) -> pd.DataFrame:
        """Must have columns 'Marker Name' and 'protein_id'."""
        pass

    @abstractmethod
    def get_raw_image_path(self, tissue_id: str) -> Path:
        pass

    @abstractmethod
    def get_raw_mask_path(self, tissue_id: str) -> Path:
        pass

    def get_split(self, tissue_id: str) -> str:
        return "test"

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.build_tissue_level_processed_dataset()
        self.build_cell_crops()
        self.print_summary()

    def build_tissue_level_processed_dataset(self) -> None:
        """
        Build images/, masks/, tissue_index, channels.csv, quantiles, means, stds,
        and sce_annotations.csv for a given dataset (required by Virtues model).
        """
        self.images_dir = self.out_dir / "images"
        self.masks_dir = self.out_dir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        channels_meta = self.load_channels_meta()
        marker_names = channels_meta["Marker Name"].astype(str).tolist()
        protein_id = channels_meta["protein_id"].astype(str).tolist()
        print(f"Loaded {len(marker_names)} channels from channels CSV.")

        df = self.load_metadata()
        if "tissue_id" not in df.columns:
            raise ValueError("load_metadata() must return column 'tissue_id'")

        quantiles_records = []
        tissues = []
        all_processed_images = []
        for tissue_id in df["tissue_id"].drop_duplicates():
            img_path = self.get_raw_image_path(tissue_id)
            mask_path = self.get_raw_mask_path(tissue_id)
            if not (img_path.exists() and mask_path.exists()):
                print(f"Skipping {tissue_id}: Missing image or mask.")
                continue

            with tifffile.TiffFile(img_path) as tf:
                arr = tf.asarray()
            if arr.ndim == 2:
                arr = arr[None, ...]
            arr = arr.astype(np.float32)
            np.save(self.images_dir / f"{tissue_id}.npy", arr)

            mask = tifffile.imread(mask_path).astype(np.int32)
            np.save(self.masks_dir / f"{tissue_id}.npy", mask)

            # quantiles
            q = np.quantile(arr, 0.99, axis=(1, 2))
            arr_clip = np.clip(arr, 0, q[:, None, None])
            arr_log = np.log1p(arr_clip)
            arr_blur = np.stack(
                [gaussian_filter(arr_log[i], sigma=1.0) for i in range(arr.shape[0])],
                axis=0,
            )

            quantiles_records.append((tissue_id, q))
            all_processed_images.append(arr_blur)
            tissues.append((tissue_id, self.get_split(tissue_id)))

        if not quantiles_records:
            print("No valid images found – nothing to save.")
            return

        # mean/std
        num_channels = all_processed_images[0].shape[0]
        total_sum = np.zeros(num_channels, dtype=np.float64)
        total_sum_sq = np.zeros(num_channels, dtype=np.float64)
        total_count = 0
        for arr_blur in all_processed_images:
            C, H, W = arr_blur.shape
            channel_sums = arr_blur.sum(axis=(1, 2))
            channel_sums_sq = (arr_blur**2).sum(axis=(1, 2))
            pixel_count = H * W
            total_sum += channel_sums
            total_sum_sq += channel_sums_sq
            total_count += pixel_count
        global_mean = total_sum / total_count
        global_var = (total_sum_sq / total_count) - (global_mean**2)
        global_std = np.sqrt(np.maximum(global_var, 0)) + 1e-9

        means_records = [(tid, global_mean.copy()) for tid, _ in quantiles_records]
        stds_records = [(tid, global_std.copy()) for tid, _ in quantiles_records]

        # tissue_index.csv
        tissue_index_df = pd.DataFrame(tissues, columns=["tissue_id", "split"])
        tissue_index_df.to_csv(self.out_dir / "tissue_index.csv", index=False)
        print(f"\nSaved tissue_index.csv with {len(tissue_index_df)} entries.")

        # channels.csv
        if len(marker_names) != num_channels:
            raise ValueError(
                f"MARKER_NAMES length ({len(marker_names)}) does not match image channels ({num_channels})."
            )
        channels_df = pd.DataFrame(
            {
                "name": marker_names,
                "protein_id": protein_id,
                "description": [""] * len(marker_names),
            }
        )
        channels_df.to_csv(self.out_dir / "channels.csv", index=False)
        print("Saved channels.csv.")

        # quantiles.csv, means.csv, stds.csv
        columns = marker_names
        quant_df = pd.DataFrame({tid: vals for tid, vals in quantiles_records}).T
        quant_df.columns = columns
        quant_df.to_csv(self.out_dir / "quantiles.csv")

        mean_df = pd.DataFrame({tid: vals for tid, vals in means_records}).T
        mean_df.columns = columns
        mean_df.to_csv(self.out_dir / "means.csv")

        std_df = pd.DataFrame({tid: vals for tid, vals in stds_records}).T
        std_df.columns = columns
        std_df.to_csv(self.out_dir / "stds.csv")
        print("Saved quantiles.csv, means.csv, stds.csv.")

        # sce_annotations.csv
        processed_tids = {t[0] for t in tissues}
        meta = df[df["tissue_id"].isin(processed_tids)].copy()
        if len(meta) == 0:
            print("No cells found for sce_annotations.csv.")
        else:
            meta["mask_id"] = -1

            for tissue_id, group in meta.groupby("tissue_id", sort=False):
                mask_path = self.masks_dir / f"{tissue_id}.npy"
                if not mask_path.exists():
                    continue
                mask = np.load(mask_path)

                for idx, row in group.iterrows():
                    row_px = row["Pos_Y"]
                    col_px = row["Pos_X"]

                    mask_value = get_mask_id(mask, row_px, col_px)
                    meta.at[idx, "mask_id"] = mask_value

                    if mask_value <= 0:
                        print(
                            f"Skipping cell in {tissue_id} with mask value {mask_value}"
                        )
                        continue

            meta = meta[meta["mask_id"] > 0].copy()
            n_before = len(meta)
            meta = meta.drop_duplicates(subset=["tissue_id", "mask_id"], keep="first")
            n_dropped = n_before - len(meta)
            if n_dropped > 0:
                print(f"Dropped {n_dropped} duplicate mask assignments.")

            print(f"Valid cells with mask_id: {len(meta):,}")

            sce_df = pd.DataFrame(
                {
                    "tissue_id": meta["tissue_id"],
                    "cell_id": meta["mask_id"],  # MUST match crop_id
                    "Pos_X": meta["Pos_X"],
                    "Pos_Y": meta["Pos_Y"],
                    "cell_category_regen": meta["celltypes"],
                    "cell_type_regen": meta["celltypes"],
                }
            )
            sce_df = sce_df.set_index("cell_id")
            sce_df.index.name = "cell_id"

            sce_df.to_csv(self.out_dir / "sce_annotations.csv", index=True)
            print(
                f"Saved sce_annotations.csv with {len(sce_df):,} rows (index = mask integer)."
            )

    def build_cell_crops(self) -> None:
        self.crops_dir = self.out_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.out_dir / "sce_annotations.csv")

        crop_records = []
        print("\nExtracting cell-centered crops ...")
        for tissue_id, group in df.groupby("tissue_id", sort=False):
            img_path = self.images_dir / f"{tissue_id}.npy"
            mask_path = self.masks_dir / f"{tissue_id}.npy"
            if not (img_path.exists() and mask_path.exists()):
                print(f" Skipping {tissue_id} - missing .npy files")
                continue

            img = np.load(img_path)  # (C, Y, X) float32
            mask = np.load(mask_path)  # (Y, X) int32
            for _, row in group.iterrows():
                row_px = float(row["Pos_Y"])
                col_px = float(row["Pos_X"])
                cell_id = int(row["cell_id"])

                patch, _ = extract_padded_patch(
                    img=img,
                    mask=mask,
                    row_px=row_px,
                    col_px=col_px,
                    cell_id=cell_id,
                    patch_size=self.patch_size,
                )

                crop_fname = f"{tissue_id}_{cell_id}.npy"
                np.save(self.crops_dir / crop_fname, patch)
                crop_records.append(
                    {
                        "tissue_id": tissue_id,
                        "crop_id": cell_id,
                        "modality": self.modality,
                        "row": int(round(row_px)),
                        "col": int(round(col_px)),
                    }
                )

        crop_df = pd.DataFrame(crop_records)
        crop_path = self.out_dir / "crop_index.csv"
        crop_df.to_csv(crop_path, index=False)
        print(f"\nSaved {len(crop_df):,} cell crops to {crop_path}")

    def print_summary(self) -> None:
        print("\n========== DATASET SUMMARY ==========")

        images_dir = self.out_dir / "images"
        masks_dir = self.out_dir / "masks"
        crops_dir = self.out_dir / "crops"

        tissue_index_path = self.out_dir / "tissue_index.csv"
        crop_index_path = self.out_dir / "crop_index.csv"
        sce_path = self.out_dir / "sce_annotations.csv"
        channels_path = self.out_dir / "channels.csv"
        quant_path = self.out_dir / "quantiles.csv"
        means_path = self.out_dir / "means.csv"
        stds_path = self.out_dir / "stds.csv"

        required_files = [
            tissue_index_path,
            crop_index_path,
            sce_path,
            channels_path,
            quant_path,
            means_path,
            stds_path,
        ]

        for f in required_files:
            if not f.exists():
                print(f"[ERROR] Missing file: {f}")
                return

        image_files = sorted(images_dir.glob("*.npy"))
        mask_files = sorted(masks_dir.glob("*.npy"))
        crop_files = sorted(crops_dir.glob("*.npy"))

        print(f"Images: {len(image_files)}")
        print(f"Masks : {len(mask_files)}")
        print(f"Crops : {len(crop_files)}")

        if len(image_files) != len(mask_files):
            print("[ERROR] Number of images != number of masks")

        tissue_index = pd.read_csv(tissue_index_path)
        crop_index = pd.read_csv(crop_index_path)
        duplicates = crop_index[
            crop_index.duplicated(subset=["tissue_id", "crop_id"], keep=False)
        ]
        if len(duplicates) > 0:
            print(
                f"\n[DEBUG] Found {len(duplicates)} duplicate (tissue_id, crop_id) records in crop_index!"
            )
            print("[DEBUG] These are causing file overwrites. Here is a sample:")
            print(duplicates.head(10))
            print()

        sce = pd.read_csv(sce_path)
        channels = pd.read_csv(channels_path)
        quant = pd.read_csv(quant_path, index_col=0)
        means = pd.read_csv(means_path, index_col=0)
        stds = pd.read_csv(stds_path, index_col=0)

        print(f"Tissues in index: {len(tissue_index)}")
        print(f"Crops in index  : {len(crop_index)}")
        print(f"SCE rows        : {len(sce)}")
        print(f"Channels        : {len(channels)}")

        if len(crop_index) != len(sce):
            print(
                f"[WARNING] crop_index rows ({len(crop_index)}) != sce rows ({len(sce)})"
            )

        if len(crop_files) != len(crop_index):
            print(
                f"[WARNING] crop files ({len(crop_files)}) != crop_index rows ({len(crop_index)})"
            )

        tissues_images = {f.stem for f in image_files}
        tissues_masks = {f.stem for f in mask_files}
        tissues_index = set(tissue_index["tissue_id"].astype(str))
        tissues_crops = set(crop_index["tissue_id"].astype(str))
        tissues_sce = set(sce["tissue_id"].astype(str))

        if tissues_images != tissues_masks:
            print("[ERROR] Tissue IDs mismatch between images and masks")

        if not tissues_index.issubset(tissues_images):
            print("[WARNING] tissue_index has tissues not in images")

        if not tissues_crops.issubset(tissues_images):
            print("[WARNING] crop_index has tissues not in images")

        if not tissues_sce.issubset(tissues_images):
            print("[WARNING] sce has tissues not in images")

        num_channels = len(channels)
        if quant.shape[1] != num_channels:
            print("[ERROR] quantiles channel mismatch")

        if means.shape[1] != num_channels:
            print("[ERROR] means channel mismatch")

        if stds.shape[1] != num_channels:
            print("[ERROR] stds channel mismatch")

        if len(image_files) > 0:
            img = np.load(image_files[0])
            if img.ndim != 3:
                print("[ERROR] image should be (C,H,W)")
            else:
                if img.shape[0] != num_channels:
                    print(
                        f"[ERROR] image channels ({img.shape[0]}) != channels.csv ({num_channels})"
                    )

        if len(mask_files) > 0:
            mask = np.load(mask_files[0])
            if mask.ndim != 2:
                print("[ERROR] mask should be (H,W)")

        if len(crop_files) > 0:
            crop = np.load(crop_files[0])
            if crop.ndim != 3:
                print("[ERROR] crop should be (C,H,W)")
            else:
                if crop.shape[1] != self.patch_size:
                    print("[ERROR] crop height != patch_size")
                if crop.shape[2] != self.patch_size:
                    print("[ERROR] crop width != patch_size")

        print("========== SUMMARY DONE ==========\n")
