import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.data import BaseDataset
from core.utils import load_config


class CordsDataset(BaseDataset):
    def __init__(self):
        config = load_config("./cords/cords_config.yaml")
        super().__init__(config)

        self.img_dir = Path(config["raw"]["img_dir"])
        self.mask_dir = Path(config["raw"]["mask_dir"])
        self.meta_path = Path(config["raw"]["meta_path"])
        self.channels_csv_path = Path(config["raw"]["channels_csv_path"])
        self.path_celltypes_map = Path(config["processed_dir"]) / "celltypes_map.json"

        self.mapping_csv_path = Path(config["mapping_csv_path"])

        self.raw_to_preprocessed_map = self._load_image_mapping()

        self.image_mask_map = self._build_image_mask_map()

    def _load_image_mapping(self) -> dict:
        """
        Parses the mapping CSV and builds a lookup table pointing from
        tissue_id to the new preprocessed image path.
        """
        mapping = {}
        if not self.mapping_csv_path.exists():
            raise FileNotFoundError(
                f"Mapping CSV not found at {self.mapping_csv_path}. "
                f"Cannot build dataset without processed images."
            )

        df = pd.read_csv(self.mapping_csv_path)

        if "dataset" in df.columns:
            df = df[df["dataset"] == "cords"]

        for _, row in df.iterrows():
            raw_path = Path(row["tiff_path"])
            new_path = Path(row["new_path"])

            filename = raw_path.name
            ac_match = re.search(r"_ac(\d+)(?:-cleaned_markers)?\.ome\.tiff$", filename)

            if ac_match:
                batch = raw_path.parent.name
                ac_num = int(ac_match.group(1))
                tissue_id = f"{batch}_ac{ac_num:03d}"
                mapping[tissue_id] = new_path

        print(f"Loaded {len(mapping)} preprocessed image mappings from CSV.")
        return mapping

    def _build_image_mask_map(self):
        """
        Dynamically finds pairs of .ome.tiff images and their corresponding masks
        based on the separate mask directory structure and the TMA prefix matching.
        Substitutes raw images with preprocessed images; excludes images without a mapping.
        """
        image_mask_map = {}
        if not self.img_dir.exists() or not self.mask_dir.exists():
            print("[WARNING] Image or mask directory does not exist.")
            return image_mask_map

        ome_batches = sorted([d for d in self.img_dir.iterdir() if d.is_dir()])
        anomalies = 0
        missing_mappings = 0

        for batch_dir in ome_batches:
            batch = batch_dir.name
            images = sorted(batch_dir.glob("*.ome.tiff"))

            if not images:
                continue

            batch_prefix = batch + "_"

            m = re.search(r"TMA_(\d+)_([A-C])(?:_\d+)?$", batch)
            if not m:
                continue

            tma_num, section = m.groups()
            mask_key = f"{tma_num}_{section}"
            mask_subdir = self.mask_dir / f"{mask_key}_mask" / "mask"

            if not mask_subdir.is_dir():
                anomalies += len(images)
                continue

            masks = list(mask_subdir.glob("*.tiff"))

            mask_by_ac = defaultdict(list)
            for mask_path in masks:
                mask_file = mask_path.name
                if not mask_file.startswith(batch_prefix):
                    continue
                ac_match = re.search(r"_s0_a(\d+)_ac_", mask_file)
                if ac_match:
                    ac_num = int(ac_match.group(1))
                    mask_by_ac[ac_num].append(mask_path)

            # Match each image
            for img_path in images:
                img_file = img_path.name
                ac_match = re.search(
                    r"_ac(\d+)(?:-cleaned_markers)?\.ome\.tiff$", img_file
                )
                if not ac_match:
                    anomalies += 1
                    continue

                ac_num_str = ac_match.group(1)
                ac_num = int(ac_num_str)

                matching = mask_by_ac.get(ac_num, [])

                if len(matching) == 1:
                    tissue_id = f"{batch}_ac{ac_num:03d}"

                    # If not mapped to a image, skip it entirely
                    if tissue_id not in self.raw_to_preprocessed_map:
                        missing_mappings += 1
                        continue

                    final_img_path = self.raw_to_preprocessed_map[tissue_id]

                    image_mask_map[tissue_id] = {
                        "image": final_img_path,
                        "mask": matching[0],
                    }
                else:
                    anomalies += 1

        print(f"Discovered {len(image_mask_map):,} image-mask pairs.")
        if anomalies > 0:
            print(
                f"[WARNING] Encountered {anomalies} anomalies (unmatched mask/image)."
            )
        if missing_mappings > 0:
            print(
                f"[WARNING] Excluded {missing_mappings} images because they lacked a preprocessed image mapping."
            )

        return image_mask_map

    def _create_celltypes_map(self, df: pd.DataFrame, save_path: Path):
        all_types = df["celltypes"].dropna().unique()
        type_to_id = {pheno: idx for idx, pheno in enumerate(sorted(all_types))}
        id_to_type = {idx: pheno for pheno, idx in type_to_id.items()}
        mapping_data = {
            "type_to_id": type_to_id,
            "id_to_type": id_to_type,
            "num_classes": len(type_to_id),
            "note": "NaN types are not included in mapping",
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        print(f"Saved cell type mapping to: {save_path}")
        print(f"Found {len(type_to_id)} unique cell types: {list(all_types)}")

    def load_metadata(self) -> pd.DataFrame:
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Could not find metadata CSV: {self.meta_path}")

        df = pd.read_csv(self.meta_path)
        print(f"Loaded raw metadata with {len(df):,} rows")

        df["BatchID"] = pd.to_numeric(df["BatchID"], errors="coerce").astype("Int64")
        df["TmaID"] = pd.to_numeric(df["TmaID"], errors="coerce").astype("Int64")
        df["acID"] = pd.to_numeric(df["acID"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["BatchID", "TmaID", "acID"])

        df["batch_name"] = df.apply(
            lambda row: f"{row['BatchID']}_LC_{row['Panel']}_TMA_{row['TmaID']}_{row['TmaBlock']}",
            axis=1,
        )
        df["ac_str"] = df["acID"].apply(lambda x: f"{x:03d}")
        df["tissue_id"] = df["batch_name"] + "_ac" + df["ac_str"]

        df["Pos_X"] = df["Center_X"]
        df["Pos_Y"] = df["Center_Y"]

        # df["celltypes"] = df["cell_type"]
        df["celltypes"] = df["cell_category"]

        unmapped = df["celltypes"].isna()
        if unmapped.any():
            print(
                f"[WARNING] {unmapped.sum():,} cells were missing or unmapped and will have NaN celltypes."
            )

        skipped_batches = {"20201222_LC_NSCLC_TMA_178_B", "20210109_LC_NSCLC_TMA_176_A"}
        df = df[~df["batch_name"].isin(skipped_batches)]

        valid_tissues = set(self.image_mask_map.keys())
        df = df[df["tissue_id"].isin(valid_tissues)]
        print(
            f"Metadata rows after filtering for valid matching images & masks: {len(df):,}"
        )

        self._create_celltypes_map(df, self.path_celltypes_map)

        return df

    def load_channels_meta(self) -> pd.DataFrame:
        if not self.channels_csv_path.exists():
            raise FileNotFoundError(
                f"Could not find channels CSV at {self.channels_csv_path}"
            )
        return pd.read_csv(self.channels_csv_path)

    def get_raw_image_path(self, tissue_id: str) -> Path:
        return self.image_mask_map[tissue_id]["image"]

    def get_raw_mask_path(self, tissue_id: str) -> Path:
        return self.image_mask_map[tissue_id]["mask"]


if __name__ == "__main__":
    CordsDataset().run()
