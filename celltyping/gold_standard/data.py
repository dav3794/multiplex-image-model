import json
from pathlib import Path

import pandas as pd
from rds2py import as_summarized_experiment, read_rds

from core.data import BaseDataset
from core.utils import load_config


class GoldStandardDataset(BaseDataset):
    def __init__(self):
        config = load_config("./gold_standard/gs_config.yaml")
        super().__init__(config)

        self.img_dir = Path(config["raw"]["img_dir"])
        self.mask_dir = Path(config["raw"]["mask_dir"])
        self.rds_path = Path(config["raw"]["rds_path"])
        self.channels_csv_path = Path(config["raw"]["channels_csv_path"])
        self.path_celltypes_map = Path(config["processed_dir"]) / "celltypes_map.json"

    def _create_celltypes_map(self, df, save_path):
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
        print(f"Found {len(type_to_id)} unique cell types: {all_types}")

    def load_metadata(self) -> pd.DataFrame:
        all_labeled_r_object = read_rds(str(self.rds_path))
        experiment = as_summarized_experiment(all_labeled_r_object)
        anndata = experiment.to_anndata()[0]
        df = anndata.obs.copy()
        df["tissue_id"] = df["image"].str.replace(".tiff", "", regex=False)
        print(f"Loaded metadata with {len(df):,} rows")

        self._create_celltypes_map(df, self.path_celltypes_map)
        return df

    def load_channels_meta(self) -> pd.DataFrame:
        if not self.channels_csv_path.exists():
            raise FileNotFoundError("Could not find channels CSV")
        return pd.read_csv(self.channels_csv_path)

    def get_raw_image_path(self, tissue_id: str) -> Path:
        return self.img_dir / f"{tissue_id}.tiff"

    def get_raw_mask_path(self, tissue_id: str) -> Path:
        return self.mask_dir / f"{tissue_id}.tiff"


if __name__ == "__main__":
    GoldStandardDataset().run()
