import json
from pathlib import Path

import pandas as pd

from core.data import BaseDataset
from core.utils import load_config

COARSE_MAP = {
    "CK^{med}ER^{lo}": "ER-Negative Epithelial Cell",
    "ER^{hi}CXCL12^{+}": "ER-Positive Epithelial Cell",
    "CD4^{+} T cells & APCs": "T Cell",  # not present in virtues map - assuming this mapping
    "CD4^{+} T cells": "T Cell",
    "Endothelial": "Stromal Cell",
    "Fibroblasts": "Stromal Cell",
    "Myofibroblasts PDPN^{+}": "Stromal Cell",
    "CD8^{+} T cells": "T Cell",
    "CK8-18^{hi}CXCL12^{hi}": "ER-Negative Epithelial Cell",
    "Myofibroblasts": "Stromal Cell",
    "CK^{lo}ER^{lo}": "ER-Negative Epithelial Cell",
    "Macrophages": "Myeloid",
    "CK^{+} CXCL12^{+}": "ER-Positive Epithelial Cell",
    "CK8-18^{hi}ER^{lo}": "ER-Negative Epithelial Cell",
    "CK8-18^{+} ER^{hi}": "ER-Positive Epithelial Cell",
    "CD15^{+}": "Myeloid",
    "MHC I & II^{hi}": "Antigen-Presenting Cell",
    "T_{Reg} & T_{Ex}": "T Cell",
    "CD57^{+}": "Natural Killer Cell",
    "Ep Ki67^{+}": "ER-Positive Epithelial Cell",
    "CK^{lo}ER^{med}": "ER-Positive Epithelial Cell",
    "Macrophages & granulocytes": "Myeloid",
    "CD38^{+} lymphocytes": "B Cell",
    "Ki67^{+}": "Myeloid",  # not present in virtues map - assuming this mapping
    "HER2^{+}": "ER-Positive Epithelial Cell",
    "B cells": "B Cell",
    "Basal": "ER-Negative Epithelial Cell",
    "Fibroblasts FSP1^{+}": "Stromal Cell",
    "Granulocytes": "Myeloid",
    "MHC I^{hi}CD57^{+}": "Natural Killer Cell",
    "Ep CD57^{+}": "ER-Negative Epithelial Cell",
    "MHC^{hi}CD15^{+}": "Myeloid",
}


class DanenbergDataset(BaseDataset):
    def __init__(self):
        config = load_config("./danenberg/dberg_config.yaml")
        super().__init__(config)

        self.img_dir = Path(config["raw"]["img_dir"])
        self.mask_dir = Path(config["raw"]["mask_dir"])
        self.meta_path = Path(config["raw"]["meta_path"])
        self.channels_csv_path = Path(config["raw"]["channels_csv_path"])
        self.path_celltypes_map = Path(config["processed_dir"]) / "celltypes_map.json"

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
        print(f"Found {len(type_to_id)} unique cell types: {all_types}")

    def load_metadata(self) -> pd.DataFrame:
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Could not find metadata CSV: {self.meta_path}")

        df = pd.read_csv(self.meta_path)

        def get_tissue_id(metabric_id, image_num) -> str:
            patient_num = str(metabric_id).replace("MB-", "")
            padded_patient = f"MB{patient_num.zfill(4)}"
            return f"{padded_patient}_{image_num}"

        df["tissue_id"] = df.apply(
            lambda row: get_tissue_id(row["metabric_id"], row["ImageNumber"]), axis=1
        )

        df["Pos_X"] = df["Location_Center_X"]
        df["Pos_Y"] = df["Location_Center_Y"]

        df["celltypes"] = df["cellPhenotype"].map(COARSE_MAP)

        unmapped = df["celltypes"].isna()
        if unmapped.any():
            print(
                f"[WARNING] {unmapped.sum():,} cells were missing or unmapped and will have NaN celltypes."
            )

        print(f"Loaded metadata with {len(df):,} rows")

        self._create_celltypes_map(df, self.path_celltypes_map)

        return df

    def load_channels_meta(self) -> pd.DataFrame:
        if not self.channels_csv_path.exists():
            raise FileNotFoundError(
                f"Could not find channels CSV at {self.channels_csv_path}"
            )
        return pd.read_csv(self.channels_csv_path)

    def get_raw_image_path(self, tissue_id: str) -> Path:
        return self.img_dir / f"{tissue_id}_FullStack.tiff"

    def get_raw_mask_path(self, tissue_id: str) -> Path:
        return self.mask_dir / f"{tissue_id}_CellMask.tiff"


if __name__ == "__main__":
    DanenbergDataset().run()
