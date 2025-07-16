#!/usr/bin/env python3
import os, csv, glob, pathlib, shutil

SRC_ROOT = "/raid_encrypted/immucan/IMC/NSCLC2/IMC1/unzipped"
DST_ROOT = "/home/szlukasik/immu-vis/data/immuvis_all2"
DATASET_NAME = "nsclc2-panel1"

for split in ("train", "test"):
    for patient_dir in glob.glob(f"{SRC_ROOT}/{split}/*"):
        src_img_dir = os.path.join(patient_dir, "img")
        dst_img_dir = os.path.join(DST_ROOT, split, DATASET_NAME, "imgs")
        pathlib.Path(dst_img_dir).mkdir(parents=True, exist_ok=True)

        for src_file in glob.glob(f"{src_img_dir}/*.tiff"):
            dst_link = os.path.join(dst_img_dir, os.path.basename(src_file))
            try:
                os.symlink(src_file, dst_link)
            except FileExistsError:
                # already linked – skip or overwrite
                pass
