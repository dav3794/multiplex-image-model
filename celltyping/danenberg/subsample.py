import pandas as pd
import numpy as np
import argparse


def subsample_csv(input_path: str, output_path: str, target_size: int = 300_000):
    df = pd.read_csv(input_path)

    df["cellPhenotype"] = df["cellPhenotype"].fillna("NaN")
    global_counts = df["cellPhenotype"].value_counts()
    target_prop = global_counts / len(df)

    phenotypes = global_counts.index
    image_key = ["metabric_id", "ImageNumber"]
    groups = df.groupby(image_key)

    image_keys = list(groups.groups.keys())
    image_data = {}
    for key in image_keys:
        group = groups.get_group(key)
        counts = group["cellPhenotype"].value_counts().reindex(phenotypes, fill_value=0)
        num_cells = len(group)
        image_data[key] = {"counts": counts, "num_cells": num_cells, "group": group}

    current_counts = pd.Series(0, index=phenotypes)
    current_total = 0
    selected_keys = []
    candidates = set(image_keys)
    while current_total < target_size and candidates:
        best_key = None
        best_mse = np.inf
        for key in candidates:
            img_counts = image_data[key]["counts"]
            img_cells = image_data[key]["num_cells"]
            tentative_counts = current_counts + img_counts
            tentative_total = current_total + img_cells
            if tentative_total == 0:
                continue
            tentative_prop = tentative_counts / tentative_total
            mse = ((tentative_prop - target_prop) ** 2).mean()
            if mse < best_mse:
                best_mse = mse
                best_key = key
        if best_key is None:
            break
        selected_keys.append(best_key)
        current_counts += image_data[best_key]["counts"]
        current_total += image_data[best_key]["num_cells"]
        candidates.remove(best_key)
    selected_df = pd.concat([image_data[key]["group"] for key in selected_keys])
    selected_df["cellPhenotype"] = selected_df["cellPhenotype"].replace("NaN", np.nan)
    selected_df.to_csv(output_path, index=False)

    n_cells = len(selected_df)
    n_images = selected_df[["metabric_id", "ImageNumber"]].drop_duplicates().shape[0]

    print("\n" + "=" * 60)
    print("SUBSAMPLING SUMMARY")
    print(f"  --> Total cells selected : {n_cells:,}")
    print(f"  --> Unique images/tissues: {n_images}")
    print(
        f"  → Average cells per image: {n_cells / n_images:.1f}"
        if n_images > 0
        else "  --> (no images)"
    )
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Subsample a CSV file.")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input CSV"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output CSV"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=300_000,
        required=True,
        help="Target number of rows",
    )

    args = parser.parse_args()

    subsample_csv(
        input_path=args.input_path,
        output_path=args.output_path,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()
