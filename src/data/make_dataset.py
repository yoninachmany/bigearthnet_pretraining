# -*- coding: utf-8 -*-
"""TODO"""
import os
import json
import logging
from pathlib import Path
import click
import pandas as pd

from tqdm import tqdm


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    make_bigearthnet_dataset(input_path, output_path)
    make_slo_lulc_dataset(input_path, output_path)


def make_bigearthnet_dataset(input_path, output_path):
    """TODO"""
    bigearthnet_path = input_path / "BigEarthNet-v1.0"
    patches = os.listdir(bigearthnet_path)
    json_file_names = [
        bigearthnet_path / patch / f"{patch}_labels_metadata.json"
        for patch in patches
    ]

    all_labels = []
    tiles = []
    for file_name in tqdm(json_file_names):
        with open(file_name) as json_file:
            data = json.load(json_file)
            all_labels.append(data["labels"])
            tiles.append(data["tile_source"])

    all_labels = ["|".join(labels) for labels in all_labels]
    tiles = ["_".join(tile.split("_")[3:6]) for tile in tiles]

    data_df = pd.DataFrame(
        data={"patch": patches, "labels": all_labels, "tile": tiles}
    )
    data_df["date"] = data_df.patch.str.split("_").str.get(2)
    data_df["patch_col"] = data_df.patch.str.split("_").str.get(3).astype(int)
    data_df["patch_row"] = data_df.patch.str.split("_").str.get(4).astype(int)

    data_df["val"] = 0
    is_val = (data_df["patch_col"] < 40) & (data_df["patch_row"] < 40)
    data_df.loc[is_val, "val"] = 1

    snow_df = pd.read_csv(
        input_path / "patches_with_seasonal_snow.csv",
        names=["patch"],
        header=None,
    )
    cloud_df = pd.read_csv(
        input_path / "patches_with_cloud_and_shadow.csv",
        names=["patch"],
        header=None,
    )

    is_snow = data_df.patch.isin(snow_df.patch.unique()).values
    data_df.loc[is_snow, "labels"] = (
        "Seasonal snow|" + data_df.loc[is_snow, "labels"]
    )

    data_df = data_df[~data_df.patch.isin(cloud_df.patch.unique())]

    data_df.to_csv(output_path / "bigearthnet_labels.csv", index=False)


def make_slo_lulc_dataset(input_path, output_path):
    """TODO"""
    slo_lulc_path = input_path / "single_frame_arrays"
    train_patches_csv = slo_lulc_path / "train_patches.csv"
    test_patches_csv = slo_lulc_path / "test_patches.csv"
    train_df = pd.read_csv(train_patches_csv, index_col=0)
    test_df = pd.read_csv(test_patches_csv, index_col=0)
    thresh_mask_frac = 0.99
    thresh_is_data_frac = 0.99
    thresh_valid_frac = 0.95
    train_df = train_df[train_df["mask_frac"] > thresh_mask_frac]
    train_df = train_df[train_df["is_data_frac"] > thresh_is_data_frac]
    train_df = train_df[train_df["valid_frac"] > thresh_valid_frac]
    test_df = test_df[test_df["mask_frac"] > thresh_mask_frac]
    test_df = test_df[test_df["is_data_frac"] > thresh_is_data_frac]
    test_df = test_df[test_df["valid_frac"] > thresh_valid_frac]
    train_df.to_csv(output_path / "slo_lulc_train_labels.csv", index=False)
    test_df.to_csv(output_path / "slo_lulc_test_labels.csv", index=False)


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    main()
