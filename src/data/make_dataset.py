# -*- coding: utf-8 -*-
import os
import json
import logging
from pathlib import Path
import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    bigearthnet_path = input_path/'BigEarthNet-v1.0'
    names = os.listdir(bigearthnet_path)
    json_file_names = [bigearthnet_path/name/f'{name}_labels_metadata.json' for name in names]

    all_labels = []
    tiles = []
    for file_name in json_file_names:
        with open(file_name) as json_file:
            data = json.load(json_file)
            all_labels.append(data['labels'])
            tiles.append(data['tile_source'])

    all_labels_w_sep = ['|'.join(labels) for labels in all_labels]
    tiles_w_sep = ['_'.join(tile.split('_')[3:6]) for tile in tiles]

    data_df = pd.DataFrame(data={'name': names, 'labels': all_labels_w_sep, 'tile': tiles_w_sep})
    data_df['date'] = data_df.name.str.split('_').str.get(2)
    data_df['patch_col'] = data_df.name.str.split('_').str.get(3).astype(int)
    data_df['patch_row'] = data_df.name.str.split('_').str.get(4).astype(int)

    data_df['val'] = 0
    is_val = (data_df['patch_col'] < 40) & (data_df['patch_row'] < 40)
    data_df.loc[is_val, 'val'] = 1

    snow_df = pd.read_csv(input_path/'patches_with_seasonal_snow.csv', names=['name'], header=None)
    cloud_df = pd.read_csv(input_path/'patches_with_cloud_and_shadow.csv', names=['name'], header=None)

    is_snow = data_df.name.isin(snow_df.name.unique()).values
    data_df.loc[is_snow, 'labels'] = 'Seasonal snow|' + data_df.loc[is_snow, 'labels']

    is_cloud = data_df.name.isin(cloud_df.name.unique()).values
    data_df.loc[is_cloud, 'labels'] = 'Cloud and cloud shadow|' + data_df.loc[is_cloud, 'labels']
    data_df.loc[is_cloud, 'val'] = -1

    data_df = data_df.drop(['patch_col', 'patch_row'], axis=1)
    data_df.to_csv(output_path/'bigearthnet_dataset.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
