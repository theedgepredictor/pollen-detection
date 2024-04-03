#######################################################################################
# Import annotation files from any kind and import them into our master reference sheet.
# Default: PascalVoc CSV
#######################################################################################

import argparse
import os

import pandas as pd

from src.consts import MEDIA_METADATA_COLUMNS
from src.services.annotations.pollen20 import run_pollen20_annotation_importer
from src.utils import get_dataframe


def runner(pollen20_import=True):
    '''
    ETL Pipeline for defining the pollen categories associated with each image (Some may be mixed but we can collect many reference slides that just have one type on them)
    :return:
    '''
    df = []

    if pollen20_import:
        pollen20_annotations_df = run_pollen20_annotation_importer()
        df.append(pollen20_annotations_df)

    if len(df) > 0:
        fs_df = get_dataframe('./data/database/pollen_media_annotations.csv')
        df = pd.concat(df, ignore_index=True)

        # Handle Upsert
        if fs_df.shape[0]!=0:
            df = pd.concat([fs_df, df], ignore_index=True).drop_duplicates(['id','source'],keep='last').reset_index().drop(columns='index')

        path = './data/database/'
        os.makedirs(path, exist_ok=True)
        df[MEDIA_METADATA_COLUMNS].to_csv(path+'pollen_media_annotations.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Pollen Image Metadata references')
    # If you have already run the raw collection pass True to skip the raw upload
    #parser.add_argument('--run_ku_import', type=bool, default=True, help='Run the KU metadata import')
    parser.add_argument('--run_pollen20_import', type=bool, default=True, help='Run the Pollen20 metadata import')
    args = parser.parse_args()
    runner(args.run_pollen20_import)
