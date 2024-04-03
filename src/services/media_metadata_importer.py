import argparse
import os

import pandas as pd

from src.consts import MEDIA_METADATA_COLUMNS
from src.services.media_metadata.kupollen import run_ku_media_metadata_importer
from src.services.media_metadata.pollen20 import run_pollen20_media_metadata_importer
from src.utils import get_dataframe


#####################################################################
# Useful for ingesting reference slide metadata for a new data source
# Add new metadata method for the new data source and add
#   the metadata dataframe to the runner method
#####################################################################


### Current metadata classes
#### - KUPollen
#### - Pollen20



def runner(ku_import=True, pollen20_import=True):
    '''
    ETL Pipeline for defining the pollen categories associated with each image (Some may be mixed but we can collect many reference slides that just have one type on them)
    :return:
    '''
    df = []
    if ku_import:
        ku_media_metadatas_df = run_ku_media_metadata_importer()
        df.append(ku_media_metadatas_df)

    if pollen20_import:
        pollen20_media_metadatas_df = run_pollen20_media_metadata_importer()
        df.append(pollen20_media_metadatas_df)



    if len(df) > 0:
        fs_df = get_dataframe('./data/database/pollen_media_metadata.csv')
        df = pd.concat(df, ignore_index=True)

        # Handle Upsert
        if fs_df.shape[0]!=0:
            df = pd.concat([fs_df, df], ignore_index=True).drop_duplicates(['id','source'],keep='last').reset_index().drop(columns='index')

        path = './data/database/'
        os.makedirs(path, exist_ok=True)
        df[MEDIA_METADATA_COLUMNS].to_csv(path+'pollen_media_metadata.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Pollen Image Metadata references')
    # If you have already run the raw collection pass True to skip the raw upload
    parser.add_argument('--run_ku_import', type=bool, default=True, help='Run the KU metadata import')
    parser.add_argument('--run_pollen20_import', type=bool, default=True, help='Run the Pollen20 metadata import')
    args = parser.parse_args()
    runner(args.run_ku_import, args.run_pollen20_import)