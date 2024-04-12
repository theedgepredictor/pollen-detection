import os
from datetime import datetime

import pandas as pd

from services.utils import name_filter, get_dataframe


class Pollen20MediaMetadata:
    def __init__(self, name: str, specific_type: str = 'MIXED'):
        self.media_type = 'IMAGE'
        self.name = name
        self.id = name_filter(name)
        self.source = 'POLLEN20L'
        self.magnification = 1000
        self.capture_datetime = datetime(2022,1,1)
        self.specific_type = specific_type
        self.last_updated = datetime.now()


def run_pollen20_media_metadata_importer():
    bboxes_df = pd.read_csv('./data/raw/pollen20L/bboxes.csv', header=None, names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class_name'])
    dataset_df = bboxes_df[['image_name', 'class_name']].drop_duplicates(['image_name'])
    media_metadatas = []
    for idx, row in dataset_df.iterrows():
        media_metadata = Pollen20MediaMetadata(name=row['image_name'], specific_type=row['class_name'].upper())
        media_metadatas.append(media_metadata.__dict__)

    pollen20_media_metadatas_df = pd.DataFrame(media_metadatas)
    common_names_df = get_dataframe('./data/database/pollen_commonnames.csv')
    if common_names_df.shape[0]==0:
        raise Exception('Need to run category_importer before running media_metadata_importer')
    pollen20_media_metadatas_df = pd.merge(pollen20_media_metadatas_df, common_names_df.rename(columns={'name': 'specific_type', 'id': 'common_name_id'}), on=['specific_type'])
    return pollen20_media_metadatas_df


