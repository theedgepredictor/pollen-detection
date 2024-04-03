import glob
import os
import re
from datetime import datetime

import pandas as pd

from src.utils import get_dataframe, name_filter


class KUMediaMetadata:
    def __init__(self, media_type: str, name: str, source: str, magnification: int, capture_datetime: datetime, specific_type: str = 'MIXED'):
        self.media_type = media_type
        self.name = name
        self.id = name_filter(name)
        self.source = source
        self.magnification = magnification
        self.capture_datetime = capture_datetime
        self.specific_type = specific_type
        self.last_updated = datetime.now()

    @classmethod
    def from_image_file(cls, file_path: str):
        file_name = os.path.basename(file_path)

        # Extract magnification
        parts = file_name.split(' on ')
        magnification_str = parts[0].split(' ')[0]
        magnification = int(magnification_str[:-1])

        datetime_str = parts[1]
        date_str = datetime_str.split(' at ')[0].strip()
        year = f"20" + date_str[-2:]
        date_str = date_str[0:-2] + year
        time_str = datetime_str.split(' at ')[1].rsplit(' ', 1)[0].replace('.', ':')
        datetime_str = f"{date_str} {time_str}"
        capture_datetime = pd.Timestamp(datetime_str).to_pydatetime()

        # Extract specific type
        specific_type = re.search(r'\((.*?)\)', file_name).group(1).upper().strip().replace('.', '')
        return cls(
            media_type=file_path.split('.')[-1],
            name=file_name,
            source='Kean University',
            magnification=magnification,
            capture_datetime=capture_datetime,
            specific_type=specific_type
        )

def run_ku_media_metadata_importer():
    media_metadatas = []
    file_paths = glob.glob('./data/raw/ku_pollen/DRB_400x/images/*')
    file_paths.extend(glob.glob('./data/raw/ku_pollen/DRB_100x/images/*'))
    for file_path in file_paths:
        media_metadata = KUMediaMetadata.from_image_file(file_path)
        media_metadatas.append(media_metadata.__dict__)

    ku_media_metadatas_df = pd.DataFrame(media_metadatas)
    common_names_df = get_dataframe('./data/database/pollen_commonnames.csv')
    if common_names_df.shape[0]==0:
        raise Exception('Need to run category_importer before running media_metadata_importer')
    ku_media_metadatas_df = pd.merge(ku_media_metadatas_df, common_names_df.rename(columns={'name': 'specific_type', 'id': 'common_name_id'}), on=['specific_type'])

    return ku_media_metadatas_df