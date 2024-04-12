import pandas as pd

from services.utils import get_dataframe


def run_pollen20_annotation_importer():
    labels_df = pd.read_csv('./data/raw/pollen20L/class_map.csv', header=None, names=['name', 'id'])
    inverse_class_dict = dict(zip(list(labels_df.name.values), list(labels_df.id.values)))
    bboxes_df = pd.read_csv('./data/raw/pollen20L/bboxes.csv', header=None, names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class_name'])
    bboxes_df['label'] = bboxes_df['class_name'].map(inverse_class_dict)
    bboxes_df.class_name = bboxes_df.class_name.str.replace('_', '').str.upper()

    common_names_df = get_dataframe('./data/database/pollen_commonnames.csv')
    if common_names_df.shape[0] == 0:
        raise Exception('Need to run category_importer before running media_metadata_importer')
    out_df = pd.merge(bboxes_df, common_names_df.rename(columns={'name': 'class_name', 'id': 'common_name_id'}).drop(columns=['classification_level']), on=['class_name'])
    out_df['source'] = 'POLLEN20L'
    return out_df