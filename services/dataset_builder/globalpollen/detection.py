import shutil
import pandas as pd
import os

def refresh(push_to_hub=False):

    ## Datasets
    POLLEN20L_DATASET_PATH = '../data/raw/pollen20L/'
    USPOLLEN_DATASET_PATH = '../data/raw/USPollen/'
    KUPOLLEN_DATASET_PATH = '../data/raw/ku_pollen/'

    ## Target Dataset
    GLOBALPOLLENDETECTION_DATASET_PATH = '../data/raw/GlobalPollenDetection/'

    ## Create master annotation from all source datasets at /bboxes.csv to Target dataset /bboxes.csv
    pollen20l_annotations = pd.read_csv(POLLEN20L_DATASET_PATH + 'bboxes.csv', header=None, names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class_name'])
    uspollen_annotations = pd.read_csv(USPOLLEN_DATASET_PATH + 'bboxes.csv')
    kupollen_annotations = pd.read_csv(KUPOLLEN_DATASET_PATH + 'bboxes.csv')

    # Concatenate annotation files
    master_annotations = pd.concat([pollen20l_annotations, uspollen_annotations, kupollen_annotations], ignore_index=True)
    image_names = list(master_annotations.image_name.values)

    ## Copy images from source datasets at /images/*.jpg to Target Dataset path /images/*.jpg
    for source_path in [POLLEN20L_DATASET_PATH, USPOLLEN_DATASET_PATH, KUPOLLEN_DATASET_PATH]:
        source_image_path = os.path.join(source_path, 'images')
        for image_file in os.listdir(source_image_path):
            if image_file in image_names:
                source_image_file = os.path.join(source_image_path, image_file)
                target_image_file = os.path.join(GLOBALPOLLENDETECTION_DATASET_PATH + 'images', image_file)
                shutil.copyfile(source_image_file, target_image_file)

    # Update detection class_names
    master_annotations['class_name'] = 'POLLEN'

    print('Refreshing Global Detection Dataset')
    dataset_df = master_annotations[['image_name', 'class_name']].drop_duplicates(['image_name'])
    print(f'  -- Images: {dataset_df.shape[0]}')
    print(f'  -- Annotations: {master_annotations.shape[0]}')

    # Save master annotation file
    master_annotations.to_csv(GLOBALPOLLENDETECTION_DATASET_PATH + 'bboxes.csv', index=False)