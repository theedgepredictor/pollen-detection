import glob
import cv2
import json
import pandas as pd
import shutil
import os
from torch.utils.data import DataLoader


class DatasetPreprocessor:
    def __init__(self, src_dir, image_dir, dest_dir, phases=["train", "valid", "test"]):
        self.src_dir = src_dir
        self.image_dir = image_dir
        self.dest_dir = dest_dir
        self.phases = phases
        self.annotation_file_name = ".json"

    def split_dataset(self, dataset_df):
        # Define the percentages for train, valid, and test splits
        train_percentage = 0.75
        valid_percentage = 0.15
        test_percentage = 0.10

        full_train_images = []
        full_valid_images = []
        full_test_images = []

        for class_name, sub_df in dataset_df.groupby(['class_name']):
            num_images = sub_df.shape[0]
            # Ensure at least one image in each phase
            num_train = max(1, int(train_percentage * num_images))
            num_valid = max(1, int(valid_percentage * num_images))
            num_test = max(1, int(test_percentage * num_images))

            # Take random sample for test and valid sets. Train will be leftover

            test_images = list(sub_df.sample(n=num_test, random_state=1).image_name.values)
            valid_images = list(sub_df[~sub_df.image_name.isin(test_images)].sample(n=num_valid, random_state=1).image_name.values)
            train_images = list(sub_df[~sub_df.image_name.isin(valid_images + test_images)].image_name.values)
            # Create directories for train, valid, and test splits
            full_train_images.extend(train_images)
            full_valid_images.extend(valid_images)
            full_test_images.extend(test_images)

        if len(self.phases) == 2:
            full_train_images = full_train_images + full_test_images
            images_phases = zip(self.phases, [full_train_images, full_valid_images])
        else:
            images_phases = zip(self.phases, [full_train_images, full_valid_images, full_test_images])

        for phase, image_names in images_phases:
            os.makedirs(os.path.join(self.dest_dir, phase), exist_ok=True)
            for image_name in image_names:
                src_path = os.path.join(self.image_dir, image_name)
                dest_path = os.path.join(self.dest_dir, phase, image_name)
                shutil.copyfile(src_path, dest_path)

    def annotate_dataset(self, categories, bboxes_df, inverse_class_dict):
        for phase in self.phases:
            root_path = os.path.join(self.dest_dir, phase)
            json_file = os.path.join(self.dest_dir, f"{phase}.json")
            res_file = {
                "categories": categories,
                "images": [],
                "annotations": []
            }
            annot_count = 0
            image_id = 0
            processed = 0

            # Obtain image_names
            file_list = glob.glob(os.path.join(root_path, "*.jpg"))

            for file in file_list:
                image_path = file
                file_name = file.replace(root_path, "").replace("\\", '').replace('/', '')
                img = cv2.imread(image_path)
                img_h, img_w, channels = img.shape
                img_elem = {
                    "file_name": file_name,
                    "height": img_h,
                    "width": img_w,
                    "id": image_id
                }
                res_file["images"].append(img_elem)
                annotations = bboxes_df[bboxes_df.image_name == file_name]
                for idx, row in annotations.iterrows():
                    key = row['image_name']
                    voc_bbox = row['x1'], row['y1'], row['x2'], row['y2']
                    x1, y1, x2, y2 = convert_voc_to_yolo(voc_bbox, img_w, img_h)
                    coords = inverse_class_dict.get(row['class_name']), x1, y1, x2, y2

                    # YOLO to COCO JSON
                    x_center = (float(coords[1]) * (img_w))
                    y_center = (float(coords[2]) * (img_h))
                    width = (float(coords[3]) * img_w)
                    height = (float(coords[4]) * img_h)
                    category_id = int(coords[0]) if len(categories) != 1 else 0

                    mid_x = int(x_center - width / 2)
                    mid_y = int(y_center - height / 2)
                    width = int(width)
                    height = int(height)

                    area = width * height
                    poly = [[mid_x, mid_y],
                            [width, height],
                            [width, height],
                            [mid_x, mid_y]]

                    annot_elem = {
                        "id": annot_count,
                        "bbox": [
                            float(mid_x),
                            float(mid_y),
                            float(width),
                            float(height)
                        ],
                        "segmentation": list([poly]),
                        "image_id": image_id,
                        "ignore": 0,
                        "category_id": category_id,
                        "iscrowd": 0,
                        "area": float(area)
                    }
                    res_file["annotations"].append(annot_elem)
                    annot_count += 1
                image_id += 1

                processed += 1
            with open(json_file, "w") as f:
                json_str = json.dumps(res_file)
                f.write(json_str)

            print("Processed {} {} images...".format(processed, phase))
        print("Done.")

    def process_dataset(self, bboxes_file, class_map_file, dataloader_batchsize=2):
        print('Processing Dataset...')
        if DATASET == 'GlobalPollen':
            bboxes_df = pd.read_csv(bboxes_file)
        else:
            bboxes_df = pd.read_csv(bboxes_file, header=None, names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class_name'])

        labels_df = pd.read_csv(class_map_file, header=None, names=['name', 'id'])

        if SERVICE == 'pollencounter':
            class_dict = {0: "pollen"}
            inverse_class_dict = {"pollen": 0}
            categories = [{"supercategory": "none", "name": 'pollen', "id": 0}]
        else:
            class_dict = dict(zip(list(map(int, labels_df.id.values)), list(labels_df.name.values)))
            inverse_class_dict = dict(zip(list(labels_df.name.values), list(map(int, labels_df.id.values))))
            categories = [{"supercategory": "none", "name": val, "id": int(key)} for key, val in class_dict.items()]

        dataset_df = bboxes_df[['image_name', 'class_name']].drop_duplicates(['image_name'])
        print(f'  -- Images: {dataset_df.shape[0]}')
        print(f'  -- Annotations: {bboxes_df.shape[0]}')
        print(f'  -- Classes: {labels_df.shape[0]}')
        print('')
        print('Splitting Dataset...')
        self.split_dataset(dataset_df)
        print('Processing Dataset...')
        self.annotate_dataset(categories, bboxes_df, inverse_class_dict)

        def collate_fn(batch):
            # DETR authors employ various image sizes during training, making it not possible
            # to directly batch together images. Hence they pad the images to the biggest
            # resolution in a given batch, and create a corresponding binary pixel_mask
            # which indicates which pixels are real/which are padding
            pixel_values = [item[0] for item in batch]
            encoding = image_processor.pad(pixel_values, return_tensors="pt")
            labels = [item[1] for item in batch]
            return {
                'pixel_values': encoding['pixel_values'],
                'pixel_mask': encoding['pixel_mask'],
                'labels': labels
            }

        train_dataset = PollenDetection(
            image_directory_path=TRAIN_DIRECTORY,
            image_processor=image_processor,
            train=True
        )
        val_dataset = PollenDetection(
            image_directory_path=VAL_DIRECTORY,
            image_processor=image_processor,
            train=False
        )
        if NOTEBOOK_MODE == 'benchmark':
            test_dataset = PollenDetection(
                image_directory_path=TEST_DIRECTORY,
                image_processor=image_processor,
                train=False
            )

        train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=dataloader_batchsize, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(dataset=val_dataset, collate_fn=collate_fn, batch_size=dataloader_batchsize, num_workers=2)
        if NOTEBOOK_MODE == 'benchmark':
            test_dataloader = DataLoader(dataset=test_dataset, collate_fn=collate_fn, batch_size=dataloader_batchsize)
        else:
            test_dataloader = None
        return train_dataloader, val_dataloader, test_dataloader, class_dict, inverse_class_dict