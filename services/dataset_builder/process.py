## Dataset starts as random unlabeled images or videos

## Master class to normalize image names, convert videos to images (Will be stored in /images directory)
import glob
import json
import os
import shutil

import pandas as pd
import supervision as sv

from services.metadata_builder.annotations.utils import convert_yolo_to_voc, convert_ls_to_yolo


class PollenDataset:
    '''
    ETL data pipeline for new images/videos to be ingested into our master datasets
    '''
    def __init__(self, root_path):
        self.root_path = root_path
        self.images_path = os.path.join(self.root_path, 'images')
        self.ls_annotations_path = os.path.join(self.root_path, 'ls_annotations')
        os.makedirs(self.images_path, exist_ok=True)

    def metadata_parser(self,file_name, custom_format=None):
        return file_name

    def gather_metadata(self):
        '''
        Behind the scenes during media copy from raw to images directory metadata will be collected
        On the image name. This will be done so that media that makes it to the images directory can have a shortened (nice) name
        The metadata will be a mapper from raw image name to processed image name. If there is a common theme around the naming
        format of a dataset update the metadata parser function to collect additional attributes from the image names
        :return:
        '''
        pass

    def media_summary(self, raw_path):
        media_files = self.list_media(raw_path)
        images = media_files['images']
        videos = media_files['videos']
        print(f'Total Media files: {len(images+videos)}')
        print(f'  - Total Images: {len(images)}')
        print(f'  - Total Videos: {len(videos)}')

    def list_media(self, raw_path):
        media_files = {
            'images':[],
            'videos':[]
        }
        for file_name in os.listdir(raw_path):
            file_path = os.path.join(raw_path, file_name)
            if os.path.isfile(file_path):
                if file_name.endswith('.mp4') or file_name.endswith('.avi'):
                    media_files['videos'].append(file_path)
                elif file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                    media_files['images'].append(file_path)
                else:
                    print(f"Ignoring file: {file_name} - Unsupported format")
        return media_files

    def load_media(self, raw_path, alt_video_fps=-1):
        '''
        Load/copy all images or videos from a directory into the images_path
        Converting videos -> images and loading images
        :param raw_path: Path to the directory containing raw media files
        :return: None
        '''
        for file_name in os.listdir(raw_path)[0:1]:
            file_path = os.path.join(raw_path, file_name)
            if os.path.isfile(file_path):
                if file_name.endswith('.mp4') or file_name.endswith('.avi'):
                    self.convert_video_to_images(file_path, fps=alt_video_fps)
                elif file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                    shutil.copy(file_path, self.images_path)
                else:
                    print(f"Ignoring file: {file_name} - Unsupported format")

    def convert_video_to_images(self, video_path, start_offset=0, frames_to_generate=-1, stride=-1):
        '''
        Convert a video to images and save them in the images_path
        :param video_path: Path to the video file
        :param start_offset: Starting frame offset (default: 0)
        :param frames_to_generate: Number of frames to generate (-1 for all frames)
        :param fps: Frames per second for sampling frames from the video (default: -1, use video FPS)
        :return: None
        '''
        video_info = sv.VideoInfo.from_video_path(video_path)
        if frames_to_generate <= 0:
            frames_to_generate = video_info.total_frames
        if stride <= 0:
            stride = video_info.fps

        source_name = video_path.split('/')[-1]
        temp, file_ext = source_name.split('.')
        output_name = temp.replace(' ', '_')
        output_video_name = output_name + '.' + file_ext
        output_image_name = output_name + '.' + 'jpg'

        with sv.ImageSink(target_dir_path=self.root_path+'ImagesFromVideos', overwrite=True) as sink:
            for index, frame in enumerate(sv.get_video_frames_generator(video_path, stride=stride, start=start_offset, end=start_offset + frames_to_generate)):
                sink.save_image(frame, image_name=f'FRAME_{index}_{output_image_name}')

        self.copy_media(self.root_path+'ImagesFromVideos', self.images_path)
        shutil.rmtree(self.root_path+'ImagesFromVideos')

    def copy_media(self, source_dir, target_dir, image_rename_dict=None):
        '''
        Copy all media files from one directory to another
        :param source_dir:
        :param target_dir:
        :param image_rename_dict: Optional dictionary to rename images
        :return:
        '''
        if image_rename_dict:
            for src_image_name, target_image_name in image_rename_dict.items():
                src_image_path = os.path.join(source_dir, src_image_name)
                target_image_path = os.path.join(target_dir, target_image_name)
                # Check if the source image exists
                if os.path.exists(src_image_path):
                    # Copy the image to the target directory with the new name
                    shutil.copyfile(src_image_path, target_image_path)
        else:
            # If image_rename_dict is None, copy images without renaming
            for filename in os.listdir(source_dir):
                src_file_path = os.path.join(source_dir, filename)
                if os.path.isfile(src_file_path):
                    shutil.copy(src_file_path, target_dir)


    def convert_ls_annotations_to_master(self):
        '''
        Dataset will be normalized by passing ls_annotations to /bbox.csv
        :return:
        '''
        bboxes_df = []
        for ann_file in glob.glob(self.ls_annotations_path + '/*.json'):
            ls_annotation = json.load(open(ann_file, "r"))
            for image_annotation in ls_annotation:
                image_name = image_annotation['image'].split('-',1)[1]
                try:
                    for label in image_annotation['label']:
                        class_name = label['rectanglelabels'][0]
                        original_width = label['original_width']
                        original_height = label['original_height']
                        x, y, w, h = convert_ls_to_yolo(
                            (
                                label['x'],
                                label['y'],
                                label['width'],
                                label['height'],
                            ),
                            original_width,
                            original_height
                        )
                        x1, y1, x2, y2 = convert_yolo_to_voc((x, y, w, h), original_width, original_height)
                        bbox = {
                            'image_name': image_name,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'class_name': class_name
                        }
                        bboxes_df.append(bbox)
                except Exception as e:
                    print(f'ls_to_master failed for: {image_name}')

        bboxes_df = pd.DataFrame(bboxes_df)

        #### Add validation here for any images that do not have annotations in the /images directory ###

        bboxes_df.to_csv(f'{self.root_path}/bboxes.csv', index=None)
        class_map_df = bboxes_df.drop_duplicates(['class_name']).copy()[['class_name']]
        class_map_df.to_csv(f'{self.root_path}/class_map.csv', index=None)

## Dataset images will be autoannotated by being added to AutoAnnotator.pre_auto_annotate

## Dataset images will be analyzed on LabelStudio for validation of AutoAnnotations

## After validation dataset image annotations will be added to ls_annotations

## At this point dataset has been loaded and can be merged to one of our global pollen dataset sources