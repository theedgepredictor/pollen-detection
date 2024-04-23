import random

import cv2

from services.active_learning_annotation_builder.utils import Prediction
from services.consts import CLASSIFICATION_CLASS_COLORS, DETECTION_CLASS_COLORS
from services.metadata_builder.annotations.utils import convert_voc_to_yolo

class AutoAnnotator:
    def __init__(self, label_studio_client, project_name, image_dir, n_tasks=None):
        self.ls = label_studio_client
        self.ls.check_connection()
        self.project_name = project_name ## GlobalPollenDetection | GlobalPollenClassification
        self.project = self._load_project()
        self.unlabeled_tasks = self.get_unlabeled_tasks(n_tasks)
        self.image_dir = image_dir

    def _make_labeling_interface(self):
        color_mapper = DETECTION_CLASS_COLORS if self.project_name == 'GlobalPollenDetection' else CLASSIFICATION_CLASS_COLORS
        labels = [f'''<Label value="{cl}" background="{color}" />''' for cl, color in color_mapper.items()]
        return f'''
            <View>
              <Image name="image" value="$image" zoom="true"/>
              <Header value="Rectangle Labels"/>
              <RectangleLabels name="label" toName="image">
                {"".join(labels)}
              </RectangleLabels>
            </View>
        '''

    def _load_project(self):
        projects = self.ls.list_projects(**{'title': self.project_name})
        if len(projects) > 0:
            project = projects[0]
        else:
            project = self.ls.start_project(
                title=self.project_name,
                label_config=self._make_labeling_interface(),
            )
        return project

    def get_unlabeled_tasks(self, n_tasks=None):
        unlabeled_tasks_ids = self.project.get_unlabeled_tasks_ids()
        if len(unlabeled_tasks_ids) == 0:
            raise Exception('No Tasks Left To Label')
        if n_tasks is not None:
            unlabeled_tasks_ids = random.sample(unlabeled_tasks_ids, n_tasks)
        return self.project.get_tasks(selected_ids=unlabeled_tasks_ids)

    def get_model_predictions(self, pipeline_model, threshold=0.4):
        image_urls = [f"{self.image_dir}/{task['data']['image'].split('-', 1)[1]}" for task in self.unlabeled_tasks]
        print(f'Generating Model Predictions for {len(image_urls)} images...')
        results = pipeline_model(image_urls, threshold=threshold)
        predictions = []
        for task, image_url, result in zip(self.unlabeled_tasks, image_urls, results):
            predictions.append(Prediction(task=task, image_url=image_url, result=result))
        return predictions

    def get_annotations(self, predictions, class_name_override=None):
        annotations = []
        for prediction in predictions:
            image = cv2.imread(prediction.image_url)
            original_height, original_width, channels = image.shape
            annotation = {
                'result': [],
                'ground_truth': True,
                'task': prediction.task['id'],
            }
            for result in prediction.result:
                voc_bbox = result['box']['xmin'], result['box']['ymin'], result['box']['xmax'], result['box']['ymax']
                yolo_x, yolo_y, yolo_w, yolo_h = convert_voc_to_yolo(voc_bbox, original_width, original_height)

                class_name = result['label'] if class_name_override is None else class_name_override
                annotation['result'].extend([{
                    "source": "$image",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": (yolo_x - yolo_w / 2) * 100,
                        "y": (yolo_y - yolo_h / 2) * 100,
                        "width": yolo_w * 100,
                        "height": yolo_h * 100,
                        "rectanglelabels": [class_name]
                    }

                }])
            annotations.append(annotation)
        return annotations

    def pre_auto_annotate(self, pipeline_model, threshold=0.4, class_name_override=None):
        responses = []
        predictions = self.get_model_predictions(pipeline_model, threshold=threshold)
        annotations = self.get_annotations(predictions, class_name_override=class_name_override)
        # return annotations
        for annotation in annotations:
            task_id = annotation['task']
            res = self.project.create_annotation(task_id, **annotation)
            responses.append(res)
        return responses