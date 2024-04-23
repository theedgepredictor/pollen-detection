import os

from label_studio_sdk import Client
from transformers import pipeline

from services.active_learning_annotation_builder.auto_annotator import AutoAnnotator


def runner(ls_project_title, image_directory, hugging_face_repo_id='Charliesgt/pollencounter_detr_resnet50', confidence_threshold=0.8, class_name_override=None):
    ls = Client(url=os.environ.get('LS_FRONTEND_URL'), api_key=os.environ.get('LS_API_KEY'))
    print(ls.check_connection())
    auto_annotator = AutoAnnotator(label_studio_client=ls, project_name=ls_project_title, image_dir=image_directory)
    pipeline_model = pipeline("object-detection", model=hugging_face_repo_id, device_map="auto")
    annotations = auto_annotator.pre_auto_annotate(pipeline_model=pipeline_model, threshold=confidence_threshold, class_name_override=class_name_override)