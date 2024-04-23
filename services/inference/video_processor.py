from typing import Dict, Set

import numpy as np
import pandas as pd
import supervision as sv
from PIL import Image
import cv2

class GlobalDetectionsManager:
    def __init__(self, id2labels):
        self.id2labels = id2labels
        self.annotations = []
        self.cropped_regions = []
        self.trackers = {} # class_id, Set[tracker_ids]

    def handle_frame(self,detections: sv.Detections, frame: np.ndarray, frame_idx: int, mode='tracked'):
        for i in range(len(detections.xyxy)):
            class_id = -1 if detections.class_id is None else detections.class_id[i]
            label = "" if self.id2labels is None else str(self.id2labels[detections.class_id[i]])
            confidence = 0 if detections.confidence is None else detections.confidence[i]
            tracker_id = -1 if detections.tracker_id is None else detections.tracker_id[i]

            if mode == 'tracked':
                if tracker_id not in self.trackers[class_id]:
                    continue
            x_min, y_min, x_max, y_max = detections.xyxy[i]
            detection = frame[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
            cropped_region = {
                "frame_idx": frame_idx,
                "class_id": class_id,
                "confidence": confidence,
                "tracker_id": tracker_id,
                "label": label,
                "detection": detection
            }
            annotation = {
                "frame_idx": frame_idx,
                "class_id": class_id,
                "confidence": confidence,
                "tracker_id": tracker_id,
                "label": label,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
            self.cropped_regions.append(cropped_region)
            self.annotations.append(annotation)

    def update_detections(self, detections: sv.Detections, frame: np.ndarray, frame_idx: int, mode='tracked'):
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            if class_id not in self.trackers:
                self.trackers[class_id] = set()  # Create an empty set if it doesn't exist
            self.trackers[class_id].add(tracker_id) # Add the tracker_id to the set
        self.handle_frame(detections, frame, frame_idx, mode)

    def finalize(self):
        return pd.DataFrame(self.annotations)


class VideoProcessor:
    def __init__(
        self,
        pipeline_model,
        source_video_path: str,
        target_video_path: str,
        confidence_threshold: float = 0.92,
        iou_threshold: float = 0.5,
        tracker_lost_track_buffer = 2,
        tracker_minimum_matching_threshold = 0.8,
        tracker_track_activation_threshold = 0.3,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.pipeline_model = pipeline_model
        self.label2id = pipeline_model.model.config.label2id
        self.id2label = pipeline_model.model.config.id2label
        self.video_info = sv.VideoInfo.from_video_path(video_path=self.source_video_path)
        ## Tracker ##
        self.tracker = sv.ByteTrack(
            lost_track_buffer = tracker_lost_track_buffer,
            frame_rate=self.video_info.fps,
            minimum_matching_threshold=tracker_minimum_matching_threshold,
            track_activation_threshold=tracker_track_activation_threshold
        )
        self.tracker.reset()
        ## Annotators ##
        self.trace_annotator = sv.TraceAnnotator(
            thickness=3,
            trace_length=self.video_info.fps * 2,
            position=sv.Position.CENTER_RIGHT,
        )
        self.line_counter = sv.LineZone(
            start=sv.Point(int(self.video_info.width * 0.3), 0),
            end=sv.Point(int(self.video_info.width * 0.3), self.video_info.height),
            triggering_anchors=[sv.Position.CENTER_RIGHT]
        )
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=3,
            text_thickness=2,
            text_scale=0.5,
            # custom_in_text='IN',
            custom_out_text='COUNT',
            display_in_count=False,
            display_out_count=True
        )
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.detections_manager = GlobalDetectionsManager(self.id2label)
        self.result_frames = []


    def process_video(self, stride=1, start_offset=0, frames_to_generate=-1, export_types = ['video','image','annotations','detections']):
        if frames_to_generate <= 0:
            frames_to_generate = self.video_info.total_frames - start_offset
        source_name = self.source_video_path.split('/')[-1]
        temp, file_ext = source_name.split('.')
        output_name = temp.replace(' ', '_')
        output_video_name = output_name + '.' + file_ext
        output_image_name = output_name + '.' + 'jpg'

        with sv.VideoSink(target_path=self.target_video_path+output_video_name, video_info=self.video_info) as sink:
            for index, frame in enumerate(sv.get_video_frames_generator(source_path=self.source_video_path, stride=stride, start=start_offset, end=start_offset + frames_to_generate)):
                result_frame = self.process_frame(frame, index)
                if 'video' in export_types:
                    sink.write_frame(frame=result_frame)
                self.result_frames.append(result_frame)
                if 'image' in export_types:
                    cv2.imwrite(f"{self.target_video_path}FRAME_{index}_{output_image_name}", result_frame)
        if 'detections' in export_types:
            for detection in self.detections_manager.annotations:
                frame_idx = detection['frame_idx']
                cropped_image = detection['detection']
                tracker_id = detection['tracker_id']
                cv2.imwrite(f"{self.target_video_path}DETECTION_{frame_idx}_{tracker_id}_{output_image_name}", cropped_image)
        detections_df = self.detections_manager.finalize()
        if 'annotations' in export_types:
            detections_df.to_csv(f"{self.target_video_path}{output_name}_annotations.csv", index=None)

    def annotate_frame(self, detections, frame, include_line=True, include_trace=True):
        ## Annotate
        labels = [
            f"{self.id2label[class_id]} ({round(confidence * 100, 2)}%)"
            for class_id, tracker_id, confidence
            in zip(detections.class_id, detections.tracker_id, detections.confidence)
        ]
        annotated_frame = frame.copy()
        if include_line:
            annotated_frame = self.line_annotator.annotate(frame=annotated_frame, line_counter=self.line_counter)
        if include_trace:
            annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame

    def model_callback(self, frame: np.ndarray):
        pass


    def process_frame(self, frame: np.ndarray, idx):
        results = self.pipeline_model(Image.fromarray(frame), threshold=self.conf_threshold)
        transformed_results = {
            'boxes': [],
            'scores': [],
            'labels': []
        }
        for result in results:
            transformed_results['boxes'].append(np.array([result['box']['xmin'], result['box']['ymin'], result['box']['xmax'], result['box']['ymax']]))
            transformed_results['scores'].append(result['score'])
            transformed_results['labels'].append(self.label2id[result['label']])
        detections = sv.Detections(
            xyxy=np.array(transformed_results['boxes']),
            confidence=np.array(transformed_results['scores']),
            class_id=np.array(transformed_results['labels']),
        )
        detections = detections.with_nms(threshold=self.iou_threshold, class_agnostic=False)
        detections = self.tracker.update_with_detections(detections)
        crossed_in, crossed_out = self.line_counter.trigger(detections=detections)
        self.detections_manager.update_detections(detections, frame, idx)
        return self.annotate_frame(detections, frame)