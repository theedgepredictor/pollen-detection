def convert_voc_to_yolo(voc_bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = voc_bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def convert_yolo_to_voc(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return x_min, y_min, x_max, y_max

def convert_coco_to_yolo(coco_bbox, img_width, img_height):
    x, y, width, height = coco_bbox
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width = width / img_width
    height = height / img_height
    return x_center, y_center, width, height

def convert_coco_to_voc(coco_bbox):
    x, y, width, height = coco_bbox
    x_min = x
    y_min = y
    x_max = x + width
    y_max = y + height
    return x_min, y_min, x_max, y_max