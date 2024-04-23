def convert_ls_to_yolo(ls_bbox, img_width, img_height):
    x, y, width, height = ls_bbox
    yolo_x = (x + width / 2) / 100
    yolo_y = (y + height / 2) / 100
    yolo_w = width / 100
    yolo_h = height / 100
    return yolo_x, yolo_y, yolo_w, yolo_h

def convert_ls_to_voc(yolo_bbox, img_width, img_height):
    x, y, width, height = yolo_bbox
    x_min = int(x * img_width / 100)
    y_min = int(y * img_height / 100)
    x_max = int((x + width) * img_width / 100)
    y_max = int((y + height) * img_height / 100)
    return x_min, y_min, x_max, y_max

def convert_yolo_to_ls(yolo_bbox, img_width, img_height):
    yolo_x, yolo_y, yolo_w, yolo_h = yolo_bbox
    x = (yolo_x - yolo_w / 2) * 100
    y = (yolo_y - yolo_h / 2) * 100
    width = yolo_w * 100
    height = yolo_h * 100
    return x, y, width, height


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