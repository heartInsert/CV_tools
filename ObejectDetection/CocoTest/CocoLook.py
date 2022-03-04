import json
import cv2
import os


def cocoLook_Byjson(img_dir, image_id, json_dir):
    with open(json_dir, 'r') as load_f:
        load_dict = json.load(load_f)
    annotations = []
    for item in load_dict['annotations']:
        if item['image_id'] == image_id:
            annotations.append(item)
    img = cv2.imread(img_dir)
    for anno in annotations:
        xmin = int(anno['bbox'][0])
        ymin = int(anno['bbox'][1])
        width = int(anno['bbox'][2])
        height = int(anno['bbox'][3])
        xmax = int(xmin + width)
        ymax = int(ymin + height)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=2)
    cv2.imshow('head', img)
    cv2.waitKey(0)


def cocoLook_ByBBox(img_dir, BBoxes):
    img = cv2.imread(img_dir)
    for bbox in BBoxes:
        x_min, y_min, width, height = bbox
        x_max = int(x_min + width)
        y_max = int(y_min + height)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), thickness=2)
    cv2.imshow('head', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    image_folder = '/home/xjz/Desktop/Coding/DL_Data/ObjectDetection/DCICweixing/training_dataset/A'
    image_name = '00014.jpg'
    image_id = "00014"
    json_dir = '/home/xjz/Desktop/Coding/DL_Data/ObjectDetection/DCICweixing/DCIC2COCO/DcicCoco.json'
    cocoLook_Byjson(os.path.join(image_folder, image_name), image_id, json_dir)
