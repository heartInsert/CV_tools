import json
import copy
import random
import numpy as np
import torch
import cv2
import os


class Image_obj(object):
    def __init__(self, img_obj):
        self.file_name = img_obj['file_name']
        self.height = img_obj['height']
        self.width = img_obj['width']
        self.id = img_obj['id']
        pass


class Bbox_obj(object):
    def __init__(self):
        pass
    

class Annotation_obj(object):
    def __init__(self, Annotation_obj):
        self.annotations = Annotation_obj
        pass

    def bboxes_width_height(self):
        bboxes = []
        for anno in self.annotations:
            x_min, y_min, width, height = anno['bbox']
            bboxes.append([x_min, y_min, width, height])
        return np.array(bboxes)

    def bboxes_xmax_ymax(self):
        bboxes = []
        for anno in self.annotations:
            x_min, y_min, width, height = anno['bbox']
            bboxes.append([x_min, y_min, x_min + width, y_min + height])
        return np.array(bboxes)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    '''
    gtbbox = torch.tensor([[10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 60.0, 20.0], [10.0, 10.0, 60.0, 20.0]]).float()
    anchor = torch.tensor([[10, 10, 20, 20], [10, 10, 30, 30]]).float()
    bbox_overlaps(gtbbox, anchor)
    bbox[x_min,y_min,x_max,y_max]
    '''
    if type(bboxes1) is np.ndarray:
        bboxes1 = torch.tensor(bboxes1).float()
        bboxes2 = torch.tensor(bboxes2).float()
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])
    lt = torch.max(bboxes1[..., :, None, :2],
                   bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:],
                   bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]
    wh = rb - lt
    overlap = wh[..., 0] * wh[..., 1].clamp(min=0)
    union = area1[..., None] + area2[..., None, :] - overlap
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap.true_divide_(union)
    return ious


def json2dict(json_file):
    images = json_file['images']
    annotations = json_file['annotations']
    image_anno_dict = {}
    for image in images[:100]:
        annotation = [anno for anno in annotations if anno['image_id'] == image['id']]
        image_anno_dict[image['file_name']] = {"image": Image_obj(image), 'annotation': Annotation_obj(annotation)}

    return image_anno_dict


def generate_bbox(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_range, y_range = img_width - width, img_height - height
    x_min = random.randint(0, x_range)
    y_min = random.randint(0, y_range)
    target_box = x_min, y_min, width, height
    stack_box = {'original_box': bbox, 'target_box': target_box}
    return stack_box


def main():
    train_json = '/home/xjz/Desktop/Coding/DL_Data/ObjectDetection/DCICweixing/training_dataset/DcicCoco_train.json'
    img_dir = '/home/xjz/Desktop/Coding/DL_Data/ObjectDetection/DCICweixing/training_dataset/A'
    target_json_dir = '/home/xjz/Desktop/Coding/DL_Data/ObjectDetection/DCICweixing/training_dataset' \
                      '/DcicCoco_augment_train.json '
    target_img_dir = '/home/xjz/Desktop/Coding/DL_Data/ObjectDetection/DCICweixing/training_dataset/augment_img'
    with open(train_json, 'r') as f:
        Train_json_file = json.load(f)
    anno_infos = Train_json_file['annotations']
    id_list = [int(anno['id']) for anno in anno_infos]
    id_max = id_list[-1]
    Train_json_file = json2dict(Train_json_file)
    target_json_file = copy.deepcopy(Train_json_file)
    keys = Train_json_file.keys()
    for key in keys:
        image_info = Train_json_file[key]['image']
        Train_annotation = Train_json_file[key]['annotation']
        for Train_anno in Train_annotation.annotations:
            if Train_anno['area'] > 900:
                continue
            n_aug = 0
            for i in range(5):
                # Train
                stack_box = generate_bbox(Train_anno['bbox'], image_info.width, image_info.height)
                x_min, y_min, width, height = stack_box['target_box']
                Train_anno_bbox = np.array([[x_min, y_min, x_min + width, y_min + height]])
                # target
                target_key = random.choice(list(keys))
                target_annotation = target_json_file[target_key]['annotation']
                ious = bbox_overlaps(Train_anno_bbox, target_annotation.bboxes_xmax_ymax())
                if ious.sum() == 0:
                    # overlap ==0
                    # x_min, y_min, width, height = Train_anno['bbox']
                    Train_img = cv2.imread(os.path.join(img_dir, image_info.file_name))

                    n_aug = n_aug + 1
                    if n_aug >= 2:
                        break
            pass

        pass

    pass


if __name__ == "__main__":
    main()
