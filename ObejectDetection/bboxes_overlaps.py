import torch
import numpy as np
import json


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    '''
    gtbbox = torch.tensor([[10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 60.0, 20.0], [10.0, 10.0, 60.0, 20.0]]).float()
    anchor = torch.tensor([[10, 10, 20, 20], [10, 10, 30, 30]]).float()
    bbox_overlaps(gtbbox, anchor)
    bbox[x_min,y_min,x_max,y_max]
    '''
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
    overlap = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2[..., None, :] - overlap
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap.true_divide_(union)
    return ious


def caculate_ious(gtbbox, predict_box):
    '''
    gtbbox:[x_min,y_min,x_max,y_max]
    predict_box:[x_min,y_min,x_max,y_max]
    '''
    gtbbox = torch.tensor(gtbbox).float()
    predict_box = torch.tensor(predict_box).float()
    ious = bbox_overlaps(gtbbox, predict_box)
    gtbbox_maxOverlap_value, gtbbox_max_index = ious.max(dim=1)
    predict_maxOverlap_value, predict_max_index = ious.max(dim=0)
    TP, FP, FN = (gtbbox_maxOverlap_value >= 0.5).sum(), (predict_maxOverlap_value < 0.5).sum(), (
            gtbbox_maxOverlap_value < 0.5).sum()

    # accuracy = TP.float().true_divide_((TP + FP + FN).float())
    return TP, FP, FN, gtbbox_maxOverlap_value < 0.5, predict_maxOverlap_value < 0.5


def json2dict(json_file):
    images = json_file['images']
    annotations = json_file['annotations']
    image_anno_dict = {}
    for image in images:
        annotation = [anno for anno in annotations if anno['image_id'] == image['id']]
        image_anno_dict[image['file_name']] = {"image": image, 'annotation': annotation}

    return image_anno_dict


def anno2bboxes(anotations):
    bboxes = []
    for anno in anotations:
        x_min, y_min, width, height = anno['bbox']
        x_max, y_max = x_min + width, y_min + height
        bboxes.append([x_min, y_min, x_max, y_max])

    return np.array(bboxes)


if __name__ == "__main__":
    gt_json_path = '/home/xjz/Desktop/Coding/PycharmProjects/competition/Help_Tools/CV_tools/ObejectDetection/data/bbox_overlap_gtbox.json'
    predict_json_path = '/home/xjz/Desktop/Coding/PycharmProjects/competition/Help_Tools/CV_tools/ObejectDetection/data/bbox_overlap_predict.json'
    with open(gt_json_path, 'r') as f:
        gt_json = json.load(f)
    with open(predict_json_path, 'r') as f:
        predict_json = json.load(f)
    gt_dict = json2dict(gt_json)
    predict_dict = json2dict(predict_json)
    keys = gt_dict.keys()
    TP_sum, FP_sum, FN_sum = 0, 0, 0
    area_list = []
    for key in keys:
        gt_bbox = anno2bboxes(gt_dict[key]['annotation'])
        predict_bbox = anno2bboxes(predict_dict[key]['annotation'])
        TP, FP, FN, gt_index, predict_index = caculate_ious(gt_bbox, predict_bbox)
        if gt_index.sum() > 0:
            for index, item in enumerate(list(gt_index.numpy())):
                if item:
                    area_list.append(gt_dict[key]['annotation'][index])
        TP_sum, FP_sum, FN_sum = TP_sum + TP, FP_sum + FP, FN_sum + FN
    accuracy = TP_sum.float().true_divide_((TP_sum + FP_sum + FN_sum).float())
    pass
print()
