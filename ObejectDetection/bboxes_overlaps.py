import torch
import numpy


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
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


if __name__ == "__main__":
    gtbbox = torch.tensor([[10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 60.0, 20.0], [10.0, 10.0, 60.0, 20.0]]).float()
    anchor = torch.tensor([[10, 10, 20, 20], [10, 10, 30, 30]]).float()
    bbox_overlaps(gtbbox, anchor)
