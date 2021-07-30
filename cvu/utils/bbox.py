"""Original Code Taken From ultralytics/yolov5
URL: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
"""
import numpy as np


def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]

    Args:
        xywh (np.ndarray): array of 4 float [center_x, center_y, width, height]

    Returns:
        np.ndarray: array of 4 float [x1, y1, x2, y2] where (x1,y1)==top-left
        and (x2,y2)==bottom-right.
    """
    xyxy = np.copy(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # top left x
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # top left y
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # bottom right x
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # bottom right y
    return xyxy


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    # Modified https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (np.ndarray[N, 4]): first box
        box2 (np.ndarray[M, 4]): second box

    Returns:
        iou (np.ndarray[N, M]): the NxM matrix containing the pairwise IoU values for
        every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.min(box1[:, None, 2:], box2[:, 2:]) -
             np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)
