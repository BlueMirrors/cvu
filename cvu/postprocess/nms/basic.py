"""Original Code Taken from https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py

"""
from typing import List, Tuple
import numpy as np


def nms_np(detections: np.ndarray, scores: np.ndarray, max_det: int,
           thresh: float) -> List[np.ndarray]:
    """Standard Non-Max Supression Algorithm for filter out detections.

    Args:
        detections (np.ndarray): bounding-boxes of shape num_detections,4
        scores (np.ndarray): confidence scores of each bounding box
        max_det (int): Maximum number of detections to keep.
        thresh (float): IOU threshold for NMS

    Returns:
        List[np.ndarray]: Filtered boxes.
    """
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # get boxes with more ious first
    order = scores.argsort()[::-1]

    # final output boxes
    keep = []

    while order.size > 0 and len(keep) < max_det:
        # pick maxmum iou box
        i = order[0]
        keep.append(i)

        # get iou
        ovr = get_iou((x1, y1, x2, y2), order, areas, idx=i)

        # drop overlaping boxes
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def get_iou(xyxy: Tuple[np.ndarray], order: np.ndarray, areas: np.ndarray,
            idx: int) -> float:
    """Helper function for nms_np to calculate IoU.

    Args:
        xyxy (Tuple[np.ndarray]): tuple of x1, y1, x2, y2 coordinates.
        order (np.ndarray): boxs' indexes sorted according to there
        confidence scores

        areas (np.ndarray): area of each box
        idx (int): base box to calculate iou for

    Returns:
        float: [description]
    """
    x1, y1, x2, y2 = xyxy
    xx1 = np.maximum(x1[idx], x1[order[1:]])
    yy1 = np.maximum(y1[idx], y1[order[1:]])
    xx2 = np.minimum(x2[idx], x2[order[1:]])
    yy2 = np.minimum(y2[idx], y2[order[1:]])

    max_width = np.maximum(0.0, xx2 - xx1 + 1)
    max_height = np.maximum(0.0, yy2 - yy1 + 1)
    inter = max_width * max_height

    return inter / (areas[idx] + areas[order[1:]] - inter)
