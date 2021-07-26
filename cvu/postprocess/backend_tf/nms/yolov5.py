"""Original Code Taken From ultralytics/yolov5
URL: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
"""
from typing import List

import numpy as np
import tensorflow as tf

from cvu.postprocess.nms.yolov5 import non_max_suppression_np


def non_max_suppression_tf(predictions: np.ndarray,
                           conf_thres: float = 0.25,
                           iou_thres: float = 0.45,
                           agnostic: bool = False,
                           multi_label: bool = False) -> List[np.ndarray]:
    """Runs Non-Maximum Suppression (NMS used in Yolov5) on inference results.

    Args:
        predictions (np.ndarray): predictions from yolov inference

        conf_thres (float, optional): confidence threshold in range 0-1.
        Defaults to 0.25.

        iou_thres (float, optional): IoU threshold in range 0-1 for NMS filtering.
        Defaults to 0.45.

        agnostic (bool, optional): agnostic to width-height. Defaults to False.
        multi_label (bool, optional): apply Multi-Label NMS. Defaults to False.

    Returns:
        List[np.ndarray]: list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    return non_max_suppression_np(predictions,
                                  conf_thres,
                                  iou_thres,
                                  agnostic,
                                  multi_label,
                                  nms=tf.image.non_max_suppression)
