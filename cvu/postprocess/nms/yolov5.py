"""Numpy Implementation of Yolov5 NMS.
Original Code Taken From ultralytics/yolov5
URL: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
"""
import time
from typing import List, Callable

import numpy as np

from cvu.utils.bbox import xywh2xyxy
from cvu.postprocess.nms import nms_np


def non_max_suppression_np(predictions: np.ndarray,
                           conf_thres: float = 0.25,
                           iou_thres: float = 0.45,
                           agnostic: bool = False,
                           multi_label: bool = False,
                           nms: Callable = nms_np) -> List[np.ndarray]:
    """Runs Non-Maximum Suppression (NMS used in Yolov5) on inference results.

    Args:
        predictions (np.ndarray): predictions from yolov inference

        conf_thres (float, optional): confidence threshold in range 0-1.
        Defaults to 0.25.

        iou_thres (float, optional): IoU threshold in range 0-1 for NMS filtering.
        Defaults to 0.45.

        agnostic (bool, optional): Perform class-agnostic NMS. Defaults to False.

        multi_label (bool, optional): apply Multi-Label NMS. Defaults to False.

        nms (Callable[[np.ndarray, np.ndarray, int, float], List[np.ndarray]]): Base NMS
        function to be applied. Defaults to nms_np.

    Returns:
        List[np.ndarray]: list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Settings
    maximum_detections = 300
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after

    # number of classes > 1 (multiple labels per box (adds 0.5ms/img))
    multi_label &= (predictions.shape[2] - 5) > 1

    start_time = time.time()
    output = [np.zeros((0, 6))] * predictions.shape[0]
    confidences = predictions[..., 4] > conf_thres

    # image index, image inference
    for batch_index, prediction in enumerate(predictions):

        # confidence
        prediction = prediction[confidences[batch_index]]

        # If none remain process next image
        if not prediction.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        prediction = detection_matrix(prediction, multi_label, conf_thres)

        # Check shape; # number of boxes
        if not prediction.shape[0]:  # no boxes
            continue

        # excess boxes
        if prediction.shape[0] > max_nms:
            prediction = prediction[np.argpartition(-prediction[:, 4],
                                                    max_nms)[:max_nms]]

        # Batched NMS
        classes = prediction[:, 5:6] * (0 if agnostic else max_wh)
        indexes = nms(prediction[:, :4] + classes, prediction[:, 4],
                      maximum_detections, iou_thres)

        # pick relevant boxes
        output[batch_index] = prediction[indexes, :]

        # check if time limit exceeded
        if (time.time() - start_time) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def detection_matrix(predictions: np.ndarray, multi_label: bool,
                     conf_thres: float) -> np.ndarray:
    """Prepare Detection Matrix for Yolov5 NMS

    Args:
        predictions (np.ndarray): one batch of predictions from yolov inference.
        multi_label (bool): apply Multi-Label NMS.
        conf_thres (float): confidence threshold in range 0-1.

    Returns:
        np.ndarray: detections matrix nx6 (xyxy, conf, cls).
    """

    # Compute conf = obj_conf * cls_conf
    predictions[:, 5:] *= predictions[:, 4:5]

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(predictions[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = (predictions[:, 5:] > conf_thres).nonzero().T
        predictions = np.concatenate(
            (box[i], predictions[i, j + 5, None], j[:, None].astype('float')),
            1)

    # best class only
    else:
        j = np.expand_dims(predictions[:, 5:].argmax(axis=1), axis=1)
        conf = np.take_along_axis(predictions[:, 5:], j, axis=1)

        predictions = np.concatenate((box, conf, j.astype('float')),
                                     1)[conf.reshape(-1) > conf_thres]

    return predictions
