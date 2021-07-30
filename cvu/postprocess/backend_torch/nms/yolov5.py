"""Original Code Taken From ultralytics/yolov5
URL: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
"""
import time
from typing import List
import torch
import torchvision

from cvu.utils.backend_torch.bbox import xywh2xyxy


def non_max_suppression_torch(predictions: torch.Tensor,
                              conf_thres: float = 0.25,
                              iou_thres: float = 0.45,
                              agnostic: bool = False,
                              multi_label: bool = False) -> List[torch.Tensor]:
    """Runs Non-Maximum Suppression (NMS) on inference results

    Args:
        predictions (torch.Tensor): predictions from yolov inference

        conf_thres (float, optional): confidence threshold in range 0-1.
        Defaults to 0.25.

        iou_thres (float, optional): IoU threshold in range 0-1 for NMS filtering.
        Defaults to 0.45.

        agnostic (bool, optional):  agnostic to width-height. Defaults to False.

        multi_label (bool, optional): apply Multi-Label NMS. Defaults to False.

    Returns:
        List[torch.Tensor]: list of detections,on (n,6) tensor per
        image [xyxy, conf, cls]
    """
    # classes=None #(commented out for later use)
    # Settings
    maximum_detections = 300
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after

    # number of classes > 1 (multiple labels per box (adds 0.5ms/img))
    multi_label &= (predictions.shape[2] - 5) > 1

    # setup
    start_time = time.time()
    confidences = predictions[..., 4] > conf_thres
    output = [torch.zeros(
        (0, 6), device=predictions.device)] * predictions.shape[0]

    # image index, image inference
    for batch_index, prediction in enumerate(predictions):

        # confidence
        prediction = prediction[confidences[batch_index]]

        # If none remain process next image
        if not prediction.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        prediction = detection_matrix(prediction, multi_label, conf_thres)

        # Filter by class
        # if classes is not None:
        #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape # number of boxes
        if not prediction.shape[0]:  # no boxes
            continue

        # excess boxes
        if prediction.shape[0] > max_nms:
            # sort by confidence
            prediction = prediction[prediction[:, 4].argsort(
                descending=True)[:max_nms]]

        # Batched NMS
        classes = prediction[:, 5:6] * (0 if agnostic else max_wh)  # classes
        indexes = torchvision.ops.nms(prediction[:, :4] + classes,
                                      prediction[:, 4], iou_thres)

        # limit detections
        if indexes.shape[0] > maximum_detections:
            indexes = indexes[:maximum_detections]

        # pick relevant boxes
        output[batch_index] = prediction[indexes, :]

        # check if time limit exceeded
        if (time.time() - start_time) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def detection_matrix(predictions: torch.Tensor, multi_label: bool,
                     conf_thres: float) -> torch.Tensor:
    """Prepare Detection Matrix for Yolov5 NMS

    Args:
        predictions (torch.Tensor): one batch of predictions from yolov inference.
        multi_label (bool): apply Multi-Label NMS.
        conf_thres (float): confidence threshold in range 0-1.

    Returns:
        torch.Tensor: detections matrix nx6 (xyxy, conf, cls).
    """

    # Compute conf = obj_conf * cls_conf
    predictions[:, 5:] *= predictions[:, 4:5]

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(predictions[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = (predictions[:, 5:] > conf_thres).nonzero(as_tuple=False).T
        predictions = torch.cat(
            (box[i], predictions[i, j + 5, None], j[:, None].float()), 1)

    # best class only
    else:
        conf, j = predictions[:, 5:].max(1, keepdim=True)

        predictions = torch.cat((box, conf, j.float()),
                                1)[conf.view(-1) > conf_thres]

    return predictions
