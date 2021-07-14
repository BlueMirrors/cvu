"""Original Code Taken From ultralytics/yolov5
URL: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
"""
import time

import numpy as np

from cvu.utils.bbox import xywh2xyxy
from cvu.postprocess.nms import nms_np


def non_max_suppression_np(prediction,
                           conf_thres=0.25,
                           iou_thres=0.45,
                           agnostic=False,
                           multi_label=False,
                           max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # confidence
        x = x[xc[xi]]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf = obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().T
            x = np.concatenate(
                (box[i], x[i, j + 5, None], j[:, None].astype('float')), 1)

        # best class only
        else:
            j = np.expand_dims(x[:, 5:].argmax(axis=1), axis=1)
            conf = np.take_along_axis(x[:, 5:], j, axis=1)

            x = np.concatenate((box, conf, j.astype('float')),
                               1)[conf.reshape(-1) > conf_thres]

        # Check shape; # number of boxes
        n = x.shape[0]

        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[np.argpartition(-x[:, 4], max_nms)[:max_nms]]

        # Batched NMS

        # classes
        c = x[:, 5:6] * (0 if agnostic else max_wh)

        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]

        # nms
        i = nms_np(boxes, scores, iou_thres, max_det)

        # limit detections
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i, :]
        if (time.time() - t) > time_limit:
            # time limit exceeded
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output
