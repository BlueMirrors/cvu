"""Original Code Taken From ultralytics/yolov5
URL: https://github.com/ultralytics/yolov5/blob/master/utils/general.py

Contains general Bound-Box postprocessing functions.
 - Scale coordinates (scale_coords)
 - Clip coordinates (clip_coords)
"""

from typing import Tuple, List

import numpy as np


def scale_coords(processed_shape: Tuple[int],
                 coords: List[int],
                 original_shape: Tuple[int],
                 ratio_pad: Tuple[int] = None) -> List[int]:
    """Rescale coords (xyxy) from processed_shape to original_shape
    Scale Coordinates according to image shape before pre-processing.

    Args:
        processed_shape (Tuple[int]): Processed-image shape.
        coords (List[int]): xyxy coordinates
        original_shape (Tuple[int]): Original-image shape.
        ratio_pad (Tuple[int], optional): Padding to achieve correct scaling.
        Defaults to None.

    Returns:
        List[int]: scaled xyxy cordinates
    """
    # calculate from original_shape
    if ratio_pad is None:
        # gain  = old / new
        gain = min(processed_shape[0] / original_shape[0],
                   processed_shape[1] / original_shape[1])

        # wh padding
        pad = ((processed_shape[1] - original_shape[1] * gain) / 2,
               (processed_shape[0] - original_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # x padding
    coords[:, [0, 2]] -= pad[0]

    # y padding
    coords[:, [1, 3]] -= pad[1]

    coords[:, :4] /= gain
    clip_coords(coords, original_shape)
    return coords


def clip_coords(boxes: List[int], img_shape: Tuple[int]) -> None:
    """Clip bounding xyxy bounding boxes to image shape (height, width)
    Clips values inplace.

    Args:
        boxes (List[int]): xyxy box
        img_shape (Tuple[int]): shape of the image for setting
        clipping limits.
    """
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def denormalize(outputs: np.ndarray, shape: Tuple[int]) -> None:
    """Denormalize outputs inplace

    Args:
        outputs (np.ndarray): inference's output
        shape (Tuple[int]): base for denormalization
    """
    outputs[..., 0] *= shape[1]  # x
    outputs[..., 1] *= shape[0]  # y
    outputs[..., 2] *= shape[1]  # w
    outputs[..., 3] *= shape[0]  # h
