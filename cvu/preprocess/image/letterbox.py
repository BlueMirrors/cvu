"""Original Code Taken From ultralytics/yolov5
URL: https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
"""
from typing import Tuple

import numpy as np
import cv2


def letterbox(img: np.ndarray,
              new_shape: Tuple[int] = (640, 640),
              color: Tuple[int] = (114, 114, 114),
              auto: bool = True,
              scale_fill: bool = False,
              scaleup: bool = True,
              stride: bool = 32) -> np.ndarray:
    """Reshape image without affecting the aspect ratio by adding minimum
    letter box type borders, and fill the border area with gray or the
    specified color. Resize and pad image while meeting stride-multiple constraints

    Args:
        img (np.ndarray): original image
        new_shape (Tuple[int], optional): shape of output image. Defaults to (640, 640).
        color (Tuple[int], optional): color to be filled in borders. Defaults to (114, 114, 114).
        auto (bool, optional): pick minimum rectangle . Defaults to True.
        scale_fill (bool, optional): strech. Defaults to False.
        scaleup (bool, optional): scale up if needed. Defaults to True.
        stride (bool, optional): used for auto. Defaults to 32.

    Returns:
        np.ndarray: resulting image
    """

    # current shape [height, width]
    shape = img.shape[:2]

    # new shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # only scale down, do not scale up (for better test mAP)
    if not scaleup:
        scale_ratio = min(scale_ratio, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = (int(round(shape[1] * scale_ratio)),
                 int(round(shape[0] * scale_ratio)))

    # wh padding
    delta_width = new_shape[1] - new_unpad[0]
    delta_height = new_shape[0] - new_unpad[1]

    # minimum rectangle
    if auto:
        # update wh padding
        delta_width = np.mod(delta_width, stride)
        delta_height = np.mod(delta_height, stride)

    # stretch
    elif scale_fill:
        delta_width, delta_height = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

        # width, height ratios
        # ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # divide padding into 2 sides
    delta_width /= 2
    delta_height /= 2

    # resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # add border
    img = cv2.copyMakeBorder(img,
                             int(round(delta_height - 0.1)),
                             int(round(delta_height + 0.1)),
                             int(round(delta_width - 0.1)),
                             int(round(delta_width + 0.1)),
                             cv2.BORDER_CONSTANT,
                             value=color)
    return img
