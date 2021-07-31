"""This file contains various drawing utility functions.
"""
from typing import Optional, Tuple

import numpy as np
import cv2

from cvu.utils.colors import random_color


def draw_bbox(image: np.ndarray,
              bbox: np.ndarray,
              title: Optional[str] = None,
              color: Optional[Tuple[int]] = None,
              thickness: Optional[int] = 3) -> None:
    """Draw Bounding Box on the given image (inplace)

    Args:
        image (np.ndarray): image to draw on
        bbox (np.ndarray): coordinates of bbox top-left and right-bottom (x1,y1,x2,y2)
        title (Optional[str], optional): title of the drawn box. Defaults to None.
        color (Optional[Tuple[int]], optional): color of the box. Defaults to None (random color)
        thickness (Optional[int], optional): thickness of the box. Defaults to 2.
    """
    # generate random color
    if color is None:
        color = random_color()

    # convert cordinates to int
    x1, y1, x2, y2 = map(int, bbox[:4])

    # add title
    if title:
        scale = min(image.shape[0], image.shape[1]) / (720 / 0.9)
        text_size = cv2.getTextSize(title, 0, fontScale=scale, thickness=1)[0]
        top_left = (x1 - thickness + 1, y1 - text_size[1] - 20)
        bottom_right = (x1 + text_size[0] + 5, y1)

        cv2.rectangle(image, top_left, bottom_right, color=color, thickness=-1)
        cv2.putText(image, title, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 2)

    # add box
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
