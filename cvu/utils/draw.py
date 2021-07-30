"""This file contains various drawing utility functions.
"""
from typing import (Optional, Tuple, List)

import numpy as np
import cv2


def random_color() -> List[float]:
    """Return a random RGB/BGR Color

    Returns:
        List[float]: list of 3 float representing Color
    """
    return list(np.random.random(size=3) * 256)  # pylint: disable=maybe-no-member


def draw_bbox(image: np.ndarray,
              bbox: np.ndarray,
              title: Optional[str] = None,
              color: Optional[Tuple[int]] = None,
              thickness: Optional[int] = 2) -> None:
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
        cv2.putText(image, title, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    min(image.shape[0], image.shape[1]) / (1280 / 0.9), color,
                    2)

    # add box
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
