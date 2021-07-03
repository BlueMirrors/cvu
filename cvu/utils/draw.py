from typing import Optional, Tuple
import random
import numpy as np
import cv2


def random_color():
    return list(np.random.random(size=3) * 256)


def draw_bbox(image: np.ndarray,
              bbox: np.ndarray,
              title: Optional[str] = None,
              color: Optional[Tuple[int]] = None,
              thickness: Optional[int] = 2):

    if color is None:
        color = random_color()

    x1, y1, x2, y2 = map(int, bbox[:4])

    if title:
        cv2.putText(image, title, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    min(image.shape[0], image.shape[1]) / (25 / 0.9), color, 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
