"""This file contains various color utility functions.
"""
from typing import Tuple
import random

# initially taken from Ultralytics color palette https://ultralytics.com/
RGB_PALETTE = ((255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
               (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
               (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
               (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
               (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199))


def random_color(bgr: bool = True) -> Tuple[int]:
    """Return a random RGB/BGR Color

    Args:
        bgr (bool, optional): whether to return bgr color or rgb.
        Defaults to True.

    Returns:
        Tuple[int]: list of 3 ints representing Color
    """
    color = random.choice(RGB_PALETTE)
    return color[::-1] if bgr else color
