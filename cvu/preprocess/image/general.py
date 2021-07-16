"""Contains genearl preprocess functions for images.
 - BGR to RGB Images (bgr_to_rgb)
 - Channels Last to Channels First (hwc_to_whc)
 - Normalize Image (normalize)
"""
import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB image.
    BGR = Blue-Green-Red
    RGB = Red-Green-Blue

    Args:
        image (np.ndarray): BGR format image.

    Returns:
        np.ndarray: RGB format image.
    """
    return image[..., ::-1]


def hwc_to_whc(image: np.ndarray) -> np.ndarray:
    """Convert channels-last to channels-first
    channels-last = Height-Width-Channels
    channels-first = Channels-Height-Width

    Args:
        image (np.ndarray): HWC format image.

    Returns:
        np.ndarray: CHW format image.
    """
    return image.transpose(2, 0, 1)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image pixels to be in the 0-1 range.

    Args:
        image (np.ndarray): Image with pixels in range 0-255

    Returns:
        np.ndarray: Normalized image with pixels in range 0-1
    """
    return image / 255.0
