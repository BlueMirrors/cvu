"""This file contains relevant util functions needed for TFLite backend setup.
Currently powered by tensorflow.lite
"""
import tensorflow as tf

__version__ = tf.__version__


def is_gpu_available() -> bool:
    """Check if GPU is available

    Returns:
        bool: True if GPU is available, False otherwise
    """
    return False
