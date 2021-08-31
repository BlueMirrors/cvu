"""This file contains relevant util functions needed for TFLite backend setup.
Currently powered by tensorflow.lite
"""
import tflite_runtime.interpreter as tflite

__version__ = None


def is_gpu_available() -> bool:
    """Check if GPU is available

    Returns:
        bool: True if GPU is available, False otherwise
    """
    return False
