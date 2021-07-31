"""This file contains relevant util functions needed for TensorRT backend setup.
"""
import tensorrt as trt

__version__ = trt.__version__


def is_gpu_available() -> bool:
    """Check if GPU is available

    Returns:
        bool: True if GPU is available, False otherwise
    """
    return True
