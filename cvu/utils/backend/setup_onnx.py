"""This file contains relevant util functions needed for ONNX backend setup.
"""
import onnxruntime

__version__ = onnxruntime.__version__


def is_gpu_available() -> bool:
    """Check if GPU is available

    Returns:
        bool: True if GPU is available, False otherwise
    """
    return onnxruntime.get_device().lower() != 'cpu'
