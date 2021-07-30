"""This file contains relevant util functions needed for Torch backend setup.
"""
import torch

__version__ = torch.__version__


def is_gpu_available() -> bool:
    """Check if GPU is available

    Returns:
        bool: True if GPU is available, False otherwise
    """
    return torch.cuda.is_available()
