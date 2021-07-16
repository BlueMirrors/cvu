import torch

__version__ = torch.__version__


def is_gpu_available():
    return torch.cuda.is_available()
