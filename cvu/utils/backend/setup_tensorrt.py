import tensorrt as trt

__version__ = trt.__version__


def is_gpu_available():
    return True
