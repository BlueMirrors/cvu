import onnxruntime

__version__ = onnxruntime.__version__


def is_gpu_available():
    return onnxruntime.get_device().lower() != 'cpu'
