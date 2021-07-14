from sys import version
import tflite

__version__ = tflite.__version__


def is_gpu_available():
    return False
