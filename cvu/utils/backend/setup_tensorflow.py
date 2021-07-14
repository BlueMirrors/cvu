import tensorflow as tf

__version__ = tf.__version__


def is_gpu_available():
    return bool(tf.config.list_physical_devices('GPU'))
