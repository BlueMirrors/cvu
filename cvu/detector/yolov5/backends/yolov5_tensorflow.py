import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from cvu.interface.model import IModel
from cvu.utils.general import get_path
from cvu.detector.yolov5.backends.common import download_weights
from cvu.postprocess.bbox import denormalize
from cvu.postprocess.backend_tf.nms.yolov5 import non_max_suppression_tf


class Yolov5(IModel):
    def __init__(self, weight: str = "yolov5s", device='auto') -> None:
        self._model = None
        self._device = None
        self._loaded = None

        self._set_device(device)
        self._load_model(weight)

    def _set_device(self, device):
        gpus = len(tf.config.list_physical_devices('GPU'))

        if (device == 'auto' or device == 'cuda') and gpus > 0:
            self._device = 'cuda:0'
            mixed_precision.set_global_policy('mixed_float16')
            return

        if device == 'tpu':
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)

                print(f"[CVU-Info] Backend: Tensorflow-{tf.__version__}-tpu")
                self._device = 'tpu'
                return

            except:
                print("[CVU-Error] Not connected to a TPU runtime",
                      "reverting to CPU")

        self._device = 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def _load_model(self, weight):
        if not os.path.exists(weight):
            weight += '_tensorflow'
            weight = get_path(__file__, "weights", weight)
            download_weights(weight, "tensorflow", unzip=True)

        load_options = None
        if self._device == 'tpu':
            load_options = tf.saved_model.SaveOptions(
                experimental_io_device="/job:localhost")

        self._loaded = tf.saved_model.load(weight, options=load_options)
        self._model = self._loaded.signatures["serving_default"]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        outputs = self._model(input_1=inputs)['tf__detect'].numpy()
        denormalize(outputs, inputs.shape[-3:])
        return self._postprocess(outputs)[0]

    def __repr__(self) -> str:
        return f"Yolov5: {self._device}"

    def _postprocess(self, outputs):
        # apply nms
        outputs = non_max_suppression_tf(outputs)
        return outputs
