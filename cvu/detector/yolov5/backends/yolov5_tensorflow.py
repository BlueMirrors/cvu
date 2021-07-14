import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from cvu.utils.google_utils import gdrive_download
from cvu.interface.model import IModel
from cvu.utils.general import load_json, get_path
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
            self._download_weights(weight)

        load_options = None
        if self._device == 'tpu':
            load_options = tf.saved_model.SaveOptions(
                experimental_io_device="/job:localhost")

        self._loaded = tf.saved_model.load(weight, options=load_options)
        self._model = self._loaded.signatures["serving_default"]

    def _download_weights(self, weight):
        if os.path.exists(weight): return

        weight_key = os.path.split(weight)[-1]
        weights_json = get_path(__file__, "weights", "tensorflow_weights.json")
        available_weights = load_json(weights_json)
        if weight_key not in available_weights:
            raise FileNotFoundError
        gdrive_download(available_weights[weight_key], weight, unzip=True)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        processed_inputs = self._preprocess(inputs)
        outputs = self._model(input_1=processed_inputs)['tf__detect'].numpy()
        self._denormalize(outputs, inputs.shape)
        return self._postprocess(outputs)[0]

    def __repr__(self) -> str:
        return f"Yolov5: {self._device}"

    def _denormalize(self, outputs, shape):
        outputs[..., 0] *= shape[1]  # x
        outputs[..., 1] *= shape[0]  # y
        outputs[..., 2] *= shape[1]  # w
        outputs[..., 3] *= shape[0]  # h

    def _preprocess(self, inputs):
        # normalize image
        inputs = inputs.astype('float32')
        inputs /= 255.

        # add batch axis if not present
        if inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=0)
        return inputs

    def _postprocess(self, outputs):
        # apply nms
        outputs = non_max_suppression_tf(outputs)
        return outputs
