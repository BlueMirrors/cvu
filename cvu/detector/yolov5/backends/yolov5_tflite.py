import os

import numpy as np
import tensorflow.lite as tflite

from cvu.interface.model import IModel
from cvu.utils.general import get_path
from cvu.detector.yolov5.backends.common import download_weights
from cvu.postprocess.backend_tf.nms.yolov5 import non_max_suppression_tf


class Yolov5(IModel):
    def __init__(self, weight: str = "yolov5s", device='auto') -> None:
        self._model = None
        self._input_details = None
        self._output_details = None
        self._input_shape = None
        self._device = None

        self._set_device(device)
        self._load_model(weight)

    def _set_device(self, device):
        # gpus = len(tf.config.list_physical_devices('GPU'))

        # if (device == 'auto' or device == 'cuda') and gpus > 0:
        #     self._device = 'cuda:0'
        #     mixed_precision.set_global_policy('mixed_float16')
        #     return

        # if device == 'tpu':
        #     try:
        #         tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        #         tf.config.experimental_connect_to_cluster(tpu)
        #         tf.tpu.experimental.initialize_tpu_system(tpu)

        #         print(f"[CVU-Info] Backend: Tensorflow-{tf.__version__}-tpu")
        #         self._device = 'tpu'
        #         return

        #     except:
        #         print("[CVU-Error] Not connected to a TPU runtime",
        #               "reverting to CPU")

        self._device = 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def _load_model(self, weight):
        if not os.path.exists(weight):
            weight += '_tflite'
            weight = get_path(__file__, "weights", weight)
            download_weights(weight, "tflite")

        # load_options = None
        # if self._device == 'tpu':
        #     load_options = tf.saved_model.SaveOptions(
        #         experimental_io_device="/job:localhost")

        self._model = tflite.Interpreter(model_path=weight)
        self._input_details = self._model.get_input_details()
        self._output_details = self._model.get_output_details()

    def _set_input_shape(self, input_shape):
        self._model.resize_tensor_input(self._input_details[0]["index"],
                                        input_shape)
        self._model.allocate_tensors()
        self._input_shape = input_shape

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if self._input_shape != inputs.shape:
            self._set_input_shape(inputs.shape)
        self._model.set_tensor(self._input_details[0]["index"], inputs)
        self._model.invoke()
        outputs = self._model.get_tensor(self._output_details[0]["index"])
        return self._postprocess(outputs)[0]

    def __repr__(self) -> str:
        return f"Yolov5: {self._device}"

    def _postprocess(self, outputs):
        # apply nms
        outputs = non_max_suppression_tf(outputs)
        return outputs