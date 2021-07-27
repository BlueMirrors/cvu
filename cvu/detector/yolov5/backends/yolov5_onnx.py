import os
import onnxruntime
import numpy as np

from cvu.interface.model import IModel
from cvu.utils.general import load_json, get_path
from cvu.utils.google_utils import gdrive_download
from cvu.postprocess.nms.yolov5 import non_max_suppression_np


class Yolov5(IModel):
    def __init__(self, weight: str = "yolov5s", device=None) -> None:
        self._model = None

        self._load_model(weight)
    
    def _load_model(self, weight: str) -> None:
        # TODO - Support for GPU
        if not os.path.exists(weight):
            weight += '_onnx'
            weight = get_path(__file__, "weights", weight)
            self._download_weights(weight)

        self._model = onnxruntime.InferenceSession(weight, None)
    
    def _download_weights(self, weight):
        if os.path.exists(weight): return

        weight_key = os.path.split(weight)[-1]
        weights_json = get_path(__file__, "weights", "onnx_weights.json")
        available_weights = load_json(weights_json)
        if weight_key not in available_weights:
            raise FileNotFoundError
        gdrive_download(available_weights[weight_key], weight)
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray: 
        inputs = self._preprocess(inputs)
        outputs = self._model.run([self._model.get_outputs()[0].name], {self._model.get_inputs()[0].name: inputs})
        return self._postprocess(outputs)[0]
    
    def __repr__(self) -> str:
        return f"Yolov5s ONNX"

    def _preprocess(self, inputs: np.ndarray) -> np.ndarray:
        # normalize image
        inputs = inputs.astype("float32")
        inputs /= 255

        # add batch if not present
        if inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=0)
        
        return inputs
    
    def _postprocess(self, outputs: np.ndarray) -> np.ndarray:
        # apply nms
        outputs = non_max_suppression_np(outputs[0])
        return outputs