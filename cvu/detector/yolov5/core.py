from typing import Union, List
from importlib import import_module

import numpy as np

from cvu.interface.core import ICore
from cvu.detector.predictions import Predictions
from cvu.preprocess.image.letterbox import letterbox
from cvu.preprocess.image.general import bgr_to_rgb, hwc_to_whc
from cvu.postprocess.bbox import scale_coords
from cvu.utils.backend import setup_backend


class Yolov5(ICore):
    _BACKEND_PKG = "cvu.detector.yolov5.backends"

    def __init__(self,
                 classes: Union[str, List[str]],
                 backend: str = "torch",
                 weight: str = "yolov5s",
                 device: str = "auto") -> None:

        self._preprocess = [letterbox, bgr_to_rgb]
        self._postprocess = []
        self._classes = classes
        self._model = None

        # setup backend and load model
        setup_backend(backend, device if device != "auto" else None)
        self._load_model(backend, weight, device)

    def __call__(self, inputs):
        # preprocess
        processed_inputs = self._apply(inputs, self._preprocess)

        # inference on backend
        outputs = self._model(processed_inputs)

        # postprocess
        outputs = self._apply(outputs, self._postprocess)

        # scale up
        outputs = self._scale(inputs.shape, processed_inputs.shape, outputs)

        # convert to preds
        return self._to_preds(outputs)

    def _scale(self, original_shape, process_shape, outputs):
        if process_shape[2] != 3:
            process_shape = process_shape[1:]

        outputs[:, :4] = scale_coords(process_shape[:2], outputs[:, :4],
                                      original_shape).round()
        return outputs

    def _to_preds(self, outputs):
        preds = Predictions()
        for *xyxy, conf, class_id in reversed(outputs):
            preds.create_and_append(xyxy, conf, class_id)
        return preds

    def _apply(self, value, functions):
        for func in functions:
            value = func(value)
        return value

    def _load_model(self, backend_name, weight, device):
        # load model
        backend = import_module(f".yolov5_{backend_name}", self._BACKEND_PKG)
        self._model = backend.Yolov5(weight, device)

        # add preprocess
        if backend_name in ['torch', 'onnx', 'tensorrt']:
            self._preprocess.append(hwc_to_whc)

        # contigousarray
        self._preprocess.append(np.ascontiguousarray)

    def __repr__(self):
        return str(self._model)
