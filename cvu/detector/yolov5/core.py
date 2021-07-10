from typing import Union, List

import numpy as np

from cvu.interface.core import ICore
from cvu.detector.predictions import Predictions
from cvu.preprocess.image.letterbox import letterbox
from cvu.preprocess.image.general import bgr_to_rgb, hwc_to_whc
from cvu.postprocess.bbox import scale_coords


class Yolov5(ICore):
    def __init__(self,
                 classes: Union[str, List[str]],
                 backend: str = "torch-jit",
                 weight: str = "yolov5s",
                 device: str = "auto",
                 inplace: bool = False) -> None:

        self._configs = {
            'inplace': inplace,
            'backend': backend,
            'classes': classes,
            'preprocess': [],
            'postprocess': []
        }

        self._model = None

        self._load_backend(backend, weight, device)

    def __call__(self, inputs):
        if not self._configs['inplace']:
            inputs = inputs.copy()

        original_shape = inputs.shape[:2]
        inputs = self._apply(inputs, self._configs['preprocess'])
        processed_shape = (inputs.shape[:2]
                           if inputs.shape[2] == 3 else inputs.shape[1:])

        outputs = self._model(inputs)

        outputs = self._apply(outputs, self._configs['postprocess'])
        outputs[:, :4] = scale_coords(processed_shape, outputs[:, :4],
                                      original_shape).round()

        return self._to_preds(outputs)

    def _to_preds(self, outputs):
        preds = Predictions()
        for *xyxy, conf, class_id in reversed(outputs):
            preds.create_and_append(xyxy, conf, class_id)
        return preds

    def _apply(self, value, functions):
        for func in functions:
            value = func(value)
        return value

    def _load_backend(self, backend, weight, device):
        preprocess = [letterbox, bgr_to_rgb]
        postprocess = []

        if backend == 'torch-jit':
            from .backends.yolov5_torch import Yolov5
            self._model = Yolov5(weight, device)

            # add preprocess
            preprocess.append(hwc_to_whc)

        # contigousarray
        preprocess.append(np.ascontiguousarray)

        self._configs['preprocess'] = preprocess
        self._configs['postprocess'] = postprocess

    def __repr__(self):
        return str(self._model)
