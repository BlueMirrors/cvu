from typing import Any, List, OrderedDict, Tuple, Union
from importlib import import_module
from cvu.detector.predictions import Predictions

import numpy as np

from cvu.preprocess.image.general import basic_preprocess, hwc_to_chw
from .core import Yolov5


class Yolov5End2End(Yolov5):
    def __init__(
        self,
        weight: str,
        classes: Union[str, List[str]] = "coco",
        backend: str = "tensorrt",
        input_shape: Tuple[int, int] = (640, 640),
        dtype: str = "fp16",
    ) -> None:
        assert backend in ["tensorrt"], f"{backend} backend not supported."
        super().__init__(
            classes=classes, backend=backend, weight=weight, input_shape=input_shape, dtype=dtype
        )

    @staticmethod
    def _scale(
        original_shape: Tuple[int], process_shape: Tuple[int], outputs: OrderedDict[str, np.ndarray]
    ) -> OrderedDict[str, np.ndarray]:
        outputs["det_boxes"] = Yolov5._scale(
            original_shape, process_shape, outputs["det_boxes"]
        )
        return outputs
    
    def _load_model(self, backend_name: str, weight: str, device: str, **kwargs: Any) -> None:
        backend = import_module(f".yolov5_{backend_name}_end2end", self._BACKEND_PKG)
        self._model = backend.Yolov5(weight, num_classes=len(self._classes), **kwargs)

        # add preprocess
        self._preprocess.append(hwc_to_chw)
        self._preprocess.append(np.ascontiguousarray)
        self._preprocess.append(basic_preprocess)
        
    def _to_preds(self, outputs: OrderedDict[str, np.ndarray]) -> Predictions:
        # create container
        preds = Predictions()

        num_dets = outputs["num_dets"][0]
        boxes, scores, classes = (
            outputs["det_boxes"][:num_dets],
            outputs["det_scores"][:num_dets],
            outputs["det_classes"][:num_dets],
        )

        # add detection
        for xyxy, conf, class_id in zip(boxes, scores, classes):
            # filter class
            if class_id in self._classes:
                preds.create_and_append(
                    xyxy, conf, class_id, class_name=self._classes[class_id]
                )
        
        return preds
