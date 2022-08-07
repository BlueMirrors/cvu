"""This file contains Yolov5's ICore implementation.

Yolov5 Core represents a common interface over which users can perform
Yolov5 powered object detection, without having to worry about the details of
different backends and their specific requirements and implementations. Core
will internally resolve and handle all internal details.

Find more about Yolov5 here from their official repository
https://github.com/ultralytics/yolov5
"""
from typing import Any, Callable, Tuple, Union, List
from importlib import import_module

import numpy as np

from cvu.interface.core import ICore
from cvu.detector.predictions import Predictions
from cvu.detector.configs import COCO_CLASSES
from cvu.preprocess.image.letterbox import letterbox
from cvu.preprocess.image.general import (basic_preprocess, bgr_to_rgb,
                                          hwc_to_chw)
from cvu.postprocess.bbox import scale_coords
from cvu.utils.backend import setup_backend


class Yolov5(ICore):
    """Implements ICore for Yolov5

    Yolov5 Core represents a common interface to perform
    Yolov5 powered object detection.
    """
    _BACKEND_PKG = "cvu.detector.yolov5.backends"

    def __init__(self,
                 classes: Union[str, List[str]],
                 backend: str = "torch",
                 weight: str = "yolov5s",
                 device: str = "auto",
                 auto_install: bool = False,
                 **kwargs: Any) -> None:
        """Initiate Yolov5 Object Detector

        Args:
            classes (Union[str, List[str]]): name of classes to be detected.
            It can be set to individual classes like 'coco', 'person', 'cat' etc.
            Alternatively, it also accepts list of classes such as ['person', 'cat'].
            For default models/weights, 'classes' is used to filter out objects
            according to provided argument from coco class. For custom models, all
            classes should be provided in original order as list of strings.

            backend (str, optional): name of the backend to be used for inference purposes.
            Defaults to "torch".

            weight (str, optional): path to weight files (according to selected backend).
            Alternatively, it also accepts identifiers (such as yolvo5s, yolov5m, etc.)
            to load pretrained models. Defaults to "yolov5s".

            device (str, optional): name of the device to be used. Valid
            devices can be "cpu", "gpu", "tpu", "auto". Defaults to "auto" which tries
            to use the device best suited for selected backend and the hardware avaibility.

            auto_install (bool, optional): auto install missing requirements for the selected
            backend.
        """
        # ICore
        super().__init__(classes, backend)

        # initiate class attributes
        if kwargs.get("input_shape", None) is not None:
            self._preprocess = [lambda image: letterbox(image, kwargs['input_shape'], auto=False),
                                bgr_to_rgb]
        else:
            self._preprocess = [letterbox, bgr_to_rgb]
        self._postprocess = []
        self._classes = {}
        self._model = None

        # setup backend and load model
        if auto_install:
            setup_backend(backend, device)
        self._load_classes(classes)
        self._load_model(backend, weight, device, **kwargs)

    def __repr__(self) -> str:
        """Returns Backend and Model Information

        Returns:
            str: information string
        """
        return str(self._model)

    def __call__(self, inputs: np.ndarray, **kwargs) -> Predictions:
        """Performs Yolov5 Object Detection on given inputs.
        Returns detected objects as Predictions object.

        Args:
            inputs (np.ndarray): image in BGR format.

        Returns:
            Predictions: detected objects.
        """
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

    @staticmethod
    def _apply(
            value: np.ndarray,
            functions: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
        """Recursively applies list of callable functions to given value

        Args:
            value (np.ndarray): input to be processed

            functions (List[Callable[[np.ndarray], np.ndarray]]): list of
            callable functions.

        Returns:
            np.ndarray: value resulting from applying all functions
        """
        for func in functions:
            value = func(value)
        return value

    @staticmethod
    def _scale(original_shape: Tuple[int], process_shape: Tuple[int],
               outputs: np.ndarray) -> np.ndarray:
        """Scale outputs based on process_shape to original_shape

        Args:
            original_shape (Tuple[int]): shape of original inputs.
            process_shape (Tuple[int]): shape of processed inputs.
            outputs (np.ndarray): outputs from yolov5 model

        Returns:
            np.ndarray: scaled outputs
        """
        # channels first, pick widht-height accordingly
        if len(process_shape) > 3:
            process_shape = process_shape[1:]

        if process_shape[2] != 3:
            process_shape = process_shape[1:]

        # scale bounding box
        outputs[:, :4] = scale_coords(process_shape[:2], outputs[:, :4],
                                      original_shape).round()
        return outputs

    def _load_model(self, backend_name: str, weight: str, device: str, **kwargs: Any) -> None:
        """Internally loads Model (backend)

        Args:
            backend_name (str): name of the backend
            weight (str): path to weight file or default identifiers
            device (str): name of target device (auto, cpu, gpu, tpu)
        """
        # load model
        backend = import_module(f".yolov5_{backend_name}", self._BACKEND_PKG)

        if backend_name != 'tensorrt':
            self._model = backend.Yolov5(weight, device)
        else:
            self._model = backend.Yolov5(weight,
                                         num_classes=len(self._classes),
                                         **kwargs)

        # add preprocess
        if backend_name in ['torch', 'onnx', 'tensorrt']:
            self._preprocess.append(hwc_to_chw)

        # contigousarray
        self._preprocess.append(np.ascontiguousarray)

        # add common preprocess
        if backend_name in ['onnx', 'tensorflow', 'tflite', 'tensorrt']:
            self._preprocess.append(basic_preprocess)

    def _load_classes(self, classes: Union[str, List[str]]) -> None:
        """Internally loads target classes

        Args:
            classes (Union[str, List[str]]): name or list of classes to be detected.
        """
        if classes == 'coco':
            classes = COCO_CLASSES

        elif isinstance(classes, str):
            classes = [classes]

        if set(classes).issubset(COCO_CLASSES):
            for i, name in enumerate(COCO_CLASSES):
                if name in classes:
                    self._classes[i] = name
        else:
            self._classes = dict(enumerate(classes))

    def _to_preds(self, outputs: np.ndarray) -> Predictions:
        """Convert Yolov5's numpy inputs to Predictions object.

        Args:
            outputs (np.ndarray): basic outputs from yolov5 inference.

        Returns:
            Predictions: detected objects
        """
        # create container
        preds = Predictions()

        # add detection
        for *xyxy, conf, class_id in outputs:
            # filter class
            if class_id in self._classes:
                preds.create_and_append(xyxy,
                                        conf,
                                        class_id,
                                        class_name=self._classes[class_id])
        return preds
