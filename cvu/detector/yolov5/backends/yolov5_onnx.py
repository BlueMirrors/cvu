"""This file contains Yolov5's IModel implementation in ONNX.
This model (onnx-backend) performs inference using ONNXRUNTIME,
on a given input numpy array, and returns result after performing
nms and other backend specific postprocessings.

Model expects normalized inputs (data-format=channels-first) with
batch axis. Model does not apply letterboxing to given inputs.
"""
import os

import numpy as np
import onnxruntime

from cvu.interface.model import IModel
from cvu.utils.general import get_path
from cvu.detector.yolov5.backends.common import download_weights
from cvu.postprocess.nms.yolov5 import non_max_suppression_np


class Yolov5(IModel):
    """Implements IModel for Yolov5 using ONNX.

    This model (onnx-backend) performs inference, using ONNXRUNTIME,
    on a numpy array, and returns result after performing NMS.

    Inputs are expected to be normalized in channels-first order
    with/without batch axis.
    """
    def __init__(self, weight: str = "yolov5s", device: str = 'auto') -> None:
        """Initiate Model

        Args:
            weight (str, optional): path to onnx weight file. Alternatively,
            it also accepts identifiers (such as yolvo5s, yolov5m, etc.) to load
            pretrained models. Defaults to "yolov5s".

            device (str, optional): name of the device to be used. Valid devices can be
            "cpu", "gpu", "cuda", "auto". Defaults to "auto" which tries to use the device
            best suited for selected backend and the hardware avaibility.
        """
        self._model = None
        self._device = device

        self._load_model(weight)

    def _load_model(self, weight: str) -> None:
        """Internally loads ONNX

        Args:
            weight (str): path to ONNX weight file or predefined-identifiers
            (such as yolvo5s, yolov5m, etc.)
        """
        # TODO - Support for GPU

        # attempt to load predefined weights
        if not os.path.exists(weight):

            # get path to pretrained weights
            weight += '.onnx'
            weight = get_path(__file__, "weights", weight)

            # download weights if not already downloaded
            download_weights(weight, "onnx")

        # load model
        self._model = onnxruntime.InferenceSession(weight, None)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Performs model inference on given inputs, and returns
        inference's output after NMS.

        Args:
            inputs (np.ndarray): normalized in channels-first format,
            with batch axis.

        Returns:
            np.ndarray: inference's output after NMS
        """
        outputs = self._model.run([self._model.get_outputs()[0].name],
                                  {self._model.get_inputs()[0].name: inputs})
        return self._postprocess(outputs)[0]

    def __repr__(self) -> str:
        """Returns Model Information

        Returns:
            str: information string
        """
        return f"Yolov5s ONNX-{self._device}"

    @staticmethod
    def _postprocess(outputs: np.ndarray) -> np.ndarray:
        """Post-process outputs from model inference.
            - Non-Max-Supression

        Args:
            outputs (np.ndarray): model inference's output

        Returns:
            np.ndarray: post-processed output
        """
        # apply nms
        outputs = non_max_suppression_np(outputs[0])
        return outputs
