"""This file contains Yolo's IDetectorModel implementation in TFLite.
This model (tflite-backend) performs inference using TFLite,
on a given input numpy array, and returns result after performing
nms and other backend specific postprocessings.

Model expects normalized inputs (data-format=channels-last) with batch
axis. Model does not apply letterboxing to given inputs.
"""
import os
from typing import Tuple, List

import numpy as np
import tensorflow.lite as tflite

from cvu.detector.interface import IDetectorModel
from cvu.postprocess.backend_tf.nms.yolov5 import non_max_suppression_tf


class Yolo(IDetectorModel):
    """Implements IDetectorModel for Yolo using TFLite.

    This model (tflite-backend) performs inference, using TFLite,
    on a numpy array, and returns result after performing NMS.

    Inputs are expected to be normalized in channels-last order with batch axis.
    """

    def __init__(self, weight: str, device='auto') -> None:
        """Initiate Model

        Args:
            weight (str, optional): path to TFLite weight files.

            device (str, optional): name of the device to be used. Valid devices can be
            "cpu", "auto". Defaults to "auto" which tries to use the
            device best suited for selected backend and the hardware avaibility.
        """
        # initiate class attributes
        self._model = None
        self._input_details = None
        self._output_details = None
        self._input_shape = None
        self._device = None

        # setup device
        self._set_device(device)

        # load SavedModel
        self._load_model(weight)

    def _set_device(self, device: str) -> None:
        """Internally setup/initiate necessary device

        Args:
            device (str): name of the device to be used.

        Raises:
            NotImplementedError: raised if invalid device is given.
        """
        if device not in ('cpu', 'auto'):
            raise NotImplementedError(
                "This Device is not yet supported with this backend,")

        # set device
        self._device = device

    def _load_model(self, weight: str) -> None:
        """Internally loads TFLite Model

        Args:
            weight (str): path to TFLite weight files
        """
        if not os.path.exists(weight):
            raise FileNotFoundError(f"Unable to locate model weights {weight}")

        # load model
        self._model = tflite.Interpreter(model_path=weight)  # pylint: disable=maybe-no-member
        self._input_details = self._model.get_input_details()
        self._output_details = self._model.get_output_details()

    def _set_input_shape(self, input_shape: Tuple[int]) -> None:
        """Resize Model's input tensors according to input_shape

        Args:
            input_shape (Tuple[int]): new target input shape
        """
        self._model.resize_tensor_input(self._input_details[0]["index"],
                                        input_shape)
        self._model.allocate_tensors()
        self._input_shape = input_shape

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Performs model inference on given inputs, and returns
        inference's output after NMS.

        Args:
            inputs (np.ndarray): normalized in channels-last format,
            with batch axis.

        Returns:
            np.ndarray: inference's output after NMS
        """
        # resize model's tensor input for new input shape
        if self._input_shape != inputs.shape:
            self._set_input_shape(inputs.shape)

        # inference
        self._model.set_tensor(self._input_details[0]["index"], inputs)
        self._model.invoke()

        # postprocess
        outputs = self._model.get_tensor(self._output_details[0]["index"])
        return self._postprocess(outputs)[0]

    def __repr__(self) -> str:
        """Returns Model Information

        Returns:
            str: information string
        """
        return f"Yolo TFLite-{self._device}"

    @staticmethod
    def _postprocess(outputs: np.ndarray) -> List[np.ndarray]:
        """Post-process outputs from model inference.
            - Non-Max-Supression

        Args:
            outputs (np.ndarray): model inference's output

        Returns:
            List[np.ndarray]: post-processed output
        """
        # apply nms
        outputs = non_max_suppression_tf(outputs)
        return outputs
