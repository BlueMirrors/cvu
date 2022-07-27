"""This file contains Yolov5's IModel implementation in PyTorch.
This model (torch-backend) performs inference using Torch-script,
on a given input numpy array, and returns result after performing
nms and other backend specific postprocessings.

Model expects unormalized inputs (data-format=channels-first) with/without
batch axis. Model does not apply letterboxing to given inputs.
"""
import os
from typing import List

import numpy as np
import torch

from cvu.interface.model import IModel
from cvu.utils.general import get_path
from cvu.detector.yolov5.backends.common import download_weights
from cvu.postprocess.backend_torch.nms.yolov5 import non_max_suppression_torch


class Yolov5(IModel):
    """Implements IModel for Yolov5 using PyTorch.

    This model (torch-backend) performs inference, using Torch-script,
    on a numpy array, and returns result after performing NMS.

    Inputs are expected to be unormalized in channels-first order
    (with/without batch axis).
    """

    def __init__(self, weight: str = "yolov5s", device='auto') -> None:
        """Initiate Model

        Args:
            weight (str, optional): path to jit-script .pt weight files. Alternatively,
            it also accepts identifiers (such as yolvo5s, yolov5m, etc.) to load
            pretrained models. Defaults to "yolov5s".

            device (str, optional): name of the device to be used. Valid devices can be
            "cpu", "gpu", "auto" or specific cuda devices such as
            "cuda:0" or "cuda:1" etc with auto_install False. Defaults to "auto" which tries
            to use the device best suited for selected backend and the hardware avaibility.
        """
        # initiate class attributes
        self._device = None
        self._model = None

        # setup device
        self._set_device(device)

        # load jit-model
        self._load_model(weight)

    def _set_device(self, device: str) -> None:
        """Internally setup torch.device

        Args:
            device (str): name of the device to be used.
        """
        if device in ('auto', 'gpu'):
            self._device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

    def _load_model(self, weight: str) -> None:
        """Internally loads JIT-Model

        Args:
            weight (str): path to jit-script .pt weight files or predefined-identifiers
            (such as yolvo5s, yolov5m, etc.)
        """
        # attempt to load predefined weights
        if not os.path.exists(weight):
            if self._device.type != 'cpu':
                weight += '.cuda'

            # get path to pretrained weights
            weight = get_path(__file__, "weights", f"{weight}.torchscript.pt")

            # download weights if not already downloaded
            download_weights(weight, "torch")

        # load model
        self._model = torch.jit.load(weight, map_location=self._device)

        # use FP16 if GPU is being used
        if self._device.type != 'cpu':
            self._model.half()

        # set model to eval mode
        self._model.eval()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Performs model inference on given inputs, and returns
        inference's output after NMS.

        Args:
            inputs (np.ndarray): unormalized in channels-first format,
            with/without batch axis.

        Returns:
            np.ndarray: inference's output after NMS
        """
        inputs = self._preprocess(inputs)
        with torch.no_grad():
            outputs = self._model(inputs)[0]
        return self._postprocess(outputs)[0].cpu().detach().numpy()

    def __repr__(self) -> str:
        """Returns Model Information

        Returns:
            str: information string
        """
        return f"Yolov5: {self._device.type}"

    def _preprocess(self, inputs: np.ndarray) -> torch.Tensor:
        """Process inputs for model inference.
            - Converts to torch tensor
            - FP16 conversion if GPU is getting used
            - Normalize
            - Add batch axis if not already present

        Args:
            inputs (np.ndarray): numpy array in channels-first format

        Returns:
            torch.Tensor: processed inputs
        """
        # create torch tensor
        inputs = torch.from_numpy(inputs).to(self._device)

        # use fp16 if available
        if self._device.type != 'cpu':
            inputs = inputs.half()
        else:
            inputs = inputs.float()

        # normalize image
        inputs /= 255.

        # add batch axis if not present
        if inputs.ndimension() == 3:
            inputs = inputs.unsqueeze(0)

        return inputs

    @staticmethod
    def _postprocess(outputs: torch.Tensor) -> List[torch.Tensor]:
        """Post-process outputs from model inference.
            - Non-Max-Supression

        Args:
            outputs (torch.Tensor): model inference's output

        Returns:
            List[torch.Tensor]: post-processed output
        """
        # apply nms
        outputs = non_max_suppression_torch(outputs)
        return outputs
