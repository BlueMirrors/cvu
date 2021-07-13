from cvu.utils.google_utils import gdrive_download
import os

import numpy as np
import torch
from torch import types

from cvu.interface.model import IModel
from cvu.utils.general import load_json, get_path
from cvu.postprocess.backend_torch.nms.yolov5 import non_max_suppression_torch


class Yolov5(IModel):
    def __init__(self, weight: str = "yolov5s", device='auto') -> None:
        self._device = None
        self._model = None

        self._set_device(device)
        self._load_model(weight)

    def _set_device(self, device):
        if device == 'auto':
            self._device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

    def _load_model(self, weight):
        if not os.path.exists(weight):
            if self._device.type != 'cpu':
                weight += '.cuda'
            weight = get_path(__file__, "weights", f"{weight}.torchscript.pt")
            self._download_weights(weight)

        self._model = torch.jit.load(weight, map_location=self._device)

        if self._device.type != 'cpu':
            self._model.half()

    def _download_weights(self, weight):
        if os.path.exists(weight): return

        weight_key = os.path.split(weight)[-1]
        weights_json = get_path(__file__, "weights", "torch_weights.json")
        available_weights = load_json(weights_json)
        if weight_key not in available_weights:
            raise FileNotFoundError
        gdrive_download(available_weights[weight_key], weight)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        inputs = self._preprocess(inputs)
        outputs = self._model(inputs)[0]
        return self._postprocess(outputs)[0].cpu().detach().numpy()

    def __repr__(self) -> str:
        return f"Yolov5: {self._device.type}"

    def _preprocess(self, inputs):
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

    def _postprocess(self, outputs):
        # apply nms
        outputs = non_max_suppression_torch(outputs)
        return outputs
