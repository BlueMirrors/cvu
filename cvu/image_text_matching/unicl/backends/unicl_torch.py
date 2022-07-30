import os
from typing import List

import numpy as np
import torch

from cvu.interface.model import IModel
from cvu.utils.general import get_path
from cvu.image_text_matching.unicl.backends.common import download_weights


class UniCL(IModel):

    def __init__(self, weight: str = "swin_b", device="auto") -> None:
        # initiate class attributes
        self._device = None
        self._model = None

        # setup device
        self._set_device(device)

        # load model
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