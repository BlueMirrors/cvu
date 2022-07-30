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
