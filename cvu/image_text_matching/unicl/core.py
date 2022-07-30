"""This file contains UniCL's ICore implementation.

UniCL Core represents a common interface over which users can perform
UniCL powered image text matching, without having to worry about the details of
different backends and their specific requirements and implementations. Core
will internally resolve and handle all internal details.

Find more about UniCL here from their official repository
https://github.com/microsoft/UniCL
"""
from importlib import import_module
from typing import Any

import numpy as np

from cvu.interface.core import ICore
from cvu.utils.backend import setup_backend


class UniCL(ICore):
    _BACKEND_PKG = "cvu.image_text_matching.unicl.backends"

    def __init__(self,
                 backend: str = "torch",
                 weight: str = "swin_b",
                 device: str = "auto",
                 auto_install: bool = False,
                 **kwargs: Any) -> None:
        # ICore
        super().__init__(backend)

        # initiate class attributes
        self._model = None

        # setup backend and load model
        if auto_install:
            setup_backend(backend, device)
        self._load_model(backend, weight, device, **kwargs)

    def __repr__(self) -> str:
        """Returns Backend and Model Information

        Returns:
            str: information string
        """
        return str(self._model)

    def _load_model(self, backend: str, weight: str, device: str,
                    **kwargs: Any) -> None:
        # load model
        backend = import_module(f".unicl_{backend}", self._BACKEND_PKG)

        self._model = backend.UniCL(weight, device)

    def __call__(self, inputs: np.ndarray, query: str, **kwargs):
        # inference on backend
        image_features, text_features, probs = self._model(
            inputs, query, **kwargs)
