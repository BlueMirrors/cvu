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

from cvu.interface.core import ICore
from cvu.utils.backend import setup_backend


class UniCL(ICore):
    __BACKEND_PKG = "cvu.image_text_matching.unicl.backends"

    def __init__(self,
                 backend: str = "torch",
                 device: str = "auto",
                 auto_install: bool = False,
                 **kwargs: Any) -> None:
        # ICore
        super().__init__()

        # initiate class attributes
        self._model = None

        # setup backend and load model
        if auto_install:
            setup_backend(backend, device)
        # self._load_model(backend, device, **kwargs)