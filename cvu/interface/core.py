"""Defines interface for CVU-Cores.
A core defines one complete method/solution for certain use cases.
For example, YoloV5 is a core of Object Detection use cases.
"""
import abc
from typing import Union, List

import numpy as np

from .predictions import IPredictions


class ICore(metaclass=abc.ABCMeta):
    """Core Interface which will be implemented for every CVU-Core.
    A core defines one complete method/solution for certain use cases.
    For example, YoloV5 is a core of Object Detection use cases.
    """

    @abc.abstractmethod
    def __init__(self, backend: str, *args, **kwargs) -> None:
        """Initiate Core.

        Args:
            backend (str): name of the backend to run core on.
        """
        ...

    @abc.abstractmethod
    def __call__(self, inputs: np.ndarray, **kwargs) -> IPredictions:
        """Execute core on inputs

        Args:
            inputs (np.ndarray): inputs to be exectued core on

        Returns:
            Predictions: results of executation
        """
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Represents core with method and configuration informations.

        Returns:
            str: formatted string with method and config info.
        """
        ...
