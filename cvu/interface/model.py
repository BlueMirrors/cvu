"""Defines interface for Model that represents CVU-Backends.
A Model combines the process for individual model inference for certain backend.
For example, YoloV5Torch can be a model of Yolov5 a core.
"""
import abc

import numpy as np


class IModel(metaclass=abc.ABCMeta):
    """Model Interface which will be implemented for every CVU-Backend.
    A Model combines the process for individual model inference
    for certain backend.
    """
    @abc.abstractmethod
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Execute core on inputs

        Args:
            inputs (np.ndarray): inputs to be exectued core on

        Returns:
            Predictions: results of executation
        """
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Represents model with method and configuration informations.

        Returns:
            str: formatted string with method and config info.
        """
        ...
