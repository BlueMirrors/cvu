"""This module contains interface definition of the detector model
"""
import abc

import numpy as np


class IDetectorModel(metaclass=abc.ABCMeta):
    """Interface of the detector model

    A model performs inference, using a certain backend runtime,
    on a numpy array, and returns result after performing NMS.

    Inputs are expected to be normalized in channels-first order
    with/without batch axis.
    """

    @abc.abstractmethod
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Performs model inference on given inputs, and returns
        inference's output after NMS.

        Args:
            inputs (np.ndarray): normalized in channels-first format,
            with batch axis.

        Returns:
            np.ndarray: inference's output after NMS
        """
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Represents model with method and configuration informations.

        Returns:
            str: formatted string with method and config info.
        """
        ...
