"""Includes interface for CVU Detectors' core.
"""
import abc
from typing import List, Union

import numpy as np

from cvu.interface.core import ICore
from cvu.detector.predictions import Predictions


class IDetector(ICore, metaclass=abc.ABCMeta):
    """Interface which will be implemented for every CVU Detector.
    A core defines one complete method/solution for certain use cases.
    For example, YoloV5 is a detector core of Object Detection use cases.
    """

    @abc.abstractmethod
    def __init__(self, classes: Union[str, List[str]], backend: str,
                 weight: str, device: str, *args, **kwargs) -> None:
        """Initiate Core.

        Args:
            classes (Union[str, List[str]]): name of classes to be detected.
            It can be set to individual classes like 'coco', 'person', 'cat' etc.
            Alternatively, it can be a list of classes such as ['person', 'cat'].
            For default weights, 'classes' is used to filter out objects
            according to provided argument from coco class (unless specified otherwise).
            
            backend (str): name of the backend to be used for inference purposes.

            weight (str): path to weight files (according to selected backend).

            device (str): name of the device to be used. Valid
            devices can be "cpu", "gpu", "tpu", "auto".

            auto_install (bool): auto install missing requirements for the selected
            backend.
        """
        ...

    @abc.abstractmethod
    def __call__(self, inputs: np.ndarray, **kwargs) -> Predictions:
        """Run object detection on given image

        Args:
            inputs (np.ndarray): image in BGR format.

        Returns:
            Predictions: detected objects.
        """
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Returns Backend and Model Information

        Returns:
            str: formatted string with method and config info.
        """
        ...
