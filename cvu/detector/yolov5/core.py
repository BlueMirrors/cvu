"""This file contains Yolov5's ICore implementation.

Yolov5 Core represents a common interface over which users can perform
Yolov5 powered object detection, without having to worry about the details of
different backends and their specific requirements and implementations. Core
will internally resolve and handle all internal details.

Find more about Yolov5 here from their official repository
https://github.com/ultralytics/yolov5
"""
import os
from typing import Any, Union, List

from cvu.detector.yolo import Yolo
from cvu.detector.yolo.common import download_weights


class Yolov5(Yolo):
    """Implements ICore for Yolov5

    Yolov5 Core represents a common interface to perform
    Yolov5 powered object detection.
    """

    def __init__(self,
                 classes: Union[str, List[str]],
                 backend: str = "torch",
                 weight: str = "yolov5s",
                 device: str = "auto",
                 auto_install: bool = False,
                 **kwargs: Any) -> None:
        """Initiate Yolov5 Object Detector

        Args:
            classes (Union[str, List[str]]): name of classes to be detected.
            It can be set to individual classes like 'coco', 'person', 'cat' etc.
            Alternatively, it also accepts list of classes such as ['person', 'cat'].
            For default models/weights, 'classes' is used to filter out objects
            according to provided argument from coco class. For custom models, all
            classes should be provided in original order as list of strings.

            backend (str, optional): name of the backend to be used for inference purposes.
            Defaults to "torch".

            weight (str, optional): path to weight files (according to selected backend).
            Alternatively, it also accepts identifiers (such as yolvo5s, yolov5m, etc.)
            to load pretrained models. Defaults to "yolov5s".

            device (str, optional): name of the device to be used. Valid
            devices can be "cpu", "gpu", "tpu", "auto". Defaults to "auto" which tries
            to use the device best suited for selected backend and the hardware avaibility.

            auto_install (bool, optional): auto install missing requirements for the selected
            backend.
        """
        if not os.path.exists(weight):
            # attemp weight download
            weight = download_weights("yolov5", weight, backend,
                                      backend == "tensorflow")

        super().__init__(classes, backend, weight, device, auto_install,
                         **kwargs)
