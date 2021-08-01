"""This file contains implementation of Prediction classe, that implements CVU's
common IPrediction interface for Object Detection. Prediction class represents a
single detected object.
"""
import numpy as np

from cvu.interface.predictions import IPrediction
from cvu.utils.draw import draw_bbox


class Prediction(IPrediction):
    """Implements CVU's common IPrediction interface for Object Detection.

    This class represents a single detected object, that encapsulates
    object's bounding box, confidence score, class id, class name and
    object-id (unique if tracker is activated). This class provides a common
    interface/structure over various detectors's (single-stage or double-stage)
    output. It also provides various common functionalities (used in general
    object detection pipeline) as public methods.
    """
    def __init__(self, obj_id: int, bbox: np.ndarray, confidence: float,
                 class_id: int, class_name: str) -> None:
        """Initiate Prediction Object

        Args:
            obj_id (int): Id of detected object. Unique if tracker is used.

            bbox (np.ndarray): Bounding box of detected object in tlbr format.
            A numpy array with 4 float/int values representing box coordinates in
            [top-left-x1, top-left-y1, bottom-right-x2, bottom-right-y2] format.

            confidence (float): Confidence Score of detected object.

            class_id (int): Class-Id of detected object. Mainly used for json output in
            Yolo format.

            class_name (str): Class-Name of detected object.
        """
        self._obj_id = obj_id
        self._bbox = bbox
        self._confidence = round(float(confidence), 2)
        self._class_id = int(class_id)
        self._class_name = class_name

    @property
    def obj_id(self) -> int:
        """Id of the detected object
        (unique if tracker is used)

        Returns:
            int: id
        """
        return self._obj_id

    @property
    def bbox(self) -> np.ndarray:
        """Bounding-Box of the detected object

        Returns:
            np.ndarray: np array with 4 float/int values
            representing bounding box in [top-left-x1,
            top-left-y1, bottom-right-x2, bottom-right-y2] format.
        """
        return self._bbox

    @property
    def confidence(self) -> float:
        """Confidence Score of detected object.

        Returns:
            float: confidence score.
        """
        return self._confidence

    @property
    def class_id(self) -> int:
        """Class-Id of detected object

        Returns:
            int: class-id
        """
        return self._class_id

    @property
    def class_name(self) -> str:
        """Class-Name of detected object.

        Returns:
            str: class-name
        """
        return self._class_name

    def __repr__(self) -> str:
        """Detected object's information

        Returns:
            str: string consisted of id, class-name,
            and bounding box's coordinates (top-left and
            bottom-right coordinates).
        """
        return "\t".join([
            f"id:{self.obj_id}", f"class={self.class_name.title()};",
            f"conf={self.confidence};",
            f"top-left=({self.bbox[0]}, {self.bbox[1]:<5});",
            f"bottom-right=({self.bbox[2]}, {self.bbox[3]})"
        ])

    def draw(self, image: np.ndarray) -> None:
        """Draws detected object on the image (inplace)

        Args:
            image (np.ndarray): BGR image
        """
        title = f"{self.obj_id}.{self.class_name.title()} ({round(100*self.confidence, 2)}%)"
        draw_bbox(image, self.bbox, title=title)
