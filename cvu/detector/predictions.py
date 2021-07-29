"""This file contains implementation of Predictions classe that implements CVU's
common IPredictions interface for Object Detection. Predictions represents a group/list
of detected objects (Prediction Objects).
"""
from typing import Iterator
from collections import Counter

import numpy as np

from cvu.interface.predictions import IPredictions
from cvu.detector.prediction import Prediction


class Predictions(IPredictions):
    """Implements CVU's common IPredictions interface for Object Detection.

    This class represents a list of detected objects which behaves similar to
    a normal list (i.e. iteration, indexing, len, etc.). It also provides a common
    interface/structure over various detectors's (single-stage or double-stage)
    output. And also includes various common functionalities (such as drawing results
    on frame) as public methods.
    """
    def __init__(self) -> None:
        """Initiate Predictions object.
        """
        # list of detected-objects (List[Prediction])
        self._objects = []

        # class-wise count of detected-objects
        # computed when count-method is used first time.
        self._count = None

    def __bool__(self) -> bool:
        """Returns True if any objects are detected,
        False otherwise.

        Returns:
            bool: True if any objects are detected
            False otherwise.
        """
        return bool(self._objects)

    def __iter__(self) -> Iterator[Prediction]:
        """Iterator to iterate over detected objects.

        Returns:
            Iterable[Prediction]: iterable object for
            getting detected-objects (asPrediction object)
        """
        return iter(self._objects)

    def __getitem__(self, key: int) -> Prediction:
        """Get Detected Object using indexing.

        Args:
            key (int): index of a detected-object

        Returns:
            Prediction: object at key
        """
        return self._objects[key]

    def __len__(self) -> int:
        """Count of detected-objects.

        Returns:
            int: count
        """
        return len(self._objects)

    def __repr__(self) -> str:
        """Information of detected-objects.

        Returns:
            str: listed information for each
            detected-objects
        """
        return '\n'.join(map(str, self._objects))

    def draw(self, image: np.ndarray) -> np.ndarray:
        """Draws detected objects on the image (inplace), and
        returns image.

        Args:
            image (np.ndarray): BGR image

        Returns:
            np.ndarray: BGR image with objects drawn on
        """
        for object_ in self._objects:
            object_.draw(image)
        return image

    def count(self) -> Counter:
        """class-wise count of detected-objects.

        Returns:
            Counter: counter of detected-objects
        """
        # if called for first time, then compute count
        if self._count is None:
            self._count = Counter(
                map(lambda obj: getattr(obj, 'class_name'), self._objects))
        return self._count

    def create_and_append(self,
                          bbox: np.ndarray,
                          confidence: float,
                          class_id: int,
                          obj_id: int = None,
                          class_name: str = None) -> None:
        """Create and add a new Prediction to current predictions' list.

        Args:
            bbox (np.ndarray): Bounding box of detected object in tlbr format.
            A numpy array with 4 float/int values representing box coordinates in
            [top-left-x1, top-left-y1, bottom-right-x2, bottom-right-y2] format.

            confidence (float): Confidence Score of detected object.

            class_id (int): Class-Id of detected object.

            obj_id (int, optional): Id of detected object. Unique if tracker is used.
            Defaults to None which set object-id by incrementing the last-object-id
            (or set to 0 if this is the first object being appended)

            class_name (str, optional): Class-Name of detected object. Defaults to None.
        """
        # create prediction
        prediction = Prediction(
            (len(self._objects) if obj_id is None else obj_id), bbox,
            confidence, class_id, class_name)

        # append
        self._objects.append(prediction)

    def append(self, object_: Prediction) -> None:
        """Add Prediction to current predictions' list.

        Args:
            object_ (Prediction): prediction to add
        """
        self._objects.append(object_)

    def remove(self, object_: Prediction) -> None:
        """Remove Prediction from current predictions' list.

        Args:
            object_ (Prediction): prediction to remove
        """
        self._objects.remove(object_)

    def clear(self) -> None:
        """Clear current predictions's list.
        """
        self._objects.clear()
