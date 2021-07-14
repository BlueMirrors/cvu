from typing import Iterator
from collections import Counter

import numpy as np

from cvu.interface.predictions import IPrediction, IPredictions
from cvu.utils.draw import draw_bbox


class Prediction(IPrediction):
    def __init__(self, obj_id: int, bbox: np.ndarray, confidence: float,
                 class_id: int, class_name: str) -> None:
        self._obj_id = obj_id
        self._bbox = bbox
        self._confidence = round(float(confidence), 2)
        self._class_id = int(class_id)
        self._class_name = class_name

    @property
    def obj_id(self) -> int:
        return self._obj_id

    @property
    def bbox(self) -> np.ndarray:
        return self._bbox

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def class_id(self) -> int:
        return self._class_id

    @property
    def class_name(self) -> str:
        return self._class_name

    def __repr__(self):
        return (f"id:{self.obj_id}; class:{self.class_name}; " +
                f"top-left:({self.bbox[0]}, {self.bbox[1]}); " +
                f"bottom-right:({self.bbox[2]}, {self.bbox[3]})")

    def draw(self, image):
        title = f"{self.obj_id}.{self.class_name}({self.confidence})"
        draw_bbox(image, self.bbox, title=title)


class Predictions(IPredictions):
    def __init__(self) -> None:
        self._objects = []
        self._count = None

    def __bool__(self) -> bool:
        return bool(self._objects)

    def __iter__(self) -> Iterator:
        return iter(self._objects)

    def __getitem__(self, key) -> Prediction:
        return self._objects[key]

    def __len__(self) -> int:
        return len(self._objects)

    def __repr__(self) -> str:
        return '\n'.join(map(str, self._objects))

    def draw(self, image) -> None:
        for object_ in self._objects:
            object_.draw(image)

    def count(self) -> dict:
        if self._count is None:
            self._count = Counter(
                map(lambda obj: getattr(obj, 'class_name'), self._objects))
        return self._count

    def create_and_append(self,
                          bbox,
                          confidence,
                          class_id,
                          obj_id=None,
                          class_name=None):

        prediction = Prediction(
            (len(self._objects) if obj_id is None else obj_id), bbox,
            confidence, class_id, class_name)

        self._objects.append(prediction)

    def append(self, object_):
        self._objects.append(object_)

    def remove(self, object_):
        self._objects.remove(object_)

    def clear(self):
        self._objects.clear()
