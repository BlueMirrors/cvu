import numpy as np

from cvu.interface.predictions import IPrediction


class Prediction(IPrediction):

    def __init__(self, output: str, confidence: float,
                 image_features: np.ndarray,
                 text_features: np.ndarray) -> None:
        self._obj_id = 1
        self._output = output
        self._confidence = confidence
        self._image_features = image_features
        self._text_features = text_features

    @property
    def obj_id(self) -> int:
        return self._obj_id

    @property
    def output(self) -> str:
        return self._output

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def image_features(self) -> np.ndarray:
        return self._image_features

    @property
    def text_features(self) -> np.ndarray:
        return self._text_features

    def __repr__(self) -> str:
        return "\t".join([
            f"id:{self.obj_id}", f"class={self.output.title()};",
            f"conf={self.confidence};"
        ])