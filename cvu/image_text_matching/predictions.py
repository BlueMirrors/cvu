from typing import Iterator
from collections import Counter

import numpy as np

from cvu.interface.predictions import IPredictions
from cvu.image_text_matching.prediction import Prediction


class Predictions(IPredictions):

    def __init__(self) -> None:
        # list of predictions
        self._outputs = []

    def __bool__(self) -> bool:
        return bool(self._outputs)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self._outputs)

    def __getitem__(self, key: int) -> Prediction:
        return self._outputs[key]

    def __len__(self) -> int:
        return len(self._outputs)

    def __repr__(self) -> str:
        return '\n'.join(map(str, self._outputs))

    def create_and_append(self, output: str, confidence: float,
                          image_features: np.ndarray,
                          text_features: np.ndarray) -> None:
        # create prediction
        prediction = Prediction(output, confidence, image_features,
                                text_features)

        # append
        self._outputs.append(prediction)
