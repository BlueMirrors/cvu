"""Defines interface for CVU-Cores' predictions. A prediction defines one
output from core's execution. For different methods/use-cases, it can
represent different things. For example, in general Object Detection,
prediction will represent single detected object. A container of multiple
predictions is the actual returned output from core.
"""
from typing import Iterator

import abc


class IPrediction(metaclass=abc.ABCMeta):
    """Prediction Interface that represents individual
    ouput from a core's execution.
    """
    @property
    @abc.abstractmethod
    def obj_id(self) -> int:
        """Unique Id that identifies individual prediction
        from a core's execution. Ids may change between
        different executions unless tracking is activated.

        Returns:
            int: object's unique id
        """
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Represent prediction

        Returns:
            str: formatted string to showcase important
            aspects of prediction
        """
        ...


class IPredictions(metaclass=abc.ABCMeta):
    """Container Interface of multiple predictions resulting
    from one core's execution.
    """
    @abc.abstractmethod
    def __bool__(self) -> bool:
        """Returns True if there are any valid predictions from
        core's execution, false otherwise.

        Returns:
            bool: True if valid predictions false otherwise.
        """
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator:
        """Iterator to be used to iterate through
        predictions.

        Yields:
            Iterator: iterator for predictions
        """
        ...

    @abc.abstractmethod
    def __getitem__(self, index: int) -> IPrediction:
        """Defines square bracket method to access specific
        predictions.

        Args:
            index (int): index of prediction to be accessed

        Returns:
            IPrediction: queried prediction
        """
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Represents all predictions

        Returns:
            str: formatted string to showcase important
            aspects of every prediction
        """
        ...
