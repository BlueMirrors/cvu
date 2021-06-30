"""Defines interface for CVU-Sinks that can be used to pipeline output
from CVU-Core to a desired output. Examples are writing output video,
writing output images, writing txt files with formatted predictions,
database, etc.
"""
import abc
from typing import List

from .predictions import IPredictions


class ISink(metaclass=abc.ABCMeta):
    """Interface for CVU-Sinks that can be used to pipeline output
    from CVU-Core to a desired output
    """
    @abc.abstractmethod
    def write(self, predictions: IPredictions, write_async: bool) -> None:
        """Write predictions to predefined output sink.

        Args:
            predictions (IPredictions): output predictions from core
            write_async (bool): whether to write asynchronously
        """
        ...

    @abc.abstractmethod
    def write_all(self, predictions: List[IPredictions],
                  write_async: bool) -> None:
        """Write multiple predictions to predefined output sink.

        Args:
            predictions (List[IPredictions]): list of output predictions from
            multiple core's execution

            write_async (bool): whether to write asynchronously
        """
        ...
