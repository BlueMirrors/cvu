"""Defines interface for CVU-Sources that can be used to pipeline data
from various sources to CVU-Core. Examples are video reading source,
image reading source, txt reading sources (that contains path/url of
other images/videos), etc.
"""
import abc

import numpy as np


class ISource(metaclass=abc.ABCMeta):
    """Interface for CVU-Sources that can be used to pipeline data
    from various sources to CVU-Core.
    """
    @abc.abstractmethod
    def read(self) -> np.ndarray:
        """Returns input for one core execution

        Returns:
            np.ndarray: input for core
        """
        ...

    @abc.abstractmethod
    def read_all(self) -> np.ndarray:
        """Returns aggregated inputs from source
        to be used for mutliple core execution.

        Returns:
            np.ndarray: aggregated inputs
        """
        ...
