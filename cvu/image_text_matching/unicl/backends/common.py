"""This file contains common functions used between different backends.
"""
import os

from cvu.utils.google_utils import gdrive_download
from cvu.utils.general import (load_json, get_path)


def download_weights(weight: str, backend: str, unzip=False) -> None:
    """Download weight if not downloaded already.

    Args:
        weight (str): path where weights should be downloaded
        backend (str): name of the backend
        unzip (bool, optional): unzip downloaded file. Defaults to False.

    Raises:
        FileNotFoundError: raised if weight is not a valid pretrained
            weight name.
    """
    # already downloaded
    if os.path.exists(weight):
        return

    # get weight's identifier key
    weight_key = os.path.split(weight)[-1]

    # get dict of all available pretrained weights
    weights_json = get_path(__file__, "weights", f"{backend}_weights.json")
    available_weights = load_json(weights_json)

    # check if a valid weight is requested
    if weight_key not in available_weights:
        raise FileNotFoundError

    # download weights
    gdrive_download(available_weights[weight_key], weight, unzip)
