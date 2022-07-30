import os
import requests

from cvu.utils.general import (load_json, get_path)


def download_weights(weight: str, backend: str):
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
    response = requests.get(available_weights[weight_key])
    if response.status_code != 200:
        raise Exception("Failed to download weights from {}".format(
            available_weights[weight_key]))

    # add extension if needed
    if os.path.splitext(weight)[-1] != '.pth':
        weight += '.pth'

    with open(weight, "wb") as f:
        f.write(response.content)