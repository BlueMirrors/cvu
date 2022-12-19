"""This file contains common functions used between different backends.
"""
import os

from cvu.utils.google_utils import gdrive_download
from cvu.utils.general import (load_json, get_path)


def get_weights_filename(weights: str, backend: str) -> str:
    save_path = ""
    if backend == "torch":
        save_path = f"{weights}_torch.torchscript.pt.pt"
    elif backend == "onnx":
        save_path = f"{weights}_onnx.onnx"
    elif backend == "tensorrt":
        save_path = f"{weights}_tensorrt.onnx"
    elif backend == "tensorflow":
        save_path = f"{weights}_tensorflow"
    elif backend == "tflite":
        save_path = f"{weights}_tflite.tflite"
    return save_path


def download_weights(
    yolo_version: str,
    weights: str,
    backend: str,
    unzip: bool = False,
) -> str:
    """Download weight if not downloaded already.

    Args:
        yolo_version (str): name of yolo version (i.e. yolov5, yolov7 etc.)
        weights (str): name of the weights (i.e. yolov5s, yolov7s etc.)
        backend (str): name of the backend
        unzip (bool): unzip downloaded file

    Raises:
        FileNotFoundError: raised if weight is not a valid pretrained
        weight name.
    
    Returns:
        save_path (str): path where file is saved
    """
    # already downloaded
    save_path = get_weights_filename(weights, backend)
    if os.path.exists(save_path):
        return save_path

    # get dict of all available pretrained weights
    weights_json = get_path(__file__, "..", "weights.json")
    available_weights = load_json(weights_json)

    weight_key = available_weights.get(yolo_version.lower(),
                                       {}).get(weights, {}).get(backend, None)

    # check if a valid weight is requested
    if not weight_key:
        raise FileNotFoundError(
            f"Invalid default weights {weights} for model {yolo_version.title()}-{backend}"
        )

    # download weights
    gdrive_download(weight_key, save_path, unzip=unzip)
    return save_path
