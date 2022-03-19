"""This file contains various general utils related to reading common-files
and resolving paths.
"""
import os
import json
import zipfile
from typing import Any, Iterable, List, Callable

import numpy as np
import cv2


def get_local_path(fname: str) -> str:
    """Returns relative path for the local file

    Args:
        fname (str): path of the file. Generally __file__ is passed which
        contains path form where function is directly or indirectly invoked

    Returns:
        str: relative path
    """
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(fname)))


def get_path(local_file: str, *args) -> str:
    """Returns resulting path from joining args behind local_file's relative path.

    Args:
        local_file (str): Generally __file__ is passed which
        contains path form where function is directly or indirectly invoked

    Returns:
        str: resulting path
    """
    return os.path.join(get_local_path(local_file), *args)


def load_json(fname: str) -> dict:
    """Loads json file in a dict object.

    Args:
        fname (str): json file path

    Raises:
        FileNotFoundError: raised when fname doesn't exists.

    Returns:
        data (dict): json file loaded into a dict.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} is not found.")

    # open JSON file
    data = {}
    with open(fname, 'r') as json_file:
        # read data
        data = json.load(json_file)

    return data


def unzip_file(filepath: str,
               destination: str = None,
               clean_up: bool = True) -> bool:
    """Unzip File and delete the original (if needed)

    Args:
        filepath (str): path to zip file

        destination (str, optional): path where file should be extracted to.
        Defaults to root dir of filepath;

        clean_up (bool, optional): [description]. delete original zip file.
        Defaults to True.

    Returns:
        bool: True if all operations were successful, false otherwise;
    """

    if not os.path.exists(filepath):
        return False

    if destination is None:
        destination = os.path.split(filepath)[0]

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(destination)

    if clean_up:
        os.remove(filepath)
    return True


def read_images_in_batch(img_dir, batchsize, preprocess=None) -> Iterable[np.ndarray]:
    """
    Read preprocessed image files in a batch.

    Args:
        img_dir (str): Path to the directory containing the images.

    Yields:
        np.ndarray: Batch of preprocessed images.
    """
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

    batch = []
    for img_file in  img_files:
        if len(batch) == batchsize:
            yield np.array(batch).squeeze(0)
            batch = []

        # Read image.
        image = cv2.imread(img_file)

        # Preprocess
        if preprocess:
            image = apply(image, preprocess)
        batch.append(image)
    yield np.array(batch)


def apply(
        value: Any,
        functions: List[Callable[[Any], Any]]) -> Any:
    """
    Recursively applies list of callable functions to given value

    Args:
        value (Any): Input to be processed.
        functions (List[Callable[[Any], Any]]): List of
        callable functions.

    Returns:
        Any: Value resulting from applying all functions.
    """
    for func in functions:
        value = func(value)
    return value
