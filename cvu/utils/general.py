import os
import json


def get_local_path(fname) -> str:
    """Returns relative path for the local file
    Args:
        fname (path of the file): Generally __file__ value of the file where
        function is directly or indirectly invoked
    Returns:
        str: relative path
    """
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(fname)))


def get_path(local_file, *args) -> str:
    """Returns resulting path from joining args
    behind local_file's relative path.
    Args:
        local_file (path of the file): Generally __file__ value of the file where
        function is directly or indirectly invoked
    Returns:
        str: resulting path
    """
    return os.path.join(get_local_path(local_file), *args)


def load_json(fname: str) -> dict:
    """Loads a json file in a dict object.
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
