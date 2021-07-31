"""Original Code Taken From https://stackoverflow.com/a/39225272

This file contains various functions for interacting with Google Products
(such as Google-Drive).
"""
import os
from typing import Optional

import requests

from cvu.utils.general import unzip_file


def gdrive_download(id_: str, filepath: str, unzip: bool = False) -> None:
    """Downloads and extracts file (if needed) from Google drive given the id of the file.

    Args:
        id_ (str): google drive file_ id
        file_name (Optional[str], optional): output file name.
        unzip: unzip the downloaded file
    """
    # gdrive URL
    url = "https://docs.google.com/uc?export=download"

    # add extension if needed
    if unzip and os.path.splitext(filepath)[-1] != '.zip':
        filepath += '.zip'

    print(f'[CVU-Info] Downloading {filepath} from {url}&id={id_}')

    session = requests.Session()

    response = session.get(url, params={'id': id_}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id_, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    # save file
    save_response_content(response, filepath)

    # unzip if needed
    if unzip:
        unzip_file(filepath)


def get_confirm_token(response: requests.models.Response) -> Optional[str]:
    """Get confirmation token from request's response

    Args:
        response (requests.models.Response): request's response

    Returns:
        Optional[str]: confirmation token if any available
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response: requests.models.Response,
                          destination: str) -> None:
    """Save response content at destination path

    Args:
        response (requests.models.Response): request's response
        destination (str): path where response content will be saved
    """
    # default
    chunk_size = 32768

    with open(destination, "wb") as response_file:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                response_file.write(chunk)
