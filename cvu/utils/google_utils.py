"""Original Code Taken From https://stackoverflow.com/a/39225272

Contains code for interacting with Google Products
(such as Google-Drive).
"""
import os
import requests

from cvu.utils.general import unzip_file


def gdrive_download(id_: str, filepath: str, unzip: bool = False):
    """Downloads and extracts file (if needed) from Google drive given the id of the file.

    Args:
        id_ (str): google drive file_ id
        file_name (Optional[str], optional): output file name.
        unzip: unzip the downloaded file

    Returns:
        str: path where downloaded file_ was extracted 
    """
    URL = "https://docs.google.com/uc?export=download"

    if unzip and os.path.splitext(filepath)[-1] != '.zip':
        filepath += '.zip'

    print(f'[CVU-Info] Downloading {filepath} from {URL}&id={id_}')

    session = requests.Session()

    response = session.get(URL, params={'id': id_}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id_, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, filepath)

    if unzip:
        unzip_file(filepath)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
