"""Original Code Taken From ultralytics/yolov5
URL: https://github.com/ultralytics/yolov5/blob/master/utils/google_utils.py

Contains code for interacting with Google Products
(such as Google-Drive).
"""
import os
import zipfile
import platform
import time
from pathlib import Path
from typing import Optional


def gdrive_download(id_: str, file_name: str, unzip=False) -> str:
    """Downloads and extracts file from Google drive given the id of the file.

    Args:
        id_ (str): google drive file_ id
        file_name (Optional[str], optional): output file name.
        unzip: unzip the downloaded file

    Returns:
        str: path where downloaded file_ was extracted 
    """
    if unzip and os.path.splitext(file_name)[-1] != '.zip':
        file_name += '.zip'

    # Downloads a file from Google Drive.
    start_time = time.time()
    file_ = Path(file_name)
    cookie = Path('cookie')  # gdrive cookie

    print(f"Downloading https://drive.google.com/uc?",
          f"export=download&id={id_} as {file_}...",
          end='',
          sep='')

    # remove existing file
    if file_.exists(): file_.unlink()

    # remove existing cookie
    if cookie.exists(): cookie.unlink()

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(
        f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id_}" > {out}'
    )
    if os.path.exists('cookie'):  # large file
        command = (
            'curl -Lb ./cookie "drive.google.com/uc?' +
            f'export=download&confirm={get_token()}&id={id_}" -o {file_}')
    else:  # small file
        command = f'curl -s -L -o {file_} "drive.google.com/uc?export=download&id={id_}"'
    result = os.system(command)  # execute, capture return

    # remove existing cookie
    if cookie.exists(): cookie.unlink()

    # Error check
    if result != 0:
        # remove partial
        if file_.exists(): file_.unlink()
        print('Download error ')  # raise Exception('Download error')
        return result

    if unzip:
        print('Unzipping...', file_name)
        # unzip
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(os.path.split(file_name)[0])

        # delete zip
        if os.path.isfile(file_name):
            os.remove(file_name)

    print(f'Done ({time.time() - start_time:.1f}s)')
    return result


def get_token(cookie: Optional[str] = "./cookie") -> str:
    """Get Download Token Value from GDrive Cookie

    Args:
        cookie (Optional[str], optional): Path to cookie. Defaults to "./cookie".

    Returns:
        str: token
    """
    with open(cookie) as cookie_file:
        for line in cookie_file:
            if "download" in line:
                return line.split()[-1]
    return ""
