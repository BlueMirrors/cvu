"""
This file contains various functions for interacting with Google Products
(such as Google-Drive).
"""
import os
from typing import Optional

import gdown

from cvu.utils.general import unzip_file


def gdrive_download(id_: str,
                    filepath: str,
                    unzip: Optional[bool] = False) -> None:
    """Downloads and extracts file (if needed) from Google drive given the id of the file.

    Args:
        id_ (str): google drive file_ id
        file_name (Optional[str], optional): output file name.
        unzip: unzip the downloaded file
    """
    # gdrive URL
    url = f"https://drive.google.com/uc?id={id_}"

    # add extension if needed
    if unzip and os.path.splitext(filepath)[-1] != '.zip':
        filepath += '.zip'

    print(f"[CVU-Info] Downloading '{filepath}' from '{url}'")
    gdown.download(url, filepath, quiet=False)

    # unzip if needed
    if unzip:
        unzip_file(filepath)
