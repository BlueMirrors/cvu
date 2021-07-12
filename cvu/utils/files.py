import os


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
