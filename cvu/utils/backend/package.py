"""This file contains utility functions to install, setup and
test various pip package and there dependencies.
"""
import os
import subprocess
import sys
import importlib
from typing import List


def install(package: str, *args) -> None:
    """Install pip-package
    args are directly passed to pip

    Args:
        package (str): name of the pip-package
    """
    try:
        print(f"[CVU-Info] Auto-Installing {package}...")
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "install", package, *args])
        print(output.decode("utf-8"))
    except subprocess.CalledProcessError:
        print(f"[CVU-Error] {package.title()} Auto-Installation Failed...")


def setup(package: str,
          device: str,
          dependencies: List[str] = None,
          version: str = None,
          args: List[str] = None) -> bool:
    """Install package and relevant dependencies if not already installed,
    and test error-free import.

    Args:
        package (str): name of the package to install and test
        device (str): name of the device
        dependencies (List[str], optional): name of dependency packages. Defaults to None.
        version (str, optional): specific version. Defaults to None.
        args (List[str], optional): specific pip install arguments. Defaults to None.

    Returns:
        bool: True if install and test ran successfully, False otherwise
    """
    # check if already installed
    try:
        attempt_import(package, device, dependencies)
        return True

    # attempt installation
    except ModuleNotFoundError:

        # add version info if needed
        if version is not None:
            package = f'{package}=={version}'

        # install dependencies
        if dependencies is not None:
            for dependncy in dependencies:
                install(dependncy)

        # pass on pip args if applicable
        if args:
            install(package, *args)
        else:
            install(package)

    # test if installation was successful
    try:
        attempt_import(package, device, dependencies)
        return True

    # failed to install properly
    except ModuleNotFoundError:
        print(f"[CVU-Error] {package.title()} Import Failed, either",
              "change backend or reinstall it properly...")

    return False


def attempt_import(package: str,
                   device: str,
                   dependencies: List[str] = None) -> None:
    """Imports the package and all given dependencies

    Args:
        package (str): package to import
        device (str): name of the device
        dependencies (List[str], optional): name of dependency packages. Defaults to None.
    """
    if device in ("cpu", "tpu"):
        # disable GPUs explicitly
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        # activate gpu again
        del os.environ["CUDA_VISIBLE_DEVICES"]

    # sanitize (for example tensorflow-gpu => tensorflow)
    package = package.split('-')[0]

    # import dependencies
    if dependencies is not None:
        for dependency in dependencies:
            importlib.import_module(dependency)

    # import package
    importlib.import_module(package)
