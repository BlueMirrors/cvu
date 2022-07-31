"""This file contains utility functions to install, setup and
test various pip package and there dependencies.
"""
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
        print(f"[CVU-Error] '{package}' Auto-Installation Failed...")


def setup_package(package_name: str,
                  import_name: str,
                  version: str = None,
                  args: List[str] = None,
                  dependencies: List[str] = None) -> bool:
    """Install package and relevant dependencies if not already installed,
    and test error-free import.

    Args:
        package_name (str): name of the package to install and test
        import_name (str): name of the module that will be imported
        version (str, optional): specific version. Defaults to None.
        args (List[str], optional): specific pip install arguments. Defaults to None.
        dependencies (List[str], optional): name of dependency packages. Defaults to None.

    Returns:
        bool: True if install and test ran successfully, False otherwise
    """
    # check if already installed
    try:
        attempt_import(import_name, dependencies)
        return True

    # attempt installation
    except ModuleNotFoundError:

        # add version info if needed
        if version is not None:
            package_name = f'{package_name}=={version}'

        # install dependencies
        if dependencies is not None:
            for dependncy in dependencies:
                install(dependncy)

        # pass on pip args if applicable
        if args:
            install(package_name, *args)
        else:
            install(package_name)

    # test if installation was successful
    try:
        attempt_import(import_name, dependencies)
        return True

    # failed to install properly
    except ModuleNotFoundError:
        print(f"[CVU-Error] Failed to install '{package_name}'",
              "please install it manually.")

    return False


def attempt_import(package: str, dependencies: List[str] = None) -> None:
    """Imports the package and all given dependencies

    Args:
        package (str): package to import
        dependencies (List[str], optional): name of dependency packages.
        Defaults to None.
    """
    # import dependencies
    if dependencies is not None:
        for dependency in dependencies:
            importlib.import_module(dependency)

    # import package
    importlib.import_module(package)
