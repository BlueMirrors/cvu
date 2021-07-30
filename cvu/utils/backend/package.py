import subprocess
import sys
import importlib


def install(package, *args):
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "install", package, *args])
        print(output.decode("utf-8"))
    except subprocess.CalledProcessError:
        print(f"[CVU-Error] {package.title()} Auto-Installation Failed...")


def setup(package, dependencies=None, version=None, args=None):
    try:
        attempt_import(package, dependencies)
        return True

    except ModuleNotFoundError:
        if version is not None:
            package = f'{package}=={version}'

        if dependencies is not None:
            for dependncy in dependencies:
                install(dependncy)

        if args:
            install(package, *args)
        else:
            install(package)

    try:
        attempt_import(package, dependencies)
        return True

    except ModuleNotFoundError:
        pass

    except Exception as error:
        print(error)

    print(f"[CVU-Error] {package.title()} Import Failed, either",
          "change backend or reinstall it properly...")
    return False


def attempt_import(package, dependencies=None):
    package = package.split('-')[0]
    if dependencies is not None:
        for dependency in dependencies:
            importlib.import_module(dependency)

    importlib.import_module(package)
