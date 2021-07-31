"""This module handles all the backend installation and setup related tasks.
"""
import importlib
from .package import setup

SUPPORTED_BACKENDS = {
    'torch': {
        'name': 'torch',
        'best-device': 'cuda',
        'dependencies': ['torchvision'],
        'version': None,
        'device-agnostic': True,
        'args': None,
        'supported-devices': ['gpu', 'cuda', 'cpu']
    },
    'tensorflow': {
        'name': 'tensorflow',
        'best-device': 'cuda',
        'dependencies': None,
        'version': None,
        'device-agnostic': False,
        'args': None,
        'supported-devices': ['gpu', 'cuda', 'cpu', 'tpu']
    },
    'tflite': {
        'name': 'tensorflow',
        'best-device': 'cpu',
        'dependencies': None,
        'version': None,
        'device-agnostic': True,
        'args': None,
        'supported-devices': ['cpu']
    },
    'onnx': {
        'name': 'onnxruntime',
        'best-device': 'cuda',
        'dependencies': ['onnx'],
        'version': None,
        'device-agnostic': False,
        'args': None,
        'supported-devices': ['gpu', 'cuda', 'cpu']
    },
    'tensorrt': {
        'name': 'nvidia-tensorrt',
        'best-device': 'cuda',
        'dependencies': ['pycuda'],
        'version': None,
        'device-agnostic': True,
        'args': ["--index-url", "https://pypi.ngc.nvidia.com"],
        'supported-devices': ['gpu', 'cuda']
    }
}


def setup_backend(backend_name: str, device: str = "auto") -> bool:
    """Setup Backend and install dependencies

    Args:
        backend_name (str): name of the backend

        device (str, optional): name of the device to use. Defaults to "auto" i.e.
        auto-select best available option.

    Raises:
        NotImplementedError: raised if invalid backend name is given.

    Returns:
        bool: True if backend setup was success, False otherwise
    """
    # check if backend_name is valid
    if backend_name not in SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"[CVU] {backend_name} not supported. Please use a valid backend.")

    # get info
    backend = SUPPORTED_BACKENDS[backend_name]
    package = backend['name']

    # flag to know if mode is auto select device
    auto_selected_device = False

    # set auto-select mode, select device
    if device == "auto":
        device = backend['best-device']
        auto_selected_device = True

    if device.split(':')[0] not in backend['supported-devices']:
        raise Exception((f"[CVU-Error] Selected device {device} is " +
                         f"incompatible with {backend['name']} backend. " +
                         "Please change the backend or device."))

    # update package name if needed (for example tensorflow vs tensorflow-gpu)
    if device not in ("cpu", "tpu") and not backend['device-agnostic']:
        package += "-gpu"

    # attempt to install best option (possibly only option for some backends)
    success = setup(package=package,
                    device=device,
                    dependencies=backend['dependencies'],
                    version=backend["version"],
                    args=backend["args"])

    # if best option failed, attempt to install other options if available
    # only applicable in auto-select mode
    if not success and auto_selected_device and not backend['device-agnostic']:
        package = package.split('-')[0]
        success = setup(package=package,
                        device=device,
                        dependencies=backend['dependencies'],
                        version=backend["version"],
                        args=backend["args"])
        if success:
            device = "cpu"

    # if successfully installed, test import and update device info
    if success:

        # import backend main inference module
        module = importlib.import_module(f".setup_{backend_name}",
                                         "cvu.utils.backend")

        # update device info
        if auto_selected_device:
            device = 'cuda' if module.is_gpu_available() else 'cpu'

        elif device not in ("cpu", "tpu") and not module.is_gpu_available():
            raise Exception(
                (f"[CVU-Error] {device} is not available, please " +
                 "make sure you selected the correct device or consider " +
                 "changing the backend or device."))

        # log
        print(f"[CVU-Info] Backend: {backend_name.title()}",
              f"-{module.__version__}-{device}",
              sep="")

    return success
