"""This module handles all the backend installation and setup related tasks.
"""
import importlib
from cvu.utils.general import (load_json, get_path)
from cvu.utils.backend.package import setup_package

SUPPORTED_BACKENDS = load_json(get_path(__file__, "backend_config.json"))


def setup_backend(backend_name: str, device: str = "auto") -> bool:
    """Setup Backend and install dependencies

    Args:
        backend_name (str): name of the backend
        device (str, optional): name of the device to use. Defaults to "auto" i.e.
        auto-select best available option.

    Raises:
        NotImplementedError: raised if invalid backend name is given.
        ValueError: raised if invalid device is selected for the backend

    Returns:
        bool: name of target device if backend setup was success, None otherwise
    """
    # check if backend_name is valid
    if backend_name not in SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"[CVU] {backend_name} not supported. Please use a valid backend.")

    backend_config = SUPPORTED_BACKENDS[backend_name]
    if device != "auto" and (device not in backend_config["device_configs"]):
        raise ValueError(
            f"[CVU] {device} is not supported for {backend_name} backend.")

    # choose target devices
    devices = [device]
    if device == "auto":
        devices = backend_config["auto_device_priority"]

    # try setting up backend for target_devices in priority order
    for target_device in devices:
        print(
            f"[CVU-INFO] Attempting to setup {backend_name} for {target_device} device"
        )
        config = backend_config["device_configs"][target_device]

        if setup_package(**config):
            # test gpu availability if gpu-backend
            if target_device == "gpu":
                module = importlib.import_module(f".setup_{backend_name}",
                                                 "cvu.utils.backend")
                if not module.is_gpu_available():
                    print("[CVU-WARNING] GPU not detected")
                    continue
            print(f"[CVU-INFO] Using backend {backend_name}-{target_device}")
            return target_device
        print(
            f"[CVU-WARNING] Failed to setup {backend_name} for {target_device} device"
        )

    print(
        f"[CVU-ERROR] Failed to setup {backend_name}.",
        "Please try to install it manually or choose different backend configuration."
    )
    return None
