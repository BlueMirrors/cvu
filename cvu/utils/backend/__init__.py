import importlib

SUPPORTED_BACKENDS = {
    'torch': 'cuda',
    'tensorflow': 'cuda',
    'tflite': 'cpu',
    'onnx': 'cuda',
    'tensorrt': 'cuda'
}


def setup_backend(backend_name, device=None):
    if backend_name not in SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"[CVU] {backend_name} not supported. Please use a valid backend.")

    backend = importlib.import_module(f".setup_{backend_name}",
                                      "cvu.utils.backend")
    backend.setup(
        SUPPORTED_BACKENDS[backend_name] if device is None else device)
