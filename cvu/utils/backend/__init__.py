import importlib
from .package import setup

SUPPORTED_BACKENDS = {
    'torch': {
        'name': 'torch',
        'device': 'cuda',
        'dependencies': ['torchvision'],
        'version': None,
        'device-agnostic': True,
        'args': None
    },
    'tensorflow': {
        'name': 'tensorflow',
        'device': 'cuda',
        'dependencies': None,
        'version': None,
        'device-agnostic': False,
        'args': None
    },
    'tflite': {
        'name': 'tflite',
        'device': 'cpu',
        'dependencies': None,
        'version': None,
        'device-agnostic': True,
        'args': None
    },
    'onnx': {
        'name': 'onnxruntime',
        'device': 'cuda',
        'dependencies': ['onnx'],
        'version': None,
        'device-agnostic': False,
        'args': None
    },
    'tensorrt': {
        'name': 'nvidia-tensorrt',
        'device': 'cuda',
        'dependencies': ['pycuda'],
        'version': None,
        'device-agnostic': True,
        'args': ["--index-url", "https://pypi.ngc.nvidia.com"]
    }
}


def setup_backend(backend_name, device=None):
    if backend_name not in SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"[CVU] {backend_name} not supported. Please use a valid backend.")

    backend = SUPPORTED_BACKENDS[backend_name]
    package = backend['name']
    if device is None:
        device = backend['device']

    if device != "cpu" and not backend['device-agnostic']:
        package += f"-gpu"

    flag = setup(package=package,
                 device=device,
                 dependencies=backend['dependencies'],
                 version=backend["version"],
                 args=backend["args"])

    if not flag and not backend['device-agnostic']:
        package = package.split('-')[0]
        flag = setup(package=package,
                     device='cpu',
                     dependencies=backend['dependencies'],
                     version=backend["version"],
                     args=backend["args"])
    if flag:
        module = importlib.import_module(f".setup_{backend_name}",
                                         "cvu.utils.backend")

        device = 'cuda' if module.is_gpu_available() else 'cpu'
        print(f"[CVU-Info] Backend: {backend_name.title()}",
              f"-{module.__version__}-{device}",
              sep="")

    return flag
