import importlib
import os


def setup(device):
    try:
        attempt_import(device)
        return
    except ModuleNotFoundError:
        os.system(
            "pip install nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com"
        )

    try:
        attempt_import(device)

    except Exception as error:
        print("[CVU-Error] TensorRT Import Failed, either",
              "change backend or reinstall it properly...")
        print(error)


def attempt_import(device):
    trt = importlib.import_module("tensorrt")
    if not trt.Builder(trt.Logger()):
        raise Exception("Builder Failed to Initialize...")

    print(f"[CVU-Info] Backend: TensorRT-{trt.__version__}-{device}")
