import importlib
import os


def setup(device):
    try:
        attempt_import(device)
        return

    except ModuleNotFoundError:
        os.system("pip install onnx")
        os.system("pip install onnxruntime" +
                  ("" if device == "cpu" else "-gpu"))

    try:
        attempt_import(device)

    except Exception as error:
        print("[CVU-Error] ONNX Import Failed, either",
              "change backend or reinstall it properly...")
        print(error)


def attempt_import(device):
    importlib.import_module("onnx")
    onnx_runtime = importlib.import_module("onnxruntime")
    if device != 'cpu' and onnx_runtime.get_device().lower() == 'cpu':
        device = 'cpu'
    print(f"[CVU-Info] Backend: ONNX-{onnx_runtime.__version__}-{device}")
