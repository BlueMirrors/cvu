import importlib
import os


def setup(device):
    try:
        attempt_import(device)
        return
    except ModuleNotFoundError:
        os.system("pip install tflite")

    try:
        attempt_import(device)

    except Exception as error:
        print("[CVU-Error] TFLite Import Failed, either",
              "change backend or reinstall it properly...")
        print(error)


def attempt_import(device):
    tflite = importlib.import_module("tflite")
    print(f"[CVU-Info] Backend: TFLite-{tflite.__version__}-{device}")
