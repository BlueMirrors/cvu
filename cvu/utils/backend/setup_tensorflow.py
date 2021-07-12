import importlib
import os


def setup(device):
    try:
        attempt_import(device)
        return
    except ModuleNotFoundError:
        os.system("pip install tensorflow" +
                  ("" if device == "cpu" else "-gpu"))

    try:
        attempt_import(device)

    except Exception as error:
        print("[CVU-Error] Tensorflow Import Failed, either",
              "change backend or reinstall it properly...")
        print(error)


def attempt_import(device):
    tf = importlib.import_module("tensorflow")
    if device != 'cpu' and not bool(tf.config.list_physical_devices('GPU')):
        device = 'cpu'
    print(f"[CVU-Info] Backend: Tensorflow-{tf.__version__}-{device}")
