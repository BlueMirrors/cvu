import importlib
import os


def setup(device):
    try:
        attempt_import(device)
        return
    except ModuleNotFoundError:
        os.system("pip install torch")

    try:
        attempt_import(device)

    except Exception as error:
        print("[CVU-Error] Torch Import Failed, either",
              "change backend or reinstall it properly...")
        print(error)


def attempt_import(device):
    torch = importlib.import_module("torch")
    if device != 'cpu' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"[CVU-Info] Backend: Torch-{torch.__version__}-{device}")
