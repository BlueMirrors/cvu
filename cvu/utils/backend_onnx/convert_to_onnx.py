"""
ONNX conversion from different frameworks.
"""

from typing import Optional, Tuple, Union
import os
import onnx
import torch


def onnx_from_torchscript(
    torchscript_model: str,
    shape: Tuple[int, int]=(640, 640),
    save_path:Optional[str]=None,
    dynamic:Optional[bool]=False) -> Union[str, None]:
    """Convert torchscript model to ONNX.
    Args:
        torchscript_model (str): path to torchscript model
        shape (Optional[Tuple[int, int]]): input shape of the model
        save_path (Optional[str]): path to save onnx model
        dynamic (Optional[bool]): bool for onnx dynamic shape conversion
    Returns:
        Optional[str]: path to converted onnx model
    Raises:
        FileNotFoundError if torchscript model not found
    """
    if not os.path.exists(torchscript_model):
        raise FileNotFoundError(f"{torchscript_model} doesnt not exist.")

    if save_path is None:
        save_path = torchscript_model.replace('torchscript', 'onnx')

    if os.path.exists(save_path):
        return save_path

    # load torchscript model
    extra_files = {'config.txt': ''}  # model metadata
    model = torch.jit.load(torchscript_model, map_location='cpu', _extra_files=extra_files)
    img = torch.zeros((1,3, *shape))

    try:

        print(f'\n[CVU-Info] starting export with onnx {onnx.__version__}...')
        torch.onnx.export(
            model,
            img,
            save_path,
            verbose=False,
            opset_version=13,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None)
        # Checks
        model_onnx = onnx.load(save_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        print(f'[CVU-Info] export success, saved as {save_path})')
        return save_path
    except Exception as exception:  # pylint: disable=broad-except
        print(f'[CVU-Info] export failure: {exception}')

    return None
