"""This file contains Yolov5's IModel implementation in ONNX.
This model (onnx-backend) performs inference using ONNXRUNTIME,
on a given input numpy array, and returns result after performing
nms and other backend specific postprocessings.

Model expects normalized inputs (data-format=channels-first) with
batch axis. Model does not apply letterboxing to given inputs.
"""
import os
from typing import List

import numpy as np
import torch
import onnx
import onnxsim
import onnxruntime

from cvu.interface.model import IModel
from cvu.utils.general import get_path
from cvu.detector.yolov5.backends.common import download_weights
from cvu.postprocess.nms.yolov5 import non_max_suppression_np


class Yolov5(IModel):
    """Implements IModel for Yolov5 using ONNX.

    This model (onnx-backend) performs inference, using ONNXRUNTIME,
    on a numpy array, and returns result after performing NMS.

    Inputs are expected to be normalized in channels-first order
    with/without batch axis.
    """

    def __init__(self, weight: str = "yolov5s", device: str = 'auto') -> None:
        """Initiate Model

        Args:
            weight (str, optional): path to onnx weight file. Alternatively,
            it also accepts identifiers (such as yolvo5s, yolov5m, etc.) to load
            pretrained models. Defaults to "yolov5s".

            device (str, optional): name of the device to be used. Valid devices can be
            "cpu", "gpu", "auto". Defaults to "auto" which tries to use the device
            best suited for selected backend and the hardware avaibility.
        """
        self._model = None
        self._device = device

        self._load_model(weight)

    def _load_model(self, weight: str) -> None:
        """Internally loads ONNX

        Args:
            weight (str): path to ONNX weight file or predefined-identifiers
            (such as yolvo5s, yolov5m, etc.)
        """
        if weight.endswith("torchscript"):
            # convert to onnx
            weight = self._onnx_from_torchscript(weight, save_path=weight.replace("torchscript", "onnx"))

        # attempt to load predefined weights
        elif not os.path.exists(weight):

            # get path to pretrained weights
            weight += '.onnx'
            weight = get_path(__file__, "weights", weight)

            # download weights if not already downloaded
            download_weights(weight, "onnx")

        # load model
        if self._device == "cpu":
            # load model on cpu
            self._model = onnxruntime.InferenceSession(
                weight, providers=["CPUExecutionProvider"])
            return

        if self._device == "gpu":
            # load model on gpu
            self._model = onnxruntime.InferenceSession(
                weight, providers=['CUDAExecutionProvider'])
            return

        # load-model with auto-device selection
        try:
            self._model = onnxruntime.InferenceSession(
                weight,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # failed to load model, try CPU runtime if applicable
        except RuntimeError as error:

            # check if multiple runtime providers are available
            multiple_providers = len(onnxruntime.get_available_providers()) > 1

            # try cpu device if multiple providers are available,
            # and device selection is set to auto
            if multiple_providers:

                print("[CVU-Info] CUDA powered Backend failed to load,",
                      "switching to CPU.")

                print(f"[CVU-Info] Backend:Onnx-{onnxruntime.__version__}-cpu")

                self._model = onnxruntime.InferenceSession(
                    weight, providers=["CPUExecutionProvider"])
                return

            # switching providers failed or is not applicable
            raise error

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Performs model inference on given inputs, and returns
        inference's output after NMS.

        Args:
            inputs (np.ndarray): normalized in channels-first format,
            with batch axis.

        Returns:
            np.ndarray: inference's output after NMS
        """
        outputs = self._model.run([self._model.get_outputs()[0].name],
                                  {self._model.get_inputs()[0].name: inputs})
        return self._postprocess(outputs)[0]

    def __repr__(self) -> str:
        """Returns Model Information

        Returns:
            str: information string
        """
        return f"Yolov5s ONNX-{self._device}"

    @staticmethod
    def _postprocess(outputs: np.ndarray) -> List[np.ndarray]:
        """Post-process outputs from model inference.
            - Non-Max-Supression

        Args:
            outputs (np.ndarray): model inference's output

        Returns:
           List[np.ndarray]: post-processed output
        """
        # apply nms
        outputs = non_max_suppression_np(outputs[0])
        return outputs

    def _onnx_from_torchscript(
        self, 
        torchscript_model: str, 
        shape: List=[640, 640], 
        save_path=None, 
        simplify=False,
        dynamic=False):
        """Convert torchscript model to ONNX.

        Args:
            torchscript_model: path to torchscript model

        Returns:

        Raises:
        
        """
        if not os.path.exists(torchscript_model):
            raise FileNotFoundError(f"{torchscript_model} doesnt not exist.")
        
        if save_path is None:
            save_path = torchscript_model.replace('torchscript', 'onnx')

        # load torchscript model
        extra_files = {'config.txt': ''}  # model metadata
        model = torch.jit.load(torchscript_model, map_location='cpu', _extra_files=extra_files)

        im = torch.zeros((1,3, *shape))

        try:

            print(f'\n[CVU-Info] starting export with onnx {onnx.__version__}...')

            torch.onnx.export(
                model,
                im,
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

            # Simplify
            if simplify:
                try:

                    print(f'[CVU-Info] simplifying with onnx-simplifier {onnxsim.__version__}...')
                    model_onnx, check = onnxsim.simplify(model_onnx,
                                                         dynamic_input_shape=dynamic,
                                                         input_shapes={'images': list(im.shape)} if dynamic else None)
                    assert check, 'assert check failed'
                    onnx.save(model_onnx, save_path)
                except Exception as e:
                    print(f'[CVU-Info] simplifier failure: {e}')
            print(f'[CVU-Info] export success, saved as {save_path})')
            return save_path
        except Exception as e:
            print(f'[CVU-Info] export failure: {e}')