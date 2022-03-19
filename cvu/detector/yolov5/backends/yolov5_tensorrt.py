"""This file contains Yolov5's IModel implementation in TensorRT.
This model (tensorRT-backend) performs inference using TensorRT,
on a given input numpy array, and returns result after performing
nms and other backend specific postprocessings.

Model expects normalized inputs (data-format=channels-first) with
batch axis. Model does not apply letterboxing to given inputs.
"""
import os
from typing import Tuple, List

import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa # pylint: disable=unused-import
import pycuda.driver as cuda

from cvu.interface.model import IModel
from cvu.preprocess.image.letterbox import letterbox
from cvu.preprocess.image.general import (basic_preprocess, bgr_to_rgb,
                                          hwc_to_chw)
from cvu.utils.general import get_path
from cvu.detector.yolov5.backends.common import download_weights
from cvu.postprocess.nms.yolov5 import non_max_suppression_np
from cvu.utils.backend_tensorrt.int8_calibrator import Int8EntropyCalibrator2


class Yolov5(IModel):
    # noqa # pylint: disable=too-many-instance-attributes
    """Implements IModel for Yolov5 using TensorRT.

    This model (tensorrt-backend) performs inference, using TensorRT,
    on a numpy array, and returns result after performing NMS.

    This model does not support runtime dynamic inputs. In other words, once
    created and first inference is done, model expects rest of the inputs to
    be of the same shape as the first input (or the input shape given at the
    initialization step).

    Inputs are expected to be normalized in channels-first order
    with/without batch axis.
    """
    def __init__(self,
                 weight: str = None,
                 num_classes: int = 80,
                 input_shape: Tuple[int, int] = None,
                 dtype: str = "fp16",
                 calib_images_dir: str = None) -> None:
        """Initialize.

        Args:
            weight (str): name of the weight file
            num_classes (int): number of classes model is trained with
            input_shape (Tuple(int, int)): input shape of the model (h, w)
            dtype (str): dtype ['fp32', 'fp16', 'int8'] for tensorrt model
            calib_images_dir (str): calibration images directory for when dtype is 'int8'
        """

        # Create a Context on this device,
        self._ctx = cuda.Device(0).make_context()
        self._logger = trt.Logger(trt.Logger.INFO)
        self._stream = cuda.Stream()

        # initiate basic class attributes
        self._weight = weight
        self._dtype = dtype
        if self._dtype == "int8":
            if calib_images_dir is None:
                raise Exception("[CVU-Error] calib_images_dir is None with dtype int8.")
            self._calib_images_dir = calib_images_dir

        # initiate model specific class attributes
        self._nc = num_classes
        self._input_shape = input_shape

        # initiate engine related class attributes
        self._engine = None
        self._context = None
        self._inputs = None
        self._outputs = None
        self._bindings = None

        # initiate engine if input_shape given
        if self._input_shape is not None:
            self._load_model(weight)
            self._allocate_buffers()

    def _deserialize_engine(self,
                            trt_engine_path: str) -> trt.tensorrt.ICudaEngine:
        """Deserialize TensorRT Cuda Engine

        Args:
            trt_engine_path (str): path to engine file

        Returns:
            trt.tensorrt.ICudaEngine: deserialized engine
        """
        with open(trt_engine_path, 'rb') as engine_file:
            with trt.Runtime(self._logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_file.read())

        return engine

    def _load_model(self, weight: str) -> None:
        """Internally loads TensorRT cuda engine and creates execution context.

        Args:
            weight (str): path to ONNX weight file, TensorRT Engine file or
            predefined-identifiers (such as yolvo5s, yolov5m, etc.)
        """
        # load default models using predefined-identifiers
        if "." not in weight:
            height, width = self._input_shape[:2]

            # get path to pretrained weights
            engine_path = get_path(__file__, "weights",
                                   f"{weight}_{height}_{width}_{self._dtype}_trt.engine")

            onnx_weight = get_path(__file__, "weights", f"{weight}_trt.onnx")

            # download onnx weights if needed, and/or generate engine file
            if not os.path.exists(engine_path):

                # download weights if not already downloaded
                download_weights(onnx_weight, "tensorrt")

                # build engine with current configs and load it
                self._engine = self._build_engine(onnx_weight, engine_path,
                                                  self._input_shape)
            else:
                # deserialize and load engine
                self._engine = self._deserialize_engine(engine_path)

        # use custom models
        else:
            # get path to weights
            engine_path = weight.replace(
                "onnx", "engine") if ".onnx" in weight else weight

            # build engine with given configs and load it
            if not os.path.exists(engine_path):
                self._engine = self._build_engine(weight, engine_path,
                                                  self._input_shape)
            else:
                # deserialize and load engine
                self._engine = self._deserialize_engine(engine_path)

        # check if engine loaded properly
        if not self._engine:
            raise Exception("[CVU-Error] Couldn't build engine successfully !")

        # create execution context
        self._context = self._engine.create_execution_context()
        if not self._context:
            raise Exception(
                "[CVU-Error] Couldn't create execution context from engine successfully !")

    @staticmethod
    def get_supported_dtypes(builder) -> List[str]:
        """Method to check if fp16 and int8 are suuported on the platform.

        Args:
            builder (trt.Bilder): a trt.Builder object

        Returns:
            list of suuported dtypes
        """
        supported_dtypes = ["fp32"]

        if builder.platform_has_fast_fp16:
            supported_dtypes.append("fp16")

        if builder.platform_has_fast_int8:
            supported_dtypes.append("int8")
        return supported_dtypes


    def _build_engine(self, onnx_weight: str, trt_engine_path: str,
                      input_shape: Tuple[int]) -> trt.tensorrt.ICudaEngine:
        """Builds and serializes TensorRT engine by parsing the onnx model.

        Args:
            onnx_weight (str): path to onnx weight file
            trt_engine_path (str): path where serialized engine file will be saved
            input_shape (Tuple[int]): input shape for network

        Raises:
            FileNotFoundError: raised if onnx weight file doesn't exists
            TypeError: raised if invalid type of weight file is given

        Returns:
            trt.tensorrt.ICudaEngine: built engine
        """

        # checks if onnx path exists
        if not os.path.exists(onnx_weight):
            raise FileNotFoundError(
                f"[CVU-Error] {onnx_weight} does not exists.")

        # check if valid onnx_weight
        if ".onnx" not in onnx_weight:
            raise TypeError(
                f"[CVU-Error] Expected onnx weight file, instead {onnx_weight} is given."
            )

        # Specify that the network should be created with an explicit batch dimension.
        batch_size = 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # build and serialize engine
        with trt.Builder(self._logger) as builder, \
             builder.create_network(batch_size) as network, \
             trt.OnnxParser(network, self._logger) as parser:

            # get supported dtypes on this platform
            supported_dtypes = Yolov5.get_supported_dtypes(builder)
            if self._dtype not in supported_dtypes:
                raise Exception(f"[CVU-Error] Invalid dtype '{self._dtype}'. "\
                    f"Please choose from {str(supported_dtypes)}")

            # setup builder config
            config = builder.create_builder_config()
            config.max_workspace_size = 64 * 1 << 20  # 64 MB
            builder.max_batch_size = 1

            print(f"[CVU-Info] Platform has {self._dtype} support.",\
                  f"Setting {self._dtype} to True")
            # fp16 quantization
            if self._dtype == "fp16":
                config.flags = 1 << (int)(trt.BuilderFlag.FP16)

            # int8 qunatization
            elif self._dtype == "int8":
                # Activate int8 mode

                config.flags = 1 << (int)(trt.BuilderFlag.INT8)
                config.int8_calibrator = Int8EntropyCalibrator2(
                    batchsize=1,
                    input_h=input_shape[0],
                    input_w=input_shape[1],
                    img_dir=self._calib_images_dir,
                    preprocess=[
                        letterbox,
                        bgr_to_rgb,
                        hwc_to_chw,
                        np.ascontiguousarray,
                        basic_preprocess
                    ]
                )

            # parse onnx model
            with open(onnx_weight, 'rb') as onnx_file:
                if not parser.parse(onnx_file.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            # set input shape
            network.get_input(0).shape = (1, 3, *input_shape)

            # build engine
            engine = builder.build_engine(network, config)
            with open(trt_engine_path, 'wb') as trt_engine_file:
                trt_engine_file.write(engine.serialize())
            print("[CVU-Info] Engine serialized and saved !")
            return engine

    def _allocate_buffers(self) -> None:
        """Allocates memory for inference using TensorRT engine.
        """
        inputs, outputs, bindings = [], [], []
        for binding in self._engine:
            size = trt.volume(self._engine.get_binding_shape(binding))
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self._engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        # set buffers
        self._inputs = inputs
        self._outputs = outputs
        self._bindings = bindings

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Performs model inference on given inputs, and returns
        inference's output after NMS.

        Args:
            inputs (np.ndarray): normalized in channels-first format,
            with/without batch axis.

        Raises:
            Exception: raised if inputs's shape doesn't not match with
            expected input shape.

        Returns:
            np.ndarray: inference's output after NMS
        """
        # set input shape and build engine if first inference
        if self._input_shape is None:
            self._input_shape = inputs.shape[-2:]
            print("[CVU-Info] Building and Optimizing TRT-Engine",
                  f"for input_shape={self._input_shape}.",
                  "This might take a few minutes for first time.")
            self._load_model(self._weight)
            self._allocate_buffers()

        # check if inputs shape match expected shape
        if inputs.shape[-2:] != self._input_shape:
            raise Exception(
                ("[CVU-Error] Invalid Input Shapes: Expected input to " +
                 f"be of shape {self._input_shape}, but got " +
                 f" input of shape {inputs.shape[-2:]}." +
                 "Please rebuild TRT Engine with correct shapes."))

        # perform inference and postprocess
        outputs = self._inference(inputs)
        preds = self._post_process(outputs)
        return preds[0]

    def _inference(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Runs inference on the given inputs.

        Args:
            inputs (np.ndarray): channels-first format,
            with/without batch axis

        Returns:
            List[np.ndarray]: inference's output (raw tensorrt output)
        """
        self._ctx.push()

        # copy inputs to input memory
        # without astype gives invalid arg error
        self._inputs[0]['host'] = np.ravel(inputs).astype(np.float32)

        # transfer data to the gpu
        for inp in self._inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self._stream)

        # run inference
        self._context.execute_async_v2(bindings=self._bindings,
                                       stream_handle=self._stream.handle)

        # fetch outputs from gpu
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self._stream)

        # synchronize stream
        self._stream.synchronize()
        self._ctx.pop()
        return [out['host'] for out in self._outputs]

    def _post_process(self, outputs: List[np.ndarray]) -> List[np.ndarray]:
        """Post-process outputs from model inference.

        Transforms tensorrt output into boxes, confs, labels and
        applies non max suppression.

        Args:
            outputs (List[np.ndarray]): raw tensorrt output tensor

        Returns:
            List[np.ndarray]: post-processed output after nms
        """
        # reshape into expected output shape
        outputs = outputs[-1].reshape((1, -1, self._nc + 5))
        return non_max_suppression_np(outputs)

    def __repr__(self) -> str:
        """Returns Model Information

        Returns:
            str: information string
        """
        return f"Yolov5s TensorRT-Cuda-{self._input_shape}"

    def __del__(self):
        """Clean up execution context stack.
        """
        try:
            self._ctx.pop()
        except pycuda.driver.LogicError as _:
            print("[CVU-Info] Context stack is already empty.")
