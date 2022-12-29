from collections import OrderedDict
import os
from typing import List, Tuple
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda

from .yolov5_tensorrt import Yolov5 as Yolov5Trt


class Yolov5(Yolov5Trt):
    def __init__(
        self,
        weight: str = None,
        num_classes: int = 80,
        input_shape: Tuple[int, int] = (640, 640),
        dtype: str = "fp16",
    ) -> None:
        # check constraints
        if input_shape is None:
            raise Exception("[CVU-Error] input_shape can't be none for end2end model.")
        if not os.path.exists(weight):
            raise FileNotFoundError(f"[CVU-Error] {weight} not found.")
        if ".onnx" in weight:
            raise NotImplementedError("[CVU-Error] conversion from onnx is not supported.")
        
        # regular init
        super().__init__(
            weight=weight, num_classes=num_classes, input_shape=tuple(input_shape), dtype=dtype
        )
    
    def _deserialize_engine(self, trt_engine_path: str) -> trt.tensorrt.ICudaEngine:
        # init NMS plugin
        trt.init_libnvinfer_plugins(self._logger, namespace="")
        return super()._deserialize_engine(trt_engine_path)

    def _allocate_buffers(self) -> None:
        inputs, outputs, bindings, output_names = [], [], [], []
        
        for binding in self._engine:
            size = trt.volume(self._engine.get_tensor_shape(binding))
            dtype = trt.nptype(self._engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self._engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})
                output_names.append(binding)

        # set buffers
        self._inputs = inputs
        self._outputs = outputs
        self._bindings = bindings
        self._output_names = output_names

    def _post_process(self, outputs: List[np.ndarray]) -> OrderedDict[str, np.ndarray]:
        outputs = OrderedDict((name, out) for name, out in zip(self._output_names, outputs))
        outputs["det_boxes"] = outputs["det_boxes"].reshape(-1, 4)
        return [outputs] # because __call__ makes outputs[0]
