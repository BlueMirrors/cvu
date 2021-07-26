import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2

from cvu.interface.model import IModel
from cvu.utils.google_utils import gdrive_download
from cvu.postprocess.nms.yolov5 import non_max_suppression_np
from cvu.utils.general import load_json, get_path


class Yolov5(IModel):
    def __init__(self,
                 weight: str = None,
                 device="cuda",
                 num_classes: int = 80,
                 conf_thres: float = 0.4,
                 iou_thres: float = 0.5,
                 fp16: bool = True,
                 input_shape=(384, 640)) -> None:

        # Create a Context on this device,
        self._ctx = cuda.Device(0).make_context()
        self._logger = trt.Logger(trt.Logger.VERBOSE)
        self._stream = cuda.Stream()

        self._nc = num_classes
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._fp16 = fp16

        self._engine = None
        self._context = None
        self._inputs = None
        self._outputs = None
        self._bindings = None

        self._load_model(weight, input_shape)
        self._allocate_buffers()

    def _deserialize_engine(self, trt_engine_path):
        with open(trt_engine_path, 'rb') as engine_file:
            with trt.Runtime(self._logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_file.read())

        return engine

    def _download_weight(self, weight):
        if os.path.exists(weight):
            return

        weight_key = os.path.split(weight)[-1]
        weights_json = get_path(__file__, "weights", "tensorrt_weights.json")
        available_weights = load_json(weights_json)
        if weight_key not in available_weights:
            raise NotImplementedError(
                f"{weight_key.split('_')[0]} is not a supported model")
        gdrive_download(available_weights[weight_key], weight)

    def _load_model(self, weight, input_shape):
        """Deserialized TensorRT cuda engine and creates execution context.
        """
        # load default models
        if "." not in weight:
            engine_path = get_path(__file__, "weights", f"{weight}_trt.engine")
            if not os.path.exists(engine_path):
                onnx_weight = engine_path.replace("engine", "onnx")
                self._download_weight(onnx_weight)
                self._engine = self._build_engine(onnx_weight, input_shape)
            else:
                self._engine = self._deserialize_engine(engine_path)

        # use custom models
        else:
            engine_path = weight.replace(
                "onnx", "engine") if ".onnx" in weight else weight

            if not os.path.exists(engine_path):
                self._engine = self._build_engine(weight, input_shape)
            else:
                self._engine = self._deserialize_engine(engine_path)

        if not self._engine:
            raise Exception("Couldn't build engine successfully !")

        self._context = self._engine.create_execution_context()
        if not self._context:
            raise Exception(
                "Couldn't create execution context from engine successfully !")

    def _build_engine(self, onnx_weight,
                      input_shape) -> trt.tensorrt.ICudaEngine:
        """Builds TensorRT engine by parsing the onnx model.
        """

        # checks to
        if not os.path.exists(onnx_weight):
            raise FileNotFoundError(f"{onnx_weight} does not exists.")
        elif ".onnx" not in onnx_weight:
            raise TypeError(
                f"Expected onnx weight file, instead {onnx_weight} is given.")

        trt_engine_path = onnx_weight.replace("onnx", "engine")

        # Specify that the network should be created with an explicit batch dimension.
        EXPLICIT_BATCH = 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(self._logger) as builder, \
             builder.create_network(EXPLICIT_BATCH) as network, \
             trt.OnnxParser(network, self._logger) as parser:

            # setup builder config
            config = builder.create_builder_config()
            config.max_workspace_size = 64 * 1 << 20  # 64 MB
            builder.max_batch_size = 1

            if self._fp16:
                if builder.platform_has_fast_fp16:
                    print("Platform has FP16 support. Setting fp16 to True")
                    config.flags = 1 << (int)(trt.BuilderFlag.FP16)

            # parse onnx model
            with open(onnx_weight, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            # set input shape
            profile = builder.create_optimization_profile()
            input_shape = (1, 3, *input_shape)
            profile.set_shape('images', input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)

            # build engine
            engine = builder.build_engine(network, config)
            with open(trt_engine_path, 'wb') as f:
                f.write(engine.serialize())
            print("Engine serialized and saved !")
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

        self._inputs = inputs
        self._outputs = outputs
        self._bindings = bindings

    def _pre_process(self, img: np.ndarray) -> np.ndarray:
        """Preprocess the input image.
        :params:
            img: numpy.ndarray, opencv image object
        
        :returns:
            img: numpy.ndarray, preprocessed opencv image object
        """
        img = img.astype('float32')
        img /= 255.0
        return img

    def __call__(self, inputs: np.ndarray) -> list:
        """Preprocess image, infer and postprocess raw
        tensorrt outputs.
        :params:
            img: numpy.ndarray, opencv image object
        
        :returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        resized = self._pre_process(inputs)
        outputs = self._inference(resized)
        preds = self._post_process(outputs)
        return preds[0]

    def _inference(self, img: np.ndarray) -> list:
        """
        Runs inference on the given image.
        :params:
            img: numpy.ndarray, preprocessed opencv image object
        
        :returns:
            raw tensorrt output as a list
        """
        self._ctx.push()

        # copy img to input memory
        # without astype gives invalid arg error
        self._inputs[0]['host'] = np.ravel(img).astype(np.float32)

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

    def _post_process(self, outputs: list) -> list:
        """
        Transforms tensorrt output into boxes, confs, labels and 
        applies non max suppression.
        :params:
            output: raw tensorrt output tensor
        Returns:
            boxes: x1,y1,x2,y2 tensor (dets, 4)
            confs: class * obj prob tensor (dets, 1) 
            classes: class type tensor (dets, 1)
        """
        # reshape into expected output shape
        outputs = outputs[-1].reshape((1, -1, self._nc + 5))
        return non_max_suppression_np(outputs,
                                      conf_thres=self._conf_thres,
                                      iou_thres=self._iou_thres)

    def __repr__(self) -> str:
        return f"Yolov5: tensorrt"

    def __del__(self):
        try:
            self._ctx.pop()
        except Exception as e:
            print("[CVU-Info] Context stack is already empty.")
