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
                 fp16: bool = True) -> None:

        # Create a Context on this device,
        self._ctx = cuda.Device(0).make_context()
        self._nc = num_classes
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._logger = trt.Logger(trt.Logger.VERBOSE)
        self._fp16 = fp16

        self._engine = None
        self._context = None
        self._inputs = None
        self._outputs = None
        self._bindings = None
        self._stream = cuda.Stream()

        self._load_model(weight)
        self._allocate_buffers()

        # refer https://github.com/ultralytics/yolov5
        # post processing config
        self._output_shapes = [(1, 3, 80, 80, self._nc + 5),
                               (1, 3, 40, 40, self._nc + 5),
                               (1, 3, 20, 20, self._nc + 5)]
        self._strides = np.array([8., 16., 32.])
        anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ])
        self._nl = len(anchors)

        self._no = self._nc + 5  # outputs per anchor
        self._na = len(anchors[0])
        a = anchors.copy().astype(np.float32)
        a = a.reshape(self._nl, -1, 2)
        self._anchors = a.copy()
        self._anchor_grid = a.copy().reshape(self._nl, 1, -1, 1, 1, 2)

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

    def _load_model(self, weight):
        """Deserialized TensorRT cuda engine and creates execution context.
        """
        # load default models
        if "." not in weight:
            engine_path = get_path(__file__, "weights", f"{weight}_trt.engine")
            if not os.path.exists(engine_path):
                onnx_weight = engine_path.replace("engine", "onnx")
                self._download_weight(onnx_weight)
                self._engine = self._build_engine(onnx_weight)
            else:
                self._engine = self._deserialize_engine(engine_path)

        # use custom models
        else:
            engine_path = weight.replace(
                "onnx", "engine") if ".onnx" in weight else weight

            if not os.path.exists(engine_path):
                self._engine = self._build_engine(weight)
            else:
                self._engine = self._deserialize_engine(engine_path)

        if not self._engine:
            raise Exception("Couldn't build engine successfully !")

        self._context = self._engine.create_execution_context()
        if not self._context:
            raise Exception(
                "Couldn't create execution context from engine successfully !")

    def _build_engine(self, onnx_weight) -> trt.tensorrt.ICudaEngine:
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
        # img = cv2.resize(img, (640, 640))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = img.transpose((2, 0, 1)).astype(np.float32)
        # img /= 255.0
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
        outputs = self._do_inference(resized)

        reshaped = []
        for output, shape in zip(outputs, self._output_shapes):
            reshaped.append(output.reshape(shape))

        preds = self._post_process(reshaped)
        return preds[0]

    def _do_inference(self, img: np.ndarray) -> list:
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
        scaled = []
        grids = []
        for out in outputs:
            out = self._sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self._make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self._strides,
                                             self._anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2)**2 * anchor

            out = out.reshape((1, 3 * width * height, 85))
            z.append(out)
        pred = np.concatenate(z, 1)
        return non_max_suppression_np(pred,
                                      conf_thres=self._conf_thres,
                                      iou_thres=self._iou_thres)

    def _make_grid(self, nx: int, ny: int) -> np.ndarray:
        """
        Create scaling tensor based on box location -> https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        
        :params:
            nx: x-axis num boxes
            ny: y-axis num boxes
        
        :returns:
            grid: tensor of shape (1, 1, nx, ny, num_classes)
        """
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def _sigmoid_v(self, array: np.ndarray) -> np.ndarray:
        return np.reciprocal(np.exp(-array) + 1.0)

    def __repr__(self) -> str:
        return f"Yolov5: tensorrt"

    def __del__(self):
        try:
            self._ctx.pop()
        except Exception as e:
            print("[CVU-Info] Context stack is already empty.")
