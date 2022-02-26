import tenosrrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

from cvu.preprocess.image.letterbox import letterbox
from cvu.preprocess.image.general import (basic_preprocess, bgr_to_rgb,
                                          hwc_to_chw)


class Int8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    def __init__(
        self, 
        batchsize: int = 1, 
        input_w: int = 1280, 
        input_h: int = 720, 
        img_dir: str = None, 
        calib_cache: str = "int8calib.cache", 
        ) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batchsize = batchsize
        self.total_imgs = 100
        self.input_w = input_w
        self.input_h = input_h
        self.img_dir = img_dir
        

        # Get all images.
        self.img_files = [os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)]

        # check if total image files are enough for calibration
        if len(self.img_files) < self.total_imgs:
            raise Exception(
                "[CVU-Error] Not enough images to perform INT8 calibration !"
            )
        
        self.img_idx = 0
        self.calib_cache = calib_cache

        # Each element of the calibration data is a float32.
        self.device_input = cuda.mem_alloc(
            trt.volume((self.batchsize, 3, self.input_w, self.input_w)) * trt.float32.itemsize
        )

        self.preprocess = [letterbox, bgr_to_rgb, hwc_to_chw, np.ascontiguousarray, basic_preprocess]
        self.batches = self.load_batches()

    @staticmethod
    def apply(
            value: np.ndarray,
            functions: List[Callable[[np.ndarray], np.ndarray]]):
        for func in functions:
            value = func(value)
        return value

    def load_batches(self):
        for i in range(self.total_imgs):
            print(f"Calib batch {i}")
            yield self.read_batch_file(self.img_files[self.img_idx:self.img_idx+self.batchsize])
            self.img_idx = self.img_idx + self.batchsize

    def read_batch_files(self, img_files):
        batch = list()
        for img_file in  img_files:
            # read image
            img = cv2.imread(img_file)

            # preprocess
            processed_inputs = self.apply(img, self.preprocess)

            batch.append(processed_inputs)
        
        batch = np.array(batch)
        return batch
    
    def get_batch_size(self):
        return self.batchsize
    
    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            data = next(self.batches)
            # Assume that self.device_input is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.calib_cache):
            with open(self.calib_cache, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.calib_cache, "wb") as f:
            f.write(cache)