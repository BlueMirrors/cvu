"""This file contains TensorRT's trt.IInt8EntropyCalibrator2 implementation.
This calibrator (tensorRT-backend) performs int8 calibration using TensorRT,
on a given set of images, and returns builds the TensorRT engine after this
calibration process is completed.
"""

import os
from typing import Callable, List, Union

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa # pylint: disable=unused-import

from cvu.utils.general import read_images_in_batch


class Int8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    """Implements trt.IInt8EntropyCalibrator2 for Yolov5.
    """
    def __init__(
        self,
        batchsize: int = 1,
        input_h: int = 640,
        input_w: int = 640,
        img_dir: str = None,
        preprocess: List[Callable] = None,
        calib_cache: str = "int8calib.cache",
        ) -> None:
        """Initialize Int8EntropyCalibrator2.

        Args:
            batchsize (int): Batchsize for the calibration process.
            input_h (int): Maximum height of the input for CUDA mem alloc.
            input_w (int): Maximum width of the input for CUDA mem alloc.
            img_dir (str): Directory containing calibration images from training dataset.
            preprocess (List[Callable]): List of preprocessing to apply.
            calib_cache (str): File to store the calibration cache.
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batchsize = batchsize
        self.input_w = input_w
        self.input_h = input_h
        self.img_dir = img_dir

        self.calib_cache = calib_cache

        # each element of the calibration data is a float32
        # get the larger dim from (h, w)
        input_dim = input_h if input_h > input_w else input_w
        self.device_input = cuda.mem_alloc(
            trt.volume((self.batchsize, 3, input_dim, input_dim)) * trt.float32.itemsize
        )

        self.preprocess = preprocess

        self.batches = read_images_in_batch(
            self.img_dir, self.batchsize, preprocess=self.preprocess
        )

    def get_batch_size(self):
        """Get batch size.
        """
        return self.batchsize

    def get_batch(self, names: List[str]) -> List[int]:    # pylint: disable=unused-argument
        """Get a batch of input for calibration.

        Args:
            names: List of file names.

        Returns:
            A list of device memory pointers set to the memory containing
            each network input data, or an empty list if there are no more
            batches for calibration.
        """
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

    def read_calibration_cache(self) -> Union[memoryview, None]:
        """Load a calibration cache.

        Returns:
            A cache object or None if there is no data.
        """
        # If there is a cache, use it instead of calibrating again. Otherwise,
        # implicitly return None.
        if os.path.exists(self.calib_cache):
            with open(self.calib_cache, "rb") as calib_cache_file:
                return calib_cache_file.read()
        return None

    def write_calibration_cache(self, cache: memoryview) -> None:
        """Save a calibration cache.

        Args:
            cache: The calibration cache to write.
        """
        with open(self.calib_cache, "wb") as calib_cache_file:
            calib_cache_file.write(cache)
