"""This file contains Yolov5's IModel implementation in Tensorflow.
This model (tensorflow-backend) performs inference using SavedModel,
on a given input numpy array, and returns result after performing
nms and other backend specific postprocessings.

Model expects normalized inputs (data-format=channels-last) with batch
axis. Model does not apply letterboxing to given inputs.
"""
import logging
import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from cvu.interface.model import IModel
from cvu.utils.general import get_path
from cvu.detector.yolov5.backends.common import download_weights
from cvu.postprocess.bbox import denormalize
from cvu.postprocess.backend_tf.nms.yolov5 import non_max_suppression_tf


class Yolov5(IModel):
    """Implements IModel for Yolov5 using Tensorflow.

    This model (tensorflow-backend) performs inference, using SavedModel,
    on a numpy array, and returns result after performing NMS.

    Inputs are expected to be normalized in channels-last order with batch axis.
    """

    def __init__(self, weight: str = "yolov5s", device='auto') -> None:
        """Initiate Model

        Args:
            weight (str, optional): path to SavedModel weight files. Alternatively,
            it also accepts identifiers (such as yolvo5s, yolov5m, etc.) to load
            pretrained models. Defaults to "yolov5s".

            device (str, optional): name of the device to be used. Valid devices can be
            "cpu", "gpu", "auto", "tpu". Defaults to "auto" which tries to use the
            device best suited for selected backend and the hardware avaibility.
        """
        # initiate class attributes
        self._model = None
        self._device = None
        self._loaded = None
        self._signature = {"input": None, "output": None}

        # disable logging
        logging.disable(logging.WARNING)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # setup device
        self._set_device(device)

        # load SavedModel
        self._load_model(weight)

    def _set_device(self, device: str) -> None:
        """Internally setup/initiate necessary device

        Args:
            device (str): name of the device to be used.
        """
        # get gpu count
        gpus = len(tf.config.list_physical_devices('GPU'))

        # set FP16 if GPU available and selected
        if device in ('auto', 'gpu') and gpus > 0:
            self._device = 'cuda:0'
            mixed_precision.set_global_policy('mixed_float16')

        # initialize and setup TPU
        elif device == 'tpu':
            # find, connect and intialize tpus
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)

            print(f"[CVU-Info] Backend: Tensorflow-{tf.__version__}-tpu")
            self._device = 'tpu'

        # use cpu
        else:
            # disable GPUs (in case CPU was manually selected)
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self._device = 'cpu'

    def _load_model(self, weight: str) -> None:
        """Internally loads SavedModel

        Args:
            weight (str): path to SavedModel weight files or predefined-identifiers
            (such as yolvo5s, yolov5m, etc.)
        """
        # attempt to load predefined weights
        if not os.path.exists(weight):
            weight += '_tensorflow'

            # get path to pretrained weights
            weight = get_path(__file__, "weights", weight)

            # download weights if not already downloaded
            download_weights(weight, "tensorflow", unzip=True)

        # set load_options needed for TPU (if needed)
        load_options = None
        if self._device == 'tpu':
            load_options = tf.saved_model.LoadOptions(
                experimental_io_device="/job:localhost")

        # load model
        self._loaded = tf.saved_model.load(weight, options=load_options)
        self._model = self._loaded.signatures["serving_default"]

        # set input and output signature names
        self._signature["input"] = list(
            self._model.structured_input_signature[1].keys())[0]
        self._signature["output"] = list(
            self._model.structured_outputs.keys())[0]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Performs model inference on given inputs, and returns
        inference's output after NMS.

        Args:
            inputs (np.ndarray): normalized in channels-last format,
            with batch axis.

        Returns:
            np.ndarray: inference's output after NMS
        """
        outputs = self._model(**{self._signature["input"]: inputs})[
                                     self._signature["output"]].numpy()
        denormalize(outputs, inputs.shape[-3:])
        return self._postprocess(outputs)[0]

    def __repr__(self) -> str:
        """Returns Model Information

        Returns:
            str: information string
        """
        return f"Yolov5-Tensorflow: {self._device}"

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
        outputs = non_max_suppression_tf(outputs)
        return outputs
