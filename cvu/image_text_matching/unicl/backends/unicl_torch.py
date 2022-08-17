import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from timm.data.transforms import _pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from cvu.interface.model import IModel
from cvu.utils.general import get_path
from cvu.image_text_matching.unicl.backends.common import download_weights
from cvu.image_text_matching.unicl.backends.config import get_config
from cvu.image_text_matching.unicl.backends.model import build_model


class UniCL(IModel):

    def __init__(self,
                 weight: str = "swin_b",
                 config: str = "swin_b",
                 device="auto") -> None:
        # initiate class attributes
        self._device = None
        self._model = None
        self._transform = None
        self._config = self._build_config(config)
        # TODO - maybe move this to a file?
        self._query = [
            "a photo of a cat", "a photo of a dog", "a photo of a person"
        ]

        # setup device
        self._set_device(device)

        # load model
        self._load_model(weight)

        # set transforms
        self._transforms()

    def _transforms(self) -> None:
        """Internally setup image transforms for the model.
        """
        # declare pytorch transforms
        self._transform = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=_pil_interp("bicubic")),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def _set_device(self, device: str) -> None:
        """Internally setup torch.device

        Args:
            device (str): name of the device to be used.
        """
        if device in ('auto', 'gpu'):
            self._device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

    def _build_config(self, config: str) -> dict:
        """Built config for the model.

        Args:
            config (str): name of the config to be used.

        Returns:
            dict: model configuration
        """
        if config == 'swin_b':
            return get_config(
                "cvu/image_text_matching/unicl/backends/configs/unicl_swin_base.yaml"
            )
        elif config == 'swin_t':
            return get_config(
                "cvu/image_text_matching/unicl/backends/configs/unicl_swin_tiny.yaml"
            )

    def _load_model(self, weight: str) -> None:
        """Internally load torch model

        Args:
            weight (str): path to torch .pth weight files or predefined-identifiers (such as swin_b, swin_t)
        """
        # attempt to load predefined weights
        if not os.path.exists(weight):

            # get path to pretrained weights
            weight = get_path(__file__, "weights", f"{weight}.pth")

            # download weights if not already downloaded
            download_weights(weight, "torch")

        # build model
        self._model = build_model(self._config).to(self._device)
        # load checkpoint
        checkpoint = torch.load(weight, map_location=self._device)
        self._model.load_state_dict(checkpoint['model'], strict=False)

        # set model to eval mode
        self._model.eval()

    def __call__(
            self, inputs: np.ndarray,
            query: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        # apply preprocessing to image
        inputs = self._preprocess(inputs)

        # apply preprocessing to query
        query = self._preprocess_query(query)

        with torch.no_grad():
            image_features, text_features, T = self._model(inputs, query)
            probs = (T * image_features @ text_features.t()).softmax(
                dim=-1).cpu().numpy()

        return image_features.cpu().numpy(), text_features.cpu().numpy(
        ), probs, self._query

    def __repr__(self) -> str:
        """Return Model Information.

        Returns:
            str: information string.
        """
        return f"UniCL: {self._device.type}"

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Apply image transforms for the model.

        Args:
            image (np.ndarray): image in BGR format.

        Returns:
            torch.Tensor: transformed image.
        """
        # get a pil image
        image = Image.fromarray(image).convert("RGB")

        # apply transforms
        return self._transform(image).unsqueeze(0).to(self._device)

    def _preprocess_query(self, query: str) -> torch.Tensor:
        # add query to the list of queries, if not already in there
        if query not in self._query:
            self._query.append(query)

        # apply tokenizer
        query = self._model.tokenizer(self._query,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=77,
                                      return_tensors='pt').to(self._device)

        return query