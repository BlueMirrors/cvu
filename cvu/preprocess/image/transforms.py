from configparser import Interpolation
import numpy as np
from PIL import Image

from torch import Tensor
from torchvision import transforms
from timm.data.transforms import _pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def transform(image: np.ndarray) -> Tensor:
    """Apply image transforms for the model.

    Args:
        image (np.ndarray): image in BGR format.

    Returns:
        Tensor: transformed image.
    """
    # declare pytorch transforms
    _transform = transforms.Compose([
        transforms.Resize((224, 224), Interpolation=_pil_interp("bicubic")),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    # get a pil image
    image = Image.fromarray(image).convert("RGB")

    # apply transforms
    return _transform(image).unsqueeze(0)