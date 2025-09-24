from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from seg.dataloaders.labels import labels as CITYSCAPES_LABELS


MODEL_ZOO: Dict[str, str] = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "segformer_b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "segformer_b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


def resolve_model_name(name_or_path: str) -> str:
    key = name_or_path.lower()
    return MODEL_ZOO.get(key, name_or_path)


def model_to_folder_name(model_name: str) -> str:
    return model_name.replace('/', '__')


def build_joint_resize(size: Tuple[int, int]) -> Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]:
    height, width = size

    def _resize(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        resized_image = image.resize((width, height), Image.BILINEAR)
        resized_mask = mask.resize((width, height), Image.NEAREST)
        return resized_image, resized_mask

    return _resize


def get_class_names() -> List[str]:
    relevant = [label for label in CITYSCAPES_LABELS if 0 <= label.trainId < 255]
    relevant.sort(key=lambda lbl: lbl.trainId)
    return [lbl.name for lbl in relevant]


def get_palette() -> np.ndarray:
    relevant = [label for label in CITYSCAPES_LABELS if 0 <= label.trainId < 255]
    relevant.sort(key=lambda lbl: lbl.trainId)
    palette = np.array([label.color for label in relevant], dtype=np.uint8)
    palette = np.vstack([palette, np.array([[0, 0, 0]], dtype=np.uint8)])
    return palette


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    tensor = image_tensor.detach().cpu()
    if tensor.ndim != 3:
        raise ValueError("Expected image tensor with shape (C, H, W)")
    if tensor.dtype.is_floating_point:
        tensor = tensor.clamp(0.0, 1.0) * 255.0
    array = tensor.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def mask_to_color(mask: torch.Tensor | np.ndarray) -> Image.Image:
    if isinstance(mask, torch.Tensor):
        mask_array = mask.detach().cpu().numpy()
    else:
        mask_array = mask

    palette = get_palette()
    mask_array = np.clip(mask_array.astype(np.int64), 0, palette.shape[0] - 1)
    color = palette[mask_array]
    return Image.fromarray(color.astype(np.uint8))
