from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Sequence

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from seg.dataloaders.labels import labels as CITYSCAPES_LABELS

if TYPE_CHECKING:  # pragma: no cover - type checking helpers only
    from torch import Tensor

LOGGER = logging.getLogger(__name__)

LEFT_IMAGE_SUFFIX = "_leftImg8bit.png"
GT_LABEL_SUFFIX = "_gtFine_labelIds.png"

IGNORE_LABEL = 255


def _build_id_to_train_id_mapping() -> np.ndarray:
    """Return a vectorised mapping from raw Cityscapes IDs to train IDs."""

    mapping = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    for label in CITYSCAPES_LABELS:
        label_id = label.id
        train_id = label.trainId
        if label_id < 0:
            continue
        if train_id < 0:
            mapping[label_id] = IGNORE_LABEL
        else:
            mapping[label_id] = train_id
    return mapping


ID_TO_TRAIN_ID = _build_id_to_train_id_mapping()


def _ensure_tensor(data) -> "Tensor":
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    return tensor


def _default_image_transform(image: Image.Image) -> "Tensor":
    """Convert an RGB PIL image into a float tensor normalised to [0, 1]."""

    tensor = F.to_tensor(image)
    return tensor.to(dtype=torch.float32)


def encode_target(mask: np.ndarray) -> np.ndarray:
    """Map raw label IDs to train IDs using the precomputed lookup."""

    encoded = ID_TO_TRAIN_ID[mask.clip(min=0, max=ID_TO_TRAIN_ID.shape[0] - 1)]
    return encoded.astype(np.uint8)


def decode_target(train_mask: np.ndarray) -> np.ndarray:
    """Convert train IDs back to RGB colours using the label list."""

    colors: List[Sequence[int]] = []
    for label in CITYSCAPES_LABELS:
        if label.trainId < 0 or label.trainId == IGNORE_LABEL:
            continue
        colors.append(label.color)

    color_lookup = np.vstack(colors + [(0, 0, 0)])
    safe_mask = np.clip(train_mask, 0, color_lookup.shape[0] - 1)
    return color_lookup[safe_mask]


@dataclass(frozen=True)
class CityscapesSample:
    image_path: Path
    mask_path: Path


class CityscapesSegmentation(Dataset):
    """Dataset that yields paired Cityscapes images and segmentation masks."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        *,
        joint_transform: Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]] | None = None,
        image_transform: Callable[[Image.Image], "Tensor"] | None = None,
        target_transform: Callable[["Tensor"], "Tensor"] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.joint_transform = joint_transform
        self.image_transform = image_transform or _default_image_transform
        self.target_transform = target_transform

        self.left_dir = self.root_dir / "leftImg8bit" / split
        self.gt_dir = self.root_dir / "gtFine" / split

        if not self.left_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.left_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.gt_dir}")

        self.samples: List[CityscapesSample] = self._collect_samples()
        if not self.samples:
            LOGGER.warning("No Cityscapes samples found under %s", self.left_dir)

    def _collect_samples(self) -> List[CityscapesSample]:
        samples: List[CityscapesSample] = []
        for image_path in sorted(self.left_dir.rglob(f"*{LEFT_IMAGE_SUFFIX}")):
            relative = image_path.relative_to(self.left_dir)
            mask_name = relative.name.replace(LEFT_IMAGE_SUFFIX, GT_LABEL_SUFFIX)
            mask_path = self.gt_dir / relative.parent / mask_name
            if not mask_path.exists():
                LOGGER.debug("Missing mask for image %s", image_path)
                continue
            samples.append(CityscapesSample(image_path=image_path, mask_path=mask_path))
        return samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple["Tensor", "Tensor"]:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            rgb_image = image.convert("RGB")
        with Image.open(sample.mask_path) as mask:
            mask_image = mask.convert("L")

        if self.joint_transform is not None:
            rgb_image, mask_image = self.joint_transform(rgb_image, mask_image)

        image_tensor = self.image_transform(rgb_image)
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("image_transform must return a torch.Tensor")

        mask_array = np.array(mask_image, dtype=np.int16)
        encoded_mask = encode_target(mask_array)
        mask_tensor = _ensure_tensor(encoded_mask).to(dtype=torch.long)

        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_tensor)

        return image_tensor, mask_tensor


def build_cityscapes_loader(
    root_dir: str | Path = "data/cityscapes",
    split: str = "train",
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    joint_transform: Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]] | None = None,
    image_transform: Callable[[Image.Image], "Tensor"] | None = None,
    target_transform: Callable[["Tensor"], "Tensor"] | None = None,
) -> DataLoader:
    """Build a `DataLoader` yielding Cityscapes image/mask pairs."""

    dataset = CityscapesSegmentation(
        root_dir=root_dir,
        split=split,
        joint_transform=joint_transform,
        image_transform=image_transform,
        target_transform=target_transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _demo() -> None:
    """Run a tiny demo that prints loader output shapes and mask stats."""

    loader = build_cityscapes_loader(
        root_dir=Path("data/cityscapes"),
        split="train",
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    images, masks = batch

    LOGGER.info("Loaded batch images shape: %s", tuple(images.shape))
    LOGGER.info("Loaded batch masks shape: %s", tuple(masks.shape))
    LOGGER.info("Mask unique values: %s", torch.unique(masks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
