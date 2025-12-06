from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

from seg.dataloaders.cityscapes import (
    IGNORE_LABEL,
    encode_target as _encode_cityscapes_target,
    _default_image_transform,
    _ensure_tensor,
)

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIX = "_rgb_anon.png"
IMAGE_REF_SUFFIX = "_rgb_ref_anon.png"
GT_LABEL_SUFFIX = "_gt_labelIds.png"
GT_TRAIN_SUFFIX = "_gt_labelTrainIds.png"

DEFAULT_WEATHERS = ("fog", "night", "rain", "snow")


def _list_weathers(weather: str | Iterable[str]) -> List[str]:
    if isinstance(weather, str):
        if weather == "all":
            return list(DEFAULT_WEATHERS)
        return [weather]
    return list(weather)


@dataclass(frozen=True)
class ACDCSample:
    image_path: Path
    mask_path: Path
    ref_image_path: Path


class ACDCSegmentation(Dataset):
    """
    Dataset that yields (image, mask, ref_image) triplets for the ACDC dataset.

    Directory structure (per weather condition):
      - rgb_anon/<weather>/<split>/**/*_rgb_anon.png
      - rgb_anon/<weather>/<split>_ref/**/*_rgb_ref_anon.png
      - gt/<weather>/<split>/**/*_gt_labelIds.png (or *_gt_labelTrainIds.png)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        *,
        weather: str | Iterable[str] = "all",
        joint_transform: Callable[[Image.Image, Image.Image, Image.Image], tuple[Image.Image, Image.Image, Image.Image]]
        | None = None,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.weathers = _list_weathers(weather)
        self.joint_transform = joint_transform
        self.image_transform = image_transform or _default_image_transform
        self.target_transform = target_transform

        self.rgb_root = self.root_dir / "rgb_anon"
        self.gt_root = self.root_dir / "gt"
        if not self.rgb_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.rgb_root}")
        if not self.gt_root.exists():
            raise FileNotFoundError(f"Mask root not found: {self.gt_root}")

        self.samples: list[ACDCSample] = self._collect_samples()
        if not self.samples:
            LOGGER.warning("No ACDC samples found under %s", self.root_dir)

    def _collect_samples(self) -> list[ACDCSample]:
        samples: list[ACDCSample] = []
        for weather in self.weathers:
            img_dir = self.rgb_root / weather / self.split
            ref_dir = self.rgb_root / weather / f"{self.split}_ref"
            mask_dir = self.gt_root / weather / self.split

            if not mask_dir.exists():
                LOGGER.debug("Mask directory missing for weather '%s': %s", weather, mask_dir)
                continue
            if not img_dir.exists():
                LOGGER.debug("Image directory missing for weather '%s': %s", weather, img_dir)
                continue
            if not ref_dir.exists():
                LOGGER.debug("Ref image directory missing for weather '%s': %s", weather, ref_dir)
                continue

            seen: set[Path] = set()

            # Prefer raw labelIds; fall back to labelTrainIds if needed.
            label_paths = list(mask_dir.rglob(f"*{GT_LABEL_SUFFIX}"))
            train_paths = list(mask_dir.rglob(f"*{GT_TRAIN_SUFFIX}"))

            def _maybe_add(mask_path: Path) -> None:
                rel = mask_path.relative_to(mask_dir)
                base = rel.name.replace(GT_LABEL_SUFFIX, "").replace(GT_TRAIN_SUFFIX, "")
                key = rel.parent / base
                if key in seen:
                    return
                image_path = img_dir / rel.parent / f"{base}{IMAGE_SUFFIX}"
                ref_image_path = ref_dir / rel.parent / f"{base}{IMAGE_REF_SUFFIX}"

                if not image_path.exists():
                    LOGGER.debug("Missing image for mask %s", mask_path)
                    return
                if not ref_image_path.exists():
                    LOGGER.debug("Missing ref image for mask %s", mask_path)
                    return

                seen.add(key)
                samples.append(ACDCSample(image_path=image_path, mask_path=mask_path, ref_image_path=ref_image_path))

            for path in sorted(label_paths):
                _maybe_add(path)
            for path in sorted(train_paths):
                _maybe_add(path)

        return samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def _encode_mask(self, mask_path: Path, mask_image: Image.Image) -> np.ndarray:
        array = np.array(mask_image, dtype=np.int16)
        if GT_TRAIN_SUFFIX in mask_path.name:
            return array.astype(np.uint8)
        return _encode_cityscapes_target(array)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        with Image.open(sample.image_path) as img:
            rgb_image = img.convert("RGB")
        with Image.open(sample.ref_image_path) as ref_img:
            ref_image = ref_img.convert("RGB")
        with Image.open(sample.mask_path) as mask:
            mask_image = mask.convert("L")

        if self.joint_transform is not None:
            try:
                rgb_image, mask_image, ref_image = self.joint_transform(rgb_image, mask_image, ref_image)
            except TypeError as exc:
                raise TypeError("joint_transform for ACDCSegmentation must accept (image, mask, ref_image)") from exc

        image_tensor = self.image_transform(rgb_image)
        ref_tensor = self.image_transform(ref_image)
        if not isinstance(image_tensor, torch.Tensor) or not isinstance(ref_tensor, torch.Tensor):
            raise TypeError("image_transform must return torch.Tensor for both image and ref_image")

        encoded_mask = self._encode_mask(sample.mask_path, mask_image)
        mask_tensor = _ensure_tensor(encoded_mask).to(dtype=torch.long)

        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_tensor)

        return ref_tensor, mask_tensor, image_tensor


def build_acdc_loader(
    root_dir: str | Path = "data/acdc",
    split: str = "train",
    *,
    weather: str | Iterable[str] = "all",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    joint_transform: Callable[[Image.Image, Image.Image, Image.Image], tuple[Image.Image, Image.Image, Image.Image]] |
    None = None,
    image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> DataLoader:
    """Build a `DataLoader` yielding (image, mask, ref_image) batches for ACDC."""

    dataset = ACDCSegmentation(
        root_dir=root_dir,
        split=split,
        weather=weather,
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
    """Tiny sanity check that loads one batch and prints shapes/stats."""

    from torchvision import transforms
    import torchvision.transforms.functional as TF
    from typing import Tuple
    from gen_ddpm.sgg_sampler import MEAN, STD

    def _build_acdc_transforms(image_size: int,) -> Tuple[Callable, Callable]:
        """Resize → center-crop to a square and normalise with dataset mean/std for the DDPM."""

        def joint_transform(image, mask, ref_image):
            image = TF.resize(
                image,
                (image_size, image_size * 2),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            image = TF.center_crop(image, image_size)

            ref_image = TF.resize(
                ref_image,
                (image_size, image_size * 2),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            ref_image = TF.center_crop(ref_image, image_size)

            mask = TF.resize(
                mask,
                (image_size, image_size * 2),
                interpolation=transforms.InterpolationMode.NEAREST,
                antialias=False,
            )
            mask = TF.center_crop(mask, image_size)

            return image, mask, ref_image

        def image_transform(image):
            tensor = TF.to_tensor(image)
            return transforms.Normalize(mean=MEAN, std=STD)(tensor)

        return joint_transform, image_transform

    joint_transform, image_transform = _build_acdc_transforms(image_size=128)

    loader = build_acdc_loader(
        root_dir=Path("data/acdc"),
        split="val",
        batch_size=2,
        shuffle=False,
        num_workers=0,
        joint_transform=joint_transform,
        image_transform=image_transform,
    )

    images, masks, refs = next(iter(loader))
    LOGGER.info("Batch images shape: %s", tuple(images.shape))
    LOGGER.info("Batch refs shape: %s", tuple(refs.shape))
    LOGGER.info("Batch masks shape: %s", tuple(masks.shape))
    LOGGER.info("Mask unique values: %s", torch.unique(masks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
