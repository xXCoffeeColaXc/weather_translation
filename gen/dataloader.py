from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Sequence

from PIL import Image
from torch.utils import data
from torchvision import transforms

SUPPORTED_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg", ".png")
DEFAULT_SPLITS: Sequence[str] = ("train", "val", "test")
DEFAULT_CONDITIONS: Sequence[str] = ("rain", "fog", "night")

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


def _normalise_extensions(extensions: Sequence[str] | None) -> Sequence[str]:
    """Ensure extensions are lowercase and prefixed with a dot."""

    if not extensions:
        return SUPPORTED_EXTENSIONS

    normalised = []
    for ext in extensions:
        if not ext:
            continue
        ext = ext.lower()
        if not ext.startswith('.'):
            ext = f'.{ext}'
        normalised.append(ext)

    return normalised or SUPPORTED_EXTENSIONS


def _iter_image_files(directory: Path, extensions: Sequence[str]) -> Iterable[Path]:
    """Yield image files under directory matching the provided extensions."""

    if not directory.exists():
        logger.debug("Image directory %s does not exist; skipping.", directory)
        return []

    normalised = tuple(ext.lower() for ext in extensions)
    return (path for path in directory.rglob('*') if path.is_file() and path.suffix.lower() in normalised)


def _collect_image_paths(
    root_dir: Path,
    conditions: Sequence[str],
    splits: Iterable[str],
    extensions: Sequence[str],
) -> List[Path]:
    """Build a sorted, de-duplicated list of image paths."""

    image_paths: List[Path] = []
    seen = set()
    for condition in conditions:
        for split in splits:
            search_root = root_dir / condition / split
            for path in _iter_image_files(search_root, extensions):
                if path not in seen:
                    seen.add(path)
                    image_paths.append(path)

    image_paths.sort()
    return image_paths


class ACDCDataset(data.Dataset):
    """Dataset loader for ACDC weather translation images."""

    def __init__(
        self,
        root_dir: str | Path,
        selected_conditions: Sequence[str] | None = None,
        transform: Callable[[Image.Image], object] | None = None,
        splits: Iterable[str] = DEFAULT_SPLITS,
        extensions: Sequence[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.selected_conditions = tuple(selected_conditions or DEFAULT_CONDITIONS)
        if not self.selected_conditions:
            raise ValueError("selected_conditions must not be empty")

        self.transform = transform
        self.splits = tuple(splits)
        if not self.splits:
            raise ValueError("splits must not be empty")

        self.extensions = tuple(_normalise_extensions(extensions))

        self._img_paths: List[Path] = _collect_image_paths(self.root_dir, self.selected_conditions, self.splits,
                                                           self.extensions)
        if not self._img_paths:
            logger.warning(
                "No images found in %s for conditions=%s, splits=%s",
                self.root_dir,
                self.selected_conditions,
                self.splits,
            )

    @property
    def image_paths(self) -> Sequence[Path]:
        """The backing list of image paths."""

        return self._img_paths

    def add_images(self, image_dir: str | Path, splits: Iterable[str] | None = None) -> None:
        """Augment the dataset with additional images from another directory."""

        additional_root = Path(image_dir).expanduser()
        target_splits = tuple(splits) if splits is not None else self.splits
        extra_paths = _collect_image_paths(additional_root, self.selected_conditions, target_splits, self.extensions)

        if not extra_paths:
            logger.debug("No additional images found under %s", additional_root)
            return

        existing = set(self._img_paths)
        for path in extra_paths:
            if path not in existing:
                self._img_paths.append(path)
                existing.add(path)

        self._img_paths.sort()

    def __len__(self) -> int:
        """Return the number of available images."""

        return len(self._img_paths)

    def __getitem__(self, idx: int) -> "torch.Tensor":
        image_path = self._img_paths[idx]

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        if self.transform is not None:
            return self.transform(rgb_image)

        image_tensor = transforms.ToTensor()(rgb_image)
        return image_tensor * 2.0 - 1.0


def build_training_transform(image_size: int) -> transforms.Compose:
    """Default augmentation pipeline for model training."""

    return transforms.Compose([
        # transforms.Resize(
        #     image_size,
        #     transforms.InterpolationMode.BILINEAR,
        # ),
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.8, 1.0),  # small zoom-in/outs
            ratio=(0.95, 1.05),  # keep near square to avoid distortions
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomCrop(image_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
    ])


def get_loader(
    image_dir: str | Path,
    selected_attrs: Sequence[str],
    image_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
    *,
    shuffle: bool = True,
    pin_memory: bool = True,
    transform: Callable[[Image.Image], object] | None = None,
) -> data.DataLoader:
    """Build a `torch.utils.data.DataLoader` for the ACDC dataset."""

    if not selected_attrs:
        raise ValueError("selected_attrs must not be empty")

    dataset_transform = transform or build_training_transform(image_size)
    dataset = ACDCDataset(
        root_dir=image_dir,
        selected_conditions=selected_attrs,
        transform=dataset_transform,
    )

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _demo() -> None:
    """Small sanity check when running the module directly."""

    image_dir = Path("data/acdc/rgb_anon")
    loader = get_loader(
        image_dir=image_dir,
        selected_attrs=("rain",),
        image_size=128,
        batch_size=16,
        num_workers=2,
    )

    first_batch = next(iter(loader))
    logger.info("Loaded batch with shape: %s", tuple(first_batch.shape))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
