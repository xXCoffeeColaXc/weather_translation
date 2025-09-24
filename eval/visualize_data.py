from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F

from gen.dataloader import ACDCDataset
from seg.dataloaders.cityscapes import (
    CityscapesSegmentation,
    decode_target as decode_cityscapes_mask,
)


LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path("eval/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pipelines represented as (height, width)
PIPELINE_SIZES: Dict[str, Tuple[int, int]] = {
    "square_512": (512, 512),
    "widescreen_768x384": (384, 768),
}


def _build_acdc_transform(size: Tuple[int, int]) -> transforms.Compose:
    """Return a deterministic transform pipeline for ACDC samples."""

    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
        ]
    )


def _denormalize_acdc(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] tensor back to a float image in [0, 1]."""

    image = ((image_tensor.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def visualize_acdc(
    root_dir: str | Path = "data/acdc/rgb_anon",
    selected_conditions: Iterable[str] | None = None,
    splits: Iterable[str] = ("train",),
    samples_per_grid: int = 4,
) -> None:
    """Generate grid plots showcasing ACDC imagery across configured pipelines."""

    root_dir = Path(root_dir)
    conditions = tuple(selected_conditions) if selected_conditions is not None else None

    for label, size in PIPELINE_SIZES.items():
        transform = _build_acdc_transform(size)
        dataset = ACDCDataset(
            root_dir=root_dir,
            selected_conditions=conditions,
            transform=transform,
            splits=splits,
        )

        if len(dataset) == 0:
            LOGGER.warning("ACDC dataset is empty for pipeline %s at %s", label, root_dir)
            continue

        num_samples = min(samples_per_grid, len(dataset))
        fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
        axes = np.atleast_1d(axes)

        for ax, idx in zip(axes, range(num_samples)):
            image_tensor = dataset[idx]
            image_np = _denormalize_acdc(image_tensor)
            ax.imshow(image_np)
            ax.set_title(f"Sample {idx}")
            ax.axis("off")

        fig.suptitle(f"ACDC samples – {label} ({size[1]}x{size[0]})")
        fig.tight_layout()
        output_path = OUTPUT_DIR / f"acdc_{label}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        LOGGER.info(
            "Saved ACDC visualization for %s to %s (samples=%d)",
            label,
            output_path,
            num_samples,
        )


def _resize_pair(size: Tuple[int, int]):
    """Return a joint transform that resizes image/mask pairs to the target size."""

    def _transform(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        resized_image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        resized_mask = F.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        return resized_image, resized_mask

    return _transform


def _cityscapes_image_transform(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image into a float tensor in [0, 1]."""

    return F.to_tensor(image).to(dtype=torch.float32)


def visualize_cityscapes(
    root_dir: str | Path = "data/cityscapes",
    split: str = "train",
    samples_per_pipeline: int = 2,
) -> None:
    """Generate demo visualizations for Cityscapes with original and colour masks."""

    root_dir = Path(root_dir)

    for label, size in PIPELINE_SIZES.items():
        joint_transform = _resize_pair(size)
        dataset = CityscapesSegmentation(
            root_dir=root_dir,
            split=split,
            joint_transform=joint_transform,
            image_transform=_cityscapes_image_transform,
        )

        if len(dataset) == 0:
            LOGGER.warning("Cityscapes dataset is empty for pipeline %s at %s", label, root_dir)
            continue

        num_samples = min(samples_per_pipeline, len(dataset))
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
        axes = np.atleast_2d(axes)

        for row_idx in range(num_samples):
            image_tensor, mask_tensor = dataset[row_idx]

            image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
            mask_np = mask_tensor.detach().cpu().numpy()
            color_mask = decode_cityscapes_mask(mask_np)

            axes[row_idx, 0].imshow(image_np)
            axes[row_idx, 0].set_title(f"Image {row_idx}")
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(color_mask.astype(np.uint8))
            axes[row_idx, 1].set_title(f"Mask {row_idx}")
            axes[row_idx, 1].axis("off")

            unique_ids = torch.unique(mask_tensor)
            LOGGER.info(
                "Cityscapes %s sample %d – mask ids: %s",
                label,
                row_idx,
                unique_ids.tolist(),
            )

        fig.suptitle(f"Cityscapes samples – {label} ({size[1]}x{size[0]})")
        fig.tight_layout()
        output_path = OUTPUT_DIR / f"cityscapes_{label}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        LOGGER.info(
            "Saved Cityscapes visualization for %s to %s (samples=%d)",
            label,
            output_path,
            num_samples,
        )


def main() -> None:
    visualize_acdc()
    visualize_cityscapes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
