from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from seg.utils.hf_utils import mask_to_color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual comparison of SD vs DDPM generations (images + masks).")
    parser.add_argument("--original", type=Path, required=True, help="Path to the original image.")
    parser.add_argument("--gt-mask", type=Path, required=True, help="Path to the ground-truth mask.")
    parser.add_argument("--ddpm-x0", type=Path, required=True, help="DDPM decoded image (x0_hat).")
    parser.add_argument("--ddpm-x0-mask", type=Path, required=True, help="Segmentation mask for DDPM x0.")
    parser.add_argument("--sd-x0", type=Path, required=True, help="Stable Diffusion decoded image (x0_hat).")
    parser.add_argument("--sd-x0-mask", type=Path, required=True, help="Segmentation mask for SD x0.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to save the comparison figure.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(14.0, 8.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 14 8).",
    )
    return parser.parse_args()


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _is_label_mask(array: np.ndarray) -> bool:
    return array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 1)


def _load_mask(path: Path) -> Image.Image:
    with Image.open(path) as img:
        array = np.array(img)
    if _is_label_mask(array):
        label_array = array if array.ndim == 2 else array[:, :, 0]
        colorized = mask_to_color(label_array.astype(np.int32))
        return colorized
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array.astype(np.uint8))
    raise ValueError(f"Unsupported mask format for {path} with shape {array.shape}")


def _plot_panel(figsize: Tuple[float, float], images: list[Tuple[str, Image.Image]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for ax, (title, img) in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
    plt.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    images = [
        ("Original", _load_image(args.original)),
        ("DDPM x0", _load_image(args.ddpm_x0)),
        ("SD x0", _load_image(args.sd_x0)),
        ("GT mask", _load_mask(args.gt_mask)),
        ("DDPM x0 mask", _load_mask(args.ddpm_x0_mask)),
        ("SD x0 mask", _load_mask(args.sd_x0_mask)),
    ]

    fig = _plot_panel(tuple(args.figsize), images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
'''
python scripts/compare_sd_ddpm.py \
  --original /home/talmacsi/BME/weather_translation_clean/eval/translation_quantitative_semantic_consistency/rain2/munster_000009_000019_leftImg8bit_xs.png \
  --gt-mask /home/talmacsi/BME/weather_translation_clean/eval/translation_quantitative_semantic_consistency/rain2/munster_000009_000019_leftImg8bit_gt.png \
  --ddpm-x0 /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/munster/munster_000009_000019_leftImg8bit.png/munster_000009_000019_sr_denoised.png \
  --ddpm-x0-mask /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/munster/munster_000009_000019_leftImg8bit.png/munster_000009_000019_sr_pred_mask.png \
  --sd-x0 /home/talmacsi/BME/weather_translation_clean/eval/translation_quantitative_semantic_consistency/rain2/munster_000009_000019_leftImg8bit_xt.png \
  --sd-x0-mask /home/talmacsi/BME/weather_translation_clean/eval/translation_quantitative_semantic_consistency/rain2/munster_000009_000019_leftImg8bit_xt_pred.png \
  --output eval/sd_ddpm_comparison.png

python scripts/compare_sd_ddpm.py \
  --original /home/talmacsi/BME/weather_translation_clean/eval/translation_quantitative_semantic_consistency/rain2/frankfurt_000000_002963_leftImg8bit_xs.png \
  --gt-mask /home/talmacsi/BME/weather_translation_clean/eval/translation_quantitative_semantic_consistency/rain2/frankfurt_000000_002963_leftImg8bit_xs_pred.png \
  --sd-x0 /home/talmacsi/BME/weather_translation_clean/eval/translation_quantitative_semantic_consistency/rain2/frankfurt_000000_002963_leftImg8bit_xt.png \
  --sd-x0-mask /home/talmacsi/BME/weather_translation_clean/eval/translation_quantitative_semantic_consistency/rain2/frankfurt_000000_002963_leftImg8bit_xt_pred.png \
  --ddpm-x0 /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/frankfurt/frankfurt_000000_002963_leftImg8bit.png/frankfurt_000000_002963_sr_denoised.png \
  --ddpm-x0-mask /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/frankfurt/frankfurt_000000_002963_leftImg8bit.png/frankfurt_000000_002963_sr_pred_mask.png \
  --output eval/sd_ddpm_comparison2.png
'''
