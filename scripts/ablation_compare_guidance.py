from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2x2 ablation grid: DDPM w/wo guidance vs SD w/wo guidance.")
    parser.add_argument("--ddpm-guided", type=Path, required=True, help="DDPM result with guidance.")
    parser.add_argument("--ddpm-plain", type=Path, required=True, help="DDPM result without guidance.")
    parser.add_argument("--sd-guided", type=Path, required=True, help="SD result with guidance.")
    parser.add_argument("--sd-plain", type=Path, required=True, help="SD result without guidance.")
    parser.add_argument(
        "--titles",
        nargs=4,
        metavar=("WITH_LORA", "WITHOUT_LORA", "WITH_LORA", "WITHOUT_LORA"),
        default=["WITH_LORA", "WITHOUT_LORA", "WITH_LORA", "WITHOUT_LORA"],
        help="Custom titles for the four cells (row-major order).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(10.0, 10.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 10 10).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to save the grid.")
    return parser.parse_args()


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _plot_grid(images: Tuple[Image.Image, Image.Image, Image.Image, Image.Image], titles: Tuple[str, str, str, str],
               figsize: Tuple[float, float]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, img, title in zip(axes.flatten(), images, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
    plt.tight_layout()
    return fig


def main() -> None:
    args = parse_args()

    images = (
        _load_rgb(args.ddpm_guided),
        _load_rgb(args.ddpm_plain),
        _load_rgb(args.sd_guided),
        _load_rgb(args.sd_plain),
    )
    titles = tuple(args.titles)  # type: ignore[assignment]
    fig = _plot_grid(images, titles, tuple(args.figsize))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved ablation grid to {args.output}")


if __name__ == "__main__":
    main()
'''

python scripts/ablation_compare_guidance.py \
  --ddpm-guided /home/talmacsi/BME/weather_translation_clean/eval_fog/cherrypicked_frankfurt_000000_009688_leftImg8bit_02/frankfurt/frankfurt_000000_009688_leftImg8bit.png \
  --ddpm-plain /home/talmacsi/BME/weather_translation_clean/eval_fog/cherrypicked_frankfurt_000000_009688_leftImg8bit_without_lora/frankfurt/frankfurt_000000_009688_leftImg8bit.png \
  --sd-guided /home/talmacsi/BME/weather_translation_clean/eval_night/cherrypicked_frankfurt_000000_013382_leftImg8bit_01/frankfurt/frankfurt_000000_013382_leftImg8bit.png \
  --sd-plain /home/talmacsi/BME/weather_translation_clean/eval_night/cherrypicked_frankfurt_000000_013382_leftImg8bit_without_lora/frankfurt/frankfurt_000000_013382_leftImg8bit.png \
  --output eval/ablation_grid_lora.png

'''
