"""
Utility to visualize diffusion debug steps.

Given a steps directory containing files like `step_000_t0501_image.png` and
`step_000_t0501_mask.png`, this script stitches all steps horizontally and
plots two rows: the generated images and their masks. Tick labels show the
step index and timestep.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _parse_steps(steps_dir: Path):
    pattern = re.compile(r"step_(\d+)_t(\d+)_image\.png$")
    entries = []
    for image_path in sorted(steps_dir.glob("step_*_image.png")):
        match = pattern.match(image_path.name)
        if not match:
            continue
        step = int(match.group(1))
        timestep = int(match.group(2))
        mask_path = steps_dir / f"step_{match.group(1)}_t{match.group(2)}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path.name}")
        entries.append({"step": step, "t": timestep, "image": image_path, "mask": mask_path})
    if not entries:
        raise FileNotFoundError(f"No step_*_image.png files found in {steps_dir}")
    entries.sort(key=lambda x: x["step"])
    return entries


def _compute_tile_height(
    entries: Iterable[dict],
    base_height: int,
    max_width: int | None,
) -> int:
    """
    Returns the tile height, optionally down-scaling if a max width is provided.
    Without max_width the plot width grows with the number of items.
    """
    widths = []
    for entry in entries:
        with Image.open(entry["image"]) as img:
            w, h = img.size
        widths.append(w * (base_height / h))
    total_width = sum(widths)
    if max_width is None or total_width <= 0:
        return max(1, int(base_height))
    scale = min(1.0, max_width / total_width)
    return max(1, int(base_height * scale))


def _load_and_resize(path: Path, height: int) -> np.ndarray:
    with Image.open(path) as img:
        w, h = img.size
        new_w = max(1, int(round(w * (height / h))))
        resized = img.resize((new_w, height), Image.BILINEAR)
        return np.asarray(resized)


def build_rows(entries: list[dict], tile_height: int):
    image_tiles = []
    mask_tiles = []
    xtick_positions = []
    xtick_labels = []
    cursor = 0

    for entry in entries:
        image_arr = _load_and_resize(entry["image"], tile_height)
        mask_arr = _load_and_resize(entry["mask"], tile_height)

        tile_width = image_arr.shape[1]
        center = cursor + tile_width / 2
        xtick_positions.append(center)
        xtick_labels.append(f"step {entry['step']}\n(t={entry['t']})")

        cursor += tile_width
        image_tiles.append(image_arr)
        mask_tiles.append(mask_arr)

    image_row = np.concatenate(image_tiles, axis=1)
    mask_row = np.concatenate(mask_tiles, axis=1)
    return image_row, mask_row, xtick_positions, xtick_labels


def plot_steps(
    image_row: np.ndarray,
    mask_row: np.ndarray,
    xtick_positions: list[float],
    xtick_labels: list[str],
    *,
    output: Path,
    dpi: int,
    show: bool,
) -> None:
    width = image_row.shape[1]
    height = image_row.shape[0] + mask_row.shape[0]
    fig_width_in = max(8.0, width / dpi)
    fig_height_in = max(4.0, height / dpi)

    fig, axes = plt.subplots(2, 1, figsize=(fig_width_in, fig_height_in), dpi=dpi, constrained_layout=True)
    axes_data = [("Images", image_row), ("Masks", mask_row)]

    for ax, (title, data) in zip(axes, axes_data):
        ax.imshow(data)
        ax.set_title(title)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_xlim(0, data.shape[1])
        ax.set_yticks([])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize step debug images and masks.")
    parser.add_argument("steps_dir", type=Path, help="Path to steps directory (contains step_*_image.png files).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <steps_dir>/steps_overview.png).",
    )
    parser.add_argument("--tile-height", type=int, default=256, help="Base height for each tile before width scaling.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Optional max combined width in pixels; if omitted, width grows with the number of steps.",
    )
    parser.add_argument("--dpi", type=int, default=100, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Display the plot in addition to saving.")
    args = parser.parse_args()

    steps_dir = args.steps_dir.expanduser().resolve()
    if not steps_dir.exists():
        raise FileNotFoundError(f"Steps directory not found: {steps_dir}")

    entries = _parse_steps(steps_dir)
    tile_height = _compute_tile_height(entries, args.tile_height, args.max_width)

    image_row, mask_row, xtick_positions, xtick_labels = build_rows(entries, tile_height)

    output_path = args.output or steps_dir / "steps_overview.png"
    plot_steps(
        image_row,
        mask_row,
        xtick_positions,
        xtick_labels,
        output=output_path,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
