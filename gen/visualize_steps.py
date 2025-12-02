"""
Utility to visualize diffusion debug steps.

Given a steps directory containing files like `step_000_t0501_image.png` and
`step_000_t0501_mask.png` (and optionally `step_000_t0501_x0.png`), this script
stitches all steps horizontally and plots the generated images, predicted x0
images (if present), and their masks. Tick labels show the step index and
timestep.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _parse_steps(steps_dir: Path) -> tuple[list[dict], bool, dict[str, bool]]:
    pattern = re.compile(r"step_(\d+)_t(\d+)_image\.png$")
    entries = []
    presence = {
        "x0": [],
        "guidance_grad": [],
        "tr_grad": [],
        "guidance_overlay": [],
        "tr_overlay": [],
    }
    for image_path in sorted(steps_dir.glob("step_*_image.png")):
        match = pattern.match(image_path.name)
        if not match:
            continue
        step = int(match.group(1))
        timestep = int(match.group(2))
        mask_path = steps_dir / f"step_{match.group(1)}_t{match.group(2)}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path.name}")
        x0_path = steps_dir / f"step_{match.group(1)}_t{match.group(2)}_x0.png"
        guidance_grad = steps_dir / f"step_{match.group(1)}_t{match.group(2)}_guidance_grad.png"
        tr_grad = steps_dir / f"step_{match.group(1)}_t{match.group(2)}_tr_grad.png"
        guidance_overlay = steps_dir / f"step_{match.group(1)}_t{match.group(2)}_guidance_overlay.png"
        tr_overlay = steps_dir / f"step_{match.group(1)}_t{match.group(2)}_tr_overlay.png"

        x0_exists = x0_path.exists()
        guidance_grad_exists = guidance_grad.exists()
        tr_grad_exists = tr_grad.exists()
        guidance_overlay_exists = guidance_overlay.exists()
        tr_overlay_exists = tr_overlay.exists()

        presence["x0"].append(x0_exists)
        presence["guidance_grad"].append(guidance_grad_exists)
        presence["tr_grad"].append(tr_grad_exists)
        presence["guidance_overlay"].append(guidance_overlay_exists)
        presence["tr_overlay"].append(tr_overlay_exists)
        entries.append(
            {
                "step": step,
                "t": timestep,
                "image": image_path,
                "mask": mask_path,
                "x0": x0_path if x0_exists else None,
                "guidance_grad": guidance_grad if guidance_grad_exists else None,
                "tr_grad": tr_grad if tr_grad_exists else None,
                "guidance_overlay": guidance_overlay if guidance_overlay_exists else None,
                "tr_overlay": tr_overlay if tr_overlay_exists else None,
            }
        )
    if not entries:
        raise FileNotFoundError(f"No step_*_image.png files found in {steps_dir}")
    entries.sort(key=lambda x: x["step"])
    # Optional artifacts: allow missing; visualizer will render black tiles when absent
    availability: dict[str, bool] = {}
    for key in presence.keys():
        if key == "x0":
            continue
        availability[key] = any(entry[key] is not None for entry in entries)

    # x0 remains strict: either all present or none
    has_x0 = all(presence["x0"]) if presence["x0"] else False
    if any(presence["x0"]) and not has_x0:
        missing = [e["image"].name for e in entries if e["x0"] is None]
        raise FileNotFoundError(f"x0 files missing for steps: {', '.join(missing)}")

    return entries, has_x0, availability


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


def build_rows(
    entries: list[dict],
    tile_height: int,
    *,
    include_x0: bool,
    include_guidance_grad: bool,
    include_tr_grad: bool,
    include_guidance_overlay: bool,
    include_tr_overlay: bool,
):
    image_tiles = []
    x0_tiles = []
    mask_tiles = []
    guidance_grad_tiles = []
    tr_grad_tiles = []
    guidance_overlay_tiles = []
    tr_overlay_tiles = []
    xtick_positions = []
    xtick_labels = []
    cursor = 0

    for entry in entries:
        image_arr = _load_and_resize(entry["image"], tile_height)
        mask_arr = _load_and_resize(entry["mask"], tile_height)
        tile_width = image_arr.shape[1]

        def _load_optional(path: Path | None) -> np.ndarray:
            if path is None:
                return np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            return _load_and_resize(path, tile_height)

        x0_arr = _load_and_resize(entry["x0"], tile_height) if include_x0 else None
        guidance_grad_arr = _load_optional(entry["guidance_grad"]) if include_guidance_grad else None
        tr_grad_arr = _load_optional(entry["tr_grad"]) if include_tr_grad else None
        guidance_overlay_arr = _load_optional(entry["guidance_overlay"]) if include_guidance_overlay else None
        tr_overlay_arr = _load_optional(entry["tr_overlay"]) if include_tr_overlay else None

        center = cursor + tile_width / 2
        xtick_positions.append(center)
        xtick_labels.append(f"step {entry['step']}\n(t={entry['t']})")

        cursor += tile_width
        image_tiles.append(image_arr)
        if include_x0:
            x0_tiles.append(x0_arr)
        mask_tiles.append(mask_arr)
        if include_guidance_grad:
            guidance_grad_tiles.append(guidance_grad_arr)
        if include_tr_grad:
            tr_grad_tiles.append(tr_grad_arr)
        if include_guidance_overlay:
            guidance_overlay_tiles.append(guidance_overlay_arr)
        if include_tr_overlay:
            tr_overlay_tiles.append(tr_overlay_arr)

    rows = [("Images", np.concatenate(image_tiles, axis=1))]
    if include_x0:
        rows.append(("x0", np.concatenate(x0_tiles, axis=1)))
    if include_guidance_grad:
        rows.append(("Guidance grad", np.concatenate(guidance_grad_tiles, axis=1)))
    if include_tr_grad:
        rows.append(("TR grad", np.concatenate(tr_grad_tiles, axis=1)))
    if include_guidance_overlay:
        rows.append(("Guidance overlay", np.concatenate(guidance_overlay_tiles, axis=1)))
    if include_tr_overlay:
        rows.append(("TR overlay", np.concatenate(tr_overlay_tiles, axis=1)))
    rows.append(("Masks", np.concatenate(mask_tiles, axis=1)))
    return rows, xtick_positions, xtick_labels


def plot_steps(
    rows: list[tuple[str, np.ndarray]],
    xtick_positions: list[float],
    xtick_labels: list[str],
    *,
    output: Path,
    dpi: int,
    show: bool,
) -> None:
    if not rows:
        raise ValueError("No rows to plot.")

    width = rows[0][1].shape[1]
    if any(row[1].shape[1] != width for row in rows):
        raise ValueError("All rows must have the same width after concatenation.")
    height = sum(row[1].shape[0] for row in rows)
    fig_width_in = max(8.0, width / dpi)
    fig_height_in = max(4.0, height / dpi)

    fig, axes = plt.subplots(len(rows), 1, figsize=(fig_width_in, fig_height_in), dpi=dpi, constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]

    for ax, (title, data) in zip(axes, rows):
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
    parser = argparse.ArgumentParser(description="Visualize step debug images, predicted x0, masks, and gradients.")
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

    entries, has_x0, availability = _parse_steps(steps_dir)
    tile_height = _compute_tile_height(entries, args.tile_height, args.max_width)

    rows, xtick_positions, xtick_labels = build_rows(
        entries,
        tile_height,
        include_x0=has_x0,
        include_guidance_grad=availability.get("guidance_grad", False),
        include_tr_grad=availability.get("tr_grad", False),
        include_guidance_overlay=availability.get("guidance_overlay", False),
        include_tr_overlay=availability.get("tr_overlay", False),
    )

    output_path = args.output or steps_dir / "steps_overview.png"
    plot_steps(
        rows,
        xtick_positions,
        xtick_labels,
        output=output_path,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
