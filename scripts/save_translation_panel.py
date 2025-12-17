from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from PIL import Image

# Default ordering for the combined panel (columns).
DEFAULT_VARIANTS = ["gt", "xs_pred", "xs", "xt_pred", "xt"]

VARIANT_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "gt": ("_gt.png",),
    "xs_pred": ("_xs_pred.png", "_xs_ped.png"),
    "xs": ("_xs.png",),
    "xt_pred": ("_xt_pred.png",),
    "xt": ("_xt.png",),
}

VARIANT_TITLES: Dict[str, str] = {
    "gt": "Ground truth",
    "xs_pred": "Xs (pred)",
    "xs": "Xs",
    "xt_pred": "Xt (pred)",
    "xt": "Xt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Combine translation visuals (gt/xs/xs_pred/xt_pred/xt) into a single horizontal panel."))
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing the image variants (e.g. eval/translation_quantitative_semantic_consistency/night2).",
    )
    parser.add_argument(
        "base_names",
        nargs="+",
        type=str,
        help=("One or more base image names without the variant suffix "
              "(e.g. frankfurt_000000_009688_leftImg8bit). Each becomes a row."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Path to save the combined figure "
              "(default: <folder>/<first_base_name>_panel.png, or _<N>rows for multiple)."),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_SUFFIXES.keys()),
        default=DEFAULT_VARIANTS,
        help="Variant keys to include, in order.",
    )
    return parser.parse_args()


def _normalize_base_name(base_name: str) -> str:
    # Strip any extension so either foo or foo.png works.
    return Path(base_name).stem


def _find_image(folder: Path, base_stem: str, suffixes: Sequence[str]) -> Path:
    for suffix in suffixes:
        candidate = folder / f"{base_stem}{suffix}"
        if candidate.exists():
            return candidate
    joined = ", ".join(suffixes)
    raise FileNotFoundError(f"No file found for {base_stem} with suffixes: {joined}")


def _load_images(image_paths: Iterable[Path]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def _compute_grid_figsize(sample_image: Image.Image, rows: int, cols: int) -> Tuple[float, float]:
    """Estimate a reasonable figure size for a grid layout."""
    width, height = sample_image.size
    aspect = width / height if height else 1.0
    row_height = 4.0
    col_width = max(3.0, row_height * aspect)
    return col_width * cols, row_height * rows


def save_panel(folder: Path, base_names: Sequence[str], variants: Sequence[str], output_path: Path) -> Path:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    if not base_names:
        raise ValueError("No base names provided.")

    base_stems = [_normalize_base_name(name) for name in base_names]

    titles: List[str] = [VARIANT_TITLES.get(variant, variant) for variant in variants]

    rows: List[List[Path]] = []
    for base_stem in base_stems:
        row_paths: List[Path] = []
        for variant in variants:
            suffixes = VARIANT_SUFFIXES.get(variant)
            if not suffixes:
                raise KeyError(f"Unknown variant key: {variant}")
            path = _find_image(folder, base_stem, suffixes)
            row_paths.append(path)
        rows.append(row_paths)

    flat_paths = [path for row in rows for path in row]
    images = _load_images(flat_paths)

    sample_image = images[0]
    fig_width, fig_height = _compute_grid_figsize(sample_image, len(base_stems), len(variants))

    fig, axes = plt.subplots(len(base_stems), len(variants), figsize=(fig_width, fig_height))

    if len(base_stems) == 1 and len(variants) == 1:
        axes_grid = [[axes]]  # type: ignore[list-item]
    elif len(base_stems) == 1:
        axes_grid = [axes]  # type: ignore[list-item]
    elif len(variants) == 1:
        axes_grid = [[ax] for ax in axes]  # type: ignore[arg-type]
    else:
        axes_grid = axes  # type: ignore[assignment]

    img_iter = iter(images)
    for row_idx, base_stem in enumerate(base_stems):
        for col_idx in range(len(variants)):
            ax = axes_grid[row_idx][col_idx]
            image = next(img_iter)
            ax.imshow(image)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(titles[col_idx])
        axes_grid[row_idx][0].set_ylabel(base_stem, rotation=90, labelpad=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    folder: Path = args.folder
    base_names: List[str] = args.base_names
    variants: List[str] = args.variants

    output: Path
    if args.output is not None:
        output = args.output
    else:
        first = _normalize_base_name(base_names[0])
        suffix = "" if len(base_names) == 1 else f"_{len(base_names)}rows"
        output = folder / f"{first}_panel{suffix}.png"

    saved_path = save_panel(folder, base_names, variants, output)
    print(f"Saved panel to {saved_path}")


if __name__ == "__main__":
    main()
'''
python scripts/save_translation_panel.py \
    eval/translation_quantitative_semantic_consistency/fog2 \
        frankfurt_000000_009688_leftImg8bit \
        lindau_000004_000019_leftImg8bit \
        munster_000025_000019_leftImg8bit \
    --output eval/translation_quantitative_semantic_consistency/fog2/fog2_panel.png
'''
