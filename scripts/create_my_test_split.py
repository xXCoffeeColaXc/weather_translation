from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

from seg.dataloaders.cityscapes import LEFT_IMAGE_SUFFIX, GT_LABEL_SUFFIX


def _read_path_list(list_file: Path | None, cli_paths: List[str]) -> List[Path]:
    paths: List[Path] = []
    if list_file is not None:
        for line in list_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            paths.append(Path(line))
    paths.extend(Path(p) for p in cli_paths)
    return paths


def _locate_root(image_path: Path) -> tuple[Path, Path]:
    """
    Given a full leftImg8bit image path, return (root_dir, left_dir).

    Example:
      /.../data/cityscapes/leftImg8bit/train/berlin/img_leftImg8bit.png
      -> (data/cityscapes, data/cityscapes/leftImg8bit)
    """
    parts = image_path.resolve().parts
    try:
        idx = parts.index("leftImg8bit")
    except ValueError as exc:
        raise ValueError(f"'leftImg8bit' not found in {image_path}") from exc
    left_dir = Path(*parts[: idx + 1])
    root_dir = left_dir.parent
    return root_dir, left_dir


def _copy_pair(image_path: Path, split_name: str, *, dry_run: bool = False) -> None:
    # Work with absolute paths to avoid relative_to mismatches.
    image_path = image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.name.endswith(LEFT_IMAGE_SUFFIX):
        raise ValueError(f"Unexpected image suffix for {image_path.name}")

    root_dir, left_dir = _locate_root(image_path)
    relative = image_path.relative_to(left_dir)
    if len(relative.parts) < 2:
        raise ValueError(f"Image path missing split/city structure: {image_path}")

    split = relative.parts[0]
    inner_path = Path(*relative.parts[1:])

    # Source mask path (keep original split for the source).
    mask_name = image_path.name.replace(LEFT_IMAGE_SUFFIX, GT_LABEL_SUFFIX)
    city_subpath = inner_path.parent
    src_mask = root_dir / "gtFine" / split / city_subpath / mask_name
    if not src_mask.exists():
        raise FileNotFoundError(f"Mask not found for {image_path} at {src_mask}")

    # Destination paths under the new split.
    dest_image = root_dir / "leftImg8bit" / split_name / inner_path
    dest_mask = root_dir / "gtFine" / split_name / city_subpath / mask_name

    if dry_run:
        print(f"[DRY RUN] Copy {image_path} -> {dest_image}")
        print(f"[DRY RUN] Copy {src_mask} -> {dest_mask}")
        return

    dest_image.parent.mkdir(parents=True, exist_ok=True)
    dest_mask.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dest_image)
    shutil.copy2(src_mask, dest_mask)


def process_paths(image_paths: Iterable[Path], split_name: str, *, dry_run: bool = False) -> None:
    for path in image_paths:
        _copy_pair(path, split_name, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a custom Cityscapes split (default: my_test) by copying "
            "selected image/mask pairs while preserving the directory structure."
        )
    )
    parser.add_argument(
        "image_paths",
        nargs="*",
        help="Full paths to Cityscapes leftImg8bit images.",
    )
    parser.add_argument(
        "-f",
        "--list-file",
        type=Path,
        help="Optional text file with one image path per line.",
    )
    parser.add_argument(
        "--split-name",
        default="my_test",
        help="Name of the new split to create under leftImg8bit/ and gtFine/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _read_path_list(args.list_file, args.image_paths)
    if not paths:
        raise SystemExit("No image paths provided.")
    process_paths(paths, args.split_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
