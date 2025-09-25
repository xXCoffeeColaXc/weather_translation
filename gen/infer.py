from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from torchvision import transforms

from gen.dataloader import ACDCDataset

LOGGER = logging.getLogger(__name__)

DTYPE = torch.float16
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DEFAULT_PROMPT = "autonomous driving city street at night, wet asphalt, headlights"
DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, warped geometry"
IMAGE_SIZE: Tuple[int, int] = (384, 768)  #(512, 512)
NUM_STEPS = 750
DEFAULT_STRENGTH = 0.65
DEFAULT_GUIDANCE = 5.0
'''
python gen/infer.py --image-path /home/talmacsi/BME/weather_translation/data/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png
'''


def build_inference_transform(size: Tuple[int, int]) -> transforms.Compose:
    """Deterministic resize + normalisation matching diffusion expectations."""

    height, width = size
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Lambda(lambda t: (t + 1) / 2),  # scale to [0, 1]
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion v1.5 img2img inference")
    parser.add_argument("--image-path", type=Path, help="Path to a source image. Overrides dataset selection.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/acdc/rgb_anon"),
        help="Root directory organised as condition/split/filename.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["night"],
        help="Dataset sub-folders to search (e.g. night rain).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        help="Dataset splits to include (e.g. train val test).",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the dataset image to use as init.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/sd15_img2img.png"),
        help="Where to save the generated image.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda, cuda:1, cpu).")
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable safety checker for faster execution.",
    )
    return parser.parse_args()


def _prepare_pipeline(device: torch.device, dtype: torch.dtype, disable_safety: bool) -> StableDiffusionImg2ImgPipeline:
    LOGGER.info("Loading pipeline %s with dtype=%s on %s", MODEL_NAME, dtype, device)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_NAME, torch_dtype=dtype, safety_checker=None)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        LOGGER.warning("Could not enable xformers attention: %s", exc)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_image_from_path(path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(path) as img:
        tensor = transform(img.convert("RGB"))
    return tensor


def _load_image_from_dataset(
    root: Path,
    selected_conditions: Sequence[str],
    splits: Sequence[str],
    index: int,
    transform: transforms.Compose,
) -> tuple[torch.Tensor, Path]:
    dataset = ACDCDataset(
        root_dir=root,
        selected_conditions=selected_conditions,
        transform=transform,
        splits=splits,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No images found under {root} for conditions={selected_conditions} and splits={splits}")

    safe_index = index % len(dataset)
    init_tensor = dataset[safe_index]
    image_path = Path(dataset.image_paths[safe_index]).resolve()
    return init_tensor, image_path


def _prepare_init_tensor(args: argparse.Namespace, transform: transforms.Compose) -> tuple[torch.Tensor, Path]:
    if args.image_path is not None:
        resolved = args.image_path.expanduser().resolve()
        LOGGER.info("Using explicit init image %s", resolved)
        tensor = _load_image_from_path(resolved, transform)
        return tensor, resolved

    tensor, path = _load_image_from_dataset(
        root=args.dataset_root.expanduser().resolve(),
        selected_conditions=tuple(args.conditions),
        splits=tuple(args.splits),
        index=args.sample_index,
        transform=transform,
    )
    LOGGER.info("Using dataset init image %s", path)
    return tensor, path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    dtype = DTYPE if device.type == "cuda" else torch.float32

    transform = build_inference_transform(IMAGE_SIZE)
    init_tensor, init_path = _prepare_init_tensor(args, transform)

    if init_tensor.ndim == 3:
        init_tensor = init_tensor.unsqueeze(0)
    if init_tensor.size(-2) != IMAGE_SIZE[0] or init_tensor.size(-1) != IMAGE_SIZE[1]:
        raise ValueError(f"Init image from {init_path} resized to unexpected shape {tuple(init_tensor.shape)}")

    pipe = _prepare_pipeline(device=device, dtype=dtype, disable_safety=args.disable_safety_checker)

    init_tensor = init_tensor.to(device=device, dtype=pipe.unet.dtype)

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    LOGGER.info("Running img2img with steps=%d strength=%.2f guidance=%.2f", args.num_inference_steps, args.strength,
                args.guidance_scale)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_tensor,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )

    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)

    LOGGER.info("Saved generated image to %s", output_path)


if __name__ == "__main__":
    main()
