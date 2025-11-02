from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

LOGGER = logging.getLogger(__name__)


def _chunked(sequence: Sequence[str], chunk_size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(sequence), chunk_size):
        yield sequence[start:start + chunk_size]


def _slugify(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    safe = safe.strip("-")
    return safe or "empty"


def _build_conditional_prompts(
    prompts: Sequence[str] | None,
    conditions: Sequence[str],
    template: str,
    samples_per_prompt: int,
) -> List[str]:
    if prompts:
        base_prompts = list(prompts)
    else:
        base_prompts = [template.format(condition=cond) for cond in conditions]

    expanded: List[str] = []
    for prompt in base_prompts:
        expanded.extend([prompt] * samples_per_prompt)
    return expanded


@dataclass
class GenerationConfig:
    prompts: List[str]
    negative_prompt: str | None
    guidance_scale: float
    subdir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a fine-tuned Stable Diffusion checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing fine-tuned weights (e.g. gen/checkpoints/sd_finetune/step-000500).",
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model to load before applying fine-tuned weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gen/samples"),
        help="Directory to save generated images and metadata.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g. cuda, cuda:1, cpu). Default picks cuda if available.",
    )
    parser.add_argument("--height", type=int, default=384)  # 512
    parser.add_argument("--width", type=int, default=720)  # 512
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=
        'low quality, low-res, blurry, overexposed, underexposed, oversaturated, noise, grainy, jpeg artifacts, watermark, text, cartoon, fisheye, warped road, unrealistic geometry, distorted cars, duplicate vehicles, stretched pedestrians, malformed people, extra limbs, floating objects, glitch, unrealistic reflections, lens flare halos, blown highlights'
    )
    parser.add_argument("--samples-per-prompt", type=int, default=2, help="Images to draw per conditional prompt.")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Explicit prompts for conditional sampling. If omitted, --conditions with --prompt-template is used.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=("night", "rain", "fog"),
        help="Weather conditions used when --prompts is not specified.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="autonomous driving scene at {condition}, cinematic, detailed, wet asphalt reflections",
        help="Template for building prompts when --prompts is not provided.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for pipeline calls.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bar while sampling.")
    parser.add_argument("--save-metadata", action="store_true", help="Persist JSON metadata alongside samples.")
    parser.add_argument("--num-unconditional", type=int, default=0, help="How many unconditional samples to draw.")
    parser.add_argument(
        "--unconditional-prompt",
        type=str,
        default="",
        help="Prompt for unconditional sampling (defaults to empty string).",
    )
    parser.add_argument(
        "--unconditional-guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale used for unconditional sampling (set to 1.0 to disable CFG).",
    )
    parser.add_argument("--unconditional-negative-prompt", type=str, default=None)
    parser.add_argument(
        "--fuse-lora",
        action="store_true",
        help="If the checkpoint contains LoRA weights, fuse them into the UNet for inference.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("auto", "fp16", "bf16", "fp32"),
        default="auto",
        help="Inference dtype for the pipeline. 'auto' picks fp16 on CUDA and fp32 otherwise.",
    )
    return parser.parse_args()


def _select_dtype(device: torch.device, dtype_flag: str) -> torch.dtype:
    if dtype_flag == "fp32":
        return torch.float32
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _load_component_if_exists(
    component_cls,
    path: Path,
    *,
    dtype: torch.dtype | None = None,
) -> object | None:
    if not path.exists():
        return None
    LOGGER.info("Loading fine-tuned component from %s", path)
    if dtype is not None:
        try:
            return component_cls.from_pretrained(path, torch_dtype=dtype)
        except TypeError:
            return component_cls.from_pretrained(path)
    return component_cls.from_pretrained(path)


def load_finetuned_pipeline(args: argparse.Namespace, device: torch.device) -> StableDiffusionPipeline:
    dtype = _select_dtype(device, args.dtype)
    LOGGER.info("Loading base pipeline %s with dtype=%s on %s", args.pretrained_model_name_or_path, dtype, device)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    ckpt_dir = args.checkpoint_dir.expanduser().resolve()

    text_encoder = _load_component_if_exists(CLIPTextModel, ckpt_dir / "text_encoder", dtype=dtype)
    if text_encoder is not None:
        pipe.text_encoder = text_encoder

    tokenizer = _load_component_if_exists(CLIPTokenizer, ckpt_dir / "tokenizer", dtype=None)
    if tokenizer is not None:
        pipe.tokenizer = tokenizer

    vae = _load_component_if_exists(AutoencoderKL, ckpt_dir / "vae", dtype=dtype)
    if vae is not None:
        pipe.vae = vae

    unet_path = ckpt_dir / "unet"
    lora_path = ckpt_dir / "unet_lora"

    if unet_path.exists():
        LOGGER.info("Loading fine-tuned UNet from %s", unet_path)
        pipe.unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
    elif lora_path.exists():
        LOGGER.info("Loading LoRA weights from %s", lora_path)
        try:
            pipe.load_lora_weights(lora_path)
        except AttributeError as exc:
            raise RuntimeError("The installed diffusers version does not support `load_lora_weights`. "
                               "Upgrade diffusers or load a checkpoint with full UNet weights.") from exc

        if args.fuse_lora:
            LOGGER.info("Fusing LoRA weights into UNet")
            try:
                pipe.fuse_lora()
            except AttributeError:
                LOGGER.warning("Diffusers version does not provide `fuse_lora`; proceeding without fusing.")
    else:
        LOGGER.warning("No UNet or LoRA weights found in %s; using base checkpoint UNet.", ckpt_dir)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        LOGGER.debug("Could not enable xformers attention: %s", exc)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=not args.progress)
    return pipe


def _prepare_generations(args: argparse.Namespace) -> List[GenerationConfig]:
    generations: List[GenerationConfig] = []

    conditional_prompts: List[str] = []
    if args.samples_per_prompt > 0:
        if args.prompts:
            conditional_prompts = _build_conditional_prompts(
                prompts=args.prompts,
                conditions=(),
                template=args.prompt_template,
                samples_per_prompt=args.samples_per_prompt,
            )
        elif args.conditions:
            conditional_prompts = _build_conditional_prompts(
                prompts=None,
                conditions=args.conditions,
                template=args.prompt_template,
                samples_per_prompt=args.samples_per_prompt,
            )

    if conditional_prompts:
        generations.append(
            GenerationConfig(
                prompts=conditional_prompts,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                subdir="conditional",
            ))

    if args.num_unconditional > 0:
        prompts = [args.unconditional_prompt] * args.num_unconditional
        generations.append(
            GenerationConfig(
                prompts=prompts,
                negative_prompt=args.unconditional_negative_prompt,
                guidance_scale=args.unconditional_guidance_scale,
                subdir="unconditional",
            ))

    if not generations:
        raise ValueError("No prompts provided. Specify --prompts or set --num-unconditional > 0.")

    return generations


def _save_image(image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _metadata_dict(**kwargs) -> dict:
    return {key: value for key, value in kwargs.items() if value is not None}


def generate_images(
    pipe: StableDiffusionPipeline,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    generations = _prepare_generations(args)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_images = sum(len(gen.prompts) for gen in generations)
    LOGGER.info("Generating %d images across %d sets", total_images, len(generations))

    progress_bar = None
    if args.progress:
        try:
            from tqdm.auto import tqdm
            progress_bar = tqdm(total=total_images, desc="Sampling")
        except ImportError:  # pragma: no cover
            LOGGER.warning("tqdm is not installed; progress bar disabled.")

    last_seed = args.seed or 0
    exec_device = getattr(pipe, "_execution_device", device)
    for generation in generations:
        subdir = output_root / generation.subdir
        negative_prompt = generation.negative_prompt
        prompts = generation.prompts
        for batch_index, prompt_batch in enumerate(_chunked(prompts, args.batch_size)):
            generator = None
            if args.seed is not None:
                generator = torch.Generator(device=exec_device)
                generator.manual_seed(last_seed + batch_index)
            result = pipe(
                prompt=list(prompt_batch),
                negative_prompt=[negative_prompt] * len(prompt_batch) if negative_prompt else None,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=generation.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            )

            for idx, image in enumerate(result.images):
                prompt_text = prompt_batch[idx]
                prompt_slug = _slugify(prompt_text)
                image_index = batch_index * args.batch_size + idx
                filename = f"{image_index:05d}_{prompt_slug}.png"
                output_path = subdir / filename
                _save_image(image, output_path)

                if args.save_metadata:
                    metadata = _metadata_dict(
                        prompt=prompt_text,
                        negative_prompt=negative_prompt,
                        guidance_scale=generation.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        seed=args.seed,
                        height=args.height,
                        width=args.width,
                    )
                    metadata_path = output_path.with_suffix(".json")
                    metadata_path.write_text(json.dumps(metadata, indent=2))

                if progress_bar is not None:
                    progress_bar.update(1)

        last_seed += math.ceil(len(prompts) / args.batch_size)

    if progress_bar is not None:
        progress_bar.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    pipe = load_finetuned_pipeline(args, device=device)
    pipe.unet.eval()
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.eval()
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.eval()

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("Image height and width must be multiples of 8 for Stable Diffusion.")

    generate_images(pipe, args, device)


if __name__ == "__main__":
    main()
