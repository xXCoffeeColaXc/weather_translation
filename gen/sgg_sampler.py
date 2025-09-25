from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from torch.cuda.amp import autocast as cuda_autocast

from seg.dataloaders.cityscapes import CityscapesSegmentation, IGNORE_LABEL
from seg.infer import ModelBundle, build_cityscapes_dataloader, load_hf_model
from seg.utils.hf_utils import build_joint_resize, tensor_to_pil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MODEL = "runwayml/stable-diffusion-v1-5"
TEACHER_MODEL = "segformer_b5"

PROMPT = "autonomous driving city street at night, wet asphalt, headlights"
NEG = "blurry, distorted, warped geometry"
SIZE = (384 // 2, 768 // 2)
STEPS = 750
STRENGTH = 0.55
CFG = 4.0

GUIDE_START, GUIDE_END = 1.0, 1.8
ETA = 0.4
GRAD_EPS = 1e-8
USE_KL = False


def save_01(x01: torch.Tensor, path: Path) -> None:
    array = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def build_sampler_loader(
    root: Path,
    split: str,
    size: Tuple[int, int],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[CityscapesSegmentation, torch.utils.data.DataLoader]:
    joint_transform = build_joint_resize(size)
    dataset = CityscapesSegmentation(
        root_dir=root,
        split=split,
        joint_transform=joint_transform,
    )

    loader = build_cityscapes_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader


def _prepare_teacher_inputs(
    image_batch: torch.Tensor,
    processor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    pixel_values = image_batch.to(dtype=torch.float32)

    desired_height, desired_width = target_size
    if getattr(processor, "do_resize", False):
        size_config = getattr(processor, "size", {}) or {}
        desired_height = size_config.get("height", desired_height)
        desired_width = size_config.get("width", desired_width)

    if pixel_values.shape[-2:] != (desired_height, desired_width):
        pixel_values = F.interpolate(
            pixel_values,
            size=(desired_height, desired_width),
            mode="bilinear",
            align_corners=False,
        )

    if getattr(processor, "do_rescale", False):
        scale = float(getattr(processor, "rescale_factor", 1.0))
        pixel_values = pixel_values * 255.0 * scale

    if getattr(processor, "do_normalize", False):
        mean = torch.tensor(
            getattr(processor, "image_mean", [0.5, 0.5, 0.5]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            getattr(processor, "image_std", [0.5, 0.5, 0.5]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).view(1, -1, 1, 1)
        pixel_values = (pixel_values - mean) / std

    return pixel_values


def _no_autocast(device_type: str):
    return cuda_autocast(enabled=False) if device_type == "cuda" else contextlib.nullcontext()


def compute_teacher_logits(
    image_batch: torch.Tensor,
    bundle: ModelBundle,
    *,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    device = bundle.device
    with _no_autocast(device.type):
        pixel_values = _prepare_teacher_inputs(image_batch.to(device=device), bundle.processor, target_size)
        pixel_values = pixel_values.to(device=device, dtype=bundle.model.dtype)
        outputs = bundle.model(pixel_values=pixel_values)
        logits = outputs.logits
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
    return logits


def ce_label(
    image_batch: torch.Tensor,
    mask: torch.Tensor,
    bundle: ModelBundle,
) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    logits = compute_teacher_logits(image_batch, bundle, target_size=mask.shape[-2:])
    target = mask.to(device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, target, ignore_index=IGNORE_LABEL)


def kl_logits(
    reference_logits: torch.Tensor,
    image_batch: torch.Tensor,
    bundle: ModelBundle,
) -> torch.Tensor:
    candidate_logits = compute_teacher_logits(image_batch, bundle, target_size=reference_logits.shape[-2:])
    log_probs = F.log_softmax(candidate_logits, dim=1)
    ref_probs = F.softmax(reference_logits.detach(), dim=1)
    return F.kl_div(log_probs, ref_probs, reduction="batchmean")


def _maybe_autocast(device_type: str, dtype: torch.dtype):
    return torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else contextlib.nullcontext()


def main(
    dataset_root: str | Path = "data/cityscapes",
    split: str = "val",
    out_dir: str | Path = "out/sgg",
    *,
    teacher_model: str = TEACHER_MODEL,
    max_samples: Optional[int] = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> None:
    device = torch.device(DEVICE)
    torch_dtype = DTYPE if device.type == "cuda" else torch.float32

    if seed is not None:
        torch.manual_seed(seed)

    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    teacher_bundle = load_hf_model(teacher_model, device=str(device))

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL, torch_dtype=torch_dtype, safety_checker=None)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        print(f"Could not enable xformers attention: {exc}")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    dataset, loader = build_sampler_loader(
        Path(dataset_root),
        split,
        SIZE,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    prompt_tokens = pipe.tokenizer(
        PROMPT,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative_tokens = pipe.tokenizer(
        NEG or "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        with _maybe_autocast(device.type, torch_dtype):
            pos_embeds = pipe.text_encoder(prompt_tokens.input_ids.to(device))[0]
            neg_embeds = pipe.text_encoder(negative_tokens.input_ids.to(device))[0]
    text_context = torch.cat([neg_embeds, pos_embeds], dim=0)

    total_steps = len(loader) if max_samples is None else min(len(loader), max_samples)
    for idx, (images, masks) in enumerate(loader):
        if idx >= total_steps:
            break

        sample_info = dataset.samples[idx]
        relative_path = sample_info.image_path.relative_to(dataset.left_dir)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image_batch = images.to(device=device, dtype=torch.float32)
        mask_batch = masks.to(device=device, dtype=torch.long)
        image_pil = tensor_to_pil(images[0])

        # encode
        with torch.no_grad():
            with _maybe_autocast(device.type, torch_dtype):
                init = pipe.image_processor.preprocess(image_pil).to(device, dtype=torch_dtype)
                latents = pipe.vae.encode(init * 2 - 1).latent_dist.sample() * pipe.vae.config.scaling_factor

        pipe.scheduler.set_timesteps(STEPS, device=device)
        timesteps = pipe.scheduler.timesteps
        t_start = int((1 - STRENGTH) * len(timesteps))
        noise = torch.randn_like(latents)
        latents = pipe.scheduler.add_noise(latents, noise, timesteps[t_start]).detach().requires_grad_(True)

        reference_logits = None
        if USE_KL:
            with torch.no_grad():
                reference_logits = compute_teacher_logits(image_batch,
                                                          teacher_bundle,
                                                          target_size=mask_batch.shape[-2:])

        ce_log: list[float] = []
        denom = max(1, len(timesteps) - t_start - 1)

        for step_index, timestep in enumerate(timesteps[t_start:]):
            latent_input = torch.cat([latents, latents], dim=0)
            with _maybe_autocast(device.type, torch_dtype):
                noise_pred = pipe.unet(latent_input, timestep, encoder_hidden_states=text_context).sample
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + CFG * (noise_text - noise_uncond)

            # scheduler step (no grad)
            with torch.no_grad():
                latents_next = pipe.scheduler.step(noise_pred, timestep, latents.detach()).prev_sample

            # ---- SGG mid-window
            progress = step_index / denom
            # if GUIDE_START <= progress <= GUIDE_END:
            #     latents_guided = latents_next.clone().detach().requires_grad_(True)
            #     with _maybe_autocast(device.type, torch_dtype):
            #         decoded = pipe.vae.decode(latents_guided / pipe.vae.config.scaling_factor).sample
            #         decoded_01 = (decoded + 1) / 2

            #     loss = ce_label(decoded_01, mask_batch, teacher_bundle)
            #     if USE_KL and reference_logits is not None:
            #         loss = loss + 0.1 * kl_logits(reference_logits, decoded_01, teacher_bundle)

            #     grad = torch.autograd.grad(loss, latents_guided, retain_graph=False)[0]
            #     grad_norm = torch.linalg.norm(grad)
            #     normalised_grad = grad / (grad_norm + GRAD_EPS)
            #     latents = (latents_next - ETA * normalised_grad).detach().requires_grad_(True)
            #     ce_log.append(float(loss.detach().cpu()))
            # else:
            latents = latents_next.detach().requires_grad_(True)

        # decode final
        with torch.no_grad():
            with _maybe_autocast(device.type, torch_dtype):
                decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
                decoded_01 = (decoded + 1) / 2

        save_01(decoded_01, output_path)
        ce_path = output_path.with_name(output_path.stem + "_ce.npy")
        np.save(ce_path, np.array(ce_log, dtype=np.float32))
        print(f"saved {relative_path} steps_with_CE: {len(ce_log)}")


if __name__ == "__main__":
    main()
