"""
Stable Diffusion XL (refiner) variant of sgg_sampler.py.

This mirrors the SGG/LCG guidance loop but swaps in StableDiffusionXLImg2ImgPipeline.
It keeps the teacher-guided loss, optional blur/TV, and class filtering.
Note: SDXL UNet expects both text embeddings and pooled embeddings plus time_ids.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
)
from PIL import Image
from torch.cuda.amp import autocast as cuda_autocast

from seg.dataloaders.cityscapes import CityscapesSegmentation, IGNORE_LABEL
from seg.infer import ModelBundle, build_cityscapes_dataloader, load_hf_model
from seg.utils.hf_utils import build_joint_resize, tensor_to_pil, mask_to_color

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

OUTDIR = "eval/plain_sgg_sdxl_v0"
PROMPT = "autonomous driving city scene at night with wet asphalt reflections, cinematic"
NEG = "blurry, cartoon, oversaturated, low detail, distorted cars, night vision, grainy, watermark, text, lens flare halos, blown highlights"
SIZE = (512, 512)
STEPS = 30
STRENGTH = 0.8
CFG = 7.5

GUIDE_START, GUIDE_END = 2.0, 1.0
ETA = 8.0
GRAD_EPS = 1e-8
USE_KL = True
CALLBACK_INTERVAL = 10
TEMPERATURE = 1.0


@dataclass
class SamplerConfig:
    prompt: str = PROMPT
    negative_prompt: str = NEG
    size: Tuple[int, int] = SIZE
    steps: int = STEPS
    strength: float = STRENGTH
    start_timestep_value: Optional[int] = None
    full_denoise_from_start: bool = False
    timestep_stride: int = 1
    timestep_spacing: str = "trailing"
    use_karras_sigmas: bool = True
    cfg: float = CFG
    guide_start: float = GUIDE_START
    guide_end: float = GUIDE_END
    eta: float = ETA
    use_kl: bool = USE_KL
    ce_temperature: float = TEMPERATURE
    grad_eps: float = GRAD_EPS
    use_cycle: bool = True
    guide_blur_sigma: float = 0.0
    guide_allowed_classes: Optional[List[int]] = None
    guide_tv_weight: float = 0.0


@dataclass
class StepOutput:
    step_index: int
    timestep: int
    progress: float
    ce_loss: Optional[float]
    decoded_image: torch.Tensor
    decoded_x0_image: torch.Tensor
    predicted_mask: torch.Tensor


def save_01(x01: torch.Tensor, path: Path) -> None:
    array = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def _maybe_autocast(device_type: str, dtype: torch.dtype):
    return torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else contextlib.nullcontext()


def _build_timesteps(scheduler, config: SamplerConfig, *, device: torch.device, verbose: bool):
    scheduler.set_timesteps(config.steps, device=device)
    full_timesteps = scheduler.timesteps
    if config.start_timestep_value is not None:
        target = torch.tensor(config.start_timestep_value, device=full_timesteps.device, dtype=full_timesteps.dtype)
        t_start_index = int(torch.argmin(torch.abs(full_timesteps - target)).item())
    else:
        total_timesteps = len(full_timesteps)
        t_start_index = int((1.0 - config.strength) * total_timesteps)
        t_start_index = max(0, min(total_timesteps - 1, t_start_index))
    timesteps = full_timesteps[t_start_index:]
    if verbose:
        start_ts_value = float(timesteps[0].item())
        print(
            f"Using {len(timesteps)}/{len(full_timesteps)} timesteps, start index {t_start_index}, start t={start_ts_value:.1f}"
        )
    return timesteps, t_start_index


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - radius
    kernel_1d = torch.exp(-0.5 * (coords / sigma)**2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_h = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1)
    channels = image.shape[1]
    blurred = F.conv2d(image, kernel_h.expand(channels, 1, 1, kernel_size), padding=(0, radius), groups=channels)
    blurred = F.conv2d(blurred, kernel_v.expand(channels, 1, kernel_size, 1), padding=(radius, 0), groups=channels)
    return blurred


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dx.abs().mean() + dy.abs().mean())


def _mask_keep_classes(mask: torch.Tensor, allowed: Optional[List[int]]) -> torch.Tensor:
    if not allowed:
        return mask
    allowed_t = torch.tensor(allowed, device=mask.device, dtype=mask.dtype)
    keep = (mask.unsqueeze(-1) == allowed_t).any(dim=-1)
    masked = mask.clone()
    masked[~keep] = IGNORE_LABEL
    return masked


def compute_teacher_logits(image_batch: torch.Tensor, bundle: ModelBundle, *, target_size: Tuple[int,
                                                                                                 int]) -> torch.Tensor:
    device = bundle.device
    with _no_autocast(device.type):
        pixel_values = _prepare_teacher_inputs(image_batch.to(device=device), bundle.processor, target_size)
        pixel_values = pixel_values.to(device=device, dtype=bundle.model.dtype)
        outputs = bundle.model(pixel_values=pixel_values)
        logits = outputs.logits
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


def _no_autocast(device_type: str):
    return cuda_autocast(enabled=False) if device_type == "cuda" else contextlib.nullcontext()


def _prepare_teacher_inputs(image_batch: torch.Tensor, processor, target_size: Tuple[int, int]) -> torch.Tensor:
    pixel_values = image_batch.to(dtype=torch.float32)
    desired_height, desired_width = target_size
    if getattr(processor, "do_resize", False):
        size_config = getattr(processor, "size", {}) or {}
        desired_height = size_config.get("height", desired_height)
        desired_width = size_config.get("width", desired_width)
    if pixel_values.shape[-2:] != (desired_height, desired_width):
        pixel_values = F.interpolate(pixel_values,
                                     size=(desired_height, desired_width),
                                     mode="bilinear",
                                     align_corners=False)
    if getattr(processor, "do_rescale", False):
        scale = float(getattr(processor, "rescale_factor", 1.0))
        pixel_values = pixel_values * 255.0 * scale
    if getattr(processor, "do_normalize", False):
        mean = torch.tensor(getattr(processor, "image_mean", [0.5, 0.5, 0.5]),
                            device=pixel_values.device,
                            dtype=pixel_values.dtype).view(1, -1, 1, 1)
        std = torch.tensor(getattr(processor, "image_std", [0.5, 0.5, 0.5]),
                           device=pixel_values.device,
                           dtype=pixel_values.dtype).view(1, -1, 1, 1)
        pixel_values = (pixel_values - mean) / std
    return pixel_values


def ce_label(image_batch: torch.Tensor,
             mask: torch.Tensor,
             bundle: ModelBundle,
             temperature: float = 1.0) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    logits = compute_teacher_logits(image_batch, bundle, target_size=mask.shape[-2:])
    target = mask.to(device=logits.device, dtype=torch.long)
    if temperature != 1.0:
        logits = logits / temperature
    return F.cross_entropy(logits, target, ignore_index=IGNORE_LABEL)


def kl_logits(reference_logits: torch.Tensor, image_batch: torch.Tensor, bundle: ModelBundle) -> torch.Tensor:
    candidate_logits = compute_teacher_logits(image_batch, bundle, target_size=reference_logits.shape[-2:])
    log_probs = F.log_softmax(candidate_logits, dim=1)
    ref_probs = F.softmax(reference_logits.detach(), dim=1)
    return F.kl_div(log_probs, ref_probs, reduction="batchmean")


def _lcg_loss(decoded_01: torch.Tensor, mask_batch: torch.Tensor,
              teacher_bundle: ModelBundle) -> Optional[torch.Tensor]:
    logits = compute_teacher_logits(decoded_01, teacher_bundle, target_size=mask_batch.shape[-2:])
    log_probs = F.log_softmax(logits, dim=1)
    classes = torch.unique(mask_batch)
    classes = classes[classes != IGNORE_LABEL]
    if classes.numel() == 0:
        return None
    per_class_losses = []
    for cls in classes:
        region_mask = (mask_batch == cls)
        if not region_mask.any():
            continue
        nll = -log_probs[:, cls][region_mask]
        per_class_losses.append(nll.mean())
    if not per_class_losses:
        return None
    return torch.stack(per_class_losses).mean()


def _encode_init_latents(pipe: StableDiffusionXLImg2ImgPipeline, image_pil: Image.Image, *, device: torch.device,
                         torch_dtype: torch.dtype) -> torch.Tensor:
    with torch.no_grad():
        with _maybe_autocast(device.type, torch_dtype):
            init = pipe.image_processor.preprocess(image_pil).to(device, dtype=torch_dtype)
            latents = pipe.vae.encode(init).latent_dist.sample() * pipe.vae.config.scaling_factor
            return latents


def _add_noise(scheduler, init_latents: torch.Tensor, timestep: torch.Tensor, *, verbose: bool) -> torch.Tensor:
    if verbose:
        print(f"Adding noise at timestep {timestep}.")
    if isinstance(timestep, torch.Tensor) and timestep.ndim == 0:
        timestep = timestep.repeat(init_latents.shape[0])
    return scheduler.add_noise(init_latents, torch.randn_like(init_latents), timestep)


def _scheduler_step_with_x0(scheduler, latents: torch.Tensor, timestep: torch.Tensor, noise_pred: torch.Tensor):
    with torch.no_grad():
        step_result = scheduler.step(noise_pred, timestep, latents.detach())
    latents_next = step_result.prev_sample if hasattr(step_result, "prev_sample") else step_result[0]
    pred_original_latents = getattr(step_result, "pred_original_sample", None)
    if pred_original_latents is None:
        alpha_prod_t = scheduler.alphas_cumprod[int(timestep)].to(device=latents.device, dtype=latents.dtype)
        sqrt_alpha_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_prod_t)
        pred_original_latents = (latents.detach() - sqrt_one_minus_alpha_t * noise_pred.detach()) / sqrt_alpha_t
    return latents_next, pred_original_latents.detach()


def build_step_logger(steps_dir: Path) -> Callable[[StepOutput], None]:
    steps_dir.mkdir(parents=True, exist_ok=True)

    def _callback(step: StepOutput) -> None:
        suffix = f"step_{step.step_index:03d}_t{step.timestep:04d}"
        save_01(step.decoded_image, steps_dir / f"{suffix}_image.png")
        save_01(step.decoded_x0_image, steps_dir / f"{suffix}_x0.png")
        mask_to_color(step.predicted_mask[0]).save(steps_dir / f"{suffix}_mask.png")

    return _callback


def decode_latents(pipe: StableDiffusionXLImg2ImgPipeline, latents: torch.Tensor, *, device_type: str,
                   torch_dtype: torch.dtype) -> torch.Tensor:
    with torch.no_grad():
        with _maybe_autocast(device_type, torch_dtype):
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
            return (decoded + 1) / 2


def predict_mask_from_image(image_batch: torch.Tensor, bundle: ModelBundle, *, target_size: Tuple[int,
                                                                                                  int]) -> torch.Tensor:
    logits = compute_teacher_logits(image_batch, bundle, target_size=target_size)
    return logits.argmax(dim=1)


def build_sampler_loader(root: Path, split: str, size: Tuple[int, int], *, batch_size: int, num_workers: int,
                         pin_memory: bool):
    joint_transform = build_joint_resize(size)
    dataset = CityscapesSegmentation(root_dir=root, split=split, joint_transform=joint_transform)
    loader = build_cityscapes_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return dataset, loader


def _encode_prompts(
    pipe: StableDiffusionXLImg2ImgPipeline,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    num_images_per_prompt: int = 1,
):
    # SDXL encode_prompt returns pos/neg embeddings and pooled variants
    prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    return prompt_embeds, pooled_prompt_embeds


def _maybe_apply_sgg(latents_next: torch.Tensor, timestep: torch.Tensor, progress: float, *, pipe,
                     mask_batch: torch.Tensor, teacher_bundle: ModelBundle, reference_logits: Optional[torch.Tensor],
                     config: SamplerConfig, device: torch.device, torch_dtype: torch.dtype, mode: str,
                     noise_pred: torch.Tensor):
    if not (config.guide_start <= progress <= config.guide_end):
        print(" No SGG guidance at this step.")
        return latents_next.detach().requires_grad_(True), None

    alpha_bar_t = pipe.scheduler.alphas_cumprod[int(timestep)]
    sigma_t = torch.sqrt(1 - alpha_bar_t)
    eta_t = config.eta * (sigma_t / pipe.scheduler.alphas_cumprod.sqrt().max())
    print(f" Applying SGG ({mode.upper()}) with eta={eta_t:.4f} at progress={progress:.3f}")

    latents_guided = latents_next.clone().detach().requires_grad_(True)
    with _maybe_autocast(device.type, torch_dtype):
        decoded = pipe.vae.decode(latents_guided / pipe.vae.config.scaling_factor).sample
        decoded_01 = (decoded + 1) / 2
        if config.guide_blur_sigma > 0:
            decoded_01 = _gaussian_blur(decoded_01, config.guide_blur_sigma)

    mask_for_loss = _mask_keep_classes(mask_batch, config.guide_allowed_classes)
    if mode == "lcg":
        loss = _lcg_loss(decoded_01, mask_for_loss, teacher_bundle)
        if loss is None:
            print(" LCG skipped (no valid class pixels); falling back to GSG.")
            loss = ce_label(decoded_01, mask_for_loss, teacher_bundle, temperature=config.ce_temperature)
    else:
        loss = ce_label(decoded_01, mask_for_loss, teacher_bundle, temperature=config.ce_temperature)
    if config.use_kl and reference_logits is not None:
        loss = loss + 0.1 * kl_logits(reference_logits, decoded_01, teacher_bundle)
    if config.guide_tv_weight > 0:
        loss = loss + config.guide_tv_weight * _total_variation(decoded_01)

    grad = torch.autograd.grad(loss, latents_guided, retain_graph=False)[0]
    grad_norm = torch.linalg.norm(grad)
    normalised_grad = grad / (grad_norm + config.grad_eps)
    latents_updated = (latents_next - eta_t * normalised_grad).detach().requires_grad_(True)
    return latents_updated, float(loss.detach().cpu())


def run_single_sample(pipe: StableDiffusionXLImg2ImgPipeline,
                      teacher_bundle: ModelBundle,
                      *,
                      image_pil: Image.Image,
                      reference_image: torch.Tensor,
                      mask_batch: torch.Tensor,
                      prompt: str,
                      negative_prompt: str,
                      config: SamplerConfig,
                      device: torch.device,
                      torch_dtype: torch.dtype,
                      callback_interval: Optional[int] = 100,
                      step_callback: Optional[Callable[[StepOutput], None]] = None,
                      verbose: bool = True):
    init_latents = _encode_init_latents(pipe, image_pil, device=device, torch_dtype=torch_dtype)
    timesteps, _ = _build_timesteps(pipe.scheduler, config, device=device, verbose=verbose)
    timestep = timesteps[0]
    latents = _add_noise(pipe.scheduler, init_latents, timestep, verbose=verbose).detach().requires_grad_(True)

    reference_logits = None
    if config.use_kl:
        with torch.no_grad():
            reference_logits = compute_teacher_logits(reference_image,
                                                      teacher_bundle,
                                                      target_size=mask_batch.shape[-2:])

    prompt_embeds, pooled_prompt_embeds = _encode_prompts(pipe, prompt, negative_prompt, device)

    _, _, height, width = latents.shape
    orig_size = (height * 8, width * 8)
    crop_top_left = (0, 0)
    target_size = orig_size
    proj_cfg = getattr(getattr(pipe, "text_encoder_2", None), "config", None)
    proj_dim = getattr(proj_cfg, "projection_dim", 0) if proj_cfg is not None else 0

    add_time_ids = pipe._get_add_time_ids(
        original_size=orig_size,
        crops_coords_top_left=crop_top_left,
        target_size=target_size,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        negative_original_size=orig_size,
        negative_crops_coords_top_left=crop_top_left,
        negative_target_size=target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=proj_dim,
    )
    add_time_ids = add_time_ids[0]
    add_time_ids = add_time_ids.to(device=device)
    add_time_ids = add_time_ids.repeat(latents.shape[0] * 2, 1)

    ce_log: list[float] = []
    steps_to_run = len(timesteps)
    denom = max(1, steps_to_run - 1)
    callback_every = max(1, int(callback_interval)) if callback_interval and callback_interval > 0 else max(
        1, steps_to_run)
    guided_cycle_idx = 0

    for step_index, timestep in enumerate(timesteps):
        print(f" Step {step_index+1}/{steps_to_run} at timestep {timestep}")
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = pipe.scheduler.scale_model_input(latent_input, timestep)
        with _maybe_autocast(device.type, torch_dtype):
            noise_pred = pipe.unet(
                latent_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                },
            ).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + config.cfg * (noise_text - noise_uncond)
            noise_pred_for_sgg = noise_pred.detach()

        latents_next, pred_original_latents = _scheduler_step_with_x0(pipe.scheduler, latents, timestep, noise_pred)
        progress = step_index / denom
        ce_loss_value: Optional[float] = None
        guidance_window = (config.guide_start <= progress <= config.guide_end)
        if guidance_window:
            phase = guided_cycle_idx % 2
            guided_cycle_idx += 1
            if phase in (0, 1):
                mode = "gsg" if phase == 0 else "lcg"
                latents, ce_loss_value = _maybe_apply_sgg(
                    latents_next,
                    timestep,
                    progress,
                    pipe=pipe,
                    mask_batch=mask_batch,
                    teacher_bundle=teacher_bundle,
                    reference_logits=reference_logits,
                    config=config,
                    device=device,
                    torch_dtype=torch_dtype,
                    mode=mode,
                    noise_pred=noise_pred_for_sgg,
                )
                print(f" Applied SGG ({mode.upper()}); CE loss={ce_loss_value:.4f}")
            else:
                print(" Guidance window active but skipping SGG this step.")
                latents = latents_next.detach().requires_grad_(True)
        else:
            print(" No SGG guidance at this step.")
            latents = latents_next.detach().requires_grad_(True)

        if ce_loss_value is not None:
            ce_log.append(ce_loss_value)

        should_log_step = (step_callback is not None and
                           (step_index % callback_every == 0 or step_index == steps_to_run - 1))
        if should_log_step:
            decoded_for_log = decode_latents(pipe, latents, device_type=device.type, torch_dtype=torch_dtype)
            decoded_x0_for_log = decode_latents(pipe,
                                                pred_original_latents,
                                                device_type=device.type,
                                                torch_dtype=torch_dtype)
            predicted_mask = predict_mask_from_image(decoded_for_log, teacher_bundle, target_size=mask_batch.shape[-2:])
            timestep_value = int(timestep.detach().cpu().item()) if isinstance(timestep,
                                                                               torch.Tensor) else int(timestep)
            step_callback(
                StepOutput(
                    step_index=step_index,
                    timestep=timestep_value,
                    progress=progress,
                    ce_loss=ce_loss_value,
                    decoded_image=decoded_for_log.detach().cpu(),
                    decoded_x0_image=decoded_x0_for_log.detach().cpu(),
                    predicted_mask=predicted_mask.detach().cpu(),
                ))

        if verbose and (step_index % max(1, config.steps // 10) == 0):
            timestep_display = int(timestep.detach().cpu().item()) if isinstance(timestep,
                                                                                 torch.Tensor) else int(timestep)
            print(f"[{step_index:02d}/{steps_to_run}] t={timestep_display}")

    decoded_final = decode_latents(pipe, latents, device_type=device.type, torch_dtype=torch_dtype)
    return decoded_final, ce_log


def main(
    dataset_root: str | Path = "data/cityscapes",
    split: str = "my_test",
    out_dir: str | Path = OUTDIR,
    *,
    teacher_model: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    max_samples: Optional[int] = 2,
    num_workers: int = 4,
    seed: Optional[int] = 42,
    config: Optional[SamplerConfig] = None,
    callback_interval: Optional[int] = CALLBACK_INTERVAL,
    step_callback: Optional[Callable[[StepOutput], None]] = None,
    verbose: bool = True,
) -> None:

    device = torch.device(DEVICE)
    torch_dtype = DTYPE if device.type == "cuda" else torch.float32
    sampler_config = config or SamplerConfig()
    if seed is not None:
        torch.manual_seed(seed)

    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    teacher_bundle = load_hf_model(teacher_model, device=str(device))

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch_dtype,
        safety_checker=None,
        add_watermarker=False,
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.unet.to(memory_format=torch.channels_last)
    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()

    dataset, loader = build_sampler_loader(
        Path(dataset_root),
        split,
        sampler_config.size,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    total_steps = len(loader) if max_samples is None else min(len(loader), max_samples)
    for idx, (images, masks) in enumerate(loader):
        if idx >= total_steps:
            break
        if idx != 0:
            continue

        sample_info = dataset.samples[idx]
        relative_path = sample_info.image_path.relative_to(dataset.left_dir)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        local_callback = step_callback
        if local_callback is None and callback_interval is not None:
            local_callback = build_step_logger(output_path.parent / f"{output_path.stem}_steps")

        image_batch = images.to(device=device, dtype=torch.float32)
        mask_batch = masks.to(device=device, dtype=torch.long)
        image_pil = tensor_to_pil(images[0])

        original_path = output_path.with_name(output_path.stem + "_original.png")
        mask_path = output_path.with_name(output_path.stem + "_mask.png")
        image_pil.save(original_path)
        mask_to_color(mask_batch[0]).save(mask_path)

        config_path = output_path.with_name(output_path.stem + "_config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(sampler_config), f, indent=4)

        decoded_01, ce_log = run_single_sample(
            pipe,
            teacher_bundle,
            image_pil=image_pil,
            reference_image=image_batch,
            mask_batch=mask_batch,
            prompt=sampler_config.prompt,
            negative_prompt=sampler_config.negative_prompt,
            config=sampler_config,
            device=device,
            torch_dtype=torch_dtype,
            callback_interval=callback_interval,
            step_callback=local_callback,
            verbose=verbose,
        )

        save_01(decoded_01, output_path)
        ce_path = output_path.with_name(output_path.stem + "_ce.npy")
        np.save(ce_path, np.array(ce_log, dtype=np.float32))

        if verbose:
            print(f"saved {relative_path} steps_with_CE: {len(ce_log)}")
            if ce_log:
                print("Last CE loss:", ce_log[-1])


if __name__ == "__main__":
    main()
