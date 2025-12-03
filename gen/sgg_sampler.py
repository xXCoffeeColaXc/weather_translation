from __future__ import annotations
import os
from dotenv import load_dotenv
import tqdm

load_dotenv()

import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from dataclasses import asdict, field
import json
import numpy as np
import torch
import torch.nn.functional as F
import math
from termcolor import colored
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from torch.cuda.amp import autocast as cuda_autocast

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusion3Pipeline

from val_sd import _select_dtype, _load_component_if_exists

from seg.dataloaders.cityscapes import CityscapesSegmentation, IGNORE_LABEL
from seg.infer import ModelBundle, build_cityscapes_dataloader, load_hf_model
from seg.utils.hf_utils import build_joint_randomresizedcrop, tensor_to_pil, mask_to_color

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MODEL = "runwayml/stable-diffusion-v1-5"
#MODEL = "stabilityai/stable-diffusion-3.5-medium"  # Alternative SD 3.5 medium
TEACHER_MODEL = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"  # "segformer_b5"
HF_TOKEN = os.environ.get("HF_TOKEN")

OUTDIR = "eval_city/benchmark_inversion_09_no_gsg_gated"

# CONDITION = "rain"
# PROMPT = f"autonomous driving scene at {CONDITION}, cinematic, detailed, wet asphalt reflections"
# PROMPT = "urban driving scene during rain, realistic and detailed with visible vehicles."
PROMPT = "autonomous driving scene at rain, realistic, detailed, road and surroundings visible. Scene contains sidewalk, building, car, person in a urban setting."
NEG = "blurry, unnatural, cartoon, oversaturated, low detail, distorted cars, grainy, lens flare halos, blown highlights, unrealistic colors"
SIZE = (512, 512)

STEPS = 100
STRENGTH = 0.5
CFG = 7.5

GUIDE_START, GUIDE_END = 2.0, 1.0
GUIDE_LAMBDA = 10.0
GUIDE_BLUR_SIGMA = 0.0
GUIDE_ALLOWED_CLASSES = [0, 1, 2, 8, 9, 10, 11, 12, 13,
                         18]  # road, sidewalk, building, vegetation, terrain, sky, rider, car, bycicle
GUIDE_CLASS_WEIGHTS = {
    13: 3.0,  # car
    0: 2.0,  # road
    2: 1.0,  # building
    1: 1.0,  # sidewalk
    8: 1.0,  # vegetation
    9: 1.0,  # terrain
    10: 1.0,  # sky
    11: 0.5,  # person
    12: 0.5,  # rider
    18: 0.5,  # bicycle
}
GUIDE_TV_WEIGHT = 0.01

TEMPERATURE = 1.0
LAMBDA_TR = 0.25

GRAD_EPS = 1e-8
LOSS_TYPE = "ce"  # "ce" or "kl" or "blend"
BLEND_WEIGHT = 0.9
CALLBACK_INTERVAL = 5
PRED_ON_X0_HAT = True
MODE = "alternate"  # "gsg" or "lcg" or "alternate"
GRAD_CLIP_NORM = 4  # e.g., 5.0 to clip by global norm
GRAD_CLIP_VALUE = None  # e.g., 0.2 to clamp elementwise


@dataclass
class SamplerConfig:
    prompt: str = PROMPT
    negative_prompt: str = NEG
    size: Tuple[int, int] = SIZE  # (width, height)
    steps: int = STEPS  # number of diffusion steps
    strength: float = STRENGTH  # denoising strength (0..1)
    start_timestep_value: Optional[int] = None  # e.g., 400 to start from that training timestep
    full_denoise_from_start: bool = False  # if True, run a contiguous chain from start_timestep_value → 0
    timestep_stride: int = 1  # stride for contiguous timesteps when full_denoise_from_start is True
    timestep_spacing: str = "trailing"  # scheduler spacing when not using the full contiguous path
    use_karras_sigmas: bool = True  # DPM++ only; must be False for custom contiguous timesteps
    cfg: float = CFG  # classifier-free guidance scale
    guide_start: float = GUIDE_START  # fractional progress to start SGG
    guide_end: float = GUIDE_END  # fractional progress to end SGG
    guide_lambda: float = GUIDE_LAMBDA  #
    lambda_tr: float = LAMBDA_TR  # weight for teacher-relative MSE on latent delta
    loss_type: str = LOSS_TYPE  # "ce" or "kl" or "blend"
    blend_weight: float = BLEND_WEIGHT  # weight for blending KL and CE losses when loss_type is "blend"
    ce_temperature: float = TEMPERATURE
    grad_eps: float = GRAD_EPS
    use_cycle: bool = True  # whether to alternate GSG/LCG/skip cycles
    guide_blur_sigma: float = GUIDE_BLUR_SIGMA  # optional Gaussian blur on guided image before teacher loss
    guide_allowed_classes: Optional[list[int]] = field(
        default_factory=lambda: GUIDE_ALLOWED_CLASSES)  # only these class ids are kept for CE; others ignored
    guide_tv_weight: float = GUIDE_TV_WEIGHT  # optional TV penalty on the guided decode
    grad_clip_norm: Optional[float] = GRAD_CLIP_NORM  # if set, clip gradients by global norm
    grad_clip_value: Optional[float] = GRAD_CLIP_VALUE  # if set, elementwise clamp gradients
    pred_on_x0_hat: bool = PRED_ON_X0_HAT  # whether to use x0 prediction in scheduler step (if supported)
    mode: str = MODE  # "gsg" or "lcg" or "alternate"
    # Class weighting for CE/KL guidance (trainIds). Defaults: car > road > building; others 0 (ignored).
    class_weights: Optional[dict[int, float]] = field(default_factory=lambda: GUIDE_CLASS_WEIGHTS)
    inverted_latents_path: Optional[str] = "munster_000009_000019_leftImg8bit_inverted_latents.pt"


@dataclass
class StepOutput:
    step_index: int
    timestep: int
    progress: float
    sgg_loss: Optional[float]
    decoded_image: torch.Tensor
    decoded_x0_image: torch.Tensor
    predicted_mask: torch.Tensor
    predicted_mask_x0: torch.Tensor
    guidance_heatmap: Optional[torch.Tensor] = None
    tr_heatmap: Optional[torch.Tensor] = None
    guidance_overlay: Optional[torch.Tensor] = None
    tr_overlay: Optional[torch.Tensor] = None


def save_01(x01: torch.Tensor, path: Path) -> None:
    array = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def _encode_init_latents(
    pipe: StableDiffusionImg2ImgPipeline,
    image_pil: Image.Image,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        with _maybe_autocast(device.type, torch_dtype):
            init = pipe.image_processor.preprocess(image_pil).to(device, dtype=torch_dtype)  # already in [-1,1]
            return pipe.vae.encode(init).latent_dist.sample() * pipe.vae.config.scaling_factor


def _build_timesteps(
    scheduler,
    config: SamplerConfig,
    *,
    device: torch.device,
    verbose: bool,
) -> tuple[torch.Tensor, int]:
    """
    Returns (timesteps, start_index). Handles contiguous mode or cropped scheduler spacing.
    """
    if config.full_denoise_from_start:
        if config.start_timestep_value is None:
            raise ValueError("full_denoise_from_start=True requires start_timestep_value to be set.")
        if getattr(scheduler.config, "use_karras_sigmas", False):
            raise ValueError("full_denoise_from_start=True requires scheduler.use_karras_sigmas=False.")
        stride = max(1, int(config.timestep_stride))
        start_t = int(min(config.start_timestep_value, scheduler.config.num_train_timesteps - 1))
        custom_timesteps = torch.arange(start_t, -1, -stride, device=device, dtype=torch.int64)
        scheduler.set_timesteps(timesteps=custom_timesteps.tolist(), device=device)
        timesteps = scheduler.timesteps
        t_start_index = 0
        if verbose:
            print(f"Using full contiguous timesteps from {start_t} → 0 with stride {stride} "
                  f"({len(timesteps)} total steps).")
        return timesteps, t_start_index

    scheduler.set_timesteps(config.steps, device=device)
    full_timesteps = scheduler.timesteps

    if config.start_timestep_value is not None:
        target = torch.tensor(config.start_timestep_value, device=full_timesteps.device, dtype=full_timesteps.dtype)
        t_start_index = int(torch.argmin(torch.abs(full_timesteps - target)).item())
    else:
        total_timesteps = len(full_timesteps)
        # steps_to_use = max(1, int(total_timesteps * config.strength))
        # t_start_index = max(total_timesteps - steps_to_use, 0)

        t_start_index = int((1.0 - config.strength) * total_timesteps)
        t_start_index = max(0, min(total_timesteps - 1, t_start_index))

    timesteps = full_timesteps[t_start_index:]

    if verbose:
        start_ts_value = float(timesteps[0].item()) if isinstance(timesteps[0], torch.Tensor) else float(timesteps[0])
        print(f"Using {len(timesteps)}/{len(full_timesteps)} timesteps, start index {t_start_index}, "
              f"start t={start_ts_value:.1f}")
        print(f"Timesteps: {timesteps}")

    return timesteps, t_start_index


def _add_noise(
    scheduler,
    init_latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    verbose: bool,
) -> torch.Tensor:
    if verbose:
        print(f"Adding noise at timestep {timestep}.")
    if isinstance(timestep, torch.Tensor) and timestep.ndim == 0:
        timestep = timestep.repeat(init_latents.shape[0])
    return scheduler.add_noise(init_latents, torch.randn_like(init_latents), timestep)


def _scheduler_step_with_x0(
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    noise_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _maybe_apply_sgg(
    latents_next: torch.Tensor,
    timestep: torch.Tensor,
    progress: float,
    *,
    pipe,
    mask_batch: torch.Tensor,
    teacher_bundle: ModelBundle,
    reference_logits: Optional[torch.Tensor],
    config: SamplerConfig,
    device: torch.device,
    torch_dtype: torch.dtype,
    mode: str = "gsg",
    noise_pred: torch.Tensor,
) -> tuple[torch.Tensor, Optional[float], Optional[dict[str, torch.Tensor]]]:
    if not (config.guide_start <= progress <= config.guide_end):
        print(" No SGG guidance at this step.")
        return latents_next.detach(), None, None

    #strongest when the image is almost pure noise
    alpha_bar_t = pipe.scheduler.alphas_cumprod[int(timestep)]  # scalar tensor
    sigma_t = torch.sqrt(1 - alpha_bar_t)  # ∝ noise level
    guide_lambda_t = config.guide_lambda * (sigma_t / pipe.scheduler.alphas_cumprod.sqrt().max())  # normalize

    latents_ori = latents_next.detach()  # teacher/base branch (no grad)
    latents_guided = latents_ori.clone().detach().requires_grad_(True)  # branch we backprop through

    if config.pred_on_x0_hat:
        sqrt_alpha_t = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_bar_t).sqrt()
        x0_hat_latents = (latents_guided - sqrt_one_minus_alpha_t * noise_pred.detach()) / sqrt_alpha_t

    with _maybe_autocast(device.type, torch_dtype):
        if config.pred_on_x0_hat:
            decoded = pipe.vae.decode(x0_hat_latents / pipe.vae.config.scaling_factor).sample  # [-1,1]
        else:
            decoded = pipe.vae.decode(latents_guided / pipe.vae.config.scaling_factor).sample  # [-1,1]
        decoded_01 = (decoded + 1) / 2  # [0,1]
        decoded_01 = decoded_01.clamp(0.0, 1.0)
        if config.guide_blur_sigma > 0:
            decoded_01 = _gaussian_blur(decoded_01, config.guide_blur_sigma)

    mask_for_loss = _mask_keep_classes(mask_batch, config.guide_allowed_classes)

    # Guidance branch selection
    if mode == "lcg":
        if config.loss_type == "kl":
            # LCG with relative KL
            guidance_loss = _lcg_relative_loss(
                decoded_01,
                mask_for_loss,
                reference_logits,
                teacher_bundle,
                temperature=config.ce_temperature,
                class_weights=config.class_weights,
            )
        elif config.loss_type == "ce":
            # Classic LCG with Cross-Entropy
            guidance_loss = _lcg_loss(
                decoded_01,
                mask_for_loss,
                teacher_bundle,
                class_weights=config.class_weights,
            )
        elif config.loss_type == "blend":
            ce_loss = _lcg_loss(
                decoded_01,
                mask_for_loss,
                teacher_bundle,
                class_weights=config.class_weights,
            )
            kl_loss = _lcg_relative_loss(
                decoded_01,
                mask_for_loss,
                reference_logits,
                teacher_bundle,
                temperature=config.ce_temperature,
                class_weights=config.class_weights,
            )
            guidance_loss = config.blend_weight * kl_loss + (1 - config.blend_weight) * ce_loss
        if guidance_loss is None:
            print(" LCG skipped (no valid class pixels); falling back to GSG.")
            guidance_loss = ce_label(decoded_01,
                                     mask_for_loss,
                                     teacher_bundle,
                                     temperature=config.ce_temperature,
                                     class_weights=config.class_weights)
    else:
        if config.loss_type == "kl":
            # GSG with relative KL
            guidance_loss = teacher_relative_loss(
                decoded_01,
                reference_logits,
                teacher_bundle,
                mask=mask_for_loss,
                temperature=config.ce_temperature,
                class_weights=config.class_weights,
            )
        elif config.loss_type == "ce":
            # Classic GSG with Cross-Entropy
            guidance_loss = ce_label(decoded_01,
                                     mask_for_loss,
                                     teacher_bundle,
                                     temperature=config.ce_temperature,
                                     class_weights=config.class_weights)
        elif config.loss_type == "blend":
            ce_loss = ce_label(decoded_01,
                               mask_for_loss,
                               teacher_bundle,
                               temperature=config.ce_temperature,
                               class_weights=config.class_weights)
            kl_loss = teacher_relative_loss(
                decoded_01,
                reference_logits,
                teacher_bundle,
                mask=mask_for_loss,
                temperature=config.ce_temperature,
                class_weights=config.class_weights,
            )
            guidance_loss = config.blend_weight * kl_loss + (1 - config.blend_weight) * ce_loss

    if config.guide_tv_weight > 0:
        guidance_loss = guidance_loss + config.guide_tv_weight * _total_variation(decoded_01)

    # ∇_z L_guidance
    guidance_grad = torch.autograd.grad(guidance_loss, latents_guided, retain_graph=False, create_graph=False)[0]
    guidance_grad_used, guidance_norm = _clip_gradient(
        guidance_grad,
        max_norm=config.grad_clip_norm,
        max_value=config.grad_clip_value,
        eps=config.grad_eps,
    )

    # Update latents_guided semantic gradient (detached). No grad needed beyond this point.
    latents_guided_1step = (latents_guided - guide_lambda_t * guidance_grad_used).detach()

    # Teacher-relative loss (Compute L_TR = ‖z_guided - z_ori‖²)
    tr_grad = 2 * (latents_guided_1step - latents_ori)  # direct gradient of MSE
    tr_grad_used, tr_norm = _clip_gradient(
        tr_grad,
        max_norm=config.grad_clip_norm,
        max_value=config.grad_clip_value,
        eps=config.grad_eps,
    )

    # Teacher-relative latent penalty to keep SGG near the base SD manifold
    latents_updated = latents_ori - guide_lambda_t * (guidance_grad_used + config.lambda_tr * tr_grad_used)

    guidance_info = (f"SGG ({mode.upper()}) Loss type={config.loss_type}, "
                     f"guidance_loss={guidance_loss.item():.4f}, "
                     f"mse_loss={(latents_guided_1step - latents_ori).pow(2).mean().item():.6f}, "
                     f"guide_lambda_t={guide_lambda_t.item():.4f}")
    norm_info = (f"Norms -> guidance_grad(L2)={guidance_norm:.4f}, "
                 f"tr_grad(L2)={tr_norm:.4f}, "
                 f"guidance_grad(max)={guidance_grad_used.abs().max().item():.4f}, "
                 f"tr_grad(max)={tr_grad_used.abs().max().item():.4f}")
    print(f" {guidance_info}")
    print(f"  {norm_info}")

    target_size = tuple(decoded_01.shape[-2:])
    guidance_heatmap = _gradient_to_heatmap(guidance_grad_used.detach(), target_size=target_size)
    tr_heatmap = _gradient_to_heatmap(tr_grad_used.detach(), target_size=target_size)
    guidance_overlay = _overlay_heatmap(decoded_01.detach(), guidance_heatmap)
    tr_overlay = _overlay_heatmap(decoded_01.detach(), tr_heatmap)

    visuals = {
        "guidance_heatmap": guidance_heatmap.detach(),
        "tr_heatmap": tr_heatmap.detach(),
        "guidance_overlay": guidance_overlay.detach(),
        "tr_overlay": tr_overlay.detach(),
        "guidance_grad_norm": guidance_norm,
        "tr_grad_norm": tr_norm,
    }

    return latents_updated.detach(), float(guidance_loss.detach().cpu()), visuals


def build_step_logger(steps_dir: Path) -> Callable[[StepOutput], None]:
    """
    Minimal default callback that saves the decoded image and the teacher mask for each logged step.
    """
    steps_dir.mkdir(parents=True, exist_ok=True)

    def _callback(step: StepOutput) -> None:
        suffix = f"step_{step.step_index:03d}_t{step.timestep:04d}"
        save_01(step.decoded_image, steps_dir / f"{suffix}_image.png")
        save_01(step.decoded_x0_image, steps_dir / f"{suffix}_x0.png")
        mask_to_color(step.predicted_mask[0]).save(steps_dir / f"{suffix}_mask.png")
        mask_to_color(step.predicted_mask_x0[0]).save(steps_dir / f"{suffix}_mask_x0.png")
        if step.guidance_heatmap is not None:
            save_01(step.guidance_heatmap, steps_dir / f"{suffix}_guidance_grad.png")
        if step.tr_heatmap is not None:
            save_01(step.tr_heatmap, steps_dir / f"{suffix}_tr_grad.png")
        if step.guidance_overlay is not None:
            save_01(step.guidance_overlay, steps_dir / f"{suffix}_guidance_overlay.png")
        if step.tr_overlay is not None:
            save_01(step.tr_overlay, steps_dir / f"{suffix}_tr_overlay.png")

    return _callback


def decode_latents(
    pipe: StableDiffusionImg2ImgPipeline,
    latents: torch.Tensor,
    *,
    device_type: str,
    torch_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        with _maybe_autocast(device_type, torch_dtype):
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
            return (decoded + 1) / 2


def predict_mask_from_image(
    image_batch: torch.Tensor,
    bundle: ModelBundle,
    *,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    logits = compute_teacher_logits(image_batch, bundle, target_size=target_size)
    return logits.argmax(dim=1)


def build_sampler_loader(
    root: Path,
    split: str,
    size: Tuple[int, int],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[CityscapesSegmentation, torch.utils.data.DataLoader]:
    joint_transform = build_joint_randomresizedcrop(size[0])
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
    #with torch.autocast(device_type=device.type, dtype=bundle.model.dtype):
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
    temperature: float = 1.0,
    class_weights: Optional[dict[int, float]] = None,
) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    logits = compute_teacher_logits(image_batch, bundle, target_size=mask.shape[-2:])
    target = mask.to(device=logits.device, dtype=torch.long)
    if temperature != 1.0:
        logits = logits / temperature
    if class_weights is None:
        return F.cross_entropy(logits, target, ignore_index=IGNORE_LABEL)
    loss_map = F.cross_entropy(logits, target, ignore_index=IGNORE_LABEL, reduction="none")
    weights = _pixel_weight_map(target, class_weights).to(device=logits.device, dtype=logits.dtype)
    valid = target != IGNORE_LABEL
    weighted = loss_map * weights
    denom = weights[valid].sum().clamp_min(1e-6)
    return weighted[valid].sum() / denom


def kl_logits(
    reference_logits: torch.Tensor,
    image_batch: torch.Tensor,
    bundle: ModelBundle,
) -> torch.Tensor:
    candidate_logits = compute_teacher_logits(image_batch, bundle, target_size=reference_logits.shape[-2:])
    log_probs = F.log_softmax(candidate_logits, dim=1)
    ref_probs = F.softmax(reference_logits.detach(), dim=1)
    return F.kl_div(log_probs, ref_probs, reduction="batchmean")


def _lcg_relative_loss(
    decoded_01: torch.Tensor,
    mask_batch: torch.Tensor,
    reference_logits: torch.Tensor,
    teacher_bundle: ModelBundle,
    *,
    temperature: float = 1.0,
    class_weights: Optional[dict[int, float]] = None,
) -> Optional[torch.Tensor]:
    """
    Label-conditional *relative* guidance:
    For each class present in mask_batch (excluding IGNORE_LABEL),
    compute a per-class KL( p_ref || p_cur ) restricted to that class region,
    then average across classes.

    p_ref   = teacher(original_image)
    p_cur   = teacher(decoded_01)
    """
    # Teacher logits for current guided image
    cand_logits = compute_teacher_logits(decoded_01, teacher_bundle, target_size=mask_batch.shape[-2:])

    # Reference teacher logits (from original image), already sized to mask resolution
    ref_logits = reference_logits

    if temperature != 1.0:
        cand_logits = cand_logits / temperature
        ref_logits = ref_logits / temperature

    # [B, C, H, W]
    cand_log_probs = F.log_softmax(cand_logits, dim=1)
    ref_probs = F.softmax(ref_logits.detach(), dim=1)

    B, C, H, W = cand_logits.shape

    # Flatten spatial dims for easier masking
    cand_log_probs_flat = cand_log_probs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    ref_probs_flat = ref_probs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    mask_flat = mask_batch.reshape(-1)  # [B*H*W]

    classes = torch.unique(mask_batch)
    classes = classes[classes != IGNORE_LABEL]
    if classes.numel() == 0:
        return None

    per_class_losses = []
    per_class_weights = []
    for cls in classes:
        # select pixels of this class
        region_mask = (mask_flat == cls)
        if not region_mask.any():
            continue
        weight = _class_weight_value(int(cls.item()), class_weights)
        if weight == 0.0:
            continue
        cand_lp_region = cand_log_probs_flat[region_mask]  # [N_c, C]
        ref_p_region = ref_probs_flat[region_mask]  # [N_c, C]

        # KL( p_ref || p_cur ) averaged over pixels in the region
        kl_c = F.kl_div(
            cand_lp_region,  # log p_cur
            ref_p_region,  # p_ref
            reduction="batchmean",
        )
        per_class_losses.append(kl_c)
        per_class_weights.append(weight)

    if not per_class_losses or not per_class_weights:
        return None

    losses = torch.stack(per_class_losses)
    if class_weights is None:
        return losses.mean()
    weight_tensor = torch.as_tensor(per_class_weights, device=losses.device, dtype=losses.dtype)
    return (losses * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)


def teacher_relative_loss(
    decoded_01: torch.Tensor,
    reference_logits: torch.Tensor,
    bundle: ModelBundle,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    class_weights: Optional[dict[int, float]] = None,
) -> torch.Tensor:
    """
    Relative teacher loss: per-pixel KL( p_ref || p_cur ) between
    teacher on original image (reference_logits) and teacher on current decoded image.

    Optionally restrict to pixels where mask != IGNORE_LABEL.
    """
    device = reference_logits.device
    # logits on current guided image
    cand_logits = compute_teacher_logits(decoded_01, bundle, target_size=reference_logits.shape[-2:])
    cand_logits = cand_logits.to(device=device)

    ref_logits = reference_logits.to(device=device)

    if temperature != 1.0:
        cand_logits = cand_logits / temperature
        ref_logits = ref_logits / temperature

    cand_log_probs = F.log_softmax(cand_logits, dim=1)  # log p_cur
    ref_probs = F.softmax(ref_logits.detach(), dim=1)  # p_ref (no grad through ref)

    # pixelwise KL (shape [B, C, H, W])
    kl_map = F.kl_div(
        cand_log_probs,
        ref_probs,
        reduction="none",
    )

    # Reduce over channels: sum_c p_ref(c) * (log p_ref(c) - log p_cur(c))
    kl_map = kl_map.sum(dim=1)  # [B, H, W]

    if mask is not None:
        # Restrict to valid pixels (and optionally apply class weights)
        valid = (mask != IGNORE_LABEL).to(device=device)  # [B, H, W]
        if class_weights is None:
            kl_map = kl_map * valid
            denom = valid.float().sum().clamp_min(1.0)
            return kl_map.sum() / denom
        weights = _pixel_weight_map(mask.to(device=device), class_weights).to(device=device, dtype=kl_map.dtype)
        weighted = kl_map * weights
        denom = (weights * valid).sum().clamp_min(1e-6)
        return weighted.sum() / denom
    return kl_map.mean()


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


def _clip_gradient(grad: torch.Tensor, *, max_norm: Optional[float], max_value: Optional[float],
                   eps: float) -> tuple[torch.Tensor, float]:
    """Apply elementwise clamp then global-norm clip (if configured)."""
    grad_clipped = grad
    if max_value is not None:
        grad_clipped = torch.clamp(grad_clipped, min=-max_value, max=max_value)
    norm = torch.linalg.norm(grad_clipped)
    if max_norm is not None and norm > max_norm:
        grad_clipped = grad_clipped * (max_norm / (norm + eps))
    return grad_clipped, float(norm.detach().cpu())


def _grad_magnitude(grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return grad.pow(2).sum(dim=1, keepdim=True).add(eps).sqrt()


def _colormap(heat: torch.Tensor) -> torch.Tensor:
    """Simple blue→cyan→yellow→red colormap for single-channel heat in [0,1]."""
    heat = heat.clamp(0, 1)
    r = heat
    g = 1 - (heat - 0.5).abs() * 2  # high in the middle
    b = 1 - heat
    return torch.cat([r, g.clamp(0, 1), b], dim=1)


def _gradient_to_heatmap(
    grad: torch.Tensor,
    *,
    target_size: tuple[int, int],
    eps: float = 1e-8,
) -> torch.Tensor:
    magnitude = _grad_magnitude(grad, eps=eps)
    magnitude = magnitude / magnitude.max().clamp_min(eps)
    heat = F.interpolate(magnitude, size=target_size, mode="bilinear", align_corners=False)
    return _colormap(heat)


def _overlay_heatmap(base: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    return (1 - alpha) * base + alpha * heatmap


def _mask_keep_classes(mask: torch.Tensor, allowed: Optional[list[int]]) -> torch.Tensor:
    if not allowed:
        return mask
    allowed_t = torch.tensor(allowed, device=mask.device, dtype=mask.dtype)
    keep = (mask.unsqueeze(-1) == allowed_t).any(dim=-1)
    masked = mask.clone()
    masked[~keep] = IGNORE_LABEL
    return masked


def _class_weight_value(cls: int, class_weights: Optional[dict[int, float]]) -> float:
    if class_weights is None:
        return 1.0
    return float(class_weights.get(int(cls), 0.0))


def _pixel_weight_map(mask: torch.Tensor, class_weights: Optional[dict[int, float]]) -> torch.Tensor:
    """
    Build a per-pixel weight map from class IDs. Defaults to all ones when no weights are given.
    """
    device = mask.device
    dtype = torch.float32 if not mask.is_floating_point() else mask.dtype
    if class_weights is None:
        return torch.ones_like(mask, dtype=dtype, device=device)
    weights = torch.zeros_like(mask, dtype=dtype, device=device)
    for cls, value in class_weights.items():
        if value == 0:
            continue
        weights = torch.where(mask == int(cls), torch.as_tensor(float(value), device=device, dtype=dtype), weights)
    weights = torch.where(mask == IGNORE_LABEL, torch.zeros((), device=device, dtype=dtype), weights)
    return weights


def _lcg_loss(
    decoded_01: torch.Tensor,
    mask_batch: torch.Tensor,
    teacher_bundle: ModelBundle,
    *,
    class_weights: Optional[dict[int, float]] = None,
) -> Optional[torch.Tensor]:
    """
    Label-conditional guidance: compute per-class NLL over each present class region, average across classes.
    Returns None if no valid class pixels.
    """
    logits = compute_teacher_logits(decoded_01, teacher_bundle, target_size=mask_batch.shape[-2:])
    log_probs = F.log_softmax(logits, dim=1)

    classes = torch.unique(mask_batch)
    classes = classes[classes != IGNORE_LABEL]
    if classes.numel() == 0:
        return None

    per_class_losses = []
    per_class_weights = []
    for cls in classes:
        region_mask = (mask_batch == cls)
        if not region_mask.any():
            continue
        # Only penalise pixels belonging to this class
        nll = -log_probs[:, cls][region_mask]
        weight = _class_weight_value(int(cls.item()), class_weights)
        if weight == 0.0:
            continue
        per_class_losses.append(nll.mean())
        per_class_weights.append(weight)

    if not per_class_losses or not per_class_weights:
        return None

    losses = torch.stack(per_class_losses)
    if class_weights is None:
        return losses.mean()
    weight_tensor = torch.as_tensor(per_class_weights, device=losses.device, dtype=losses.dtype)
    return (losses * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)


## Inversion
@torch.no_grad()
def invert(
    pipe,
    latents,
    device,
    torch_dtype,
    guidance_scale=CFG,
    num_inference_steps=STEPS,
    do_classifier_free_guidance=True,
):
    # change prompt to be sunny:
    SUNNY_PROMPT = PROMPT.replace("rain", "sunny clear day")
    prompt_tokens = pipe.tokenizer(
        SUNNY_PROMPT,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative_tokens = pipe.tokenizer(
        NEG,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        with _maybe_autocast(device.type, torch_dtype):
            pos_embeds = pipe.text_encoder(prompt_tokens.input_ids.to(device))[0]
            neg_embeds = pipe.text_encoder(negative_tokens.input_ids.to(device))[0]
    text_embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm.tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        with _maybe_autocast(device.type, torch_dtype):
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents -
                   (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() /
                                                         alpha_t.sqrt()) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


def run_single_sample(
    pipe: StableDiffusionImg2ImgPipeline,  # SD-1.5 pipeline; we only use its submodules
    teacher_bundle: ModelBundle,  # frozen pixel classifier (SegFormer*)
    *,
    image_pil: Image.Image,  # input source image (sunny)
    reference_image: torch.Tensor,  # optional: for KL vs sunny teacher logits
    mask_batch: torch.Tensor,  # source label map (H×W, int)
    text_context: torch.Tensor,  # [uncond; cond] text embeddings for CFG
    config: SamplerConfig,  # steps, strength, cfg, guide window, etc.
    device: torch.device,
    torch_dtype: torch.dtype,
    callback_interval: Optional[int] = 100,
    step_callback: Optional[Callable[[StepOutput], None]] = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, list[float]]:

    inverted_latents_path = Path(config.inverted_latents_path)

    if inverted_latents_path.exists():
        inverted_latents = torch.load(inverted_latents_path, map_location=device)
    else:
        init_latents = _encode_init_latents(pipe, image_pil, device=device, torch_dtype=torch_dtype)
        inversion_steps = int(config.steps)
        inverted_latents = invert(pipe,
                                  init_latents,
                                  device,
                                  torch_dtype,
                                  num_inference_steps=inversion_steps,
                                  do_classifier_free_guidance=True)
        torch.save(inverted_latents, inverted_latents_path)

    # Rebuild the timestep schedule here (invert mutates the scheduler), and make sure
    # we only index into available inverted latents.
    timesteps, t_start_idx = _build_timesteps(pipe.scheduler, config, device=device, verbose=verbose)
    print(f"Start index: {t_start_idx}")
    max_available_idx = inverted_latents.shape[0] - 1
    if t_start_idx > max_available_idx:
        if verbose:
            print(
                colored(
                    f"Clamping start index from {t_start_idx} to {max_available_idx} "
                    f"to match inverted latents.", "yellow"))
        t_start_idx = max_available_idx
    latents = inverted_latents[-(t_start_idx + 1)][None]

    # --- (Optional) reference teacher logits for extra KL stabilization --------
    reference_logits = None
    if config.loss_type in ("kl", "blend"):
        with torch.no_grad():
            reference_logits = compute_teacher_logits(reference_image,
                                                      teacher_bundle,
                                                      target_size=mask_batch.shape[-2:])

    sgg_loss_log: list[float] = []
    steps_to_run = len(timesteps)
    print(f"Running {steps_to_run}")
    denom = max(1, steps_to_run - 1)
    callback_every = max(1, int(callback_interval)) if callback_interval and callback_interval > 0 else max(
        1, steps_to_run)
    guided_cycle_idx = 0

    protected = {11: 0.1, 12: 0.1, 13: 0.1}  # person, rider, car; tune as needed
    weights = torch.ones_like(mask_batch, dtype=torch.float32, device=mask_batch.device)
    for cls, w in protected.items():
        weights = torch.where(mask_batch == cls, w, weights)

    weights_latent = F.interpolate(weights.unsqueeze(1), size=latents.shape[-2:], mode="nearest")

    # === Main reverse loop: predict noise → scheduler step → (optional) SGG ===
    for step_index, timestep in enumerate(timesteps):
        print(colored(f" Step {step_index+1}/{steps_to_run} at timestep {timestep}", "cyan"))
        # 1) UNet forward with CFG:
        #    duplicate latents for [uncond, cond]; predict eps_uncond, eps_text; mix with cfg.
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = pipe.scheduler.scale_model_input(latent_input, timestep)
        with _maybe_autocast(device.type, torch_dtype):
            noise_pred = pipe.unet(latent_input, timestep, encoder_hidden_states=text_context).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + (config.cfg * weights_latent) * (noise_text - noise_uncond)
            noise_pred_for_sgg = noise_pred.detach()

        latents_next, pred_original_latents = _scheduler_step_with_x0(pipe.scheduler, latents, timestep, noise_pred)

        # Progress (0..1) inside the guided window
        progress = step_index / denom
        sgg_loss_value: Optional[float] = None
        grad_visuals: Optional[dict[str, torch.Tensor]] = None
        guidance_window = (config.guide_start <= progress <= config.guide_end)
        if guidance_window:
            phase = guided_cycle_idx % 2  # 0: GSG, 1: LCG, 2-3: skip
            guided_cycle_idx += 1
            if phase in (0, 1):
                # mode = "gsg"  # "gsg" if phase == 0 else "lcg"
                if config.mode == "alternate":
                    mode = "gsg" if phase == 0 else "lcg"
                elif config.mode in ("gsg", "lcg"):
                    mode = config.mode
                latents, sgg_loss_value, grad_visuals = _maybe_apply_sgg(
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
                # print(f" Applied SGG ({mode.upper()}) this step; loss={sgg_loss_value:.4f}")
            else:
                print(" Guidance window active but skipping SGG this step.")
                latents = latents_next.detach()
        else:
            print(" No SGG guidance at this step.")
            latents = latents_next.detach()

        if sgg_loss_value is not None:
            sgg_loss_log.append(sgg_loss_value)

        # (Optional) logging every few steps: decode preview + teacher prediction
        should_log_step = (step_callback is not None and
                           (step_index % callback_every == 0 or step_index == steps_to_run - 1))
        if should_log_step:
            decoded_for_log = decode_latents(pipe, latents, device_type=device.type, torch_dtype=torch_dtype)
            decoded_x0_for_log = decode_latents(
                pipe,
                pred_original_latents,
                device_type=device.type,
                torch_dtype=torch_dtype,
            )
            predicted_mask = predict_mask_from_image(decoded_for_log, teacher_bundle, target_size=mask_batch.shape[-2:])
            # replace the predicted_mask line
            predicted_mask_x0 = predict_mask_from_image(decoded_x0_for_log,
                                                        teacher_bundle,
                                                        target_size=mask_batch.shape[-2:])

            timestep_value = int(timestep.detach().cpu().item()) if isinstance(timestep,
                                                                               torch.Tensor) else int(timestep)
            step_callback(
                StepOutput(
                    step_index=step_index,
                    timestep=timestep_value,
                    progress=progress,
                    sgg_loss=sgg_loss_value,
                    decoded_image=decoded_for_log.detach().cpu(),
                    decoded_x0_image=decoded_x0_for_log.detach().cpu(),
                    predicted_mask=predicted_mask.detach().cpu(),
                    predicted_mask_x0=predicted_mask_x0.detach().cpu(),
                    guidance_heatmap=grad_visuals.get("guidance_heatmap").detach().cpu()
                    if grad_visuals and grad_visuals.get("guidance_heatmap") is not None else None,
                    tr_heatmap=grad_visuals.get("tr_heatmap").detach().cpu()
                    if grad_visuals and grad_visuals.get("tr_heatmap") is not None else None,
                    guidance_overlay=grad_visuals.get("guidance_overlay").detach().cpu()
                    if grad_visuals and grad_visuals.get("guidance_overlay") is not None else None,
                    tr_overlay=grad_visuals.get("tr_overlay").detach().cpu()
                    if grad_visuals and grad_visuals.get("tr_overlay") is not None else None,
                ))

        if verbose and (step_index % max(1, config.steps // 10) == 0):
            timestep_display = int(timestep.detach().cpu().item()) if isinstance(timestep,
                                                                                 torch.Tensor) else int(timestep)
            print(f"[{step_index:02d}/{steps_to_run}] t={timestep_display}")

    # Final decode to image space
    decoded_final = decode_latents(pipe, latents, device_type=device.type, torch_dtype=torch_dtype)
    return decoded_final, sgg_loss_log


def _maybe_autocast(device_type: str, dtype: torch.dtype):
    return torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else contextlib.nullcontext()


def load_finetuned_pipeline(device: torch.device, dtype: torch.dtype) -> StableDiffusionPipeline:
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    progress = False

    print("Loading base pipeline %s with dtype=%s on %s", pretrained_model_name_or_path, dtype, device)

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    ckpt_dir = Path(
        "gen/checkpoints/sd_lora_finetune_100ep_lora128_dyna_maskprompts/step-036000").expanduser().resolve()

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
        print("Loading fine-tuned UNet from %s", unet_path)
        pipe.unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
    elif lora_path.exists():
        print("Loading LoRA weights from %s", lora_path)
        try:
            #pipe.load_lora_weights(lora_path)
            pipe.unet.load_attn_procs(lora_path)
            # Optional: verify by checking keys (class-name checks are brittle)
            has_lora = any("lora" in k.lower() for k in pipe.unet.state_dict().keys())
            print("LoRA params present in UNet state: %s", has_lora)

        except AttributeError as exc:
            raise RuntimeError("The installed diffusers version does not support `load_lora_weights`. "
                               "Upgrade diffusers or load a checkpoint with full UNet weights.") from exc
    else:
        print("No UNet or LoRA weights found in %s; using base checkpoint UNet.", ckpt_dir)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        print("Could not enable xformers attention: %s", exc)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=not progress)
    return pipe


def main(
    dataset_root: str | Path = "data/cityscapes",
    split: str = "my_test",
    out_dir: str | Path = OUTDIR,
    *,
    teacher_model: str = TEACHER_MODEL,
    max_samples: Optional[int] = 3,
    num_workers: int = 1,
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

    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL,
    #                                                       torch_dtype=torch_dtype,
    #                                                       safety_checker=None,
    #                                                       use_auth_token=HF_TOKEN)

    pipe = load_finetuned_pipeline(device, dtype=torch_dtype)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        print(f"Could not enable xformers attention: {exc}")

    if not sampler_config.full_denoise_from_start:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        from diffusers import DPMSolverMultistepScheduler
        use_karras = sampler_config.use_karras_sigmas and not sampler_config.full_denoise_from_start
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",  # smaller noise jumps
            use_karras_sigmas=use_karras,  # disable if using custom contiguous timesteps
            timestep_spacing=sampler_config.timestep_spacing,  # trailing/leading/linspace
            rescale_betas_zero_snr=True,  # flattens SNR so early steps are less destructive
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_attention_slicing()  # chunk attention to reduce peak memory
    pipe.enable_vae_slicing()  # decode in slices to save VRAM
    pipe.enable_vae_tiling()  # tile VAE for large images
    pipe.unet.to(memory_format=torch.channels_last)

    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    dataset, loader = build_sampler_loader(
        Path(dataset_root),
        split,
        sampler_config.size,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    prompt_tokens = pipe.tokenizer(
        sampler_config.prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative_tokens = pipe.tokenizer(
        sampler_config.negative_prompt or "",
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

        if idx != 2:
            continue

        sample_info = dataset.samples[idx]
        relative_path = sample_info.image_path.relative_to(dataset.left_dir)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If no custom callback was provided but a logging interval was, attach a simple logger that
        # saves intermediate decoded images and teacher masks.
        local_callback = step_callback
        if local_callback is None and callback_interval is not None:
            local_callback = build_step_logger(output_path.parent / f"{output_path.stem}_steps")

        image_batch = images.to(device=device, dtype=torch.float32)
        mask_batch = masks.to(device=device, dtype=torch.long)
        image_pil = tensor_to_pil(images[0])

        original_path = output_path.with_name(output_path.stem + "_original.png")
        mask_path = output_path.with_name(output_path.stem + "_mask.png")
        image_pil.save(original_path)
        # save_mask(mask_batch[0], mask_path)
        mask_to_color(mask_batch[0]).save(mask_path)

        # Save config as json
        config_path = output_path.with_name(output_path.stem + "_config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(sampler_config), f, indent=4)

        decoded_01, sgg_log = run_single_sample(
            pipe,
            teacher_bundle,
            image_pil=image_pil,
            reference_image=image_batch,
            mask_batch=mask_batch,
            text_context=text_context,
            config=sampler_config,
            device=device,
            torch_dtype=torch_dtype,
            callback_interval=callback_interval,
            step_callback=local_callback,
            verbose=verbose,
        )

        save_01(decoded_01, output_path)
        sgg_path = output_path.with_name(output_path.stem + "_sgg_loss.npy")
        np.save(sgg_path, np.array(sgg_log, dtype=np.float32))

        if verbose:
            print(f"saved {relative_path} steps_with_SGG: {len(sgg_log)}")
            print("Last SGG loss:", sgg_log[-1])


if __name__ == "__main__":
    main()
