from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from PIL import Image

from gen_ddpm.linear_noise_scheduler import LinearNoiseScheduler
from gen.sgg_sampler import (
    _mask_keep_classes,
    _lcg_loss,
    _lcg_relative_loss,
    ce_label,
    teacher_relative_loss,
    compute_teacher_logits,
)
from gen_ddpm.model import UNet
from seg.dataloaders.cityscapes import CityscapesSegmentation
from seg.dataloaders.acdc import ACDCSegmentation, build_acdc_loader
from seg.utils.hf_utils import mask_to_color
from seg.infer import load_hf_model, ModelBundle
from srgan_model.models import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN, STD = [0.4865, 0.4998, 0.4323], [0.2326, 0.2276, 0.2659]

OUTDIR = "gen_ddpm/ddpm_final_experiments/benchmark_002"

input_paths = [
    "frankfurt_000000_002963_leftImg8bit",
    "frankfurt_000000_009688_leftImg8bit",
    "frankfurt_000000_013382_leftImg8bit",
    "lindau_000004_000019_leftImg8bit",
    "lindau_000013_000019_leftImg8bit",
    "lindau_000047_000019_leftImg8bit",
    "munster_000009_000019_leftImg8bit",
]

GUIDE_LAMBDA = 10.0
GUIDE_START = 0.0
GUIDE_END = 1.0
GUIDE_ALLOWED_CLASSES = [0, 1, 2, 8, 9, 10, 13, 14]
GUIDE_CLASS_WEIGHTS = {
    13: 3.0,  # car
    0: 1.5,  # road
    2: 1.5,  # building
    1: 1.5,  # sidewalk
    8: 1.5,  # vegetation
    9: 1.0,  # terrain
    10: 1.0,  # sky
}
GUIDE_CLASS_WEIGHTS = None
TEMPERATURE = 1.0
LOSS_TYPE = "ce"
BLEND_WEIGHT = 0.1
PRED_ON_X0_HAT = False
STEPS = 400
MODE = "alternate"
GRAD_CLIP_NORM: Optional[float] = 1.5  # clip gradients by global norm if set
GRAD_CLIP_VALUE: Optional[float] = None  # clamp elementwise if set
GRAD_EPS = 1e-8


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
    predicted_without_guidance: Optional[torch.Tensor] = None
    predicted_mask_without_guidance: Optional[torch.Tensor] = None
    guidance_heatmap: Optional[torch.Tensor] = None
    tr_heatmap: Optional[torch.Tensor] = None
    guidance_overlay: Optional[torch.Tensor] = None
    tr_overlay: Optional[torch.Tensor] = None


@dataclass
class SamplerConfig:
    # data
    dataset_root: str = "data/cityscapes"  # "data/cityscapes"
    split: str = "my_test"
    image_size: int = 128
    batch_size: int = 1
    num_workers: int = 1
    max_samples: Optional[int] = 20

    # diffusion
    diffusion_steps: int = STEPS  # how many forward steps to apply before reversing
    num_timesteps: int = 1000  # training schedule length
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # model/checkpoints
    checkpoint: str = "gen_ddpm/checkpoints/1000-checkpoint.ckpt"
    srgan_checkpoint: str = "srgan_model/weights/swift_srgan_4x.pth.tar"
    teacher_model: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"

    guidance_lr: float = GUIDE_LAMBDA
    guide_start: float = GUIDE_START
    guide_end: float = GUIDE_END
    guide_allowed_classes: Optional[list[int]] = field(default_factory=lambda: GUIDE_ALLOWED_CLASSES)
    guide_class_weights: Optional[dict[int, float]] = field(default_factory=lambda: GUIDE_CLASS_WEIGHTS)
    temperature: float = TEMPERATURE
    loss_type: str = LOSS_TYPE  # "ce", "kl", or "blend"
    blend_weight: float = BLEND_WEIGHT
    pred_on_x0_hat: bool = PRED_ON_X0_HAT
    mode: str = MODE
    grad_clip_norm: Optional[float] = GRAD_CLIP_NORM
    grad_clip_value: Optional[float] = GRAD_CLIP_VALUE
    grad_eps: float = GRAD_EPS
    seed: Optional[int] = 42

    # logging
    output_dir: str = OUTDIR
    log_every: Optional[int] = None
    sanity_check: bool = False


def _build_acdc_transforms(image_size: int,) -> Tuple[Callable, Callable]:
    """Resize → center-crop to a square and normalise with dataset mean/std for the DDPM."""

    def joint_transform(image, mask, ref_image):
        image = TF.resize(
            image,
            (image_size, image_size * 2),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        image = TF.center_crop(image, image_size)

        ref_image = TF.resize(
            ref_image,
            (image_size, image_size * 2),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        ref_image = TF.center_crop(ref_image, image_size)

        mask = TF.resize(
            mask,
            (image_size, image_size * 2),
            interpolation=transforms.InterpolationMode.NEAREST,
            antialias=False,
        )
        mask = TF.center_crop(mask, image_size)

        return image, mask, ref_image

    def image_transform(image):
        tensor = TF.to_tensor(image)
        return transforms.Normalize(mean=MEAN, std=STD)(tensor)

    return joint_transform, image_transform


def _build_cityscapes_transforms(image_size: int,) -> Tuple[Callable, Callable]:
    """Resize → center-crop to a square and normalise with dataset mean/std for the DDPM."""

    def joint_transform(image, mask):
        image = TF.resize(
            image,
            (image_size, image_size * 2),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        image = TF.center_crop(image, image_size)

        mask = TF.resize(
            mask,
            (image_size, image_size * 2),
            interpolation=transforms.InterpolationMode.NEAREST,
            antialias=False,
        )
        mask = TF.center_crop(mask, image_size)
        return image, mask

    def image_transform(image):
        tensor = TF.to_tensor(image)
        return transforms.Normalize(mean=MEAN, std=STD)(tensor)

    return joint_transform, image_transform


def _denormalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """Convert from (x - mean)/std space to [0,1] for SRGAN + teacher.

    This is differentiable: gradients w.r.t. the normalized tensor will
    correctly pass through this operation.
    """
    mean = torch.tensor(MEAN, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    std = torch.tensor(STD, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0.0, 1.0)


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


def _maybe_set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_batch_image(batch: torch.Tensor, path: Path, *, nrow: Optional[int] = None) -> None:
    batch_01 = _denormalize_to_01(batch)
    grid = make_grid(batch_01, nrow=nrow or batch_01.shape[0])
    pil_image = transforms.ToPILImage()(grid.cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)


def save_01(x01: torch.Tensor, path: Path) -> None:
    tensor = x01[0].detach().cpu().clamp(0, 1)
    pil_image = transforms.ToPILImage()(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)


def build_step_logger(steps_dir: Path) -> Callable[[StepOutput], None]:
    """Save decoded image/x0 and teacher masks for each logged step."""
    steps_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def _callback(step: StepOutput) -> None:
        suffix = f"step_{step.step_index:03d}_t{step.timestep:04d}"

        if step.decoded_image is not None:
            save_01(step.decoded_image, steps_dir / f"{suffix}_image.png")
        if step.decoded_x0_image is not None:
            save_01(step.decoded_x0_image, steps_dir / f"{suffix}_x0.png")

        if step.predicted_mask is not None:
            mask_to_color(step.predicted_mask[0].cpu()).save(steps_dir / f"{suffix}_mask.png")
        if step.predicted_mask_x0 is not None:
            mask_to_color(step.predicted_mask_x0[0].cpu()).save(steps_dir / f"{suffix}_mask_x0.png")

        if step.predicted_without_guidance is not None:
            save_01(step.predicted_without_guidance, steps_dir / f"{suffix}_image_wo_guidance.png")
        if step.predicted_mask_without_guidance is not None:
            mask_to_color(step.predicted_mask_without_guidance[0].cpu()).save(
                steps_dir / f"{suffix}_mask_wo_guidance.png"
            )

        if step.guidance_heatmap is not None:
            save_01(step.guidance_heatmap, steps_dir / f"{suffix}_guidance_grad.png")
        if step.tr_heatmap is not None:
            save_01(step.tr_heatmap, steps_dir / f"{suffix}_tr_grad.png")
        if step.guidance_overlay is not None:
            save_01(step.guidance_overlay, steps_dir / f"{suffix}_guidance_overlay.png")
        if step.tr_overlay is not None:
            save_01(step.tr_overlay, steps_dir / f"{suffix}_tr_overlay.png")

    return _callback


def _move_scheduler_to_device(scheduler: LinearNoiseScheduler, device: torch.device) -> None:
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_cum_prod = scheduler.alpha_cum_prod.to(device)
    scheduler.sqrt_alpha_cum_prod = scheduler.sqrt_alpha_cum_prod.to(device)
    scheduler.one_minus_cum_prod = scheduler.one_minus_cum_prod.to(device)
    scheduler.sqrt_one_minus_alpha_cum_prod = scheduler.sqrt_one_minus_alpha_cum_prod.to(device)
    scheduler.max_sigma = scheduler.max_sigma.to(device)


def run_shape_sanity_check(
    model: UNet,
    scheduler: LinearNoiseScheduler,
    image_size: int,
    device: torch.device,
) -> None:
    """Lightweight smoke test to ensure model, scheduler, and time embeddings agree on shapes."""
    with torch.no_grad():
        x = torch.randn(1, 3, image_size, image_size, device=device)
        t = torch.tensor([scheduler.num_timesteps - 1], device=device, dtype=torch.long)
        time_embed = scheduler.one_minus_cum_prod.to(device)[t].view(-1, 1, 1, 1)
        out = model(x, time_embed)
        assert out.shape == x.shape, f"UNet output {out.shape} != input {x.shape}"

        xt = scheduler.add_noise2(x, torch.randn_like(x), t)
        assert xt.shape == x.shape, f"add_noise2 output {xt.shape} != input {x.shape}"


def apply_segmentation_guidance(
    x_t: torch.Tensor,
    mask: torch.Tensor,
    srgan: Generator,
    teacher_bundle: ModelBundle,
    *,
    guidance_lr_t: torch.Tensor,
    mode: str,
    allowed_classes: Optional[list[int]],
    class_weights: Optional[dict[int, float]],
    temperature: float,
    loss_type: str,
    blend_weight: float,
    reference_logits: Optional[torch.Tensor],
    pred_on_x0_hat: bool,
    eps_xt: Optional[torch.Tensor],
    sigma_t: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    grad_clip_norm: Optional[float],
    grad_clip_value: Optional[float],
    grad_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute semantic guidance gradient w.r.t. the *diffusion variable* x_t.

    If pred_on_x0_hat=True:
      - teacher sees x0_hat = (x_t - sigma_t * eps_xt) / sqrt(alpha_bar),
      - gradient is still taken w.r.t x_t via chain rule.

    Otherwise:
      - teacher sees x_t (decoded to [0,1] and upscaled by SRGAN),
      - gradient is w.r.t x_t directly.
    """
    # Detach from UNet / previous steps, but keep a gradient graph w.r.t x_t itself.
    x_var = x_t.detach().requires_grad_(True)

    if pred_on_x0_hat:
        if eps_xt is None:
            raise ValueError("eps_xt is required when pred_on_x0_hat=True.")
        # eps_xt is detached so we do NOT backprop into the UNet.
        x_teacher_base = (x_var - sigma_t * eps_xt.detach()) / sqrt_alpha_bar
    else:
        x_teacher_base = x_var

    # Decode for SRGAN + segmentation teacher.
    sr_input = _denormalize_to_01(x_teacher_base)
    sr_image = srgan(sr_input)  # [B, 3, 512, 512]

    target_size = sr_image.shape[-2:]
    mask_up = F.interpolate(
        mask.float().unsqueeze(1),
        size=target_size,
        mode="nearest",
    ).squeeze(1).long()
    mask_up = _mask_keep_classes(mask_up, allowed_classes)

    # Compute semantic loss (GSG or LCG, CE/KL/blend)
    if mode == "lcg":
        if loss_type == "kl":
            loss = _lcg_relative_loss(
                sr_image,
                mask_up,
                reference_logits,
                teacher_bundle,
                temperature=temperature,
                class_weights=class_weights,
            )
        elif loss_type == "ce":
            loss = _lcg_loss(
                sr_image,
                mask_up,
                teacher_bundle,
                class_weights=class_weights,
            )
        else:
            ce_loss = _lcg_loss(
                sr_image,
                mask_up,
                teacher_bundle,
                class_weights=class_weights,
            )
            kl_loss = _lcg_relative_loss(
                sr_image,
                mask_up,
                reference_logits,
                teacher_bundle,
                temperature=temperature,
                class_weights=class_weights,
            )
            if ce_loss is None and kl_loss is None:
                loss = None
            elif ce_loss is None:
                loss = kl_loss
            elif kl_loss is None:
                loss = ce_loss
            else:
                loss = blend_weight * kl_loss + (1.0 - blend_weight) * ce_loss

        # Fallback if no valid class-region was found.
        if loss is None:
            loss = ce_label(
                sr_image,
                mask_up,
                teacher_bundle,
                temperature=temperature,
                class_weights=class_weights,
            )
    else:  # GSG
        if loss_type == "kl":
            loss = teacher_relative_loss(
                sr_image,
                reference_logits,
                teacher_bundle,
                mask=mask_up,
                temperature=temperature,
                class_weights=class_weights,
            )
        elif loss_type == "ce":
            loss = ce_label(
                sr_image,
                mask_up,
                teacher_bundle,
                temperature=temperature,
                class_weights=class_weights,
            )
        else:
            ce_loss = ce_label(
                sr_image,
                mask_up,
                teacher_bundle,
                temperature=temperature,
                class_weights=class_weights,
            )
            kl_loss = teacher_relative_loss(
                sr_image,
                reference_logits,
                teacher_bundle,
                mask=mask_up,
                temperature=temperature,
                class_weights=class_weights,
            )
            loss = blend_weight * kl_loss + (1.0 - blend_weight) * ce_loss

    # If still None (very degenerate case), skip guidance
    if loss is None:
        return x_t.detach(), torch.tensor(0.0, device=x_t.device)

    #Gradient of loss w.r.t *x_t*
    # grad_x = torch.autograd.grad(
    #     loss,
    #     x_var,
    #     retain_graph=False,
    #     create_graph=False,
    # )[0]

    grad_x = torch.autograd.grad(
        loss,
        sr_image,
        retain_graph=False,
        create_graph=False,
    )[0]
    grad_x = F.interpolate(
        grad_x,
        size=(128, 128),
        mode="bilinear",
        align_corners=False,
    )

    grad_x_clipped, grad_norm = _clip_gradient(
        grad_x,
        max_norm=grad_clip_norm,
        max_value=grad_clip_value,
        eps=grad_eps,
    )

    # grad_x_clipped = grad_x
    # grad_norm = float(torch.linalg.norm(grad_x_clipped).detach().cpu())

    # Gradient descent on the semantic loss (move towards *lower* CE/NLL/KL)
    guided = x_var - guidance_lr_t * grad_x_clipped

    # Optional diagnostics
    print(f"Segmentation guidance ({mode}): "
          f"loss={loss.item():.4f}, "
          f"grad_norm={grad_norm:.4f}, "
          f"grad_x min/max={grad_x_clipped.min().item():.6f}/{grad_x_clipped.max().item():.6f}, "
          f"guidance_lr_t mean={guidance_lr_t.mean().item():.6f}")

    return guided.detach(), loss.detach()


def load_model(checkpoint: str, device: torch.device) -> UNet:
    model = UNet().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def load_srgan(checkpoint: str, device: torch.device) -> Generator:
    srgan = Generator(upscale_factor=4).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    srgan.load_state_dict(ckpt["model"])
    srgan.eval()
    for p in srgan.parameters():
        p.requires_grad_(False)
    return srgan


def build_loader(config: SamplerConfig) -> tuple[CityscapesSegmentation, DataLoader]:
    joint_transform, image_transform = _build_cityscapes_transforms(config.image_size)

    #joint_transform, image_transform = _build_acdc_transforms(image_size=config.image_size)

    # loader = build_acdc_loader(
    #     root_dir=config.dataset_root,
    #     split=config.split,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers,
    #     joint_transform=joint_transform,
    #     image_transform=image_transform,
    # )

    dataset = CityscapesSegmentation(
        root_dir=config.dataset_root,
        split=config.split,
        joint_transform=joint_transform,
        image_transform=image_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=DEVICE.type == "cuda",
    )
    return loader.dataset, loader


@torch.no_grad()
def forward_diffuse(
    x0: torch.Tensor,
    scheduler: LinearNoiseScheduler,
    steps: int,
    *,
    log_every: int = 0,
) -> torch.Tensor:
    """Jump directly to timestep steps - 1 in the forward process (no incremental noise)."""
    if steps < 1:
        raise ValueError("steps must be >= 1 for forward diffusion.")
    t_value = steps - 1
    t = torch.full((x0.size(0),), t_value, device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = scheduler.add_noise2(x0, noise, t)
    return xt


def reverse_denoise(
    xt: torch.Tensor,
    model: UNet,
    scheduler: LinearNoiseScheduler,
    steps: int,
    *,
    mask: torch.Tensor,
    srgan: Generator,
    teacher_bundle: ModelBundle,
    guidance_lr: float,
    guide_start: float,
    guide_end: float,
    reference_logits: Optional[torch.Tensor],
    allowed_classes: Optional[list[int]],
    class_weights: Optional[dict[int, float]],
    temperature: float,
    loss_type: str,
    blend_weight: float,
    pred_on_x0_hat: bool,
    grad_clip_norm: Optional[float],
    grad_clip_value: Optional[float],
    grad_eps: float,
    mode: str = "alternate",
    log_every: int = 0,
    step_logger: Optional[Callable[[StepOutput], None]] = None,
) -> torch.Tensor:
    """
    Reverse the diffusion chain x_t → x_0 using the trained DDPM model with SRGAN + segmentation guidance.

    Guidance is always applied to x_t, even if the teacher predicts on x0_hat.
    """
    x = xt
    x_wo_guidance = xt.clone()
    total_steps = steps
    steps_to_run = steps
    denom = max(1, steps_to_run - 1)

    for idx, step in enumerate(tqdm(reversed(range(steps)), desc="DDPM Denoising")):
        # print(f"Denoising step {idx + 1}/{total_steps}, t={step}...")
        t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
        progress = idx / denom

        alpha_bar = scheduler.alpha_cum_prod.to(x.device)[t].view(-1, 1, 1, 1)
        sqrt_alpha_bar = scheduler.sqrt_alpha_cum_prod.to(x.device)[t].view(-1, 1, 1, 1)
        sigma_t = scheduler.sqrt_one_minus_alpha_cum_prod.to(x.device)[t].view(-1, 1, 1, 1)

        # Time embedding for UNet (same encoding as during training).
        t_embed = scheduler.one_minus_cum_prod.to(x.device)[t].view(-1, 1, 1, 1)

        # Noise prediction on current x_t (no gradients needed here).
        with torch.no_grad():
            eps_xt = model(x, t_embed)
            eps_xt_wo_guidance = model(x_wo_guidance, t_embed)

        # Guidance schedule: stronger at later, cleaner steps (small t).
        # sqrt_alpha_bar is ~1 at t=0 (clean), and small at large t (noisy).
        max_sigma = scheduler.max_sigma.to(x.device)
        guidance_lr_t = guidance_lr * (sigma_t / max_sigma)
        #guidance_lr_t = guidance_lr * sqrt_alpha_bar

        if mode == "alternate":
            selected_mode = "gsg" if idx % 2 == 0 else "lcg"
        elif mode in ("gsg", "lcg"):
            selected_mode = mode

        # Semantic guidance step (gradients only through teacher + SRGAN into x_t).
        guidance_window = guide_start <= progress <= guide_end
        if guidance_window:
            x_guided, loss = apply_segmentation_guidance(
                x,
                mask,
                srgan,
                teacher_bundle,
                guidance_lr_t=guidance_lr_t,
                mode=selected_mode,
                allowed_classes=allowed_classes,
                class_weights=class_weights,
                temperature=temperature,
                loss_type=loss_type,
                blend_weight=blend_weight,
                reference_logits=reference_logits,
                pred_on_x0_hat=pred_on_x0_hat,
                eps_xt=eps_xt,
                sigma_t=sigma_t,
                sqrt_alpha_bar=sqrt_alpha_bar,
                grad_clip_norm=grad_clip_norm,
                grad_clip_value=grad_clip_value,
                grad_eps=grad_eps,
            )
        else:
            x_guided, loss = x, None
        x = x_guided

        # Recompute noise prediction for the *guided* x_t, optionally log, then take the DDPM step.
        with torch.no_grad():
            eps_guided = model(x, t_embed)
            #print(f"Step {idx}/{total_steps - 1} (t={step}): ")
            should_log_step = idx == 0 or step == 0 or (log_every and step % log_every == 0)
            # Logging of teacher outputs (optional)
            if step_logger is not None and should_log_step:
                with torch.inference_mode():
                    sr_image = srgan(_denormalize_to_01(x))
                    logits = compute_teacher_logits(
                        sr_image,
                        teacher_bundle,
                        target_size=sr_image.shape[-2:],
                    )
                    pred_mask = logits.argmax(dim=1)

                    # x0_hat based on guided x_t and eps_guided
                    x0_hat = (x - sigma_t * eps_guided) / sqrt_alpha_bar
                    x0_image = srgan(_denormalize_to_01(x0_hat))
                    logits_x0 = compute_teacher_logits(
                        x0_image,
                        teacher_bundle,
                        target_size=x0_image.shape[-2:],
                    )
                    pred_mask_x0 = logits_x0.argmax(dim=1)

                    # Also run teacher on unguided x_t for comparison
                    sr_image_wo_guidance = srgan(_denormalize_to_01(x_wo_guidance))
                    logits = compute_teacher_logits(
                        sr_image_wo_guidance,
                        teacher_bundle,
                        target_size=sr_image_wo_guidance.shape[-2:],
                    )
                    pred_mask_wo_guidance = logits.argmax(dim=1)

                step_logger(
                    StepOutput(
                        step_index=idx,
                        timestep=step,
                        progress=progress,
                        sgg_loss=float(loss.detach().cpu().item()) if loss is not None else None,
                        decoded_image=sr_image.detach().cpu(),
                        decoded_x0_image=x0_image.detach().cpu(),
                        predicted_mask=pred_mask.detach().cpu(),
                        predicted_mask_x0=pred_mask_x0.detach().cpu(),
                        predicted_without_guidance=x_wo_guidance.detach().cpu(),
                        predicted_mask_without_guidance=pred_mask_wo_guidance.detach().cpu(),
                    ))

            mean, sigma = scheduler.sample_prev_timestep2(x, eps_guided, t)
            x = mean if step == 0 else mean + sigma

            mean_wo, sigma_wo = scheduler.sample_prev_timestep2(x_wo_guidance, eps_xt_wo_guidance, t)
            x_wo_guidance = mean_wo if step == 0 else mean_wo + sigma_wo

    return x, x_wo_guidance


def run_pipeline(
    model: UNet,
    scheduler: LinearNoiseScheduler,
    image_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    *,
    srgan: Generator,
    teacher_bundle: ModelBundle,
    steps: int,
    guidance_lr: float,
    guide_start: float,
    guide_end: float,
    allowed_classes: Optional[list[int]],
    class_weights: Optional[dict[int, float]],
    temperature: float,
    loss_type: str,
    blend_weight: float,
    pred_on_x0_hat: bool,
    grad_clip_norm: Optional[float],
    grad_clip_value: Optional[float],
    grad_eps: float,
    mode: str = "alternate",
    log_every: int,
    step_logger: Optional[Callable[[StepOutput], None]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reference logits from original clean images for KL-type relative losses.
    with torch.inference_mode():
        ref_sr = srgan(_denormalize_to_01(image_batch))
        ref_target = ref_sr.shape[-2:]
        reference_logits = compute_teacher_logits(
            ref_sr,
            teacher_bundle,
            target_size=ref_target,
        )

    noised = forward_diffuse(image_batch, scheduler, steps, log_every=log_every)

    denoised, denoised_wo_guidance = reverse_denoise(
        noised,
        model,
        scheduler,
        steps,
        mask=mask_batch,
        srgan=srgan,
        teacher_bundle=teacher_bundle,
        guidance_lr=guidance_lr,
        guide_start=guide_start,
        guide_end=guide_end,
        reference_logits=reference_logits,
        allowed_classes=allowed_classes,
        class_weights=class_weights,
        temperature=temperature,
        loss_type=loss_type,
        blend_weight=blend_weight,
        pred_on_x0_hat=pred_on_x0_hat,
        grad_clip_norm=grad_clip_norm,
        grad_clip_value=grad_clip_value,
        grad_eps=grad_eps,
        mode=mode,
        log_every=log_every,
        step_logger=step_logger,
    )

    return noised, denoised, denoised_wo_guidance


def save_outputs(
    out_dir: Path,
    *,
    base_name: str,
    original: torch.Tensor,
    sr_original: torch.Tensor,
    sr_denoised: torch.Tensor,
    sr_denoised_wo_guidance: torch.Tensor,
    sr_pred_mask: torch.Tensor,
    sr_pred_mask_wo: torch.Tensor,
    mask: torch.Tensor,
    # ref_image: torch.Tensor,
    noised: torch.Tensor,
    denoised: torch.Tensor,
    final_timestep: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_batch_image(original, out_dir / f"{base_name}_orig.png")
    # _save_batch_image(ref_image, out_dir / f"{base_name}_ref.png")
    save_01(sr_original, out_dir / f"{base_name}_sr_orig.png")
    save_01(sr_denoised, out_dir / f"{base_name}_sr_denoised.png")
    mask_to_color(sr_pred_mask[0]).save(out_dir / f"{base_name}_sr_pred_mask.png")
    save_01(sr_denoised_wo_guidance, out_dir / f"{base_name}_sr_denoised_wo_guidance.png")
    mask_to_color(sr_pred_mask_wo[0]).save(out_dir / f"{base_name}_sr_pred_mask_wo_guidance.png")
    mask_to_color(mask[0]).save(out_dir / f"{base_name}_mask.png")
    _save_batch_image(denoised, out_dir / f"{base_name}_denoised.png")


def save_01(x01: torch.Tensor, path: Path) -> None:
    array = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def main() -> None:
    config = SamplerConfig()

    _maybe_set_seed(config.seed)

    if config.diffusion_steps < 1:
        raise ValueError("diffusion_steps must be >= 1.")
    if config.diffusion_steps > config.num_timesteps:
        raise ValueError("diffusion_steps cannot exceed num_timesteps.")

    model = load_model(config.checkpoint, DEVICE)
    srgan = load_srgan(config.srgan_checkpoint, DEVICE)

    teacher_bundle = load_hf_model(config.teacher_model, device=str(DEVICE))
    for p in teacher_bundle.model.parameters():
        p.requires_grad_(False)

    scheduler = LinearNoiseScheduler(
        config.num_timesteps,
        config.beta_start,
        config.beta_end,
    )
    _move_scheduler_to_device(scheduler, DEVICE)

    if config.sanity_check:
        run_shape_sanity_check(model, scheduler, config.image_size, DEVICE)

    dataset, loader = build_loader(config)

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_counter = 0

    for batch_idx, (batch_images, batch_masks) in enumerate(loader):
        batch_images = batch_images.to(device=DEVICE, dtype=torch.float32)
        batch_masks = batch_masks.to(device=DEVICE)
        batch_size = batch_images.size(0)

        if config.max_samples is not None and sample_counter >= config.max_samples:
            break
        if batch_size == 0:
            continue

        rand_idx = torch.randint(0, batch_size, (1,)).item()

        image = batch_images[rand_idx:rand_idx + 1]
        mask = batch_masks[rand_idx:rand_idx + 1]
        # ref_image = batch_ref_images[rand_idx:rand_idx + 1]

        dataset_idx = min(
            len(dataset.samples) - 1,
            batch_idx * config.batch_size + rand_idx,
        )
        #sample_info = dataset.samples[dataset_idx]
        # relative = Path(sample_info.image_path).relative_to(dataset.rgb_root).with_suffix("")

        # sample_dir = output_root / relative
        # sample_dir.mkdir(parents=True, exist_ok=True)
        # base_name = relative.stem
        #relative = sample_info.image_path.relative_to(dataset.left_dir)

        sample_info = dataset.samples[batch_idx]
        relative = sample_info.image_path.relative_to(dataset.left_dir)
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not output_path.stem in input_paths:
            continue

        sample_dir = output_path
        sample_dir.mkdir(parents=True, exist_ok=True)
        base_name = relative.stem.replace("_leftImg8bit", "")

        step_logger = build_step_logger(sample_dir / "steps") if config.log_every else None

        # Denoising with segmentation guidance requires gradients; surrounding IO remains inference-only.
        noised, denoised, denoised_wo_guidance = run_pipeline(
            model,
            scheduler,
            image,
            mask,
            srgan=srgan,
            teacher_bundle=teacher_bundle,
            steps=config.diffusion_steps,
            guidance_lr=config.guidance_lr,
            guide_start=config.guide_start,
            guide_end=config.guide_end,
            allowed_classes=config.guide_allowed_classes,
            class_weights=config.guide_class_weights,
            temperature=config.temperature,
            loss_type=config.loss_type,
            blend_weight=config.blend_weight,
            pred_on_x0_hat=config.pred_on_x0_hat,
            grad_clip_norm=config.grad_clip_norm,
            grad_clip_value=config.grad_clip_value,
            grad_eps=config.grad_eps,
            mode=config.mode,
            log_every=config.log_every,
            step_logger=step_logger,
        )

        with torch.inference_mode():
            sr_original = srgan(_denormalize_to_01(image))
            sr_denoised = srgan(_denormalize_to_01(denoised))
            sr_denoised_wo_guidance = srgan(_denormalize_to_01(denoised_wo_guidance))

            sr_pred_logits = compute_teacher_logits(
                sr_denoised,
                teacher_bundle,
                target_size=sr_denoised.shape[-2:],
            )
            sr_pred_mask = sr_pred_logits.argmax(dim=1)

            sr_pred_logits_wo = compute_teacher_logits(
                sr_denoised_wo_guidance,
                teacher_bundle,
                target_size=sr_denoised_wo_guidance.shape[-2:],
            )
            sr_pred_mask_wo = sr_pred_logits_wo.argmax(dim=1)

        save_outputs(
            sample_dir,
            base_name=base_name,
            original=image.cpu(),
            # ref_image=ref_image.cpu(),
            sr_original=sr_original.cpu(),
            mask=mask.cpu(),
            sr_pred_mask=sr_pred_mask.cpu(),
            sr_pred_mask_wo=sr_pred_mask_wo.cpu(),
            noised=noised.cpu(),
            denoised=denoised.cpu(),
            sr_denoised=sr_denoised.cpu(),
            sr_denoised_wo_guidance=sr_denoised_wo_guidance.cpu(),
            final_timestep=config.diffusion_steps - 1,
        )

        if sample_counter == 0:
            config_path = sample_dir / f"{base_name}_config.json"
            config_path.write_text(json.dumps(asdict(config), indent=2))

        sample_counter += 1
        if config.max_samples is not None and sample_counter >= config.max_samples:
            break


if __name__ == "__main__":
    main()
