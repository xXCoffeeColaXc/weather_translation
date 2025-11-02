from __future__ import annotations

import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from torch.cuda.amp import autocast as cuda_autocast

from seg.dataloaders.cityscapes import CityscapesSegmentation, IGNORE_LABEL
from seg.infer import ModelBundle, build_cityscapes_dataloader, load_hf_model
from seg.utils.hf_utils import build_joint_resize, tensor_to_pil, mask_to_color

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MODEL = "runwayml/stable-diffusion-v1-5"
TEACHER_MODEL = "segformer_b0"  # "segformer_b5"

PROMPT = "nighttime urban street, realistic car headlights and taillights, wet asphalt reflections, street lamps glowing warm amber, detailed building facades, cinematic contrast, foggy atmosphere, volumetric light beams"
NEG = "overexposed, underexposed, blurry, cartoon, oversaturated, low detail, distorted cars, night vision, grainy, watermark, text, lens flare halos, blown highlights"
SIZE = (384, 768)
STEPS = 26
STRENGTH = 0.55
CFG = 3

GUIDE_START, GUIDE_END = 0.2, 1.0
ETA = 0.2
GRAD_EPS = 1e-8
USE_KL = False


@dataclass
class SamplerConfig:
    prompt: str = PROMPT
    negative_prompt: str = NEG
    size: Tuple[int, int] = SIZE
    steps: int = STEPS
    strength: float = STRENGTH
    cfg: float = CFG
    guide_start: float = GUIDE_START
    guide_end: float = GUIDE_END
    eta: float = ETA
    use_kl: bool = USE_KL
    grad_eps: float = GRAD_EPS


@dataclass
class StepOutput:
    step_index: int
    timestep: int
    progress: float
    ce_loss: Optional[float]
    decoded_image: torch.Tensor
    predicted_mask: torch.Tensor


def save_01(x01: torch.Tensor, path: Path) -> None:
    array = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


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
    callback_interval: Optional[int] = None,
    step_callback: Optional[Callable[[StepOutput], None]] = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, list[float]]:
    # --- Encode the input image to latents (LD(M) step) -----------------------
    # Use VAE encoder to map x0 ∈ [0,1] → z0 (latents), scaling by VAE factor.
    with torch.no_grad():
        with _maybe_autocast(device.type, torch_dtype):
            init = pipe.image_processor.preprocess(image_pil).to(device, dtype=torch_dtype)
            init_latents = pipe.vae.encode(init * 2 - 1).latent_dist.sample() * pipe.vae.config.scaling_factor

    # --- Build the reverse-time schedule and inject SDEdit noise --------------
    # strength picks the starting timestep; larger = more noise = stronger edit.
    pipe.scheduler.set_timesteps(config.steps, device=device)
    timesteps = pipe.scheduler.timesteps
    total_timesteps = len(timesteps)
    t_start_index = int((1.0 - config.strength) * total_timesteps)
    t_start_index = max(0, min(total_timesteps - 1, t_start_index))
    noise = torch.randn_like(init_latents)
    timestep = timesteps[t_start_index]
    if isinstance(timestep, torch.Tensor) and timestep.ndim == 0:
        timestep = timestep.repeat(init_latents.shape[0])
    # z_t = add_noise(z0, t) — SDEdit initialization
    latents = pipe.scheduler.add_noise(init_latents, noise, timestep).detach().requires_grad_(True)

    # --- (Optional) reference teacher logits for extra KL stabilization --------
    reference_logits = None
    if config.use_kl:
        with torch.no_grad():
            reference_logits = compute_teacher_logits(reference_image,
                                                      teacher_bundle,
                                                      target_size=mask_batch.shape[-2:])

    ce_log: list[float] = []
    steps_to_run = total_timesteps - t_start_index
    denom = max(1, steps_to_run - 1)

    # === Main reverse loop: predict noise → scheduler step → (optional) SGG ===
    for step_index, timestep in enumerate(timesteps[t_start_index:]):
        # 1) UNet forward with CFG:
        #    duplicate latents for [uncond, cond]; predict eps_uncond, eps_text; mix with cfg.
        latent_input = torch.cat([latents, latents], dim=0)
        with _maybe_autocast(device.type, torch_dtype):
            noise_pred = pipe.unet(latent_input, timestep, encoder_hidden_states=text_context).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + config.cfg * (noise_text - noise_uncond)

        # 2) One scheduler update: z_{t-1} from z_t and predicted noise.
        with torch.no_grad():
            latents_next = pipe.scheduler.step(noise_pred, timestep, latents.detach()).prev_sample

        # Progress (0..1) inside the guided window
        progress = step_index / denom
        if verbose:
            print(f"Step {step_index}/{denom} progress {progress:.3f}")

        ce_loss_value: Optional[float] = None

        # 3) SGG (GSG variant here): decode → teacher CE → backprop to latents
        #    Paper’s Eq.(10): L_global(x_k^φ, y) = CE(g(x_k^φ), y)
        if config.guide_start <= progress <= config.guide_end:
            # --- Precompute constants for SGG guidance -------------------------------
            t_index = step_index + t_start_index
            alpha_bar_t = pipe.scheduler.alphas_cumprod[t_index]  # scalar tensor
            sigma_t = torch.sqrt(1 - alpha_bar_t)  # ∝ noise level
            eta_t = config.eta * (sigma_t / pipe.scheduler.alphas_cumprod.sqrt().max())  # normalize

            # Detach the scheduler’s z_{t-1}, then re-enable grads only on this node
            latents_guided = latents_next.clone().detach().requires_grad_(True)

            # Decode latents to image space x̂_0 ∈ [0,1] for the teacher
            with _maybe_autocast(device.type, torch_dtype):
                decoded = pipe.vae.decode(latents_guided / pipe.vae.config.scaling_factor).sample  # [-1,1]
                decoded_01 = (decoded + 1) / 2  # [0,1]

            # Global CE to source labels (GSG). For LCG you’d CE per-class on masked regions.
            loss = ce_label(decoded_01, mask_batch, teacher_bundle)

            # Optional: add KL against teacher logits on reference image (stabilizer; not in paper)
            if config.use_kl and reference_logits is not None:
                loss = loss + 0.1 * kl_logits(reference_logits, decoded_01, teacher_bundle)

            # ∂loss/∂z_{t-1}: gradient in latent space (proxy for Eq.(7) mean adjustment)
            grad = torch.autograd.grad(loss, latents_guided, retain_graph=False)[0]
            grad_norm = torch.linalg.norm(grad)
            normalised_grad = grad / (grad_norm + config.grad_eps)

            # 4) Apply the guidance step.
            #    Paper: adjust μ_θ by λ Σ_θ ∇_x L and then sample. We approximate with step size η.
            latents = (latents_next - eta_t * normalised_grad).detach().requires_grad_(True)

            ce_loss_value = float(loss.detach().cpu())
            ce_log.append(ce_loss_value)
        else:
            # No guidance at this step: just continue the reverse trajectory.
            latents = latents_next.detach().requires_grad_(True)

        # (Optional) logging every few steps: decode preview + teacher prediction
        should_log_step = (step_callback is not None and
                           (step_index % callback_interval == 0 or step_index == steps_to_run - 1))
        if should_log_step:
            decoded_for_log = decode_latents(pipe, latents, device_type=device.type, torch_dtype=torch_dtype)
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
                    predicted_mask=predicted_mask.detach().cpu(),
                ))

        if verbose and (step_index % max(1, config.steps // 10) == 0):
            timestep_display = int(timestep.detach().cpu().item()) if isinstance(timestep,
                                                                                 torch.Tensor) else int(timestep)
            print(f"[{step_index:02d}/{steps_to_run}] t={timestep_display}")

    # Final decode to image space
    decoded_final = decode_latents(pipe, latents, device_type=device.type, torch_dtype=torch_dtype)
    return decoded_final, ce_log


def _maybe_autocast(device_type: str, dtype: torch.dtype):
    return torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else contextlib.nullcontext()


def main(
    dataset_root: str | Path = "data/cityscapes",
    split: str = "val",
    out_dir: str | Path = "eval/sgg",
    *,
    teacher_model: str = TEACHER_MODEL,
    max_samples: Optional[int] = 2,
    num_workers: int = 0,
    seed: Optional[int] = None,
    config: Optional[SamplerConfig] = None,
    callback_interval: Optional[int] = None,
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

    teacher_bundle = load_hf_model(teacher_model, device=str(device), torch_dtype=torch_dtype)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL, torch_dtype=torch_dtype, safety_checker=None)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        print(f"Could not enable xformers attention: {exc}")

    #pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
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

        sample_info = dataset.samples[idx]
        relative_path = sample_info.image_path.relative_to(dataset.left_dir)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image_batch = images.to(device=device, dtype=torch.float32)
        mask_batch = masks.to(device=device, dtype=torch.long)
        image_pil = tensor_to_pil(images[0])

        original_path = output_path.with_name(output_path.stem + "_original.png")
        mask_path = output_path.with_name(output_path.stem + "_mask.png")
        image_pil.save(original_path)
        # save_mask(mask_batch[0], mask_path)
        mask_to_color(mask_batch[0]).save(mask_path)

        decoded_01, ce_log = run_single_sample(
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
            step_callback=step_callback,
            verbose=verbose,
        )

        save_01(decoded_01, output_path)
        ce_path = output_path.with_name(output_path.stem + "_ce.npy")
        np.save(ce_path, np.array(ce_log, dtype=np.float32))
        if verbose:
            print(f"saved {relative_path} steps_with_CE: {len(ce_log)}")


if __name__ == "__main__":
    main()
