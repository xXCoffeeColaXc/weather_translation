from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from gen_ddpm.linear_noise_scheduler import LinearNoiseScheduler
from gen_ddpm.model import UNet
from seg.dataloaders.cityscapes import CityscapesSegmentation, IGNORE_LABEL
from seg.utils.hf_utils import mask_to_color
from seg.infer import load_hf_model, ModelBundle
from srgan_model.models import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.4865, 0.4998, 0.4323], [0.2326, 0.2276, 0.2659]


@dataclass
class SamplerConfig:
    # data
    dataset_root: str = "data/cityscapes"
    split: str = "val"
    image_size: int = 128
    batch_size: int = 1
    num_workers: int = 4
    max_samples: Optional[int] = 2

    # diffusion
    diffusion_steps: int = 50  # how many forward steps to apply before reversing
    num_timesteps: int = 1000  # training schedule length
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # model/checkpoints
    checkpoint: str = "gen_ddpm/checkpoints/1000-checkpoint.ckpt"
    srgan_checkpoint: str = "srgan_model/weights/swift_srgan_4x.pth.tar"
    teacher_model: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    guidance_lr: float = 0.1

    # logging
    output_dir: str = "gen_ddpm/ddpm_sgg_samples"
    log_every: int = 10  # save intermediate steps every N iterations


def _build_cityscapes_transforms(image_size: int,) -> Tuple[Callable, Callable]:
    """
    Resize → center-crop to a square and normalise with dataset mean/std for the DDPM.
    """

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
    mean = torch.tensor(MEAN, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    std = torch.tensor(STD, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0.0, 1.0)


def _save_batch_image(batch: torch.Tensor, path: Path, *, nrow: Optional[int] = None) -> None:
    batch_01 = _denormalize_to_01(batch)
    grid = make_grid(batch_01, nrow=nrow or batch_01.shape[0])
    pil_image = transforms.ToPILImage()(grid.cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)


def _save_history(history: List[Tuple[int, torch.Tensor]], path: Path, prefix: str) -> None:
    if not history:
        return
    path.mkdir(parents=True, exist_ok=True)
    for step, tensor in sorted(history, key=lambda pair: pair[0]):
        _save_batch_image(tensor, path / f"{prefix}_t{step:04d}.png")


def _move_scheduler_to_device(scheduler: LinearNoiseScheduler, device: torch.device) -> None:
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_cum_prod = scheduler.alpha_cum_prod.to(device)
    scheduler.sqrt_alpha_cum_prod = scheduler.sqrt_alpha_cum_prod.to(device)
    scheduler.one_minus_cum_prod = scheduler.one_minus_cum_prod.to(device)
    scheduler.sqrt_one_minus_alpha_cum_prod = scheduler.sqrt_one_minus_alpha_cum_prod.to(device)


def _prepare_teacher_inputs(
    image_batch: torch.Tensor,
    processor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    pixel_values = image_batch

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


def compute_teacher_logits(
    image_batch: torch.Tensor,
    bundle: ModelBundle,
    *,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    pixel_values = _prepare_teacher_inputs(image_batch.to(bundle.device), bundle.processor, target_size)
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
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    target = mask.to(device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, target, ignore_index=IGNORE_LABEL)


def _lcg_loss(
    mask_batch: torch.Tensor,
    logits: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Label-conditional guidance: compute per-class NLL over each present class region, average across classes.
    Returns None if no valid class pixels.
    """
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
        # Only penalise pixels belonging to this class
        nll = -log_probs[:, cls][region_mask]
        per_class_losses.append(nll.mean())

    if not per_class_losses:
        return None

    return torch.stack(per_class_losses).mean()


def apply_segmentation_guidance(
    sample: torch.Tensor,
    mask: torch.Tensor,
    srgan: Generator,
    teacher_bundle: ModelBundle,
    *,
    guidance_lr: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Upscale sample → run segmentation → guidance loss → downsample gradient to update sample.
    Alternates between:
    - GSG: CE on full mask.
    - LCG: CE averaged per present class regions.
    """
    sr_input = _denormalize_to_01(sample).detach().requires_grad_(True)
    sr_image = srgan(sr_input)  # [B,3,512,512]
    target_size = sr_image.shape[-2:]

    logits = compute_teacher_logits(sr_image, teacher_bundle, target_size=target_size)

    mask_up = F.interpolate(mask.float().unsqueeze(1), size=target_size, mode="nearest").squeeze(1).long()

    if mode == "lcg":
        loss = _lcg_loss(mask_up, logits)
        if loss is None:
            loss = ce_label(logits, mask_up)
    else:  # gsg
        loss = ce_label(logits, mask_up)

    grad_sr = torch.autograd.grad(loss, sr_image, retain_graph=False, create_graph=False)[0]
    grad_base = F.interpolate(grad_sr, size=sample.shape[-2:], mode="bilinear", align_corners=False)

    guided = sample + guidance_lr * grad_base.to(sample.device)
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
    return dataset, loader


@torch.no_grad()
def forward_diffuse(
    x0: torch.Tensor,
    scheduler: LinearNoiseScheduler,
    steps: int,
    *,
    log_every: int = 0,
) -> tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
    """
    Jump directly to timestep `steps - 1` in the forward process (no incremental noise).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1 for forward diffusion.")

    t_value = steps - 1
    t = torch.full((x0.size(0),), t_value, device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = scheduler.add_noise2(x0, noise, t)

    history: List[Tuple[int, torch.Tensor]] = []
    if log_every:
        history.append((t_value, xt.detach().cpu()))

    return xt, history


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
    log_every: int = 0,
) -> tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
    """
    Reverse the diffusion chain x_t → x_0 using the trained DDPM model with SRGAN + segmentation guidance.
    """
    x = xt
    history: List[Tuple[int, torch.Tensor]] = []

    for idx, step in enumerate(tqdm(reversed(range(steps)), desc="DDPM Denoising")):
        t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
        time_embed = scheduler.one_minus_cum_prod.to(x.device)[t].view(-1, 1, 1, 1)
        noise_pred = model(x, time_embed)

        mean, sigma, _ = scheduler.sample_prev_timestep2(x, noise_pred, t)
        sample = mean if step == 0 else mean + sigma

        # --- SRGAN upscale → segmentation CE guidance ---
        sample = sample.detach()
        sample.requires_grad_(True)

        mode = "gsg"  # "gsg" if idx % 2 == 0 else "lcg"

        guided_sample, _ = apply_segmentation_guidance(
            sample,
            mask,
            srgan,
            teacher_bundle,
            mode=mode,
            guidance_lr=guidance_lr,
        )

        x = guided_sample.detach()

        if log_every and (step % log_every == 0 or step == 0):
            history.append((step, x.detach().cpu()))

    return x, history


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
    log_every: int,
) -> tuple[torch.Tensor, torch.Tensor, List[Tuple[int, torch.Tensor]], List[Tuple[int, torch.Tensor]]]:
    noised, forward_history = forward_diffuse(image_batch, scheduler, steps, log_every=log_every)
    denoised, reverse_history = reverse_denoise(
        noised,
        model,
        scheduler,
        steps,
        mask=mask_batch,
        srgan=srgan,
        teacher_bundle=teacher_bundle,
        guidance_lr=guidance_lr,
        log_every=log_every,
    )
    return noised, denoised, forward_history, reverse_history


def save_outputs(
    out_dir: Path,
    *,
    base_name: str,
    original: torch.Tensor,
    mask: torch.Tensor,
    noised: torch.Tensor,
    denoised: torch.Tensor,
    forward_history: List[Tuple[int, torch.Tensor]],
    reverse_history: List[Tuple[int, torch.Tensor]],
    final_timestep: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_batch_image(original, out_dir / f"{base_name}_orig.png")
    mask_to_color(mask[0]).save(out_dir / f"{base_name}_mask.png")
    _save_batch_image(noised, out_dir / f"{base_name}_noised_t{final_timestep:04d}.png")
    _save_batch_image(denoised, out_dir / f"{base_name}_denoised.png")

    _save_history(forward_history, out_dir / "steps" / "forward", prefix="forward")
    _save_history(reverse_history, out_dir / "steps" / "reverse", prefix="reverse")


def parse_args() -> SamplerConfig:
    parser = argparse.ArgumentParser(description="Run DDPM noising/denoising on Cityscapes samples.")
    parser.add_argument("--dataset-root", type=str, default="data/cityscapes")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=0.0001)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--checkpoint", type=str, default="gen_ddpm/checkpoints/1000-checkpoint.ckpt")
    parser.add_argument("--srgan-checkpoint", type=str, default="srgan_model/weights/swift_srgan_4x.pth.tar")
    parser.add_argument("--teacher-model", type=str, default="nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
    parser.add_argument("--guidance-lr", type=float, default=1.2)  # 0.8 was good
    parser.add_argument("--output-dir", type=str, default="gen_ddpm/ddpm_sgg_samples/run7_gsg_lcg")
    parser.add_argument("--log-every", type=int, default=100)

    args = parser.parse_args()
    max_samples = args.max_samples if args.max_samples >= 0 else None

    return SamplerConfig(
        dataset_root=args.dataset_root,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=max_samples,
        diffusion_steps=args.steps,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        checkpoint=args.checkpoint,
        srgan_checkpoint=args.srgan_checkpoint,
        teacher_model=args.teacher_model,
        guidance_lr=args.guidance_lr,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )


def main() -> None:
    config = parse_args()

    if config.diffusion_steps < 1:
        raise ValueError("diffusion_steps must be >= 1.")
    if config.diffusion_steps > config.num_timesteps:
        raise ValueError("diffusion_steps cannot exceed num_timesteps.")

    model = load_model(config.checkpoint, DEVICE)
    srgan = load_srgan(config.srgan_checkpoint, DEVICE)
    teacher_bundle = load_hf_model(config.teacher_model, device=str(DEVICE))
    for p in teacher_bundle.model.parameters():
        p.requires_grad_(False)
    scheduler = LinearNoiseScheduler(config.num_timesteps, config.beta_start, config.beta_end)
    _move_scheduler_to_device(scheduler, DEVICE)

    dataset, loader = build_loader(config)
    output_root = Path(config.output_dir)
    sample_counter = 0

    for batch_idx, (batch_images, batch_masks) in enumerate(loader):
        batch_images = batch_images.to(device=DEVICE, dtype=torch.float32)
        batch_masks = batch_masks.to(device=DEVICE)

        batch_size = batch_images.size(0)
        if config.max_samples is not None and sample_counter >= config.max_samples:
            break

        rand_idx = torch.randint(0, batch_size, (1,)).item()
        image = batch_images[rand_idx:rand_idx + 1]
        mask = batch_masks[rand_idx:rand_idx + 1]

        noised, denoised, forward_history, reverse_history = run_pipeline(
            model,
            scheduler,
            image,
            mask,
            srgan=srgan,
            teacher_bundle=teacher_bundle,
            steps=config.diffusion_steps,
            guidance_lr=config.guidance_lr,
            log_every=config.log_every,
        )

        dataset_idx = min(len(dataset.samples) - 1, batch_idx * config.batch_size + rand_idx)
        sample_info = dataset.samples[dataset_idx]
        relative = sample_info.image_path.relative_to(dataset.left_dir)
        sample_dir = output_root / relative.parent
        base_name = relative.stem.replace("_leftImg8bit", "")

        save_outputs(
            sample_dir,
            base_name=base_name,
            original=image.cpu(),
            mask=mask.cpu(),
            noised=noised.cpu(),
            denoised=denoised.cpu(),
            forward_history=forward_history,
            reverse_history=reverse_history,
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
