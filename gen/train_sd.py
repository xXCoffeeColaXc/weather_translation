from __future__ import annotations
import contextlib
import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

from gen.dataloader import ACDCDataset, build_training_transform

logger = get_logger(__name__)

SUPPORTED_SCHEDULERS = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
)


def parse_condition_prompts(
    items: Sequence[str],
    *,
    prompt_template: str,
) -> Dict[str, str]:
    """Parse condition-specific prompts from CLI input."""

    mapping: Dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid condition prompt '{raw}'. Expected format: condition=prompt text")
        condition, prompt = raw.split("=", 1)
        condition = condition.strip()
        prompt = prompt.strip()
        if not condition:
            raise ValueError(f"Invalid condition in mapping '{raw}'")
        mapping[condition] = prompt or prompt_template.format(condition=condition)
    return mapping


class PromptedACDCDataset(Dataset[dict]):
    """Wrap `ACDCDataset` to attach textual prompts."""

    def __init__(
        self,
        dataset: ACDCDataset,
        *,
        prompt_map: Dict[str, str],
        prompt_template: str,
    ) -> None:
        self.dataset = dataset
        self.prompt_map = dict(prompt_map)
        self.prompt_template = prompt_template
        self._root = dataset.root_dir

    def __len__(self) -> int:
        return len(self.dataset)

    def _condition_from_path(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self._root.resolve())
        except ValueError:
            # fall back to parent folder name if relative path fails
            return path.parent.name
        return relative.parts[0] if relative.parts else path.parent.name

    def __getitem__(self, idx: int) -> dict:
        pixel_values = self.dataset[idx]
        image_path = Path(self.dataset.image_paths[idx])
        condition = self._condition_from_path(image_path)
        prompt = self.prompt_map.get(condition, self.prompt_template.format(condition=condition))
        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "condition": condition,
            "image_path": str(image_path),
        }


def collate_batch(examples: Iterable[dict]) -> Dict[str, object]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}


def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: Sequence[str],
    device: torch.device,
    *,
    dtype: torch.dtype,
    train_text_encoder: bool,
) -> torch.Tensor:
    text_inputs = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)

    if train_text_encoder:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]

    return encoder_hidden_states.to(dtype=dtype)


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific model revision to use.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/acdc/rgb_anon"),
        help="Root directory following condition/split/filename structure.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["rain", "fog", "night"],
        help="Subset of weather conditions to include.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to consider (e.g. train val).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gen/checkpoints/sd_finetune"),
        help="Where to write checkpoints and logs.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="autonomous driving scene at {condition}, cinematic, detailed, wet asphalt reflections",
        help="Fallback prompt template. The token '{condition}' will be replaced with the folder name.",
    )
    parser.add_argument(
        "--condition-prompt",
        nargs="*",
        default=[],
        help="Optional overrides in the form condition=prompt text.",
    )
    parser.add_argument("--train-batch-size", type=int, default=2, help="Per device batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of passes through the dataset.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Total optimisation steps. Overrides epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=SUPPORTED_SCHEDULERS)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-text-encoder", action="store_true", help="Also fine-tune the CLIP text encoder.")
    parser.add_argument("--use-lora", action="store_true", help="Train LoRA adapters instead of full UNet.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--checkpointing-steps", type=int, default=500, help="Checkpoint frequency in global steps.")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to an accelerator state directory.")
    parser.add_argument(
        "--validation-prompts",
        nargs="*",
        default=[
            "autonomous driving city scene at night with wet asphalt reflections, cinematic, high detail",
            "dense urban street in heavy rain, headlights, realistic lighting, photorealistic",
            "foggy morning city traffic, diffused lights, detailed, immersive atmosphere",
        ],
    )
    parser.add_argument("--validation-steps", type=int, default=None, help="Frequency for validation image sampling.")
    parser.add_argument("--validation-num-images", type=int, default=2)
    parser.add_argument("--sample-num-inference-steps", type=int, default=30)
    parser.add_argument("--sample-guidance-scale", type=float, default=7.5)
    parser.add_argument("--report-to", nargs="*", default=[], help="Optional accelerate trackers, e.g. tensorboard.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on the ACDC weather dataset.")
    add_training_args(parser)
    args = parser.parse_args()
    return args


def prepare_models(
    args: argparse.Namespace,
) -> tuple[
        AutoencoderKL,
        CLIPTextModel,
        CLIPTokenizer,
        UNet2DConditionModel,
        DDPMScheduler,
        AdamW,
        torch.optim.lr_scheduler.LambdaLR,
]:
    logger.info("Loading Stable Diffusion weights from %s", args.pretrained_model_name_or_path)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.use_lora:
        try:
            from peft import LoraConfig
        except ImportError as exc:  # pragma: no cover - environment without LoRA support
            raise RuntimeError(
                "LoRA fine-tuning requires diffusers>=0.21.0. Upgrade diffusers or omit --use-lora.") from exc

        if not hasattr(unet, "add_adapter"):
            raise RuntimeError("The installed diffusers version does not expose `UNet2DConditionModel.add_adapter`. "
                               "Upgrade diffusers or disable --use-lora.")

        unet.requires_grad_(False)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=args.lora_dropout,
            bias="none",
            init_lora_weights="gaussian",
        )
        unet.add_adapter(lora_config)

    trainable_params = [param for param in unet.parameters() if param.requires_grad]
    if args.train_text_encoder:
        trainable_params += [param for param in text_encoder.parameters() if param.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
    )

    warmup_steps = int(0.05 * args.max_train_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,  # or args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        print(f"Could not enable xformers attention: {exc}")

    unet.enable_gradient_checkpointing()
    unet.to(memory_format=torch.channels_last)

    return vae, text_encoder, tokenizer, unet, noise_scheduler, optimizer, lr_scheduler


def build_dataloader(args: argparse.Namespace) -> DataLoader[Dict[str, object]]:
    transform = build_training_transform(image_size=512)
    dataset = ACDCDataset(
        root_dir=args.dataset_root,
        selected_conditions=args.conditions,
        splits=args.splits,
        transform=transform,
    )

    prompt_map = parse_condition_prompts(args.condition_prompt, prompt_template=args.prompt_template)
    prompted = PromptedACDCDataset(dataset, prompt_map=prompt_map, prompt_template=args.prompt_template)

    loader = DataLoader(
        prompted,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return loader


def maybe_sample_validation(
    args: argparse.Namespace,
    accelerator: Accelerator,
    pipeline: StableDiffusionPipeline,
    step: int,
) -> None:
    if args.validation_steps is None or step % args.validation_steps != 0 or step == 0:
        return

    if not accelerator.is_main_process:
        return

    # --- Save current train/eval state (because pipeline shares modules) ---
    unet_was_training = pipeline.unet.training
    te_was_training = getattr(pipeline, "text_encoder", None) and pipeline.text_encoder.training

    # Switch to eval for deterministic inference (turns off dropout, etc.)
    pipeline.unet.eval()
    if getattr(pipeline, "text_encoder", None) is not None:
        pipeline.text_encoder.eval()
    if getattr(pipeline, "vae", None) is not None:
        pipeline.vae.eval()

    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(args.seed + step)

    sample_dir = args.output_dir / "validation" / f"step-{step:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else (
        torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32)

    with torch.inference_mode():
        autocast_ctx = torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()
        with autocast_ctx:
            for idx, prompt in enumerate(args.validation_prompts):
                for copy in range(args.validation_num_images):
                    image = pipeline(
                        prompt=prompt,
                        num_inference_steps=args.sample_num_inference_steps,
                        guidance_scale=args.sample_guidance_scale,
                        generator=generator,
                    ).images[0]
                    out_path = sample_dir / f"{idx:02d}-{copy:02d}.png"
                    image.save(out_path)

    # Restore original train/eval states
    if unet_was_training:
        pipeline.unet.train()
    if getattr(pipeline, "text_encoder", None) is not None and te_was_training:
        pipeline.text_encoder.train()
    if getattr(pipeline, "vae", None) is not None and unet_was_training:
        pipeline.vae.train()


def save_checkpoint(
    args: argparse.Namespace,
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    vae: AutoencoderKL,
    step: int,
    is_final: bool = False,
) -> None:
    if not accelerator.is_main_process:
        return

    tag = "final" if is_final else f"step-{step:06d}"
    output_dir = args.output_dir / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving checkpoint to %s", output_dir)
    unet_to_save = accelerator.unwrap_model(unet)
    if args.use_lora:
        unet_to_save.save_attn_procs(output_dir / "unet_lora")
    else:
        unet_to_save.save_pretrained(output_dir / "unet")

    if args.train_text_encoder:
        accelerator.unwrap_model(text_encoder).save_pretrained(output_dir / "text_encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    vae.save_pretrained(output_dir / "vae")

    # Persist accelerator state for resuming optimizers and schedulers.
    accelerator.save_state(output_dir / "accelerator_state")


def create_pipeline_for_validation(
    args: argparse.Namespace,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
) -> StableDiffusionPipeline:
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        revision=args.revision,
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def train(args: argparse.Namespace) -> None:
    if args.max_train_steps is None and args.num_epochs is None:
        raise ValueError("You must specify either --max-train-steps or --num-epochs.")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to else None,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir / "logs"),
    )

    logging.basicConfig(level=logging.INFO)
    if accelerator.is_main_process:
        logger.setLevel(logging.INFO)
    else:  # pragma: no cover
        logger.setLevel(logging.ERROR)

    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2, default=str))

    if args.seed is not None:
        set_seed(args.seed)

    train_dataloader = build_dataloader(args)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        if args.num_epochs is None:
            raise ValueError("Provide --num-epochs when --max-train-steps is not set.")
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    accelerator.init_trackers("sd-finetune", config=vars(args))

    (
        vae,
        text_encoder,
        tokenizer,
        unet,
        noise_scheduler,
        optimizer,
        lr_scheduler,
    ) = prepare_models(args)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=torch.float32)

    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        (
            unet,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    params_to_clip = [param for param in unet.parameters() if param.requires_grad]
    if args.train_text_encoder:
        params_to_clip += [param for param in text_encoder.parameters() if param.requires_grad]

    if args.resume_from:
        logger.info("Loading training state from %s", args.resume_from)
        accelerator.load_state(args.resume_from)

    unet.train()
    if args.train_text_encoder:
        text_encoder.train()

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("Effective batch size: %d", total_batch_size)

    # progress_bar = accelerator.progress_bar(range(args.max_train_steps), total=args.max_train_steps)
    # progress_bar.set_description("Training")
    progress_bar = tqdm(range(args.max_train_steps), total=args.max_train_steps, desc="Training")

    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, args.num_epochs or math.ceil(args.max_train_steps / num_update_steps_per_epoch)):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = encode_prompts(
                    tokenizer,
                    text_encoder,
                    batch["prompts"],
                    accelerator.device,
                    dtype=unet.dtype,
                    train_text_encoder=args.train_text_encoder,
                )

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train/loss": loss.detach().float().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        args,
                        accelerator,
                        unet,
                        text_encoder,
                        tokenizer,
                        vae,
                        global_step,
                    )

                if (args.validation_prompts and args.validation_steps and global_step % args.validation_steps == 0 and
                        global_step != 0):
                    pipeline = create_pipeline_for_validation(args, accelerator, vae, text_encoder, tokenizer, unet)
                    maybe_sample_validation(args, accelerator, pipeline, global_step)
                    del pipeline

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    save_checkpoint(
        args,
        accelerator,
        unet,
        text_encoder,
        tokenizer,
        vae,
        global_step,
        is_final=True,
    )
    accelerator.end_training()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
