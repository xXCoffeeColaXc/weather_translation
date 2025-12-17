from __future__ import annotations
"""
Gradio UI for Stable Diffusion img2img with helper knobs and modern layout.

Features:
- Upload an image, enter prompt/negative prompt.
- Configure steps, strength (noise), CFG scale, scheduler, precision.
- VRAM-friendly toggles: attention slicing, VAE tiling, xFormers (if installed).
- Generates an image and shows it alongside the original.
"""

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image

import numpy as np

from seg.dataloaders.cityscapes import IGNORE_LABEL, encode_target
from seg.dataloaders.labels import labels as CITYSCAPES_LABELS
from seg.infer import load_hf_model
from gen.sgg_sampler import compute_teacher_logits

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_LORA = str(Path("gen/checkpoints/sd_lora_finetune_100ep_lora128_dyna_maskprompts/step-036000/unet_lora"))
TEACHER_MODEL = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
DEFAULT_GUIDE_LAMBDA = 5.0
DEFAULT_TEMPERATURE = 1.1
DEFAULT_MODE = "alternate"
DEFAULT_GRAD_CLIP_NORM = 1.5
DEFAULT_LOSS_TYPE = "blend"
DEFAULT_BLEND_WEIGHT = 0.1
DEFAULT_LAMBDA_TR = 0.5


@dataclass(frozen=True)
class PipeKey:
    model_id: str
    lora_path: Optional[str]
    dtype: torch.dtype
    attn_slicing: bool
    vae_tiling: bool
    xformers: bool


PIPE_CACHE: Dict[PipeKey, StableDiffusionImg2ImgPipeline] = {}
PIPE_LOCK = threading.Lock()
TEACHER_CACHE: Dict[str, object] = {}

# Precompute an encoded RGB -> trainId mapping for fast color mask decoding.
_COLOR_WEIGHTS = np.array([1, 256, 256 * 256], dtype=np.int64)
_ENCODED_COLOR_TO_TRAIN_ID: Dict[int, int] = {
    int(np.dot(np.array(label.color, dtype=np.int64), _COLOR_WEIGHTS)): label.trainId
    for label in CITYSCAPES_LABELS
    if label.trainId >= 0 and label.trainId != IGNORE_LABEL
}
_ENCODED_COLOR_TO_TRAIN_ID[int(np.dot(np.array((0, 0, 0), dtype=np.int64), _COLOR_WEIGHTS))] = IGNORE_LABEL


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "fp16" and device.type == "cuda":
        return torch.float16
    return torch.float32


def _apply_memory_toggles(pipe: StableDiffusionImg2ImgPipeline, *, attn_slicing: bool, vae_tiling: bool) -> None:
    if attn_slicing:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()
    if vae_tiling:
        pipe.enable_vae_tiling()
    else:
        pipe.disable_vae_tiling()


def _maybe_enable_xformers(pipe: StableDiffusionImg2ImgPipeline) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        # xFormers not installed or not supported; ignore quietly.
        pass


def _load_teacher(model_id: str, device: torch.device):
    if model_id in TEACHER_CACHE:
        return TEACHER_CACHE[model_id]
    bundle = load_hf_model(model_id, device=str(device))
    TEACHER_CACHE[model_id] = bundle
    return bundle


def _load_pipeline(
    model_id: str,
    lora_path: Optional[str],
    precision: str,
    attn_slicing: bool,
    vae_tiling: bool,
    use_xformers: bool,
) -> StableDiffusionImg2ImgPipeline:
    device = _pick_device()
    dtype = _get_dtype(precision, device)
    key = PipeKey(model_id, lora_path, dtype, attn_slicing, vae_tiling, use_xformers)

    with PIPE_LOCK:
        if key in PIPE_CACHE:
            return PIPE_CACHE[key]

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        _apply_memory_toggles(pipe, attn_slicing=attn_slicing, vae_tiling=vae_tiling)
        if use_xformers:
            _maybe_enable_xformers(pipe)
        if lora_path:
            lp = Path(lora_path).expanduser()
            if lp.exists():
                try:
                    pipe.load_lora_weights(lp)
                except Exception:
                    # Fallback for older diffusers versions.
                    pipe.unet.load_attn_procs(lp)
            # If the path doesn't exist, silently proceed with base weights.
        pipe.to(device)
        PIPE_CACHE[key] = pipe
        return pipe


def _set_scheduler(pipe: StableDiffusionImg2ImgPipeline, scheduler_name: str) -> None:
    if scheduler_name.lower() == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name.lower() in {"dpm++", "dpm"}:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _resize_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _color_mask_to_train_ids(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError("Expected color mask with shape (H, W, 3)")
    encoded = mask.reshape(-1, 3).astype(np.int64) @ _COLOR_WEIGHTS
    train_ids = np.full(encoded.shape[0], IGNORE_LABEL, dtype=np.int64)
    for color_code, train_id in _ENCODED_COLOR_TO_TRAIN_ID.items():
        train_ids[encoded == color_code] = train_id
    return train_ids.reshape(mask.shape[0], mask.shape[1])


def _prepare_mask(mask_img: Image.Image, target_size: Tuple[int, int]) -> torch.Tensor:
    mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)
    mask_array = np.array(mask_img)
    if mask_array.ndim == 2:
        max_value = int(mask_array.max()) if mask_array.size else 0
        if max_value <= 18 or max_value == IGNORE_LABEL:
            train_mask = mask_array.astype(np.uint8)
        else:
            train_mask = encode_target(mask_array)
    elif mask_array.ndim == 3:
        train_mask = _color_mask_to_train_ids(mask_array)
    else:
        raise ValueError(f"Unsupported mask shape {mask_array.shape} for mask.")
    return torch.as_tensor(train_mask, dtype=torch.long)


def _teacher_logits(images: list[Image.Image], bundle, target_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    inputs = bundle.processor(images=images, return_tensors="pt").to(device)
    outputs = bundle.model(**inputs)
    logits = outputs.logits
    if logits.shape[-2:] != target_size:
        logits = torch.nn.functional.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


def _apply_sgg_refinement(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    reference_pil: Image.Image,
    mode: str,
    *,
    guide_lambda: float,
    temperature: float,
    grad_clip_norm: float,
    loss_type: str,
    blend_weight: float,
    lambda_tr: float,
    teacher_bundle,
    device: torch.device,
    info: str,
) -> tuple[Image.Image, str]:
    if mode == "none":
        return image_pil, info

    target_size = (image_pil.height, image_pil.width)
    img_t = torch.from_numpy(np.array(image_pil).astype(np.float32) / 255.0).to(device)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).requires_grad_(True)
    mask_t = _prepare_mask(mask_pil, (image_pil.width, image_pil.height)).to(device)
    mask_t = mask_t.unsqueeze(0)  # BxHxW

    ref_logits = None
    if mode in ("gsg", "alternate"):
        with torch.inference_mode():
            ref_tensor = torch.from_numpy(np.array(reference_pil).astype(np.float32) / 255.0)
            ref_tensor = ref_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            ref_logits = compute_teacher_logits(ref_tensor, teacher_bundle, target_size=mask_t.shape[-2:])

    if img_t.grad is not None:
        img_t.grad.zero_()

    # Forward through teacher on current image (keeps grad).
    cur_logits = compute_teacher_logits(img_t, teacher_bundle, target_size=mask_t.shape[-2:])

    ce_loss = torch.nn.functional.cross_entropy(cur_logits, mask_t, ignore_index=IGNORE_LABEL)
    kl_loss = None
    if ref_logits is not None:
        p = torch.nn.functional.log_softmax(cur_logits / temperature, dim=1)
        q = torch.nn.functional.softmax(ref_logits / temperature, dim=1)
        kl_loss = torch.nn.functional.kl_div(p, q, reduction="batchmean") * (temperature**2)
        kl_loss = lambda_tr * kl_loss

    if loss_type == "ce":
        loss = ce_loss
    elif loss_type == "kl" and kl_loss is not None:
        loss = kl_loss
    else:
        loss = ce_loss if kl_loss is None else blend_weight * kl_loss + (1.0 - blend_weight) * ce_loss

    if mode == "lcg":
        loss = ce_loss
    elif mode == "gsg" and kl_loss is not None:
        loss = kl_loss
    elif mode == "alternate" and kl_loss is not None:
        loss = kl_loss

    loss.backward()
    grad = img_t.grad
    if grad is None:
        return image_pil, info + " | SGG: no gradient (skipped)"
    grad_norm = torch.norm(grad).clamp(min=1e-6)
    if grad_clip_norm is not None and grad_clip_norm > 0:
        grad = torch.clamp(grad, min=-grad_clip_norm, max=grad_clip_norm)
    with torch.no_grad():
        img_t -= guide_lambda * grad / grad_norm
        img_t.clamp_(0.0, 1.0)

    with torch.no_grad():
        out = img_t.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        out_img = Image.fromarray((out * 255).astype(np.uint8))
    info = (info + f" | SGG: {mode} λ={guide_lambda:.2f} "
            f"temp={temperature} blend={blend_weight} clip={grad_clip_norm} λ_tr={lambda_tr}")
    return out_img, info


def generate_image(
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        model_id: str,
        steps: int,
        strength: float,
        cfg: float,
        scheduler_name: str,
        width: int,
        height: int,
        seed: int,
        precision: str,
        attn_slicing: bool,
        vae_tiling: bool,
        use_xformers: bool,
        lora_path: str,
        sgg_mode: str,
        sgg_temperature: float,
        sgg_guide_lambda: float,
        sgg_grad_clip_norm: float,
        sgg_loss_type: str,
        sgg_blend_weight: float,
        sgg_lambda_tr: float,
        gt_mask: Optional[Image.Image],
        progress=gr.Progress(track_tqdm=True),
):
    if init_image is None:
        raise gr.Error("Please upload an input image.")

    device = _pick_device()
    pipe = _load_pipeline(model_id, lora_path or None, precision, attn_slicing, vae_tiling, use_xformers)
    _set_scheduler(pipe, scheduler_name)

    generator = None
    used_seed = seed
    if seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        used_seed = torch.seed() % (2**31)
        generator = torch.Generator(device=device).manual_seed(used_seed)

    resized = _resize_image(init_image, (width, height))

    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                dtype=pipe.unet.dtype if device.type == "cuda" else torch.float32):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resized,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )

    gen_image = output.images[0]
    info = (f"Model: {model_id} | LoRA: {lora_path or 'none'} | Scheduler: {scheduler_name.upper()} | "
            f"Steps: {steps} | Strength: {strength:.2f} | CFG: {cfg:.2f} | Seed: {used_seed} | "
            f"Device: {device.type} | Precision: {pipe.unet.dtype}")
    if sgg_mode != "none":
        if gt_mask is None:
            raise gr.Error("Segmentation guidance requires a ground-truth mask upload.")
        teacher = _load_teacher(TEACHER_MODEL, device)
        gen_image, info = _apply_sgg_refinement(
            gen_image,
            gt_mask,
            resized,
            sgg_mode,
            guide_lambda=sgg_guide_lambda,
            temperature=sgg_temperature,
            grad_clip_norm=sgg_grad_clip_norm,
            loss_type=sgg_loss_type,
            blend_weight=sgg_blend_weight,
            lambda_tr=sgg_lambda_tr,
            teacher_bundle=teacher,
            device=device,
            info=info,
        )
    return gen_image, info


def build_demo() -> gr.Blocks:
    device = _pick_device()

    with gr.Blocks(css=_CUSTOM_CSS) as demo:
        gr.Markdown("# Stable Diffusion Img2Img Weather Translation")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=520, elem_classes=["panel"]):
                gr.Markdown("### Input")
                image_input = gr.Image(type="pil", label="Upload image", height=360)
                gt_mask = gr.Image(type="pil",
                                   label="Ground-truth mask (required for SGG)",
                                   height=180,
                                   image_mode="RGB")

            with gr.Column(scale=1, min_width=520, elem_classes=["panel"]):
                gr.Markdown("### Output")
                output_image = gr.Image(type="pil", label="Generated image", height=360)
                run_info = gr.Textbox(label="Run info", interactive=False)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=520, elem_classes=["panel"]):
                gr.Markdown("### Prompt")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the target scene...",
                    value=
                    "ultra-detailed photorealistic rainy urban driving scene, heavy rain falling, wet reflective asphalt with glistening puddles, realistic raindrop streaks on the lens, diffused headlights and taillights reflecting on the road, muted colors, soft building and tree silhouettes in moody atmosphere low visibility, atmospheric fog, cinematic lighting, photorealistic",
                    lines=2,
                )
            with gr.Column(scale=1, min_width=520, elem_classes=["panel"]):
                gr.Markdown("### Negative Prompt")
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    placeholder="blurry, low quality, distorted geometry, artifacts",
                    value=
                    "clear sky, sunny or bright daylight, crisp visibility, dry road, snow, oversaturated, cartoon, low detail, distorted cars, blown highlights, strong contrast, color banding, lens flare halos, noisy textures, muddy shadows, warped edges, unnatural lighting, artifacts, traffic lights floating in the sky, streetlights floating in the sky, lights in the sky",
                    lines=2,
                )

        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=520, elem_classes=["panel"]):
                gr.Markdown("### Generation Settings")
                model_id = gr.Textbox(
                    label="Base model (HF id or path)",
                    value=DEFAULT_MODEL,
                    info="Base checkpoint loaded before applying LoRA adapters.",
                )
                lora_path = gr.Textbox(
                    label="LoRA weights (directory)",
                    value=DEFAULT_LORA,
                    info="Leave empty to disable. Defaults to local fine-tuned LoRA.",
                )
                scheduler_name = gr.Dropdown(
                    ["ddim", "dpm++"],
                    value="ddim",
                    label="Scheduler",
                    info="DDIM: stable & classic; DPM++: faster/stronger guidance.",
                )
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                with gr.Row():
                    steps = gr.Slider(10, 75, value=100, step=1, label="Steps")
                    strength = gr.Slider(0.05,
                                         0.95,
                                         value=0.50,
                                         step=0.01,
                                         label="Strength",
                                         info="Higher = more change from source")
                cfg = gr.Slider(0.5, 15.0, value=7.5, step=0.1, label="CFG scale")
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed (-1 for random)",
                )

            with gr.Column(scale=1, min_width=520, elem_classes=["panel"]):
                gr.Markdown("### Performance & Precision")
                precision = gr.Radio(
                    ["fp16", "fp32"],
                    value="fp16",
                    label="Precision",
                    info="FP16 requires CUDA; automatically falls back to FP32 on CPU.",
                )
                attn_slicing = gr.Checkbox(True, label="Enable attention slicing (lower VRAM)")
                vae_tiling = gr.Checkbox(False, label="Enable VAE tiling (large images)")
                use_xformers = gr.Checkbox(False, label="Enable xFormers attention (if installed)")
                sgg_mode = gr.Dropdown(
                    ["none", "lcg", "gsg", "alternate"],
                    value=DEFAULT_MODE,
                    label="Segmentation Guidance",
                    info="LCG: label-consistency CE; GSG: relative KL; Alternate toggles each step.",
                )
                sgg_guide_lambda = gr.Slider(0.1,
                                             10.0,
                                             value=DEFAULT_GUIDE_LAMBDA,
                                             step=0.1,
                                             label="Guide lambda (step size)")
                sgg_temperature = gr.Slider(0.5,
                                            5.0,
                                            value=DEFAULT_TEMPERATURE,
                                            step=0.1,
                                            label="SGG temperature (GSG)")
                sgg_grad_clip_norm = gr.Slider(0.1,
                                               5.0,
                                               value=DEFAULT_GRAD_CLIP_NORM,
                                               step=0.1,
                                               label="Grad clip norm")
                sgg_loss_type = gr.Dropdown(
                    ["blend", "ce", "kl"],
                    value=DEFAULT_LOSS_TYPE,
                    label="Loss type for guidance",
                )
                sgg_blend_weight = gr.Slider(0.0,
                                             1.0,
                                             value=DEFAULT_BLEND_WEIGHT,
                                             step=0.05,
                                             label="Blend weight (KL share)")
                sgg_lambda_tr = gr.Slider(0.0,
                                          2.0,
                                          value=DEFAULT_LAMBDA_TR,
                                          step=0.05,
                                          label="Lambda TR (relative weight)")

                generate_btn = gr.Button("Generate", variant="primary")
                clear_btn = gr.Button("Clear")
                gr.Markdown(
                    "Tip: First run downloads weights. If you hit CUDA OOM, lower width/height or enable tiling.")

        generate_btn.click(
            fn=generate_image,
            inputs=[
                image_input,
                prompt,
                negative_prompt,
                model_id,
                steps,
                strength,
                cfg,
                scheduler_name,
                width,
                height,
                seed,
                precision,
                attn_slicing,
                vae_tiling,
                use_xformers,
                lora_path,
                sgg_mode,
                sgg_temperature,
                sgg_guide_lambda,
                sgg_grad_clip_norm,
                sgg_loss_type,
                sgg_blend_weight,
                sgg_lambda_tr,
                gt_mask,
            ],
            outputs=[output_image, run_info],
        )

        clear_btn.click(
            lambda: (None, None, "", -1, None),
            inputs=None,
            outputs=[image_input, output_image, run_info, seed, gt_mask],
        )

    return demo


_CUSTOM_CSS = """
.gradio-container {max-width: 1300px; margin: auto;}
.gradio-container .md {font-size: 16px;}
.gr-button {height: 48px; font-weight: 600;}
.panel {background: transparent; border: 1px solid #2b2b2b; padding: 16px; border-radius: 12px;}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Gradio UI for SD img2img.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL.")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
