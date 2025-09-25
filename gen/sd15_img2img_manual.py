# sd15_img2img_manual.py
# Usage example:
#   python sd15_img2img_manual.py \
#     --image input.jpg --prompt "night rainy city street, wet asphalt, headlights" \
#     --out out.png --width 768 --height 432 --steps 30 --strength 0.55 --cfg 7.5

import argparse, os, math, numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, DPMSolverMultistepScheduler

# ---------------------------
# Helpers
# ---------------------------


def load_image(path, w, h):
    """Load RGB image and resize (bicubic) to desired size."""
    im = Image.open(path).convert("RGB")
    im = im.resize((w, h), Image.BICUBIC)
    return im


def pil_to_01(img, device, dtype):
    """PIL -> float tensor in [0,1], shape 1x3xHxW."""
    t = torch.from_numpy(np.array(img)).to(device)
    t = t.permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(dtype)


def save_01(t01, path):
    """Tensor [0,1] BCHW -> PIL and save."""
    arr = (t01[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def seed_everything(seed):
    if seed is None:
        return
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    g.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    return g


# ---------------------------
# Core img2img w/ manual loop
# ---------------------------


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    # 1) Load pipeline (we only use its submodules and tokenizer)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None,  # skip NSFW checker for research
        variant=None).to(device)

    # Swap scheduler if requested (DDIM is simple & stable; DPM-Solver is faster)
    if args.scheduler.lower() == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler.lower() == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError("scheduler must be 'ddim' or 'dpm'.")

    # 16GB VRAM tips
    if args.enable_attn_slicing:
        pipe.enable_attention_slicing()  # chunk attention to reduce peak memory
    if args.enable_vae_slicing:
        pipe.enable_vae_slicing()  # decode in slices to save VRAM
    if args.enable_vae_tiling:
        pipe.enable_vae_tiling()  # tile VAE for large images
    if args.channels_last:
        pipe.unet.to(memory_format=torch.channels_last)

    # Optional xformers (if installed)
    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("[warn] xformers not available:", e)

    # 2) Prepare inputs
    W, H = args.width, args.height
    init_pil = load_image(args.image, W, H)
    init_01 = pil_to_01(init_pil, device, dtype)

    # 3) Encode to **latents** with VAE (scale to [-1,1] expected by VAE)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype) if args.fp16 else torch.no_grad():
        init_latents = pipe.vae.encode(init_01 * 2 - 1).latent_dist.sample() * pipe.vae.config.scaling_factor
        # Shape: [1, 4, H/8, W/8]

    # 4) Text conditioning (classifier-free guidance (CFG))
    #    - We encode two prompts: negative (uncond) and positive (cond).
    tok = pipe.tokenizer(args.prompt,
                         padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True,
                         return_tensors="pt")
    ntok = pipe.tokenizer(args.negative_prompt or "",
                          padding="max_length",
                          max_length=pipe.tokenizer.model_max_length,
                          truncation=True,
                          return_tensors="pt")
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype) if args.fp16 else torch.no_grad():
        cond_emb = pipe.text_encoder(tok.input_ids.to(device))[0]
        uncond_emb = pipe.text_encoder(ntok.input_ids.to(device))[0]
        ctx = torch.cat([uncond_emb, cond_emb], dim=0)  # (2, seq_len, dim)

    # 5) Prepare scheduler timesteps and add **noise** (SDEdit strength)
    #    - strength ∈ [0,1]: how much we move from the source image.
    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps  # descending list of T indices
    # Where to start from (SDEdit): later t => more noise injected
    t_start_index = int((1.0 - args.strength) * len(timesteps))
    t_start_index = max(0, min(len(timesteps) - 1, t_start_index))

    gen = seed_everything(args.seed)
    noise = torch.randn_like(init_latents)
    latents = pipe.scheduler.add_noise(init_latents, noise, timesteps[t_start_index])

    # 6) Manual denoising loop (x_{t} -> x_{t-1})
    #    - At each step: UNet predicts noise -> scheduler does an update -> new latents
    for i, t in enumerate(timesteps[t_start_index:]):
        # Classifier-free guidance: run UNet on [uncond, cond] latents then combine
        latent_in = torch.cat([latents, latents], dim=0)
        if args.guidance_rescale:
            # Optional: rescale trick from Stable Diffusion 2.x paper (keeps saturation in check)
            # We'll apply after CFG combination below.
            pass

        with torch.autocast(device_type=device, dtype=dtype) if args.fp16 else torch.no_grad():
            noise_pred = pipe.unet(latent_in, t, encoder_hidden_states=ctx).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        # CFG: eps = eps_u + s * (eps_c - eps_u)
        eps = noise_uncond + args.cfg * (noise_text - noise_uncond)

        if args.guidance_rescale:
            # (optional) rescale based on predicted noise norms
            # See: https://arxiv.org/abs/2305.08891 (Guidance Rescaling)
            std_text = noise_text.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            std_uncond = noise_uncond.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            scale = std_uncond / std_text
            eps = noise_uncond + args.cfg * (scale * (noise_text - noise_uncond))

        # Scheduler update: produce x_{t-1}
        with torch.no_grad():
            latents = pipe.scheduler.step(eps, t, latents).prev_sample

        if args.verbose and (i % max(1, args.steps // 10) == 0):
            print(f"[{i:02d}/{len(timesteps)-t_start_index}] t={int(t)}")

    # 7) Decode **latents** back to image
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype) if args.fp16 else torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor
        img = pipe.vae.decode(latents).sample  # [-1,1]
        img01 = (img + 1) / 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_01(img01, args.out)
    print("Saved:", args.out)


# ---------------------------
# CLI
# ---------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Manual SD-1.5 img2img (encode + noise + denoise loop + decode)")
    # Model & IO
    p.add_argument("--model",
                   type=str,
                   default="runwayml/stable-diffusion-v1-5",
                   help="Hugging Face model id or local path")
    p.add_argument("--image", type=str, required=True, help="input image (PIL-readable)")
    p.add_argument("--out", type=str, default="out.png", help="output image path")
    # Prompts
    p.add_argument("--prompt", type=str, default="a realistic night city street, headlights")
    p.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted geometry")
    # Size & steps
    p.add_argument("--width", type=int, default=768 // 2)
    p.add_argument("--height", type=int, default=432 // 2)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--strength", type=float, default=0.05, help="SDEdit noise level [0..1]; higher = more change")
    p.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale")
    p.add_argument("--scheduler", type=str, default="ddim", choices=["ddim", "dpm"], help="Sampler type")
    # Performance / memory
    p.add_argument("--fp16", action="store_true", default=True, help="use float16 autocast")
    p.add_argument("--enable-attn-slicing", action="store_true", default=True)
    p.add_argument("--enable-vae-slicing", action="store_true", default=True)
    p.add_argument("--enable-vae-tiling", action="store_true", default=False)
    p.add_argument("--channels-last", action="store_true", default=True)
    p.add_argument("--enable-xformers", action="store_true", default=False)
    p.add_argument("--guidance-rescale", action="store_true", default=False, help="optional guidance rescale trick")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
