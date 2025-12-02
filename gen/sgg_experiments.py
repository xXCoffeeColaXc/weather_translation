from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import logging
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline

from gen.sgg_sampler import (
    CALLBACK_INTERVAL,
    DTYPE,
    DEVICE,
    TEACHER_MODEL,
    SamplerConfig,
    StepOutput,
    build_sampler_loader,
    load_finetuned_pipeline,
    predict_mask_from_image,
    run_single_sample,
    save_01,
)
from seg.infer import load_hf_model
from seg.utils.hf_utils import mask_to_color, tensor_to_pil


class _Tee(io.TextIOBase):

    def __init__(self, *streams):
        super().__init__()
        self.streams = streams

    def write(self, data):  # type: ignore[override]
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):  # type: ignore[override]
        for stream in self.streams:
            if hasattr(stream, "flush"):
                stream.flush()


"""
example usage:
python -m gen.sgg_experiments \
    --dataset-root data/cityscapes \
    --split my_test \ 
    --steps-list 100 \
    --strengths 0.5 0.6 0.7 0.8 \
    --cfg-scales 3.0 5.0 7.5 \
    --etas 6.0 8.0 10.0 15.0 \
    --temperatures 1.0 1.2 \
    --guide-tv-weights 0.0 0.01 0.02 \ 
    --lambda-tr-values 0.0 0.1 0.2 0.5 \
    --use-kl-flags true \
    --callback-interval 5
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGG sampler experiments across multiple configurations.")
    parser.add_argument("--dataset-root",
                        type=Path,
                        default=Path("data/cityscapes"),
                        help="Root directory of Cityscapes data.")
    parser.add_argument("--split", default="val", help="Dataset split to use.")
    parser.add_argument("--out-dir",
                        type=Path,
                        default=Path("eval/sgg_experiments"),
                        help="Directory to store experiment outputs.")
    parser.add_argument("--teacher-model", default=TEACHER_MODEL, help="Segmentation teacher model alias or path.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    parser.add_argument("--sample-index",
                        type=int,
                        default=0,
                        help="Dataset sample index to use for every configuration.")
    parser.add_argument("--config-file",
                        type=Path,
                        default=None,
                        help="JSON file with a list of configuration dictionaries.")
    parser.add_argument("--strengths", type=float, nargs="+", default=None, help="List of SDE strengths to explore.")
    parser.add_argument("--cfg-scales", type=float, nargs="+", default=None, help="List of CFG scales to explore.")
    parser.add_argument("--steps-list",
                        type=int,
                        nargs="+",
                        default=None,
                        help="List of diffusion step counts to explore.")
    parser.add_argument("--guide-starts",
                        type=float,
                        nargs="+",
                        default=None,
                        help="List of guide window start fractions.")
    parser.add_argument("--guide-ends", type=float, nargs="+", default=None, help="List of guide window end fractions.")
    parser.add_argument("--etas", type=float, nargs="+", default=None, help="List of ETA values to explore.")
    parser.add_argument("--temperatures",
                        type=float,
                        nargs="+",
                        default=None,
                        help="List of CE/KL temperatures to explore.")
    parser.add_argument("--guide-tv-weights",
                        type=float,
                        nargs="+",
                        default=None,
                        help="List of TV weights for guided images.")
    parser.add_argument("--lambda-tr-values",
                        type=float,
                        nargs="+",
                        default=None,
                        help="List of teacher-relative lambda values to explore.")
    parser.add_argument("--use-kl-flags",
                        type=str,
                        nargs="+",
                        default=None,
                        help="List of booleans for enabling KL guidance (e.g. true false).")
    parser.add_argument("--callback-interval",
                        "--mask-save-interval",
                        dest="callback_interval",
                        type=int,
                        default=CALLBACK_INTERVAL,
                        help="Step interval for saving intermediate decoded/x0/mask outputs.")
    parser.add_argument("--quiet-sampler",
                        action="store_true",
                        help="Reduce verbose per-step prints inside the sampler loop.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG).")
    return parser.parse_args()


def setup_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("sgg_experiments")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_configs_from_file(path: Path) -> List[SamplerConfig]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Configuration file must contain a list, received {type(payload)!r}")

    valid_keys = {field.name for field in fields(SamplerConfig)}
    configs: List[SamplerConfig] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Config entry at index {index} is not a dictionary: {entry!r}")
        filtered = {key: value for key, value in entry.items() if key in valid_keys}
        configs.append(SamplerConfig(**filtered))
    return configs


def _parse_bool_sequence(values: Sequence[str] | None, *, default: bool) -> List[bool]:
    if values is None:
        return [default]

    parsed: List[bool] = []
    for raw in values:
        if isinstance(raw, bool):
            parsed.append(raw)
            continue

        if isinstance(raw, (int, float)) and raw in (0, 1):
            parsed.append(bool(raw))
            continue

        lowered = str(raw).strip().lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            parsed.append(True)
        elif lowered in {"0", "false", "f", "no", "n"}:
            parsed.append(False)
        else:
            raise ValueError(f"Cannot parse boolean flag from {raw!r}. Use true/false or 1/0.")
    return parsed


def generate_grid_configs(
    *,
    strengths: Sequence[float] | None,
    cfg_scales: Sequence[float] | None,
    steps_list: Sequence[int] | None,
    guide_starts: Sequence[float] | None,
    guide_ends: Sequence[float] | None,
    etas: Sequence[float] | None,
    temperatures: Sequence[float] | None,
    guide_tv_weights: Sequence[float] | None,
    lambda_tr_values: Sequence[float] | None,
    use_kl_flags: Sequence[str] | None,
) -> List[SamplerConfig]:
    base = SamplerConfig()
    strengths = strengths or [base.strength]
    cfg_scales = cfg_scales or [base.cfg]
    steps_list = steps_list or [base.steps]
    guide_starts = guide_starts or [base.guide_start]
    guide_ends = guide_ends or [base.guide_end]
    etas = etas or [base.eta]
    temperatures = temperatures or [base.ce_temperature]
    guide_tv_weights = guide_tv_weights or [base.guide_tv_weight]
    lambda_tr_values = lambda_tr_values or [base.lambda_tr]
    use_kl_flags_bool = _parse_bool_sequence(use_kl_flags, default=base.use_kl)

    configs: List[SamplerConfig] = []
    seen: set[tuple[float, float, int, float, float, float, float, float, float, bool]] = set()
    for strength, cfg_scale, steps, guide_start, guide_end, eta, temperature, tv_weight, lambda_tr, use_kl in itertools.product(
            strengths,
            cfg_scales,
            steps_list,
            guide_starts,
            guide_ends,
            etas,
            temperatures,
            guide_tv_weights,
            lambda_tr_values,
            use_kl_flags_bool,
    ):
        if guide_end < guide_start:
            continue
        key = (float(strength), float(cfg_scale), int(steps), float(guide_start), float(guide_end), float(eta),
               float(temperature), float(tv_weight), float(lambda_tr), bool(use_kl))
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            SamplerConfig(
                strength=float(strength),
                cfg=float(cfg_scale),
                steps=int(steps),
                guide_start=float(guide_start),
                guide_end=float(guide_end),
                eta=float(eta),
                ce_temperature=float(temperature),
                guide_tv_weight=float(tv_weight),
                lambda_tr=float(lambda_tr),
                use_kl=bool(use_kl),
            ))
    return configs


def summarise_config(config: SamplerConfig) -> str:

    def fmt(value: float, digits: int = 2) -> str:
        return f"{value:.{digits}f}".replace(".", "p")

    return (f"s{fmt(config.strength)}_cfg{fmt(config.cfg, 1)}_steps{config.steps:03d}_"
            f"gs{fmt(config.guide_start)}_ge{fmt(config.guide_end)}_eta{fmt(config.eta, 2)}_"
            f"temp{fmt(config.ce_temperature, 2)}_tv{fmt(config.guide_tv_weight, 3)}_"
            f"ltr{fmt(config.lambda_tr, 2)}_kl{int(bool(config.use_kl))}")


def build_text_context(
    pipe: StableDiffusionImg2ImgPipeline,
    config: SamplerConfig,
    device: torch.device,
) -> torch.Tensor:
    prompt_tokens = pipe.tokenizer(
        config.prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative_tokens = pipe.tokenizer(
        config.negative_prompt or "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        pos_embeds = pipe.text_encoder(prompt_tokens.input_ids.to(device))[0]
        neg_embeds = pipe.text_encoder(negative_tokens.input_ids.to(device))[0]
    return torch.cat([neg_embeds, pos_embeds], dim=0)


def create_step_callback(
    *,
    config_name: str,
    steps_dir: Path,
    logger: logging.Logger,
    state: dict,
) -> Callable[[StepOutput], None]:
    steps_dir.mkdir(parents=True, exist_ok=True)

    def _callback(step: StepOutput) -> None:
        suffix = f"step_{step.step_index:03d}_t{step.timestep:04d}"
        decoded_path = steps_dir / f"{suffix}_decoded.png"
        decoded_x0_path = steps_dir / f"{suffix}_x0.png"
        mask_path = steps_dir / f"{suffix}_mask.png"
        save_01(step.decoded_image, decoded_path)
        save_01(step.decoded_x0_image, decoded_x0_path)
        mask_image = mask_to_color(step.predicted_mask[0])
        mask_image.save(mask_path)
        state["last_mask"] = step.predicted_mask
        records = state.setdefault("step_records", [])
        guidance_heat_path = steps_dir / f"{suffix}_guidance_grad.png" if step.guidance_heatmap is not None else None
        tr_heat_path = steps_dir / f"{suffix}_tr_grad.png" if step.tr_heatmap is not None else None
        guidance_overlay_path = steps_dir / f"{suffix}_guidance_overlay.png" if step.guidance_overlay is not None else None
        tr_overlay_path = steps_dir / f"{suffix}_tr_overlay.png" if step.tr_overlay is not None else None
        if step.guidance_heatmap is not None:
            save_01(step.guidance_heatmap, guidance_heat_path)
        if step.tr_heatmap is not None:
            save_01(step.tr_heatmap, tr_heat_path)
        if step.guidance_overlay is not None:
            save_01(step.guidance_overlay, guidance_overlay_path)
        if step.tr_overlay is not None:
            save_01(step.tr_overlay, tr_overlay_path)
        records.append({
            "step_index": step.step_index,
            "timestep": step.timestep,
            "progress": step.progress,
            "sgg_loss": step.sgg_loss,
            "decoded_path": str(decoded_path),
            "decoded_x0_path": str(decoded_x0_path),
            "mask_path": str(mask_path),
            "guidance_heatmap_path": str(guidance_heat_path) if guidance_heat_path else None,
            "tr_heatmap_path": str(tr_heat_path) if tr_heat_path else None,
            "guidance_overlay_path": str(guidance_overlay_path) if guidance_overlay_path else None,
            "tr_overlay_path": str(tr_overlay_path) if tr_overlay_path else None,
        })
        ce_repr = f"{step.sgg_loss:.6f}" if step.sgg_loss is not None else "NA"
        logger.info(
            "%s | step=%03d t=%d progress=%.3f sgg_loss=%s",
            config_name,
            step.step_index,
            step.timestep,
            step.progress,
            ce_repr,
        )

    return _callback


def prepare_pipeline(device: torch.device, torch_dtype: torch.dtype,
                     logger: logging.Logger) -> StableDiffusionImg2ImgPipeline:
    pipe = load_finetuned_pipeline(device, dtype=torch_dtype)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        logger.warning("Could not enable xformers attention: %s", exc)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.unet.to(memory_format=torch.channels_last)

    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe


def ensure_dataset_sample(
    *,
    dataset_root: Path,
    split: str,
    size: Sequence[int],
    sample_index: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Path, Path]:
    dataset, _ = build_sampler_loader(
        dataset_root,
        split,
        tuple(size),
        batch_size=1,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"Sample index {sample_index} out of range (dataset has {len(dataset)} samples)")

    image_tensor, mask_tensor = dataset[sample_index]
    sample_info = dataset.samples[sample_index]
    return image_tensor, mask_tensor, sample_info.image_path, sample_info.mask_path


def try_load_teacher(
    teacher_model: str,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    logger: logging.Logger,
):
    try:
        return load_hf_model(teacher_model, device=str(device), torch_dtype=torch_dtype)  # type: ignore[arg-type]
    except TypeError:
        logger.info("Falling back to load_hf_model without torch_dtype parameter.")
        return load_hf_model(teacher_model, device=str(device))  # type: ignore[arg-type]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(args.out_dir / "experiments.log", args.log_level)
    logger.info("Starting SGG experiment runner")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.config_file:
        configs = load_configs_from_file(args.config_file)
    else:
        try:
            configs = generate_grid_configs(
                strengths=args.strengths,
                cfg_scales=args.cfg_scales,
                steps_list=args.steps_list,
                guide_starts=args.guide_starts,
                guide_ends=args.guide_ends,
                etas=args.etas,
                temperatures=args.temperatures,
                guide_tv_weights=args.guide_tv_weights,
                lambda_tr_values=args.lambda_tr_values,
                use_kl_flags=args.use_kl_flags,
            )
        except ValueError as exc:
            logger.error("Failed to build configuration grid: %s", exc)
            return

    if not configs:
        logger.error("No configurations provided. Exiting.")
        return

    base_size = configs[0].size
    if any(config.size != base_size for config in configs):
        logger.warning("Configs specify multiple image sizes. Using size from each config when loading samples.")

    device = torch.device(DEVICE)
    torch_dtype = DTYPE if device.type == "cuda" else torch.float32

    teacher_bundle = try_load_teacher(
        args.teacher_model,
        device=device,
        torch_dtype=torch_dtype,
        logger=logger,
    )
    pipe = prepare_pipeline(device, torch_dtype, logger)

    logger.info("Loaded teacher %s and diffusion pipeline on %s", teacher_bundle.model_name, device)
    logger.info("Running %d configuration(s) on sample index %d", len(configs), args.sample_index)

    sample_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, Path, Path]] = {}
    callback_interval = args.callback_interval if args.callback_interval and args.callback_interval > 0 else None

    for config_index, config in enumerate(configs, start=1):
        logger.info("==== Configuration %d/%d ====", config_index, len(configs))
        logger.info(
            "Params: strength=%.3f cfg=%.2f steps=%d guide=(%.3f, %.3f) eta=%.3f temp=%.2f tv=%.3f lambda_tr=%.3f use_kl=%s",
            config.strength,
            config.cfg,
            config.steps,
            config.guide_start,
            config.guide_end,
            config.eta,
            config.ce_temperature,
            config.guide_tv_weight,
            config.lambda_tr,
            config.use_kl,
        )

        size_key = tuple(config.size)
        if size_key not in sample_cache:
            sample_cache[size_key] = ensure_dataset_sample(
                dataset_root=args.dataset_root,
                split=args.split,
                size=config.size,
                sample_index=args.sample_index,
                num_workers=args.num_workers,
                device=device,
            )
        image_tensor, mask_tensor, image_path, mask_path = sample_cache[size_key]

        image_batch = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
        mask_batch = mask_tensor.unsqueeze(0).to(device=device, dtype=torch.long)
        image_pil = tensor_to_pil(image_tensor)

        config_name = summarise_config(config)
        config_dir = args.out_dir / f"{config_index:02d}_{config_name}"
        steps_dir = config_dir / "steps"
        config_dir.mkdir(parents=True, exist_ok=True)

        input_path = config_dir / "input.png"
        gt_mask_path = config_dir / "gt_mask.png"
        tensor_to_pil(image_tensor).save(input_path)
        mask_to_color(mask_tensor).save(gt_mask_path)

        metadata = {
            "config": asdict(config),
            "config_index": config_index,
            "config_name": config_name,
            "sample_index": args.sample_index,
            "dataset_image_path": str(image_path),
            "dataset_mask_path": str(mask_path),
            "callback_interval": callback_interval,
        }
        with (config_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        text_context = build_text_context(pipe, config, device)

        callback_state: dict[str, object] = {}
        step_callback = create_step_callback(
            config_name=config_name,
            steps_dir=steps_dir,
            logger=logger,
            state=callback_state,
        )

        stdout_path = config_dir / "stdout.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle:
            tee_out = _Tee(sys.stdout, stdout_handle)
            tee_err = _Tee(sys.stderr, stdout_handle)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                decoded_01, sgg_log = run_single_sample(
                    pipe,
                    teacher_bundle,
                    image_pil=image_pil,
                    reference_image=image_batch,
                    mask_batch=mask_batch,
                    text_context=text_context,
                    config=config,
                    device=device,
                    torch_dtype=torch_dtype,
                    callback_interval=callback_interval,
                    step_callback=step_callback,
                    verbose=not args.quiet_sampler,
                )

        decoded_path = config_dir / "decoded.png"
        save_01(decoded_01, decoded_path)

        sgg_path = config_dir / "sgg_losses.npy"
        np.save(sgg_path, np.array(sgg_log, dtype=np.float32))

        sgg_json_path = config_dir / "sgg_losses.json"
        with sgg_json_path.open("w", encoding="utf-8") as handle:
            json.dump([float(x) for x in sgg_log], handle, indent=2)

        records_path: Path | None = config_dir / "step_records.jsonl" if callback_state.get("step_records") else None
        if callback_state.get("step_records") and records_path is not None:
            with records_path.open("w", encoding="utf-8") as handle:
                for record in callback_state["step_records"]:
                    json.dump(record, handle)
                    handle.write("\n")

        if "last_mask" in callback_state:
            final_mask_path = config_dir / "final_pred_mask.png"
            mask_to_color(callback_state["last_mask"][0]).save(final_mask_path)
        else:
            logger.warning("%s | No callback state recorded; skipping final mask export.", config_name)
            final_logits = predict_mask_from_image(
                decoded_01,
                teacher_bundle,
                target_size=mask_batch.shape[-2:],
            )
            mask_to_color(final_logits[0]).save(config_dir / "final_pred_mask.png")

        loss_stats = {
            "count": len(sgg_log),
            "min": float(np.min(sgg_log)) if sgg_log else None,
            "max": float(np.max(sgg_log)) if sgg_log else None,
            "mean": float(np.mean(sgg_log)) if sgg_log else None,
        }
        summary = {
            "config_name": config_name,
            "config_index": config_index,
            "decoded_path": str(decoded_path),
            "sgg_losses_npy": str(sgg_path),
            "sgg_losses_json": str(sgg_json_path),
            "step_records": str(records_path) if records_path else None,
            "stdout_log": str(stdout_path),
            "loss_stats": loss_stats,
            "callback_interval": callback_interval,
        }
        with (config_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        logger.info(
            "%s | completed with %d SGG loss entries. Outputs saved under %s",
            config_name,
            len(sgg_log),
            config_dir,
        )

    logger.info("All configurations completed.")


if __name__ == "__main__":
    main()
