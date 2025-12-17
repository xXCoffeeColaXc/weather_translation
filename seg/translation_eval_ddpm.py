from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from seg.dataloaders.cityscapes import IGNORE_LABEL, encode_target
from seg.dataloaders.labels import labels as CITYSCAPES_LABELS
from seg.infer import infer_batch, load_hf_model
from seg.utils.hf_utils import get_class_names, mask_to_color
from seg.utils.metrics import compute_per_class_iou, intersection_and_union

LOGGER = logging.getLogger("translation_eval_ddpm")

# Precompute an encoded RGB -> trainId mapping for fast color mask decoding.
_COLOR_WEIGHTS = np.array([1, 256, 256 * 256], dtype=np.int64)


def _encode_color(color: Sequence[int]) -> int:
    return int(np.dot(np.array(color, dtype=np.int64), _COLOR_WEIGHTS))


_ENCODED_COLOR_TO_TRAIN_ID: Dict[int, int] = {
    _encode_color(label.color): label.trainId
    for label in CITYSCAPES_LABELS
    if label.trainId >= 0 and label.trainId != IGNORE_LABEL
}
_ENCODED_COLOR_TO_TRAIN_ID[_encode_color((0, 0, 0))] = IGNORE_LABEL


@dataclass
class DomainMetrics:
    mean_iou: float
    overall_accuracy: float
    per_class_iou: Dict[str, float]


@dataclass
class TranslationEvaluationSummary:
    model_name: str
    weather: str
    samples: int
    source: DomainMetrics
    translated: DomainMetrics


@dataclass
class DDPMTranslationSample:
    original: Path
    translated: Path
    pred_mask: Path
    mask: Path
    sample_id: str


@dataclass
class SampleMetrics:
    folder: str
    sample_id: str
    original: str
    translated: str
    mask: str
    source: DomainMetrics
    translated_metrics: DomainMetrics


class RunningMetrics:

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes, dtype=torch.float64)
        self.union = torch.zeros(num_classes, dtype=torch.float64)
        self.correct = 0.0
        self.labeled = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        area_intersect, area_union, _, _ = intersection_and_union(pred,
                                                                  target,
                                                                  self.num_classes,
                                                                  ignore_index=IGNORE_LABEL)
        self.intersection += area_intersect
        self.union += area_union

        valid_mask = target != IGNORE_LABEL
        self.correct += (pred[valid_mask] == target[valid_mask]).sum().item()
        self.labeled += valid_mask.sum().item()

    def summarise(self, class_names: List[str]) -> DomainMetrics:
        per_class = compute_per_class_iou(self.intersection, self.union)
        per_class_dict = {name: round(value * 100.0, 2) for name, value in zip(class_names, per_class.tolist())}

        mean_iou = round(float(per_class.mean() * 100.0), 2)
        overall_accuracy = round(float(self.correct / max(self.labeled, 1.0) * 100.0), 2)
        return DomainMetrics(mean_iou=mean_iou, overall_accuracy=overall_accuracy, per_class_iou=per_class_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate DDPM-translated samples using stored SR images and predicted masks. "
                     "Expects *_sr_orig.png, *_sr_denoised.png, *_sr_pred_mask.png, and *_mask.png in each folder."))
    parser.add_argument("selected_weather", type=str, help="Weather tag for the current run (e.g., fog, rain, night).")
    parser.add_argument("folder_paths", nargs="+", help="Folders containing DDPM samples and masks.")
    parser.add_argument("--model", type=str, default="segformer_b3", help="Segmentation teacher alias or HF repo id.")
    parser.add_argument("--height", type=int, default=512, help="Resize height for predictions and masks.")
    parser.add_argument("--width", type=int, default=512, help="Resize width for predictions and masks.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., cpu, cuda:0).")
    parser.add_argument("--output-json", type=str, default=None, help="Where to write metrics as JSON.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/translation_quantitative_semantic_consistency",
        help="Directory to store saved visuals (grouped by weather).",
    )
    parser.add_argument("--save-visuals", action="store_true", help="Persist visuals for each sample.")
    parser.add_argument("--no-save-visuals", dest="save_visuals", action="store_false")
    parser.set_defaults(save_visuals=True)
    parser.add_argument("--log-level",
                        type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    if (args.height is None) != (args.width is None):
        parser.error("Both --height and --width must be provided together.")
    return args


def _color_mask_to_train_ids(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError("Expected color mask with shape (H, W, 3)")

    encoded = mask.reshape(-1, 3).astype(np.int64) @ _COLOR_WEIGHTS
    train_ids = np.full(encoded.shape[0], IGNORE_LABEL, dtype=np.uint8)
    for color_code, train_id in _ENCODED_COLOR_TO_TRAIN_ID.items():
        train_ids[encoded == color_code] = train_id
    return train_ids.reshape(mask.shape[0], mask.shape[1])


def _load_mask(mask_path: Path, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    with Image.open(mask_path) as mask_img:
        width, height = target_size[1], target_size[0]
        mask_img = mask_img.resize((width, height), Image.NEAREST)
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
        raise ValueError(f"Unsupported mask shape {mask_array.shape} for {mask_path}")

    height, width = train_mask.shape[0], train_mask.shape[1]
    return torch.as_tensor(train_mask, dtype=torch.long), (height, width)


def _load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _find_sample(folder: Path) -> DDPMTranslationSample:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    orig_candidates = sorted(folder.glob("*_sr_orig.png"))
    if not orig_candidates:
        raise FileNotFoundError(f"No *_sr_orig.png found in {folder}")
    original_path = orig_candidates[0]

    translated_candidates = sorted(folder.glob("*_sr_denoised.png"))
    if not translated_candidates:
        raise FileNotFoundError(f"No *_sr_denoised.png found in {folder}")
    translated_path = translated_candidates[0]

    pred_candidates = sorted(folder.glob("*_sr_pred_mask.png")) or sorted(folder.glob("*_sr_ped_mask.png"))
    if not pred_candidates:
        raise FileNotFoundError(f"No *_sr_pred_mask.png (or *_sr_ped_mask.png) found in {folder}")
    pred_mask = pred_candidates[0]

    mask_candidates = [
        p for p in folder.glob("*_mask.png")
        if "sr_pred" not in p.name and "sr_ped" not in p.name and "sr_denoised" not in p.name
    ]
    mask_candidates = sorted(mask_candidates)
    if not mask_candidates:
        raise FileNotFoundError(f"No ground-truth *_mask.png found in {folder}")
    mask_path = mask_candidates[0]

    # Derive sample_id from the ground truth mask name (drop trailing _mask).
    sample_id = mask_path.name.replace("_mask.png", "")

    return DDPMTranslationSample(
        original=original_path,
        translated=translated_path,
        pred_mask=pred_mask,
        mask=mask_path,
        sample_id=sample_id,
    )


def _persist_visuals(
    output_dir: Path,
    sample_id: str,
    original: Image.Image,
    translated: Image.Image,
    gt_mask: torch.Tensor,
    pred_translated: torch.Tensor,
    pred_original: Optional[torch.Tensor] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    original.save(output_dir / f"{sample_id}_xs.png")
    translated.save(output_dir / f"{sample_id}_xt.png")
    mask_to_color(gt_mask).save(output_dir / f"{sample_id}_gt.png")
    mask_to_color(pred_translated).save(output_dir / f"{sample_id}_xt_pred.png")
    if pred_original is not None:
        mask_to_color(pred_original).save(output_dir / f"{sample_id}_xs_pred.png")


def _summarise(summary: TranslationEvaluationSummary) -> None:
    LOGGER.info("Model: %s | Weather: %s | Samples: %d", summary.model_name, summary.weather, summary.samples)

    def _log_block(prefix: str, metrics: DomainMetrics) -> None:
        LOGGER.info("%s Mean IoU: %.2f%% | Overall acc: %.2f%%", prefix, metrics.mean_iou, metrics.overall_accuracy)
        LOGGER.info("%s Per-class IoU:", prefix)
        for cls, val in metrics.per_class_iou.items():
            LOGGER.info("  %s: %.2f%%", cls.ljust(18), val)

    _log_block("[Xs]", summary.source)
    _log_block("[Xt]", summary.translated)


def _domain_to_dict(metrics: DomainMetrics) -> Dict[str, object]:
    return {
        "mean_iou": metrics.mean_iou,
        "overall_accuracy": metrics.overall_accuracy,
        "per_class_iou": metrics.per_class_iou,
    }


def _persist_json(summary: TranslationEvaluationSummary, json_path: Path) -> None:
    payload = {
        "model": summary.model_name,
        "weather": summary.weather,
        "samples": summary.samples,
        "source": _domain_to_dict(summary.source),
        "translated": _domain_to_dict(summary.translated),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote metrics to %s", json_path)


def _persist_per_path_json(per_path: List[SampleMetrics], output_json: Path) -> None:
    per_path_json = output_json.with_name(output_json.stem + "_per_path.json")
    payload = [{
        "folder": sample.folder,
        "sample_id": sample.sample_id,
        "original": sample.original,
        "translated": sample.translated,
        "mask": sample.mask,
        "source": _domain_to_dict(sample.source),
        "translated": _domain_to_dict(sample.translated_metrics),
    } for sample in per_path]
    per_path_json.parent.mkdir(parents=True, exist_ok=True)
    per_path_json.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote per-path metrics to %s", per_path_json)


def _compute_single_metrics(pred: torch.Tensor, target: torch.Tensor, class_names: List[str]) -> DomainMetrics:
    num_classes = len(class_names)
    inter, union, _, _ = intersection_and_union(pred, target, num_classes, ignore_index=IGNORE_LABEL)
    per_class = compute_per_class_iou(inter, union)
    per_class_dict = {name: round(value * 100.0, 2) for name, value in zip(class_names, per_class.tolist())}

    valid_mask = target != IGNORE_LABEL
    correct = (pred[valid_mask] == target[valid_mask]).sum().item()
    labeled = valid_mask.sum().item()
    overall_accuracy = round(float(correct / max(labeled, 1) * 100.0), 2)
    mean_iou = round(float(per_class.mean() * 100.0), 2)
    return DomainMetrics(mean_iou=mean_iou, overall_accuracy=overall_accuracy, per_class_iou=per_class_dict)


def evaluate(args: argparse.Namespace) -> Tuple[TranslationEvaluationSummary, List[SampleMetrics]]:
    bundle = load_hf_model(args.model, device=args.device)
    class_names = get_class_names()
    num_classes = len(class_names)

    metrics_xs = RunningMetrics(num_classes)
    metrics_xt = RunningMetrics(num_classes)
    per_path_metrics: List[SampleMetrics] = []

    output_dir = Path(args.output_dir) / args.selected_weather
    target_size = (args.height, args.width)

    samples_processed = 0
    for folder_str in args.folder_paths:
        folder = Path(folder_str)
        try:
            sample = _find_sample(folder)
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping %s: %s", folder, exc)
            continue

        mask_tensor, _ = _load_mask(sample.mask, target_size)
        pred_xt, _ = _load_mask(sample.pred_mask, target_size)

        original_image = _load_image(sample.original)
        translated_image = _load_image(sample.translated)

        # Run teacher on the stored SR original image.
        pred_xs = infer_batch(
            bundle,
            [original_image],
            target_size=target_size,
        )[0]

        metrics_xs.update(pred_xs, mask_tensor)
        metrics_xt.update(pred_xt, mask_tensor)
        samples_processed += 1

        sample_source_metrics = _compute_single_metrics(pred_xs, mask_tensor, class_names)
        sample_translated_metrics = _compute_single_metrics(pred_xt, mask_tensor, class_names)
        per_path_metrics.append(
            SampleMetrics(folder=str(folder),
                          sample_id=sample.sample_id,
                          original=str(sample.original),
                          translated=str(sample.translated),
                          mask=str(sample.mask),
                          source=sample_source_metrics,
                          translated_metrics=sample_translated_metrics))

        if args.save_visuals:
            _persist_visuals(
                output_dir,
                sample.sample_id,
                original_image,
                translated_image,
                mask_tensor,
                pred_xt,
                pred_original=pred_xs,
            )

    source_metrics = metrics_xs.summarise(class_names)
    translated_metrics = metrics_xt.summarise(class_names)

    return TranslationEvaluationSummary(
        model_name=bundle.model_name,
        weather=args.selected_weather,
        samples=samples_processed,
        source=source_metrics,
        translated=translated_metrics,
    ), per_path_metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    summary, per_path_metrics = evaluate(args)
    _summarise(summary)

    if args.output_json:
        output_json = Path(args.output_dir) / args.selected_weather / args.output_json
        _persist_json(summary, output_json)
        _persist_per_path_json(per_path_metrics, output_json)


if __name__ == "__main__":
    main()
'''
python seg/translation_eval_ddpm.py \
  all_ddpm \
  /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/frankfurt/frankfurt_000000_002963_leftImg8bit.png \
  /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/frankfurt/frankfurt_000000_009688_leftImg8bit.png \
  /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/frankfurt/frankfurt_000000_013382_leftImg8bit.png \
  /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/lindau/lindau_000004_000019_leftImg8bit.png \
  /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/lindau/lindau_000013_000019_leftImg8bit.png \
  /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/lindau/lindau_000047_000019_leftImg8bit.png \
  /home/talmacsi/BME/weather_translation_clean/gen_ddpm/ddpm_final_experiments/benchmark_004/munster/munster_000009_000019_leftImg8bit.png \
  --output-json ddpm_metrics.json \
  --output-dir eval/ddpm_translation_consistency

'''
