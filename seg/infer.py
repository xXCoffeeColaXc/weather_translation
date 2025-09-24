from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from seg.utils.hf_utils import (
    build_joint_resize,
    get_class_names,
    mask_to_color,
    model_to_folder_name,
    resolve_model_name,
    tensor_to_pil,
)
from seg.utils.metrics import compute_per_class_iou, intersection_and_union, update_confusion_matrix

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model_name: str
    processor: AutoImageProcessor
    model: AutoModelForSemanticSegmentation
    device: torch.device


@dataclass
class EvaluationSummary:
    model_name: str
    samples: int
    per_class_iou: Dict[str, float]
    mean_iou: float
    overall_accuracy: float
    confusion_matrix: List[List[int]]


def load_hf_model(alias_or_path: str, *, device: Optional[str] = None) -> ModelBundle:
    model_name = resolve_model_name(alias_or_path)
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_name)

    LOGGER.info("Loading Hugging Face model %s on %s", model_name, torch_device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model.to(torch_device)
    model.eval()

    return ModelBundle(model_name=model_name, processor=processor, model=model, device=torch_device)


def infer_batch(
    bundle: ModelBundle,
    images: List[Image.Image],
    *,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    inputs = bundle.processor(images=images, return_tensors="pt")
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = bundle.model(**inputs).logits

    if target_size is not None:
        logits = torch.nn.functional.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    preds = logits.argmax(dim=1).cpu()
    return preds


def _prepare_images(batch: torch.Tensor) -> List[Image.Image]:
    return [tensor_to_pil(img) for img in batch]


def evaluate_dataloader(
    bundle: ModelBundle,
    dataloader: DataLoader,
    *,
    output_dir: Path,
    target_size: Tuple[int, int],
    max_samples: Optional[int] = None,
    save_visuals: bool = True,
) -> EvaluationSummary:
    class_names = get_class_names()
    num_classes = len(class_names)

    total_intersection = torch.zeros(num_classes, dtype=torch.float64)
    total_union = torch.zeros(num_classes, dtype=torch.float64)
    total_target = torch.zeros(num_classes, dtype=torch.float64)
    overall_correct = 0.0
    overall_labeled = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_folder = output_dir / model_to_folder_name(bundle.model_name)
    model_folder.mkdir(parents=True, exist_ok=True)

    samples_processed = 0

    iterator = dataloader
    progress = None
    try:
        from tqdm import tqdm

        progress = tqdm(total=len(dataloader), desc="Evaluating")
    except ImportError:  # pragma: no cover - tqdm optional
        progress = None

    with torch.inference_mode():
        for batch_index, (images, masks) in enumerate(iterator):
            if max_samples is not None and samples_processed >= max_samples:
                break

            current_batch = images.size(0)
            if max_samples is not None:
                remaining = max_samples - samples_processed
                if remaining <= 0:
                    break
                if current_batch > remaining:
                    images = images[:remaining]
                    masks = masks[:remaining]
                    current_batch = remaining

            image_list = _prepare_images(images)
            preds = infer_batch(bundle, image_list, target_size=target_size)

            masks = masks[:current_batch].cpu()

            area_intersect, area_union, _, area_target = intersection_and_union(preds, masks, num_classes)

            total_intersection += area_intersect
            total_union += area_union
            total_target += area_target

            valid_mask = masks != 255
            overall_correct += (preds[valid_mask] == masks[valid_mask]).sum().item()
            overall_labeled += valid_mask.sum().item()

            update_confusion_matrix(confusion, preds, masks, num_classes)

            if save_visuals:
                _persist_batch_visuals(
                    model_folder,
                    batch_index,
                    image_list,
                    masks,
                    preds,
                    samples_processed,
                )

            samples_processed += current_batch
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    per_class_iou = compute_per_class_iou(total_intersection, total_union)
    per_class = {name: round(value * 100.0, 2) for name, value in zip(class_names, per_class_iou.tolist())}

    mean_iou = round(float(per_class_iou.mean() * 100.0), 2)
    overall_accuracy = round((overall_correct / max(overall_labeled, 1.0)) * 100.0, 2)

    return EvaluationSummary(
        model_name=bundle.model_name,
        samples=samples_processed,
        per_class_iou=per_class,
        mean_iou=mean_iou,
        overall_accuracy=overall_accuracy,
        confusion_matrix=confusion.tolist(),
    )


def _persist_batch_visuals(
    base_dir: Path,
    batch_index: int,
    images: List[Image.Image],
    masks: torch.Tensor,
    preds: torch.Tensor,
    offset: int,
) -> None:
    for idx, (image, mask_tensor, pred_tensor) in enumerate(zip(images, masks, preds)):
        sample_idx = offset + idx
        prefix = f"sample_{sample_idx:05d}"
        image_path = base_dir / f"{prefix}_image.png"
        mask_path = base_dir / f"{prefix}_mask.png"
        pred_path = base_dir / f"{prefix}_pred.png"

        image.save(image_path)
        mask_to_color(mask_tensor).save(mask_path)
        mask_to_color(pred_tensor).save(pred_path)


def build_cityscapes_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


__all__ = [
    "MODEL_ZOO",
    "ModelBundle",
    "EvaluationSummary",
    "build_joint_resize",
    "build_cityscapes_dataloader",
    "load_hf_model",
    "evaluate_dataloader",
    "resolve_model_name",
]
