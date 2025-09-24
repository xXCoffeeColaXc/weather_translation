from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from seg.dataloaders.cityscapes import CityscapesSegmentation
from seg.infer import (
    EvaluationSummary,
    build_cityscapes_dataloader,
    build_joint_resize,
    evaluate_dataloader,
    load_hf_model,
)
from seg.utils.hf_utils import resolve_model_name

LOGGER = logging.getLogger("seg_eval_hf")
'''
python seg/seg_eval_hf.py segformer_b5 --root-dir data/cityscapes --split val --output-json eval/logs/segformer_b5_val.json
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace SegFormer on Cityscapes")
    parser.add_argument(
        "model",
        type=str,
        help="Model alias or Hugging Face repo id (segformer_b0, segformer_b2, segformer_b5)",
    )
    parser.add_argument("--root-dir", type=str, default="data/cityscapes", help="Cityscapes root directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--height", type=int, default=1024, help="Resize height before inference")
    parser.add_argument("--width", type=int, default=1024, help="Resize width before inference")
    parser.add_argument("--device", type=str, default=None, help="Computation device override (e.g. cpu, cuda:0)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for smoke tests")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--save-visuals", action="store_true", help="Persist predictions (default: true)")
    parser.add_argument("--no-save-visuals", dest="save_visuals", action="store_false")
    parser.set_defaults(save_visuals=True)
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON path for metrics")
    parser.add_argument("--log-level",
                        type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--pred-dir",
                        type=str,
                        default="eval/predictions",
                        help="Directory to store prediction visualisations")
    return parser.parse_args()


def load_dataset(
    root_dir: Path,
    split: str,
    resize: Tuple[int, int],
) -> CityscapesSegmentation:
    joint_transform = build_joint_resize(resize)
    return CityscapesSegmentation(
        root_dir=root_dir,
        split=split,
        joint_transform=joint_transform,
    )


def summarise(result: EvaluationSummary) -> None:
    LOGGER.info("Model: %s", result.model_name)
    LOGGER.info("Samples processed: %d", result.samples)
    LOGGER.info("Overall accuracy: %.2f%%", result.overall_accuracy)
    LOGGER.info("Mean IoU: %.2f%%", result.mean_iou)
    LOGGER.info("Per-class IoU:")
    for cls, val in result.per_class_iou.items():
        LOGGER.info("  %s: %.2f%%", cls.ljust(18), val)


def persist_json(result: EvaluationSummary, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": result.model_name,
        "samples": result.samples,
        "overall_accuracy": result.overall_accuracy,
        "mean_iou": result.mean_iou,
        "per_class_iou": result.per_class_iou,
        "confusion_matrix": result.confusion_matrix,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote metrics to %s", json_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Cityscapes root not found: {root}")

    resize = (args.height, args.width)
    dataset = load_dataset(root, args.split, resize)

    dataloader = build_cityscapes_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    bundle = load_hf_model(args.model, device=args.device)

    output_dir = Path(args.pred_dir)
    summary = evaluate_dataloader(
        bundle,
        dataloader,
        output_dir=output_dir,
        target_size=resize,
        max_samples=args.max_samples,
        save_visuals=args.save_visuals,
    )

    summarise(summary)

    if args.output_json:
        persist_json(summary, Path(args.output_json))


if __name__ == "__main__":
    main()
