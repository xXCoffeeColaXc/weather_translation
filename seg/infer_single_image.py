"""
Run semantic segmentation on a single image and visualize the prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from seg.infer import load_hf_model
from seg.utils.hf_utils import mask_to_color


def predict_single(bundle, image: Image.Image) -> torch.Tensor:
    inputs = bundle.processor(images=image, return_tensors="pt")
    inputs = {k: v.to(bundle.device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = bundle.model(**inputs).logits

    target_size = image.size[::-1]  # PIL gives (W, H); interpolate expects (H, W)
    logits = torch.nn.functional.interpolate(
        logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return logits.argmax(dim=1)[0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer a single image with a HF segmentation model.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
        help="Model alias or HF repo id (defaults to SegFormer-B3 Cityscapes).",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu).")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the predicted mask.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bundle = load_hf_model(args.model, device=args.device)

    image = Image.open(image_path).convert("RGB")
    pred_mask = predict_single(bundle, image)
    pred_color = mask_to_color(pred_mask)

    if args.save is not None:
        save_path = args.save.expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pred_color.save(save_path)
        print(f"Saved prediction to {save_path}")

    if not args.no_show:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        axes[0].imshow(image)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(pred_color)
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        plt.show()


if __name__ == "__main__":
    main()
