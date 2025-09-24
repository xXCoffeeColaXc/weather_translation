from __future__ import annotations

from typing import Tuple

import torch


def intersection_and_union(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if preds.shape != targets.shape:
        raise ValueError("Preds and targets must have identical shapes")

    preds = preds.view(-1)
    targets = targets.view(-1)

    valid_mask = targets != ignore_index
    preds = preds[valid_mask]
    targets = targets[valid_mask]

    intersect = preds[preds == targets]

    area_intersect = torch.histc(
        intersect.float(), bins=num_classes, min=0, max=num_classes - 1
    )
    area_pred = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_target = torch.histc(
        targets.float(), bins=num_classes, min=0, max=num_classes - 1
    )
    area_union = area_pred + area_target - area_intersect

    return (
        area_intersect.to(torch.float64),
        area_union.to(torch.float64),
        area_pred.to(torch.float64),
        area_target.to(torch.float64),
    )


def update_confusion_matrix(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: int = 255,
) -> None:
    if confusion.shape != (num_classes, num_classes):
        raise ValueError("Confusion matrix has unexpected shape")

    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)
    valid_mask = targets != ignore_index
    preds = preds[valid_mask]
    targets = targets[valid_mask]

    indices = targets * num_classes + preds
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion += counts.view(num_classes, num_classes).to(confusion.device)


def compute_per_class_iou(
    intersections: torch.Tensor, unions: torch.Tensor
) -> torch.Tensor:
    epsilon = torch.finfo(torch.float64).eps
    return intersections / torch.clamp(unions, min=epsilon)

