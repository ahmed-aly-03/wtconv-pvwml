"""Loss functions for the multitask (script 1) model.

L = L_cls + lambda * L_seg

L_cls: standard cross-entropy (optionally class-weighted for the imbalanced
       PVWML classes).
L_seg: standard segmentation loss = BCE-with-logits + soft Dice, the default
       combination for imbalanced pixel-wise medical segmentation (Dice
       alone can be unstable early in training; BCE alone under-weights the
       typically tiny lesion class -- summing them is the standard fix).
lambda defaults to 0.3, matching the block diagram.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    """BCE-with-logits + soft Dice, masked to samples that actually have a
    segmentation label (`has_mask`). Returns 0 (no gradient contribution)
    when no sample in the batch has a mask, so classification-only batches
    are handled gracefully."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, seg_logits: torch.Tensor, seg_targets: torch.Tensor, has_mask: torch.Tensor) -> torch.Tensor:
        if has_mask.sum() == 0:
            return seg_logits.sum() * 0.0

        mask_sel = has_mask.bool()
        logits_sel = seg_logits[mask_sel]
        targets_sel = seg_targets[mask_sel]

        bce = self.bce(logits_sel, targets_sel).mean()
        dice = self.dice(logits_sel, targets_sel)
        return self.bce_weight * bce + self.dice_weight * dice


class MultiTaskLoss(nn.Module):
    """L = L_cls + lambda * L_seg"""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, seg_loss_weight: float = 0.3):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.seg_loss = SegmentationLoss()
        self.seg_loss_weight = seg_loss_weight

    def forward(self, outputs, labels, seg_targets, has_mask):
        cls = self.cls_loss(outputs["cls_logits"], labels)
        seg = self.seg_loss(outputs["seg_logits"], seg_targets, has_mask)
        total = cls + self.seg_loss_weight * seg
        return total, {"cls_loss": cls.item(), "seg_loss": seg.item(), "total_loss": total.item()}
