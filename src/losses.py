"""Loss functions for the multitask (script 1) model.

L = L_cls + lambda * L_seg

L_cls: Asymmetric Loss (ASL) by default (Ridnik et al., ICCV 2021), applied
       one-vs-rest across the classes; falls back to plain cross-entropy via
       --cls-loss ce. See AsymmetricLoss below for why.
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


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) -- Ridnik et al., "Asymmetric Loss for
    Multi-Label Classification", ICCV 2021 (Eq. 7 in the paper).

    Designed for multi-label classification, applied here one-vs-rest for
    single-label multi-class classification (the paper explicitly validates
    this use case too, Sec. 4.2). Each class is treated as an independent
    binary classification (via sigmoid, not softmax), with two asymmetric
    mechanisms applied only to the *negative* side of each binary decision:

      1. Asymmetric focusing (gamma_neg > gamma_pos): down-weights easy
         negatives (e.g. an obviously-Non-vascular scan being confidently
         "not vMCIAD") far more aggressively than positives, instead of the
         focal loss's single symmetric gamma.
      2. Probability margin (clip): hard-discards *very* easy negatives
         below a shifted probability threshold entirely (zero gradient).

    This targets a specific, common failure mode with plain (weighted)
    cross-entropy on severely imbalanced classes: easy-negative gradients
    from the majority classes dominate training and push the decision
    boundary around, which shows up as poor precision on the rare class
    (lots of majority-class examples getting misrouted into it) even when
    its recall looks fine. Static class weights don't fix this -- they
    rescale the loss but don't change *which* examples dominate the
    gradient -- which is why ASL intentionally does not combine with class
    weighting (the paper found the two interact badly; Sec. 2.2).

    Defaults (gamma_neg=4, gamma_pos=0, clip=0.05) match the paper's best
    combined configuration (Table 2).
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=logits.shape[-1]).to(logits.dtype)

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss = loss * torch.pow(1 - pt, one_sided_gamma)

        return -loss.sum(dim=1).mean()


def build_classification_loss(
    cls_loss: str,
    class_weights: Optional[torch.Tensor] = None,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 0.0,
    asl_clip: float = 0.05,
) -> nn.Module:
    if cls_loss == "asymmetric":
        return AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)
    if cls_loss == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    raise ValueError(f"Unknown cls_loss '{cls_loss}', expected 'asymmetric' or 'ce'")


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

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        seg_loss_weight: float = 0.3,
        cls_loss: str = "asymmetric",
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
    ):
        super().__init__()
        self.cls_loss = build_classification_loss(cls_loss, class_weights, asl_gamma_neg, asl_gamma_pos, asl_clip)
        self.seg_loss = SegmentationLoss()
        self.seg_loss_weight = seg_loss_weight

    def forward(self, outputs, labels, seg_targets, has_mask):
        cls = self.cls_loss(outputs["cls_logits"], labels)
        seg = self.seg_loss(outputs["seg_logits"], seg_targets, has_mask)
        total = cls + self.seg_loss_weight * seg
        return total, {"cls_loss": cls.item(), "seg_loss": seg.item(), "total_loss": total.item()}
