"""Joint segmentation + classification model built on the WTConvNeXt backbone
from BGU-CS-VIL/WTConv, matching the block diagram:

    image -> Encoder -> BN (bottleneck) -> Decoder -> segmentation
                          |         ^
              (skip connections from 3 encoder stages)
                          |
                          v
                         GP -> Linear -> classification

The encoder is the *unmodified* WTConvNeXt backbone (tiny/small/base) from the
vendored `wtconvnext` package, so ImageNet-pretrained weights load directly
into it. The bottleneck reuses the repo's own `WTConvNeXtBlock` so the new
code stays consistent with the base model rather than introducing a
different conv layer. The decoder is a standard U-Net-style upsampling path
with skip connections, since WTConv does not ship a decoder.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d

from wtconvnext.wtconvnext import WTConvNeXtBlock, wtconvnext_base, wtconvnext_small, wtconvnext_tiny

_MODEL_FNS = {
    "wtconvnext_tiny": wtconvnext_tiny,
    "wtconvnext_small": wtconvnext_small,
    "wtconvnext_base": wtconvnext_base,
}


class WTConvNeXtEncoder(nn.Module):
    """Wraps WTConvNeXt to expose per-stage feature maps for skip connections."""

    def __init__(self, variant: str = "wtconvnext_tiny", in_chans: int = 3, drop_path_rate: float = 0.0):
        super().__init__()
        if variant not in _MODEL_FNS:
            raise ValueError(f"Unknown variant '{variant}', expected one of {list(_MODEL_FNS)}")
        self.backbone = _MODEL_FNS[variant](
            pretrained=False, in_chans=in_chans, num_classes=0, drop_path_rate=drop_path_rate
        )
        self.dims: List[int] = [info["num_chs"] for info in self.backbone.feature_info]
        self.reductions: List[int] = [info["reduction"] for info in self.backbone.feature_info]

    def load_pretrained(self, checkpoint_path: str, map_location="cpu") -> None:
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Drop the ImageNet classifier head; num_classes=0 here means the
        # encoder has no head to load it into.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        missing = [m for m in missing if not m.startswith("head.")]
        print(f"Loaded pretrained encoder from {checkpoint_path}")
        if missing:
            print(f"  missing keys: {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone.stem(x)
        feats = []
        for stage in self.backbone.stages:
            x = stage(x)
            feats.append(x)
        return feats  # [stride4, stride8, stride16, stride32], shallow -> deep


class Bottleneck(nn.Module):
    """A couple of the base model's own WTConvNeXtBlocks applied to the
    deepest encoder features, shared by both the classification and
    segmentation branches (the "BN" box in the diagram)."""

    def __init__(self, channels: int, depth: int = 2, wt_levels: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(*[
            WTConvNeXtBlock(channels, channels, kernel_size=5, wt_levels=wt_levels)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DecoderUpBlock(nn.Module):
    """Upsample, project to skip's channel count, concat with the encoder
    skip connection, then fuse with a couple of conv layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_ch: int, num_seg_classes: int, out_stride: int):
        super().__init__()
        self.out_stride = out_stride
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_ch),
            nn.GELU(),
        )
        self.classify = nn.Conv2d(in_ch, num_seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        x = self.refine(x)
        return self.classify(x)


class ClassificationHead(nn.Module):
    """GP (global pool) -> Linear, branching off the bottleneck features."""

    def __init__(self, in_ch: int, num_classes: int, drop_rate: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


class MultiTaskWTConvNeXt(nn.Module):
    """Encoder -> Bottleneck -> {Decoder -> segmentation, GP+Linear -> classification}."""

    def __init__(
        self,
        variant: str = "wtconvnext_tiny",
        num_classes: int = 4,
        num_seg_classes: int = 1,
        in_chans: int = 3,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        bottleneck_depth: int = 2,
    ):
        super().__init__()
        self.encoder = WTConvNeXtEncoder(variant, in_chans=in_chans, drop_path_rate=drop_path_rate)
        dims = self.encoder.dims  # e.g. [96, 192, 384, 768]
        self.stem_stride = self.encoder.reductions[0]  # e.g. 4

        self.bottleneck = Bottleneck(dims[3], depth=bottleneck_depth)
        self.classifier = ClassificationHead(dims[3], num_classes, drop_rate=drop_rate)

        self.up1 = DecoderUpBlock(dims[3], dims[2], dims[2])
        self.up2 = DecoderUpBlock(dims[2], dims[1], dims[1])
        self.up3 = DecoderUpBlock(dims[1], dims[0], dims[0])
        self.seg_head = SegmentationHead(dims[0], num_seg_classes, out_stride=self.stem_stride)

    def load_pretrained_encoder(self, checkpoint_path: str) -> None:
        self.encoder.load_pretrained(checkpoint_path)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        in_size = x.shape[-2:]
        f0, f1, f2, f3 = self.encoder(x)

        bn = self.bottleneck(f3)
        cls_logits = self.classifier(bn)

        d = self.up1(bn, f2)
        d = self.up2(d, f1)
        d = self.up3(d, f0)
        seg_logits = self.seg_head(d, out_size=in_size)

        return {"cls_logits": cls_logits, "seg_logits": seg_logits}


def build_multitask_model(
    variant: str,
    num_classes: int,
    num_seg_classes: int = 1,
    pretrained_encoder_path: Optional[str] = None,
    **kwargs,
) -> MultiTaskWTConvNeXt:
    model = MultiTaskWTConvNeXt(
        variant=variant, num_classes=num_classes, num_seg_classes=num_seg_classes, **kwargs
    )
    if pretrained_encoder_path:
        model.load_pretrained_encoder(pretrained_encoder_path)
    return model
