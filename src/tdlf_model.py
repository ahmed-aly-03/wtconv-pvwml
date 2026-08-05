"""Model wrappers for TDLF finetuning (script 2).

The backbone normalizes internally so PGD/AWP can perturb raw [0,1] pixel
tensors directly and epsilon stays meaningful in the same units the paper
uses (epsilon = 8/255).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import IMAGENET_MEAN, IMAGENET_STD
from wtconvnext.wtconvnext import wtconvnext_base, wtconvnext_small, wtconvnext_tiny

_MODEL_FNS = {
    "wtconvnext_tiny": wtconvnext_tiny,
    "wtconvnext_small": wtconvnext_small,
    "wtconvnext_base": wtconvnext_base,
}


class NormalizeWrapper(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class TDLFBackbone(nn.Module):
    """WTConvNeXt feature extractor with input normalization built in."""

    def __init__(self, variant: str = "wtconvnext_tiny", in_chans: int = 3, drop_path_rate: float = 0.0):
        super().__init__()
        if variant not in _MODEL_FNS:
            raise ValueError(f"Unknown variant '{variant}', expected one of {list(_MODEL_FNS)}")
        self.backbone = _MODEL_FNS[variant](
            pretrained=False, in_chans=in_chans, num_classes=0, drop_path_rate=drop_path_rate
        )
        self.normalize = NormalizeWrapper(IMAGENET_MEAN, IMAGENET_STD)
        self.num_features = self.backbone.num_features

    def load_pretrained(self, checkpoint_path: str, map_location="cpu") -> None:
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        missing = [m for m in missing if not m.startswith("head.")]
        print(f"Loaded pretrained backbone from {checkpoint_path}")
        if missing:
            print(f"  missing keys: {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x in [0, 1]. Returns pre-pool feature map (B, C, H, W)."""
        x = self.normalize(x)
        return self.backbone.forward_features(x)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward(x)
        return feat.mean(dim=(2, 3))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: int = 128):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class TDLFRepresentationModel(nn.Module):
    """Stage 1: backbone + projection head, trained with the adversarial
    supervised contrastive loss. `forward` returns L2-normalized embeddings."""

    def __init__(self, backbone: TDLFBackbone, proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.projection = ProjectionHead(backbone.num_features, out_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.pooled(x)
        return self.projection(feat)


class TDLFClassifier(nn.Module):
    """Stage 2: frozen backbone + linear classifier head (Eq. 11)."""

    def __init__(self, backbone: TDLFBackbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(not backbone_frozen):
            feat = self.backbone.pooled(x)
        return self.fc(feat)
