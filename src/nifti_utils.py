"""NIfTI volume -> 2D PNG slice extraction for the classification pipeline.

Mirrors the naming convention of the earlier "swml_vols_middle50_4class"
dataset: extract the middle fraction (default 50%) of axial slices from
each registered FLAIR volume, since the whole-volume top/bottom slices
mostly contain skull/neck or very little brain tissue.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import nibabel as nib
import numpy as np
from PIL import Image


@dataclass
class SliceExtractionConfig:
    slice_fraction: float = 0.5  # fraction of axial slices to keep, centered
    img_size: int = 224
    min_nonzero_frac: float = 0.02  # skip slices that are almost entirely background
    intensity_low_pct: float = 1.0
    intensity_high_pct: float = 99.0


def _pad_to_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=arr.dtype)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    padded[y0:y0 + h, x0:x0 + w] = arr
    return padded


def extract_slices(
    nifti_path: str,
    cfg: SliceExtractionConfig = SliceExtractionConfig(),
) -> List[np.ndarray]:
    """Returns a list of uint8 (img_size, img_size) grayscale slice arrays."""
    img = nib.load(nifti_path)
    img = nib.as_closest_canonical(img)  # consistent RAS+ orientation across scanners/cohorts
    data = img.get_fdata()

    if data.ndim == 4:
        data = data[..., 0]  # some FLAIR exports carry a singleton time/echo dim
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D (or 4D-with-singleton) volume, got shape {data.shape} for {nifti_path}")

    total_slices = data.shape[2]
    keep = max(1, int(round(total_slices * cfg.slice_fraction)))
    start = (total_slices - keep) // 2
    end = start + keep

    # Normalize using whole-volume intensity percentiles (not per-slice) so
    # brightness stays consistent slice-to-slice within the same subject.
    nonzero = data[data > 0]
    if nonzero.size == 0:
        return []
    lo, hi = np.percentile(nonzero, [cfg.intensity_low_pct, cfg.intensity_high_pct])
    if hi <= lo:
        hi = lo + 1e-6

    slices = []
    for z in range(start, end):
        sl = data[:, :, z]
        nonzero_frac = float((sl > 0).mean())
        if nonzero_frac < cfg.min_nonzero_frac:
            continue

        sl = np.clip(sl, lo, hi)
        sl = (sl - lo) / (hi - lo)
        sl = (sl * 255).astype(np.uint8)
        sl = np.rot90(sl)  # canonical axial "head up" display orientation
        sl = _pad_to_square(sl)
        slices.append(sl)

    return slices


def save_slices_as_png(
    slices: List[np.ndarray],
    out_dir: str,
    base_name: str,
    img_size: int = 224,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i, sl in enumerate(slices):
        image = Image.fromarray(sl, mode="L").resize((img_size, img_size), Image.BICUBIC)
        out_path = os.path.join(out_dir, f"{base_name}_slice{i:03d}.png")
        image.save(out_path)
        written.append(out_path)
    return written


def extract_and_save(
    nifti_path: str,
    out_dir: str,
    base_name: str,
    cfg: SliceExtractionConfig = SliceExtractionConfig(),
) -> List[str]:
    slices = extract_slices(nifti_path, cfg)
    return save_slices_as_png(slices, out_dir, base_name, cfg.img_size)
