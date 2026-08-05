"""Heuristic periventricular white-matter-lesion (PVWML) pseudo-mask extraction
from single-slice FLAIR images.

IMPORTANT — read before trusting these masks for anything clinical:
This is an *unsupervised heuristic*, not a validated segmentation model and not
radiologist ground truth. It exists only because no real PVWML annotations are
available. It assumes:
  - The input is a FLAIR slice (lesions are hyperintense relative to normal
    white matter; CSF/ventricles are comparatively dark/suppressed).
  - The slice is roughly a mid-ventricular axial slice with the brain
    approximately centered in the frame (true for "middle N%" volume crops).
  - The image has already been skull-stripped/cropped similarly to the rest
    of the dataset (no bright non-brain structures near the image border).

The method:
  1. Segment brain tissue from background via Otsu thresholding + largest
     connected component.
  2. Find ventricle/CSF as the darkest connected regions near the image
     center within the brain mask (periventricular lesions are, by
     definition, adjacent to the ventricles).
  3. Dilate the ventricle mask outward to form a periventricular band.
  4. Flag brain pixels within that band whose intensity is in the upper
     percentile of brain-tissue intensities (hyperintense, consistent with
     WML on FLAIR).
  5. Clean up with morphological opening/closing and small-object removal.

Treat the output as *weak supervision* only. Spot-check a sample of outputs
(see `generate_pseudo_masks.py --preview-n`) before trusting them, and
prefer real annotations the moment any become available.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_dilation, binary_opening, disk, remove_small_objects


@dataclass
class PseudoMaskConfig:
    ventricle_center_frac: float = 0.55  # central box (fraction of H/W) to search for ventricles
    ventricle_percentile: float = 12.0   # brain pixels below this percentile => CSF/ventricle candidate
    periventricular_dilation_frac: float = 0.06  # dilation radius as a fraction of image size
    hyperintense_percentile: float = 88.0  # brain pixels above this percentile => candidate lesion
    min_lesion_area_frac: float = 0.0003  # drop connected components smaller than this fraction of image area


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lbl = label(mask)
    if lbl.max() == 0:
        return mask
    areas = [(r.area, r.label) for r in regionprops(lbl)]
    biggest_label = max(areas)[1]
    return lbl == biggest_label


def _brain_mask(gray: np.ndarray) -> np.ndarray:
    nonzero = gray[gray > 0]
    if nonzero.size == 0:
        return np.zeros_like(gray, dtype=bool)
    thresh = threshold_otsu(nonzero)
    mask = gray > thresh * 0.5
    mask = ndi.binary_fill_holes(mask)
    mask = _largest_component(mask)
    mask = binary_closing(mask, disk(3))
    return mask


def _ventricle_mask(gray: np.ndarray, brain: np.ndarray, cfg: PseudoMaskConfig) -> np.ndarray:
    h, w = gray.shape
    cy, cx = h / 2, w / 2
    half_h, half_w = h * cfg.ventricle_center_frac / 2, w * cfg.ventricle_center_frac / 2
    center_box = np.zeros_like(brain, dtype=bool)
    y0, y1 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    x0, x1 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    center_box[y0:y1, x0:x1] = True

    brain_vals = gray[brain]
    if brain_vals.size == 0:
        return np.zeros_like(brain, dtype=bool)
    low_thresh = np.percentile(brain_vals, cfg.ventricle_percentile)

    candidate = brain & center_box & (gray <= low_thresh)
    candidate = binary_opening(candidate, disk(1))
    candidate = remove_small_objects(candidate, min_size=max(4, int(0.00005 * h * w)))
    return candidate


def _hyperintense_mask(gray: np.ndarray, brain: np.ndarray, cfg: PseudoMaskConfig) -> np.ndarray:
    brain_vals = gray[brain]
    if brain_vals.size == 0:
        return np.zeros_like(brain, dtype=bool)
    high_thresh = np.percentile(brain_vals, cfg.hyperintense_percentile)
    return brain & (gray >= high_thresh)


def generate_pvwml_pseudo_mask(
    image: np.ndarray,
    cfg: PseudoMaskConfig = PseudoMaskConfig(),
) -> np.ndarray:
    """Returns a boolean (H, W) pseudo-mask for periventricular hyperintensities.

    `image` should be a single-channel float array (any range); it is
    min-max normalized internally.
    """
    gray = image.astype(np.float32)
    if gray.max() > gray.min():
        gray = (gray - gray.min()) / (gray.max() - gray.min())

    brain = _brain_mask(gray)
    if brain.sum() == 0:
        return np.zeros_like(gray, dtype=bool)

    ventricles = _ventricle_mask(gray, brain, cfg)
    h, w = gray.shape
    dilation_radius = max(1, int(round(cfg.periventricular_dilation_frac * min(h, w))))
    band = binary_dilation(ventricles, disk(dilation_radius)) & brain & ~ventricles

    hyperintense = _hyperintense_mask(gray, brain, cfg)

    lesion = band & hyperintense
    lesion = binary_opening(lesion, disk(1))
    lesion = binary_closing(lesion, disk(2))
    lesion = remove_small_objects(lesion, min_size=max(4, int(cfg.min_lesion_area_frac * h * w)))

    return lesion
