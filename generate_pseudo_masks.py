"""Batch-generate heuristic PVWML pseudo-masks for train/val/test FLAIR slices.

These are weak, unsupervised pseudo-labels (see src/pseudo_masks.py docstring
for the exact heuristic and its assumptions) -- NOT clinical ground truth.
Always inspect --preview-n before trusting them for training.

Usage:
    python generate_pseudo_masks.py \
        --data-dir /path/to/swml_vols_middle50_4class \
        --output-dir /path/to/swml_vols_middle50_4class_pseudomasks \
        --preview-n 24
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pseudo_masks import PseudoMaskConfig, generate_pvwml_pseudo_mask  # noqa: E402


def iter_images(split_dir):
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                yield class_name, fname, os.path.join(class_dir, fname)


def make_preview(image_gray, mask, out_path):
    h, w = image_gray.shape
    rgb = np.stack([image_gray] * 3, axis=-1)
    overlay = rgb.copy()
    overlay[mask, 0] = 255
    overlay[mask, 1] = np.clip(overlay[mask, 1] * 0.3, 0, 255)
    overlay[mask, 2] = np.clip(overlay[mask, 2] * 0.3, 0, 255)
    canvas = np.concatenate([rgb, overlay], axis=1)
    Image.fromarray(canvas.astype(np.uint8)).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--preview-n", type=int, default=16, help="Save N overlay previews per split for sanity-checking.")
    parser.add_argument("--ventricle-percentile", type=float, default=PseudoMaskConfig.ventricle_percentile)
    parser.add_argument("--hyperintense-percentile", type=float, default=PseudoMaskConfig.hyperintense_percentile)
    args = parser.parse_args()

    cfg = PseudoMaskConfig(
        ventricle_percentile=args.ventricle_percentile,
        hyperintense_percentile=args.hyperintense_percentile,
    )

    for split in args.splits:
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping missing split: {split_dir}")
            continue

        out_split_dir = os.path.join(args.output_dir, split)
        preview_dir = os.path.join(args.output_dir, "_previews", split)
        os.makedirs(preview_dir, exist_ok=True)

        coverage_stats = []
        n_previewed = 0
        for class_name, fname, path in iter_images(split_dir):
            out_class_dir = os.path.join(out_split_dir, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            image = Image.open(path).convert("L")
            image_arr = np.array(image, dtype=np.float32)

            mask = generate_pvwml_pseudo_mask(image_arr, cfg)
            coverage_stats.append(mask.mean())

            out_name = os.path.splitext(fname)[0] + ".png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_class_dir, out_name))

            if n_previewed < args.preview_n:
                make_preview(image_arr, mask, os.path.join(preview_dir, f"{class_name}_{out_name}"))
                n_previewed += 1

        if coverage_stats:
            coverage_stats = np.array(coverage_stats)
            print(
                f"[{split}] {len(coverage_stats)} masks written to {out_split_dir} | "
                f"mean lesion-pixel coverage: {coverage_stats.mean() * 100:.3f}% "
                f"(min {coverage_stats.min() * 100:.3f}%, max {coverage_stats.max() * 100:.3f}%) | "
                f"previews: {preview_dir}"
            )

    print("\nDone. Open a few files in the _previews/ folders before trusting these masks for training.")


if __name__ == "__main__":
    main()
