"""Script 1: train the joint encoder/decoder/classifier model from the block
diagram (Encoder -> BN -> Decoder -> segmentation, BN -> GP -> Linear ->
classification), built on the WTConvNeXt backbone.

L = L_cls + lambda * L_seg   (lambda defaults to 0.3)

If you don't pass --mask-dir, the model still trains end-to-end but the
segmentation branch receives no gradient (see src/losses.py::SegmentationLoss)
-- it stays architecturally in place for when real PVWML masks exist.
Masks produced by generate_pseudo_masks.py are heuristic pseudo-labels, not
ground truth; read that script's docstring before using them here.

Example:
    python train_multitask.py \
        --data-dir /path/to/swml_vols_middle50_4class \
        --model-name wtconvnext_tiny \
        --pretrained-encoder-path /path/to/WTConvNeXt_tiny_5_300e_ema.pth \
        --num-classes 4 \
        --output-dir ./outputs_multitask
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from datasets import compute_class_weights, get_multitask_dataloaders  # noqa: E402
from losses import MultiTaskLoss  # noqa: E402
from multitask_model import build_multitask_model  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    probs = (torch.sigmoid(logits) > 0.5).float()
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct = 0, 0
    running = {"cls_loss": 0.0, "seg_loss": 0.0, "total_loss": 0.0}
    n_batches = 0

    for images, labels, masks, has_mask in loader:
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)
        has_mask = has_mask.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss, parts = criterion(outputs, labels, masks, has_mask)
        loss.backward()
        optimizer.step()

        for k in running:
            running[k] += parts[k]
        n_batches += 1

        preds = torch.argmax(outputs["cls_logits"], dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_acc = correct / total
    for k in running:
        running[k] /= n_batches
    return epoch_acc, running


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    running = {"cls_loss": 0.0, "seg_loss": 0.0, "total_loss": 0.0}
    n_batches = 0
    dice_scores = []
    all_preds, all_labels = [], []

    for images, labels, masks, has_mask in loader:
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)
        has_mask = has_mask.to(device)

        outputs = model(images)
        _, parts = criterion(outputs, labels, masks, has_mask)

        for k in running:
            running[k] += parts[k]
        n_batches += 1

        preds = torch.argmax(outputs["cls_logits"], dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if has_mask.sum() > 0:
            mask_sel = has_mask.bool()
            dice_scores.append(dice_score(outputs["seg_logits"][mask_sel], masks[mask_sel]))

    epoch_acc = correct / total
    for k in running:
        running[k] /= n_batches
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else None
    return epoch_acc, running, mean_dice, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, default=None,
                         help="Optional dir mirroring --data-dir with segmentation masks. Omit to train classification-only.")
    parser.add_argument("--model-name", type=str, default="wtconvnext_tiny", choices=["wtconvnext_tiny", "wtconvnext_small", "wtconvnext_base"])
    parser.add_argument("--pretrained-encoder-path", type=str, default=None,
                         help="Local checkpoint for the WTConvNeXt encoder (e.g. the ImageNet weights linked in the WTConv README).")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--num-seg-classes", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--seg-loss-weight", type=float, default=0.3, help="lambda in L = L_cls + lambda * L_seg")
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="./outputs_multitask")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_multitask_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        mask_dir=args.mask_dir,
        num_workers=args.num_workers,
    )
    print("Classes:", class_names)
    if len(class_names) != args.num_classes:
        raise ValueError(f"Expected {args.num_classes} classes, found {len(class_names)}: {class_names}")

    class_weights = compute_class_weights(train_loader.dataset, args.num_classes).to(device)
    print("Class weights (inverse frequency):", class_weights.tolist())

    model = build_multitask_model(
        variant=args.model_name,
        num_classes=args.num_classes,
        num_seg_classes=args.num_seg_classes,
        pretrained_encoder_path=args.pretrained_encoder_path,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    criterion = MultiTaskLoss(class_weights=class_weights, seg_loss_weight=args.seg_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = os.path.join(args.output_dir, f"{args.model_name}_multitask.pth")
    history_path = ckpt_path.replace(".pth", "_epoch_history.csv")
    metrics_path = ckpt_path.replace(".pth", "_validation_metrics.txt")

    best_acc = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(args.epochs):
        train_acc, train_parts = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_parts, val_dice, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        dice_str = f"{val_dice:.4f}" if val_dice is not None else "n/a"
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Train Loss: {train_parts['total_loss']:.4f} (cls {train_parts['cls_loss']:.4f}, seg {train_parts['seg_loss']:.4f}) | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_parts['total_loss']:.4f} | Val Acc: {val_acc:.4f} | Val Dice: {dice_str}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_total_loss": train_parts["total_loss"],
            "train_cls_loss": train_parts["cls_loss"],
            "train_seg_loss": train_parts["seg_loss"],
            "train_acc": train_acc,
            "val_total_loss": val_parts["total_loss"],
            "val_cls_loss": val_parts["cls_loss"],
            "val_seg_loss": val_parts["seg_loss"],
            "val_acc": val_acc,
            "val_dice": val_dice if val_dice is not None else "",
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "state_dict": best_weights,
                "best_val_acc": best_acc,
                "class_names": class_names,
                "model_name": args.model_name,
                "num_classes": args.num_classes,
                "num_seg_classes": args.num_seg_classes,
            }, ckpt_path)

    model.load_state_dict(best_weights)
    val_acc, val_parts, val_dice, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    with open(history_path, "w") as f:
        f.write("epoch,train_total_loss,train_cls_loss,train_seg_loss,train_acc,val_total_loss,val_cls_loss,val_seg_loss,val_acc,val_dice\n")
        for row in history:
            f.write(",".join(str(row[k]) for k in [
                "epoch", "train_total_loss", "train_cls_loss", "train_seg_loss", "train_acc",
                "val_total_loss", "val_cls_loss", "val_seg_loss", "val_acc", "val_dice",
            ]) + "\n")

    with open(metrics_path, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"lambda (seg_loss_weight): {args.seg_loss_weight}\n")
        f.write(f"Best Validation Accuracy: {best_acc:.6f}\n")
        f.write(f"Final Validation Dice: {val_dice}\n\n")
        f.write("Class Names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"{idx}: {name}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nCheckpoint saved to: {ckpt_path}")
    print(f"Epoch history saved to: {history_path}")
    print(f"Validation metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
