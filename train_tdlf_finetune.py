"""Script 2: finetune a standard wtconvnext (tiny/small) backbone using the
two-stage Decoupled Learning Framework (TDLF) from:
"Decoupling Representation Learning and Classifier for Long-Tailed
Adversarial Training", Pattern Recognition 172 (2026) 112607.

Stage 1 (representation learning): the backbone is finetuned with an
adversarial supervised contrastive loss (Eq. 4) over a 4-view construction
per sample -- clean_i, adv_i, clean_j, adv_j (Eq. 6-8) -- where the
adversarial views are found via PGD in raw pixel space, plus adversarial
weight perturbation (AWP, Eq. 9-10). This is the actual "loss function in
the paper".

Stage 2 (classifier learning): the backbone is frozen and a linear
classifier head is trained with class-balanced sampling (Sec. 4.2, Eq. 11).
Default loss is Asymmetric Loss (Ridnik et al., ICCV 2021, one-vs-rest;
see src/losses.py::AsymmetricLoss) rather than plain cross-entropy -- a run
with --classifier-loss ce on this dataset showed the classic easy-negative-
domination failure (vMCIAD precision 0.12: constant false-positive flooding
from the majority classes despite decent recall). Pass --classifier-loss ce
or balanced_softmax to compare against the older behavior.

Both stages print a classification report + confusion matrix and save them,
along with per-epoch history, into --output-dir.

Example:
    python train_tdlf_finetune.py \
        --data-dir /path/to/swml_vols_middle50_4class \
        --model-name wtconvnext_tiny \
        --pretrained-path /path/to/WTConvNeXt_tiny_5_300e_ema.pth \
        --num-classes 4 \
        --output-dir ./outputs_tdlf
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from datasets import get_classification_dataloaders_raw, get_two_view_dataloader  # noqa: E402
from losses import AsymmetricLoss  # noqa: E402
from tdlf_losses import AdversarialWeightPerturbation, SupervisedContrastiveLoss, generate_adversarial_views  # noqa: E402
from tdlf_model import TDLFBackbone, TDLFClassifier, TDLFRepresentationModel  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BalancedSoftmaxLoss(nn.Module):
    """Optional alternative classifier loss (ablated in the paper, Table 10)."""

    def __init__(self, class_counts: torch.Tensor):
        super().__init__()
        log_prior = torch.log(class_counts.float() / class_counts.sum())
        self.register_buffer("log_prior", log_prior)

    def forward(self, logits, labels):
        return F.cross_entropy(logits + self.log_prior, labels)


def train_stage1(args, device):
    print("\n========== Stage 1: adversarial supervised contrastive representation learning ==========\n")

    backbone = TDLFBackbone(args.model_name, drop_path_rate=args.drop_path_rate)
    if args.pretrained_path:
        backbone.load_pretrained(args.pretrained_path)
    model = TDLFRepresentationModel(backbone, proj_dim=args.proj_dim).to(device)

    scl = SupervisedContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage1_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.stage1_epochs))
    awp = AdversarialWeightPerturbation(model, gamma=args.awp_gamma) if args.use_awp else None

    train_loader, _ = get_two_view_dataloader(args.data_dir, args.img_size, args.batch_size, args.num_workers)

    history = []
    for epoch in range(args.stage1_epochs):
        model.train()
        running_loss, n_batches = 0.0, 0

        for x_i, x_j, labels in train_loader:
            x_i, x_j, labels = x_i.to(device), x_j.to(device), labels.to(device)

            if args.pgd_steps > 0:
                adv_i, adv_j = generate_adversarial_views(
                    model, scl, x_i, x_j, labels,
                    epsilon=args.pgd_epsilon, step_size=args.pgd_step_size, num_steps=args.pgd_steps,
                )
            else:
                adv_i, adv_j = x_i, x_j

            views = torch.cat([adv_i, adv_j, x_i, x_j], dim=0)
            view_labels = labels.repeat(4)

            optimizer.zero_grad()
            embeddings = model(views)
            loss = scl(embeddings, view_labels)
            loss.backward()

            if awp is not None:
                awp.perturb()
                optimizer.zero_grad()
                embeddings2 = model(views)
                loss2 = scl(embeddings2, view_labels)
                loss2.backward()
                awp.restore()

            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(1, n_batches)
        history.append({"epoch": epoch + 1, "scl_loss": epoch_loss})
        print(f"Stage 1 | Epoch [{epoch + 1}/{args.stage1_epochs}] | SCL Loss: {epoch_loss:.4f}")

    history_path = os.path.join(args.output_dir, f"{args.model_name}_tdlf_stage1_history.csv")
    with open(history_path, "w") as f:
        f.write("epoch,scl_loss\n")
        for row in history:
            f.write(f"{row['epoch']},{row['scl_loss']:.6f}\n")
    print(f"Stage 1 history saved to: {history_path}")

    # Save the inner WTConvNeXt module's state dict, not the TDLFBackbone
    # wrapper's (which would prefix every key with "backbone." and include
    # the non-learned normalize buffers) -- this must match what
    # TDLFBackbone.load_pretrained() loads into on the other end.
    backbone_ckpt_path = os.path.join(args.output_dir, f"{args.model_name}_tdlf_stage1_backbone.pth")
    torch.save({"backbone_state_dict": backbone.backbone.state_dict(), "model_name": args.model_name}, backbone_ckpt_path)
    print(f"Stage 1 backbone checkpoint saved to: {backbone_ckpt_path}")

    return backbone


@torch.no_grad()
def evaluate_stage2(model, loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    running_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return correct / total, running_loss / max(1, n_batches), all_labels, all_preds


def train_stage2(args, device, backbone):
    print("\n========== Stage 2: frozen-encoder classifier learning ==========\n")

    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    model = TDLFClassifier(backbone, args.num_classes).to(device)

    train_loader, val_loader, train_dataset = get_classification_dataloaders_raw(
        args.data_dir, args.img_size, args.batch_size, args.num_workers,
        class_balanced=args.class_balanced_sampling,
    )
    class_names = train_dataset.classes
    print("Classes:", class_names)
    print(f"Class-balanced sampling: {args.class_balanced_sampling} | Classifier loss: {args.classifier_loss}")
    if len(class_names) != args.num_classes:
        raise ValueError(f"Expected {args.num_classes} classes, found {len(class_names)}: {class_names}")

    if args.classifier_loss == "asymmetric":
        criterion = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos, clip=args.asl_clip).to(device)
    elif args.classifier_loss == "balanced_softmax":
        from collections import Counter
        counts = Counter(t for _, t in train_dataset.samples)
        class_counts = torch.tensor([counts.get(c, 0) for c in range(args.num_classes)], dtype=torch.float32)
        criterion = BalancedSoftmaxLoss(class_counts).to(device)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.stage2_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.stage2_epochs))

    ckpt_path = os.path.join(args.output_dir, f"{args.model_name}_tdlf_stage2_classifier.pth")
    history_path = ckpt_path.replace(".pth", "_epoch_history.csv")
    metrics_path = ckpt_path.replace(".pth", "_validation_metrics.txt")

    best_acc = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(args.stage2_epochs):
        model.train()
        running_loss, correct, total, n_batches = 0.0, 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, n_batches)
        train_acc = correct / total

        val_acc, val_loss, y_true, y_pred = evaluate_stage2(model, val_loader, criterion, device)

        print(
            f"Stage 2 | Epoch [{epoch + 1}/{args.stage2_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "state_dict": best_weights,
                "best_val_acc": best_acc,
                "class_names": class_names,
                "model_name": args.model_name,
                "num_classes": args.num_classes,
            }, ckpt_path)

    model.load_state_dict(best_weights)
    val_acc, val_loss, y_true, y_pred = evaluate_stage2(model, val_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nBest validation accuracy for Stage 2: {best_acc:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    with open(history_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']:.6f},{row['train_acc']:.6f},{row['val_loss']:.6f},{row['val_acc']:.6f}\n")

    with open(metrics_path, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Classifier loss: {args.classifier_loss}\n")
        f.write(f"Best Validation Accuracy: {best_acc:.6f}\n\n")
        f.write("Class Names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"{idx}: {name}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nStage 2 checkpoint saved to: {ckpt_path}")
    print(f"Stage 2 epoch history saved to: {history_path}")
    print(f"Stage 2 validation metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="wtconvnext_tiny", choices=["wtconvnext_tiny", "wtconvnext_small", "wtconvnext_base"])
    parser.add_argument("--pretrained-path", type=str, default=None, help="Local checkpoint to initialize the backbone before TDLF finetuning.")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-2)

    # Stage 1: representation learning
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--pgd-epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-step-size", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=5, help="Paper default is 10; lowered for cost. Set 0 to disable adversarial views (plain SupCon).")
    parser.add_argument("--use-awp", action="store_true", default=True)
    parser.add_argument("--no-awp", dest="use_awp", action="store_false")
    parser.add_argument("--awp-gamma", type=float, default=0.01)

    # Stage 2: classifier learning
    parser.add_argument("--stage2-epochs", type=int, default=25)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-loss", type=str, default="asymmetric", choices=["asymmetric", "ce", "balanced_softmax"],
                         help="Asymmetric Loss (Ridnik et al., ICCV 2021, one-vs-rest) by default -- targets the easy-negative-domination failure mode plain CE showed on this dataset. 'ce'/'balanced_softmax' remain available for comparison.")
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--class-balanced-sampling", action="store_true", default=True)
    parser.add_argument("--no-class-balanced-sampling", dest="class_balanced_sampling", action="store_false",
                         help="ASL does not require resampling to handle imbalance (paper avoids combining static/sampling-based rebalancing with its asymmetric weighting); worth trying if ASL + sampling together over-corrects.")

    parser.add_argument("--skip-stage1", action="store_true", help="Skip straight to Stage 2 using --pretrained-path as the (already-finetuned) backbone.")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    if args.skip_stage1:
        backbone = TDLFBackbone(args.model_name, drop_path_rate=args.drop_path_rate)
        if args.pretrained_path:
            backbone.load_pretrained(args.pretrained_path)
        backbone = backbone.to(device)
    else:
        backbone = train_stage1(args, device)

    train_stage2(args, device, backbone)

    print("\nTDLF finetuning complete.")


if __name__ == "__main__":
    main()
