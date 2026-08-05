"""Dataset utilities shared by both training scripts.

Layout expected on disk (matches the existing project's ImageFolder layout):

    data_dir/
        train/<class_name>/*.png
        val/<class_name>/*.png
        test/<class_name>/*.png

For script 1 (multitask_model), segmentation masks are optional. If given,
`mask_dir` must mirror the same relative structure as `data_dir`, e.g.:

    mask_dir/
        train/<class_name>/*.png   # same filename as the source image, single-channel
        val/<class_name>/*.png
        test/<class_name>/*.png

Any image without a corresponding mask file is treated as classification-only
(the segmentation loss term is skipped for that sample via the `has_mask` flag).
"""
import os
from collections import Counter
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PairedTransform:
    """Applies identical geometric augmentation to an (image, mask) pair.

    Using independent torchvision transforms on image and mask would sample
    different random flip/rotation parameters for each, silently misaligning
    the segmentation labels. This applies one sampled transform to both.
    """

    def __init__(
        self,
        img_size: int,
        train: bool,
        hflip_p: float = 0.5,
        rotation_degrees: float = 10.0,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        self.img_size = img_size
        self.train = train
        self.hflip_p = hflip_p
        self.rotation_degrees = rotation_degrees
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Optional[Image.Image]):
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BICUBIC)
        if mask is not None:
            mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if torch.rand(1).item() < self.hflip_p:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)

            angle = float(torch.empty(1).uniform_(-self.rotation_degrees, self.rotation_degrees).item())
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BICUBIC)
            if mask is not None:
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        if mask is not None:
            mask = TF.to_tensor(mask)
            mask = (mask > 0.5).float()

        return image, mask


class SegClsDataset(Dataset):
    """ImageFolder-style classification dataset with optional paired masks."""

    def __init__(self, root: str, transform: PairedTransform, mask_root: Optional[str] = None):
        base = datasets.ImageFolder(root)
        self.root = root
        self.mask_root = mask_root
        self.transform = transform
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx
        self.samples = base.samples

    def __len__(self):
        return len(self.samples)

    def _mask_path_for(self, image_path: str) -> str:
        rel = os.path.relpath(image_path, self.root)
        rel = os.path.splitext(rel)[0] + ".png"
        return os.path.join(self.mask_root, rel)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")

        mask = None
        has_mask = False
        if self.mask_root is not None:
            mask_path = self._mask_path_for(path)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                has_mask = True

        image, mask = self.transform(image, mask)
        if mask is None:
            mask = torch.zeros(1, image.shape[-2], image.shape[-1])

        return image, target, mask, has_mask


def get_multitask_dataloaders(
    data_dir: str,
    img_size: int,
    batch_size: int,
    mask_dir: Optional[str] = None,
    num_workers: int = 2,
):
    train_transform = PairedTransform(img_size, train=True)
    eval_transform = PairedTransform(img_size, train=False)

    train_mask_root = os.path.join(mask_dir, "train") if mask_dir else None
    val_mask_root = os.path.join(mask_dir, "val") if mask_dir else None

    train_dataset = SegClsDataset(
        os.path.join(data_dir, "train"), train_transform, mask_root=train_mask_root
    )
    val_dataset = SegClsDataset(
        os.path.join(data_dir, "val"), eval_transform, mask_root=val_mask_root
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    if train_mask_root:
        n_with_mask = sum(1 for p, _ in train_dataset.samples if os.path.exists(train_dataset._mask_path_for(p)))
        print(f"Segmentation masks found for {n_with_mask}/{len(train_dataset)} training images.")
    else:
        print("No mask_dir provided: training classification-only (segmentation loss disabled).")

    return train_loader, val_loader, train_dataset.classes


def get_classification_dataloaders(data_dir: str, img_size: int, batch_size: int, num_workers: int = 2):
    """Plain classification loaders (used by script 2 / TDLF finetuning)."""
    from torchvision import transforms

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset


class TwoViewDataset(Dataset):
    """Returns two independently-augmented views of the same image plus its
    label, for the TDLF adversarial supervised contrastive stage (script 2).
    Both views are raw [0,1] tensors (no normalization) -- the TDLF backbone
    normalizes internally so PGD can perturb pixels directly and epsilon
    stays meaningful in [0,1] units, matching the paper's epsilon=8/255.
    """

    def __init__(self, root: str, img_size: int):
        from torchvision import transforms

        base = datasets.ImageFolder(root)
        self.samples = base.samples
        self.loader = base.loader
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx

        # "Moderate" augmentation only (Sec 4.1.3 of the TDLF paper: aggressive
        # augmentation hurt robust representation learning on long-tailed data).
        self.view_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        view1 = self.view_transform(image)
        view2 = self.view_transform(image)
        return view1, view2, target


def get_two_view_dataloader(data_dir: str, img_size: int, batch_size: int, num_workers: int = 2):
    dataset = TwoViewDataset(os.path.join(data_dir, "train"), img_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return loader, dataset


def get_classification_dataloaders_raw(data_dir: str, img_size: int, batch_size: int, num_workers: int = 2, class_balanced: bool = False):
    """Same as get_classification_dataloaders but without normalization
    (values stay in [0,1]) -- used for TDLF stage 2, where the backbone
    normalizes internally."""
    from torchvision import transforms

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms)

    if class_balanced:
        sampler = make_class_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights: N / (num_classes * count[c])."""
    targets = [t for _, t in dataset.samples]
    counts = Counter(targets)
    n = len(targets)
    weights = torch.ones(num_classes)
    for c in range(num_classes):
        count_c = counts.get(c, 0)
        weights[c] = n / (num_classes * count_c) if count_c > 0 else 0.0
    return weights


def make_class_balanced_sampler(dataset) -> WeightedRandomSampler:
    """Two-stage class-balanced sampling (TDLF paper, Sec 4.2 / Eq. 11 context):
    a class is drawn uniformly, then an instance uniformly within that class.
    Equivalent to per-sample weight 1 / count(class) with replacement sampling.
    """
    targets = [t for _, t in dataset.samples]
    counts = Counter(targets)
    weights = [1.0 / counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
