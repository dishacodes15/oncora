"""
dataset.py
Preprocessing pipeline for the Brain Tumor MRI dataset.
Handles: resizing, normalization, augmentation, train/val split, DataLoaders.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ---- Config ----
DATA_DIR = "../datasets"          # adjust if your script lives elsewhere
IMG_SIZE = 224                    # standard input size for ResNet18 / EfficientNet
BATCH_SIZE = 32
VAL_SPLIT = 0.15                  # carve a validation set out of Training/

# ImageNet normalization stats - required because we're using pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---- Transforms ----
# Training set: augmentation + normalization
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Test/val set: NO augmentation, only resize + normalize
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def get_dataloaders():
    """
    Builds train, validation, and test DataLoaders from the Training/ and
    Testing/ folders. Returns loaders + class names.
    """

    # Load full training folder (we'll split it ourselves)
    full_train_data = datasets.ImageFolder(
        root=f"{DATA_DIR}/Training", transform=train_transforms
    )

    # Testing/ folder is kept completely separate - used only for final evaluation
    test_data = datasets.ImageFolder(
        root=f"{DATA_DIR}/Testing", transform=eval_transforms
    )

    class_names = full_train_data.classes
    print(f"Classes found: {class_names}")
    print(f"Total training images: {len(full_train_data)}")
    print(f"Total testing images: {len(test_data)}")

    # Split Training/ into train + validation
    val_size = int(len(full_train_data) * VAL_SPLIT)
    train_size = len(full_train_data) - val_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    # Validation set should NOT have augmentation - override its transform
    val_data.dataset.transform = eval_transforms

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check - run this file directly to confirm everything loads
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    images, labels = next(iter(train_loader))
    print(f"\nOne batch shape: {images.shape}")   # should be [32, 3, 224, 224]
    print(f"Labels in this batch: {labels[:10]}")
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
