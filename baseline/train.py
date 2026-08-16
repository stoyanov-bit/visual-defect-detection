import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from defect_detection.dataset import (
    DefectDataset,
    collect_samples,
    split_samples,
    train_transform,
    eval_transform,
)

from defect_detection.models import (
    DefectCNN,
)

from defect_detection.training import (
    train_model,
)


# ==================================================
# Configuration
# ==================================================

DATA_DIR = "data/raw/bottle"

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-3


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {device}"
)


# ==================================================
# Collect samples
# ==================================================

samples = collect_samples(
    DATA_DIR
)


train_samples, val_samples, test_samples = (
    split_samples(
        samples
    )
)


print(
    f"Training images: "
    f"{len(train_samples)}"
)

print(
    f"Validation images: "
    f"{len(val_samples)}"
)

print(
    f"Test images: "
    f"{len(test_samples)}"
)


# ==================================================
# Datasets
# ==================================================

train_dataset = DefectDataset(
    train_samples,
    transform=train_transform,
)

val_dataset = DefectDataset(
    val_samples,
    transform=eval_transform,
)

test_dataset = DefectDataset(
    test_samples,
    transform=eval_transform,
)


# ==================================================
# DataLoaders
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ==================================================
# Model
# ==================================================

model = DefectCNN()

model = model.to(
    device
)


# ==================================================
# Weighted loss
# ==================================================

number_good = sum(
    label == 0
    for _, label in train_samples
)

number_defective = sum(
    label == 1
    for _, label in train_samples
)


pos_weight_value = (
    number_good
    / number_defective
)


print(
    f"Good training samples: "
    f"{number_good}"
)

print(
    f"Defective training samples: "
    f"{number_defective}"
)

print(
    f"Positive class weight: "
    f"{pos_weight_value:.3f}"
)


pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=device,
)


criterion = nn.BCEWithLogitsLoss()


# ==================================================
# Optimizer
# ==================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==================================================
# Train
# ==================================================

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS,
)