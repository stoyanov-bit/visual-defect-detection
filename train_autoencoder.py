from pathlib import Path

import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from defect_detection.anomaly_dataset import (
    AnomalyDataset,
    collect_good_training_samples,
    train_transform,
    eval_transform,
)

from defect_detection.autoencoder import (
    ConvolutionalAutoencoder,
)

from defect_detection.anomaly_training import (
    train_autoencoder,
)

from defect_detection.plotting import (
    show_reconstructions,
    plot_learning_curves,
)


# ==================================================
# Configuration
# ==================================================

DATA_DIR = "data/raw/bottle"

MODEL_DIR = Path(
    "models"
)

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
# Data
# ==================================================

train_samples, val_samples = (
    collect_good_training_samples(
        DATA_DIR,
        val_size=0.2,
        random_state=42,
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


train_dataset = AnomalyDataset(
    train_samples,
    transform=train_transform,
)

val_dataset = AnomalyDataset(
    val_samples,
    transform=eval_transform,
)


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


# ==================================================
# Model
# ==================================================

model = ConvolutionalAutoencoder().to(
    device
)

criterion = nn.MSELoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==================================================
# Training
# ==================================================

model, history = train_autoencoder(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS,
)


# ==================================================
# Save
# ==================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model_path = (
    MODEL_DIR
    / "autoencoder.pt"
)

torch.save(
    model.state_dict(),
    model_path,
)

print()
print(
    f"Model saved to: {model_path}"
)


# ==================================================
# Visualisation
# ==================================================

plot_learning_curves(
    history,
    save_path=(
        "results/autoencoder/"
        "learning_curves.png"
    ),
)


show_reconstructions(
    model=model,
    data_loader=val_loader,
    device=device,
    number_images=5,
)