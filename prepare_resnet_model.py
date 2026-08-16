from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader

from defect_detection.anomaly_dataset import (
    AnomalyDataset,
    collect_good_training_samples,
    resnet_transform,
)

from defect_detection.resnet_features import (
    ResNetFeatureExtractor,
)

from defect_detection.feature_anomaly import (
    extract_features,
    calculate_nearest_neighbor_distances,
    calculate_threshold,
)


DATA_DIR = "data/raw/bottle"

OUTPUT_DIR = Path(
    "models"
)

BATCH_SIZE = 16


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


train_samples, val_samples = (
    collect_good_training_samples(
        DATA_DIR,
        val_size=0.2,
        random_state=42,
    )
)


train_dataset = AnomalyDataset(
    train_samples,
    transform=resnet_transform,
)

val_dataset = AnomalyDataset(
    val_samples,
    transform=resnet_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


model = ResNetFeatureExtractor().to(
    device
)

model.eval()


train_features, _, _ = extract_features(
    model,
    train_loader,
    device,
)

val_features, _, _ = extract_features(
    model,
    val_loader,
    device,
)


val_distances = (
    calculate_nearest_neighbor_distances(
        val_features,
        train_features,
    )
)


threshold = calculate_threshold(
    val_distances,
    percentile=95,
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_DIR / "train_features.npy",
    train_features,
)


np.save(
    OUTPUT_DIR / "threshold.npy",
    np.array([threshold]),
)


print(
    f"Train features saved: "
    f"{train_features.shape}"
)

print(
    f"Threshold: {threshold:.4f}"
)