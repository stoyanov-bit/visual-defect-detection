import argparse

import numpy as np
import torch

from PIL import Image

from defect_detection.anomaly_dataset import (
    resnet_transform,
)

from defect_detection.resnet_features import (
    ResNetFeatureExtractor,
)

from defect_detection.feature_anomaly import (
    calculate_nearest_neighbor_distances,
)


TRAIN_FEATURES_PATH = (
    "models/train_features.npy"
)

THRESHOLD_PATH = (
    "models/threshold.npy"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


parser = argparse.ArgumentParser(
    description=(
        "Detect visual anomalies in "
        "bottle images."
    )
)

parser.add_argument(
    "image",
    help="Path to bottle image",
)

args = parser.parse_args()


# Load normal reference features

train_features = np.load(
    TRAIN_FEATURES_PATH
)

threshold = float(
    np.load(
        THRESHOLD_PATH
    )[0]
)


# Load feature extractor

model = ResNetFeatureExtractor().to(
    device
)

model.eval()


# Load image

image = Image.open(
    args.image
).convert("RGB")

image = resnet_transform(
    image
)

image = (
    image
    .unsqueeze(0)
    .to(device)
)


# Extract feature

with torch.no_grad():

    feature = model(
        image
    )

feature = (
    feature
    .cpu()
    .numpy()
)


# Calculate nearest-neighbour score

distance = (
    calculate_nearest_neighbor_distances(
        feature,
        train_features,
    )[0]
)


prediction = (
    "DEFECTIVE"
    if distance >= threshold
    else "GOOD"
)


print()
print(
    f"Prediction   : {prediction}"
)

print(
    f"Anomaly score: {distance:.4f}"
)

print(
    f"Threshold    : {threshold:.4f}"
)