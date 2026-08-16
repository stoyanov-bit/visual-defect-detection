import numpy as np
import torch


# ==================================================
# Feature extraction
# ==================================================

def extract_features(
    model,
    data_loader,
    device,
):

    model.eval()

    features = []
    labels = []
    defect_types = []

    with torch.no_grad():

        for (
            images,
            batch_labels,
            batch_defect_types,
        ) in data_loader:

            images = images.to(device)

            batch_features = model(
                images
            )

            features.append(
                batch_features
                .cpu()
                .numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

            defect_types.extend(
                list(batch_defect_types)
            )

    features = np.concatenate(
        features,
        axis=0,
    )

    return (
        features,
        np.asarray(labels),
        defect_types,
    )


# ==================================================
# Normal center
# ==================================================

def calculate_normal_center(
    features,
):

    return np.mean(
        features,
        axis=0,
    )


# ==================================================
# Center distance
# ==================================================

def calculate_feature_distances(
    features,
    normal_center,
):

    return np.linalg.norm(
        features - normal_center,
        axis=1,
    )


# ==================================================
# Nearest-neighbour distance
# ==================================================

def calculate_nearest_neighbor_distances(
    features,
    train_features,
):

    distances = []

    for feature in features:

        sample_distances = np.linalg.norm(
            train_features - feature,
            axis=1,
        )

        nearest_distance = np.min(
            sample_distances
        )

        distances.append(
            nearest_distance
        )

    return np.asarray(
        distances
    )


# ==================================================
# Threshold
# ==================================================

def calculate_threshold(
    validation_distances,
    percentile=95,
):

    return float(
        np.percentile(
            validation_distances,
            percentile,
        )
    )


# ==================================================
# Classification
# ==================================================

def classify_anomalies(
    distances,
    threshold,
):

    distances = np.asarray(
        distances
    )

    return (
        distances >= threshold
    ).astype(int)