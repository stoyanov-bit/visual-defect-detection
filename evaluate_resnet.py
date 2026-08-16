import numpy as np
import torch
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from defect_detection.anomaly_dataset import (
    AnomalyDataset,
    collect_good_training_samples,
    collect_test_samples,
    resnet_transform,
)

from defect_detection.resnet_features import (
    ResNetFeatureExtractor,
)

from defect_detection.feature_anomaly import (
    extract_features,
    calculate_normal_center,
    calculate_feature_distances,
    calculate_nearest_neighbor_distances,
    calculate_threshold,
    classify_anomalies,
)

from defect_detection.plotting import (
    plot_anomaly_distributions,
    plot_roc_curve,
    plot_confusion_matrix,
)


DATA_DIR = "data/raw/bottle"
BATCH_SIZE = 16


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

test_samples = collect_test_samples(
    DATA_DIR
)


train_dataset = AnomalyDataset(
    train_samples,
    transform=resnet_transform,
)

val_dataset = AnomalyDataset(
    val_samples,
    transform=resnet_transform,
)

test_dataset = AnomalyDataset(
    test_samples,
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ==================================================
# ResNet feature extractor
# ==================================================

model = ResNetFeatureExtractor().to(
    device
)

model.eval()


# ==================================================
# Extract features once
# ==================================================

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

test_features, test_labels, defect_types = (
    extract_features(
        model,
        test_loader,
        device,
    )
)


print(
    "Training feature shape:",
    train_features.shape,
)


# ==================================================
# Evaluation helper
# ==================================================

def evaluate_method(
    name,
    val_scores,
    test_scores,
):

    threshold = calculate_threshold(
        val_scores,
        percentile=95,
    )

    predictions = classify_anomalies(
        test_scores,
        threshold,
    )

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    precision = precision_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        test_labels,
        test_scores,
    )

    cm = confusion_matrix(
        test_labels,
        predictions,
    )


    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Validation mean score: "
        f"{np.mean(val_scores):.4f}"
    )

    print(
        f"Threshold : {threshold:.4f}"
    )

    print(
        f"Accuracy  : {accuracy:.3f}"
    )

    print(
        f"Precision : {precision:.3f}"
    )

    print(
        f"Recall    : {recall:.3f}"
    )

    print(
        f"F1 Score  : {f1:.3f}"
    )

    print(
        f"ROC-AUC   : {auc:.3f}"
    )

    print()
    print("Confusion matrix:")
    print(cm)


    print()
    print(
        "Mean feature distance by class:"
    )

    for defect_type in [
        "good",
        "broken_large",
        "broken_small",
        "contamination",
    ]:

        indices = [
            i
            for i, sample_type
            in enumerate(defect_types)
            if sample_type == defect_type
        ]

        class_scores = (
            test_scores[indices]
        )

        print(
            f"{defect_type:15s}: "
            f"{np.mean(class_scores):.4f}"
        )


    print()
    print(
        "Detection rate by defect type:"
    )

    for defect_type in [
        "broken_large",
        "broken_small",
        "contamination",
    ]:

        indices = [
            i
            for i, sample_type
            in enumerate(defect_types)
            if sample_type == defect_type
        ]

        detected = sum(
            predictions[i] == 1
            for i in indices
        )

        total = len(indices)

        print(
            f"{defect_type:15s}: "
            f"{detected}/{total} "
            f"({detected / total:.1%})"
        )


    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "predictions": predictions,
    }


# ==================================================
# Method 1: Center distance
# ==================================================

normal_center = calculate_normal_center(
    train_features
)

val_center_scores = calculate_feature_distances(
    val_features,
    normal_center,
)

test_center_scores = calculate_feature_distances(
    test_features,
    normal_center,
)


center_results = evaluate_method(
    "RESNET18 - CENTER DISTANCE",
    val_center_scores,
    test_center_scores,
)


# ==================================================
# Method 2: Nearest neighbour
# ==================================================

val_nn_scores = (
    calculate_nearest_neighbor_distances(
        val_features,
        train_features,
    )
)

test_nn_scores = (
    calculate_nearest_neighbor_distances(
        test_features,
        train_features,
    )
)


nn_results = evaluate_method(
    "RESNET18 - NEAREST NEIGHBOUR",
    val_nn_scores,
    test_nn_scores,
)


# ==================================================
# Comparison
# ==================================================

print()
print("=" * 60)
print("RESNET METHOD COMPARISON")
print("=" * 60)

print(
    f"{'Metric':12s}"
    f"{'Center':>15s}"
    f"{'Nearest NN':>15s}"
)

print("-" * 42)

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
]:

    print(
        f"{metric.capitalize():12s}"
        f"{center_results[metric]:>15.3f}"
        f"{nn_results[metric]:>15.3f}"
    )


# ==================================================
# Choose best method based on F1
# ==================================================

if nn_results["f1"] > center_results["f1"]:

    final_name = "Nearest Neighbour"
    final_scores = test_nn_scores
    final_results = nn_results

else:

    final_name = "Center Distance"
    final_scores = test_center_scores
    final_results = center_results


print()
print(
    f"Selected method: {final_name}"
)


# ==================================================
# Final plots
# ==================================================

plot_anomaly_distributions(
    errors=final_scores,
    defect_types=defect_types,
    threshold=final_results["threshold"],
    save_path=(
        "results/resnet/"
        "feature_distance_distribution.png"
    ),
)


plot_roc_curve(
    labels=test_labels,
    scores=final_scores,
    save_path=(
        "results/resnet/"
        "roc_curve.png"
    ),
)


plot_confusion_matrix(
    labels=test_labels,
    predictions=final_results["predictions"],
    save_path=(
        "results/resnet/"
        "confusion_matrix.png"
    ),
)

from pathlib import Path


results_dir = Path(
    "results/resnet"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True,
)


metrics = {
    "method": final_name,
    "accuracy": float(
        final_results["accuracy"]
    ),
    "precision": float(
        final_results["precision"]
    ),
    "recall": float(
        final_results["recall"]
    ),
    "f1": float(
        final_results["f1"]
    ),
    "roc_auc": float(
        final_results["auc"]
    ),
    "threshold": float(
        final_results["threshold"]
    ),
}


with open(
    results_dir / "metrics.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )