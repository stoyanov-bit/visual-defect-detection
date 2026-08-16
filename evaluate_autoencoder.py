import numpy as np
import torch

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
    eval_transform,
)

from defect_detection.autoencoder import (
    ConvolutionalAutoencoder,
)

from defect_detection.anomaly_evaluation import (
    reconstruction_errors,
    topk_reconstruction_errors,
    calculate_threshold,
    classify_anomalies,
)

from defect_detection.plotting import (
    plot_anomaly_distributions,
)


DATA_DIR = "data/raw/bottle"
MODEL_PATH = "models/autoencoder.pt"

BATCH_SIZE = 16


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# Data
# ==================================================

_, val_samples = collect_good_training_samples(
    DATA_DIR,
    val_size=0.2,
    random_state=42,
)

test_samples = collect_test_samples(
    DATA_DIR
)


val_dataset = AnomalyDataset(
    val_samples,
    transform=eval_transform,
)

test_dataset = AnomalyDataset(
    test_samples,
    transform=eval_transform,
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

model = ConvolutionalAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model = model.to(
    device
)

model.eval()


# ==================================================
# Evaluation helper
# ==================================================

def evaluate_method(
    name,
    val_errors,
    test_errors,
    test_labels,
    defect_types,
):

    threshold = calculate_threshold(
        val_errors,
        percentile=95,
    )

    predictions = classify_anomalies(
        test_errors,
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
        test_errors,
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
        f"Threshold : {threshold:.6f}"
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
        "Mean score by class:"
    )

    for defect_type in [
        "good",
        "broken_large",
        "broken_small",
        "contamination",
    ]:

        class_errors = [
            error
            for error, sample_type
            in zip(
                test_errors,
                defect_types,
            )
            if sample_type == defect_type
        ]

        print(
            f"{defect_type:15s}: "
            f"{np.mean(class_errors):.6f}"
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
# Global MSE
# ==================================================

val_global, _, _ = reconstruction_errors(
    model,
    val_loader,
    device,
)

test_global, test_labels, defect_types = (
    reconstruction_errors(
        model,
        test_loader,
        device,
    )
)

global_results = evaluate_method(
    "GLOBAL MSE",
    val_global,
    test_global,
    test_labels,
    defect_types,
)


# ==================================================
# Top 1 %
# ==================================================

val_top, _, _ = topk_reconstruction_errors(
    model,
    val_loader,
    device,
    top_fraction=0.01,
)

test_top, _, _ = topk_reconstruction_errors(
    model,
    test_loader,
    device,
    top_fraction=0.01,
)

top_results = evaluate_method(
    "TOP 1% RECONSTRUCTION ERROR",
    val_top,
    test_top,
    test_labels,
    defect_types,
)


# ==================================================
# Plots
# ==================================================

plot_anomaly_distributions(
    test_top,
    defect_types,
    top_results["threshold"],
    save_path=(
        "results/autoencoder/"
        "anomaly_distribution_top1percent.png"
    ),
)


# ==================================================
# Comparison
# ==================================================

print()
print("=" * 60)
print("METHOD COMPARISON")
print("=" * 60)

print(
    f"{'Metric':12s}"
    f"{'Global MSE':>15s}"
    f"{'Top 1%':>15s}"
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
        f"{global_results[metric]:>15.3f}"
        f"{top_results[metric]:>15.3f}"
    )