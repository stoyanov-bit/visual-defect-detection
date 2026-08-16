from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _prepare_save_path(
    save_path,
):

    if save_path is None:
        return None

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return save_path


# ==================================================
# Reconstructions
# ==================================================

def show_reconstructions(
    model,
    data_loader,
    device,
    number_images=5,
):

    model.eval()

    images, _, _ = next(
        iter(data_loader)
    )

    images = images.to(
        device
    )

    reconstructions = model(
        images
    )

    images = (
        images
        .cpu()
        .detach()
    )

    reconstructions = (
        reconstructions
        .cpu()
        .detach()
    )

    number_images = min(
        number_images,
        len(images),
    )

    fig, axes = plt.subplots(
        2,
        number_images,
        figsize=(
            3 * number_images,
            6,
        ),
    )

    for i in range(number_images):

        original = (
            images[i]
            .permute(1, 2, 0)
        )

        reconstruction = (
            reconstructions[i]
            .permute(1, 2, 0)
        )

        axes[0, i].imshow(
            original
        )

        axes[0, i].set_title(
            "Original"
        )

        axes[0, i].axis(
            "off"
        )

        axes[1, i].imshow(
            reconstruction
        )

        axes[1, i].set_title(
            "Reconstruction"
        )

        axes[1, i].axis(
            "off"
        )

    plt.tight_layout()
    plt.show()


# ==================================================
# Learning curves
# ==================================================

def plot_learning_curves(
    history,
    save_path=None,
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Reconstruction Loss"
    )

    plt.title(
        "Autoencoder Training"
    )

    plt.legend()

    plt.tight_layout()

    save_path = _prepare_save_path(
        save_path
    )

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# ==================================================
# Anomaly score distributions
# ==================================================

def plot_anomaly_distributions(
    errors,
    defect_types,
    threshold,
    save_path=None,
):

    defect_classes = [
        "good",
        "broken_large",
        "broken_small",
        "contamination",
    ]

    plt.figure(
        figsize=(9, 6)
    )

    for defect_class in defect_classes:

        class_errors = [
            error
            for error, sample_type
            in zip(
                errors,
                defect_types,
            )
            if sample_type == defect_class
        ]

        plt.hist(
            class_errors,
            bins=15,
            alpha=0.5,
            label=defect_class,
        )

    plt.axvline(
        threshold,
        linestyle="--",
        label="Threshold",
    )

    plt.xlabel(
        "Anomaly Score"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Anomaly Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    save_path = _prepare_save_path(
        save_path
    )

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# ==================================================
# ROC curve
# ==================================================

def plot_roc_curve(
    labels,
    scores,
    save_path=None,
):

    fpr, tpr, _ = roc_curve(
        labels,
        scores,
    )

    auc = roc_auc_score(
        labels,
        scores,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    save_path = _prepare_save_path(
        save_path
    )

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# ==================================================
# Confusion matrix
# ==================================================

def plot_confusion_matrix(
    labels,
    predictions,
    save_path=None,
):

    cm = confusion_matrix(
        labels,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Good",
            "Defective",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    save_path = _prepare_save_path(
        save_path
    )

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()