import numpy as np
import torch


# ==================================================
# Global reconstruction error
# ==================================================

def reconstruction_errors(
    model,
    data_loader,
    device,
):

    model.eval()

    errors = []
    labels = []
    defect_types = []

    with torch.no_grad():

        for (
            images,
            batch_labels,
            batch_defect_types,
        ) in data_loader:

            images = images.to(device)

            reconstructions = model(
                images
            )

            batch_errors = torch.mean(
                (images - reconstructions) ** 2,
                dim=(1, 2, 3),
            )

            errors.extend(
                batch_errors
                .cpu()
                .numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

            defect_types.extend(
                list(batch_defect_types)
            )

    return (
        np.asarray(errors),
        np.asarray(labels),
        defect_types,
    )


# ==================================================
# Top-k reconstruction error
# ==================================================

def topk_reconstruction_errors(
    model,
    data_loader,
    device,
    top_fraction=0.01,
):

    model.eval()

    errors = []
    labels = []
    defect_types = []

    with torch.no_grad():

        for (
            images,
            batch_labels,
            batch_defect_types,
        ) in data_loader:

            images = images.to(device)

            reconstructions = model(
                images
            )

            pixel_errors = (
                images
                - reconstructions
            ) ** 2

            # Average RGB channels
            pixel_errors = pixel_errors.mean(
                dim=1
            )

            pixel_errors = pixel_errors.flatten(
                start_dim=1
            )

            k = max(
                1,
                int(
                    pixel_errors.shape[1]
                    * top_fraction
                ),
            )

            top_errors = torch.topk(
                pixel_errors,
                k=k,
                dim=1,
            ).values

            batch_errors = top_errors.mean(
                dim=1
            )

            errors.extend(
                batch_errors
                .cpu()
                .numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

            defect_types.extend(
                list(batch_defect_types)
            )

    return (
        np.asarray(errors),
        np.asarray(labels),
        defect_types,
    )


# ==================================================
# Threshold
# ==================================================

def calculate_threshold(
    errors,
    percentile=95,
):

    errors = np.asarray(
        errors
    )

    return float(
        np.percentile(
            errors,
            percentile,
        )
    )


# ==================================================
# Classification
# ==================================================

def classify_anomalies(
    errors,
    threshold,
):

    errors = np.asarray(
        errors
    )

    return (
        errors >= threshold
    ).astype(int)