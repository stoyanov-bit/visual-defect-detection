import copy

import torch


def train_autoencoder(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
):

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")

    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(epochs):

        # ==================================================
        # Training
        # ==================================================

        model.train()

        train_loss = 0.0
        train_total = 0

        for images, _, _ in train_loader:

            images = images.to(device)

            optimizer.zero_grad()

            reconstructions = model(
                images
            )

            loss = criterion(
                reconstructions,
                images,
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * images.size(0)
            )

            train_total += (
                images.size(0)
            )

        train_loss /= train_total

        # ==================================================
        # Validation
        # ==================================================

        model.eval()

        val_loss = 0.0
        val_total = 0

        with torch.no_grad():

            for images, _, _ in val_loader:

                images = images.to(device)

                reconstructions = model(
                    images
                )

                loss = criterion(
                    reconstructions,
                    images,
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                val_total += (
                    images.size(0)
                )

        val_loss /= val_total

        # ==================================================
        # History
        # ==================================================

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        # ==================================================
        # Best model
        # ==================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    # Restore best model
    model.load_state_dict(
        best_model_state
    )

    return model, history