import torch


def calculate_accuracy(
    outputs,
    labels,
):

    probabilities = torch.sigmoid(
        outputs
    )

    predictions = (
        probabilities >= 0.5
    ).float()

    correct = (
        predictions == labels
    ).sum().item()

    return correct


def train_model(
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
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ==========================================
        # TRAINING
        # ==========================================

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = (
                labels
                .float()
                .unsqueeze(1)
                .to(device)
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * images.size(0)
            )

            train_correct += (
                calculate_accuracy(
                    outputs,
                    labels,
                )
            )

            train_total += (
                images.size(0)
            )

        train_loss /= train_total

        train_accuracy = (
            train_correct
            / train_total
        )


        # ==========================================
        # VALIDATION
        # ==========================================

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_predicted_good = 0
        val_predicted_defective = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device
                )

                labels = (
                    labels
                    .float()
                    .unsqueeze(1)
                    .to(device)
                )

                outputs = model(images)
                probabilities = torch.sigmoid(outputs)

                predictions = (
                    probabilities >= 0.5
                ).int()

                val_predicted_good += (
                    predictions == 0
                ).sum().item()

                val_predicted_defective += (
                    predictions == 1
                ).sum().item()

                loss = criterion(
                    outputs,
                    labels,
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                val_correct += (
                    calculate_accuracy(
                        outputs,
                        labels,
                    )
                )

                val_total += (
                    images.size(0)
                )

        val_loss /= val_total

        val_accuracy = (
            val_correct
            / val_total
        )


        # ==========================================
        # SAVE HISTORY
        # ==========================================

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )


        # ==========================================
        # PRINT RESULTS
        # ==========================================

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.3f} | "
            f"Val Acc: {val_accuracy:.3f} | "
            f"Pred: G={val_predicted_good}, "
            f"D={val_predicted_defective}"
        )

    return history