from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights


# ==================================================
# Autoencoder transforms
# ==================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ==================================================
# ResNet transform
# ==================================================

resnet_transform = (
    ResNet18_Weights.DEFAULT.transforms()
)


# ==================================================
# Dataset
# ==================================================

class AnomalyDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label, defect_type = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, defect_type


# ==================================================
# Training / validation data
# ==================================================

def collect_good_training_samples(
    data_dir,
    val_size=0.2,
    random_state=42,
):

    data_dir = Path(data_dir)

    good_dir = (
        data_dir
        / "train"
        / "good"
    )

    samples = [
        (path, 0, "good")
        for path in good_dir.glob("*.png")
    ]

    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    return train_samples, val_samples


# ==================================================
# Test data
# ==================================================

def collect_test_samples(data_dir):

    data_dir = Path(data_dir)

    test_dir = (
        data_dir
        / "test"
    )

    samples = []

    # Good images
    for path in (
        test_dir
        / "good"
    ).glob("*.png"):

        samples.append(
            (
                path,
                0,
                "good",
            )
        )

    # Defective images
    defect_classes = [
        "broken_large",
        "broken_small",
        "contamination",
    ]

    for defect_type in defect_classes:

        defect_dir = (
            test_dir
            / defect_type
        )

        for path in defect_dir.glob("*.png"):

            samples.append(
                (
                    path,
                    1,
                    defect_type,
                )
            )

    return samples