from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


# --------------------------------------------------
# Image transformations
# --------------------------------------------------

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


# --------------------------------------------------
# Collect image paths and labels
# --------------------------------------------------

def collect_samples(data_dir):
    """
    Collect bottle images and assign binary labels.

    Label 0: good
    Label 1: defective
    """

    data_dir = Path(data_dir)

    samples = []

    # Good training images
    train_good_dir = (
        data_dir / "train" / "good"
    )

    for image_path in train_good_dir.glob("*.png"):
        samples.append(
            (image_path, 0)
        )

    # Good test images
    test_good_dir = (
        data_dir / "test" / "good"
    )

    for image_path in test_good_dir.glob("*.png"):
        samples.append(
            (image_path, 0)
        )

    # Bottle defect classes
    defect_classes = [
        "broken_large",
        "broken_small",
        "contamination",
    ]

    for defect_class in defect_classes:

        defect_dir = (
            data_dir
            / "test"
            / defect_class
        )

        for image_path in defect_dir.glob("*.png"):
            samples.append(
                (image_path, 1)
            )

    return samples


# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------

def split_samples(
    samples,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):

    if abs(
        train_size
        + val_size
        + test_size
        - 1.0
    ) > 1e-8:

        raise ValueError(
            "train_size + val_size + "
            "test_size must equal 1.0"
        )

    labels = [
        label
        for _, label in samples
    ]

    train_samples, temp_samples = train_test_split(
        samples,
        train_size=train_size,
        random_state=random_state,
        stratify=labels,
    )

    temp_labels = [
        label
        for _, label in temp_samples
    ]

    relative_val_size = (
        val_size
        / (val_size + test_size)
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        train_size=relative_val_size,
        random_state=random_state,
        stratify=temp_labels,
    )

    return (
        train_samples,
        val_samples,
        test_samples,
    )


# --------------------------------------------------
# PyTorch Dataset
# --------------------------------------------------

class DefectDataset(Dataset):

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

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label