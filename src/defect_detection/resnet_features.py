import torch

from torch import nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


class ResNetFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
        )

        model = resnet18(
            weights=weights
        )

        # Remove final classification layer.
        # Output after global average pooling:
        # 512 features per image.
        self.features = nn.Sequential(
            *list(
                model.children()
            )[:-1]
        )

        for parameter in (
            self.features.parameters()
        ):
            parameter.requires_grad = False

        self.features.eval()

    def forward(self, x):

        features = self.features(
            x
        )

        features = torch.flatten(
            features,
            start_dim=1,
        )

        return features