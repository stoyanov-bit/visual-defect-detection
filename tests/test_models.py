import torch

from defect_detection.autoencoder import (
    ConvolutionalAutoencoder,
)

from defect_detection.resnet_features import (
    ResNetFeatureExtractor,
)


def test_autoencoder_output_shape():

    model = ConvolutionalAutoencoder()

    x = torch.rand(
        2,
        3,
        224,
        224,
    )

    output = model(
        x
    )

    assert output.shape == (
        2,
        3,
        224,
        224,
    )


def test_resnet_feature_shape():

    model = ResNetFeatureExtractor()

    x = torch.rand(
        2,
        3,
        224,
        224,
    )

    output = model(
        x
    )

    assert output.shape == (
        2,
        512,
    )