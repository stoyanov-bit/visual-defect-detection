from torch import nn


class ConvolutionalAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # ==================================================
        # Encoder
        # ==================================================

        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        # ==================================================
        # Decoder
        # ==================================================

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstruction = self.decoder(
            latent
        )

        return reconstruction