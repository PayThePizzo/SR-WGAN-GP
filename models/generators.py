"""Generators Module."""

import torch.nn as nn


class Generator(nn.Module):
    """Super-resolution image generator using convolutional and upsampling layers.

    This generator network takes low-resolution images as input and produces
    higher-resolution outputs. It uses convolutional layers with ReLU activations,
    bilinear upsampling, and batch normalization to progressively extract features
    and increase image resolution.

    Attributes:
        gen (nn.Sequential): The sequential model containing convolutional,
            normalization, activation, and upsampling layers.
    """
    def __init__(self, img_channels):
        """Initialize Wasserstein GAN generator.

        Args:
            img_channels (int): Number of input and output image channels, e.g., 1 for
                grayscale or 3 for RGB._
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # First Feature Extraction Layer
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Second Feature Extraction Layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # First Upsampling Stage
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Second Upsampling Stage
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),

            # Output Layer
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        """Forward pass of the generator.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W) representing a batch of
                low-resolution images.

        Returns:
            Tensor: Output tensor of shape (N, C, 64, 64), the super-resolved images.
        """
        return self.gen(x)

    pass
