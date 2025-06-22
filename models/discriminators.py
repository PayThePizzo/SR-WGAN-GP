"""Discriminators Module."""

import torch.nn as nn


class Critic(nn.Module):
    """Wasserstein GAN critic for scoring image realism.

    This convolutional network assigns a scalar "realism" score to input images.
    It outputs unbounded real values to approximate the Wasserstein distance
    between real and generated image distributions.

    Args:
        img_channels (int): Number of channels in the input image (e.g., 1 for
            grayscale, 3 for RGB).

    Attributes:
        critic (nn.Sequential): Sequential model of convolutional layers and
            LeakyReLU activations that outputs a 1D score.
    """
    def __init__(self, img_channels):
        """Initialize the Wasserstein GAN critic.

        Args:
            img_channels (int): Number of channels in the input image (e.g., 1 for
                grayscale, 3 for RGB).
        """
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        """Forward pass of the critic.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,
                C is the number of channels, and H and W are spatial dimensions.

        Returns:
            Tensor: Flattened tensor of shape (N,) containing critic scores for each
            image in the batch.
        """
        return self.critic(x).view(-1)

    pass
