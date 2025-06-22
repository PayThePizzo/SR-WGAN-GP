"""Gradient Penalty Module."""

import torch
from torch import nn


def gradient_penalty(critic, real, fake, device):
    """Compute the WGAN-GP gradient penalty between real and fake samples.

    This penalty enforces the 1-Lipschitz constraint required by the Wasserstein
    GAN by penalizing the critic's gradient norm when applied to inputs that lie
    along straight lines (interpolations) between real and generated samples.

    The interpolated samples are computed as:
        interpolated = epsilon * real + (1 - epsilon) * fake

    According to the WGAN-GP paper (Gulrajani et al., 2017), the optimal critic
    in a WGAN has gradient norm 1 almost everywhere, especially along the lines
    connecting real and fake data points. This penalty encourages smooth, stable
    gradients that improve generator training and ensure valid Wasserstein
    distance estimation.

    Args:
        critic (nn.Module): The critic (discriminator) network returning scalar
            scores for input images.
        real (Tensor): A batch of real images of shape (N, C, H, W).
        fake (Tensor): A batch of generated images, typically from the generator.
        device (torch.device): The device on which to perform computations.

    Returns:
        Tensor: A scalar tensor representing the mean squared gradient penalty.
    """
    # Ensure `fake` matches the size of `real`
    batch_size, c, h, w = real.shape
    fake = nn.functional.interpolate(fake, size=(h, w), mode='bilinear', align_corners=False)

    # Sample interpolation coefficient (to interpolate between real and fake images)
    epsilon = torch.rand((batch_size, 1, 1, 1), device=device, requires_grad=True)

    # Creates mixed images between real and fake
    interpolated = real * epsilon + fake * (1 - epsilon)

    # Get scalar output for each interpolated image
    interpolated_scores = critic(interpolated)

    # Get gradients wrt interpolated
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_scores, device=device),
        create_graph=True,
        retain_graph=True
    )[0]

    # Compute gradient norms
    gradients = gradients.view(gradients.shape[0], -1)  # Flatten each image gradient
    gradient_norm = gradients.norm(2, dim=1)            # L2 norm of each gradient

    # Compute final penalty
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty
