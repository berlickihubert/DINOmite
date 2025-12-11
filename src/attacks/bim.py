"""
Basic Iterative Method (BIM) / Iterative FGSM (I-FGSM) attack implementation.

Based on: Kurakin et al., "Adversarial examples in the physical world" (2017)
Paper: https://arxiv.org/abs/1607.02533

BIM is an iterative version of FGSM that applies FGSM multiple times
with a smaller step size.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def bim_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 10,
) -> torch.Tensor:
    """
    Basic Iterative Method (BIM) / Iterative FGSM attack.

    BIM applies FGSM multiple times with a smaller step size,
    clipping the perturbation after each step to stay within
    the epsilon ball.

    Args:
        model: Target model to attack
        images: Input images tensor (B, C, H, W) in range [0, 1]
        labels: True labels tensor (B,)
        eps: Maximum perturbation (L∞ norm)
        alpha: Step size per iteration
        steps: Number of iterations

    Returns:
        Adversarial images tensor (B, C, H, W)

    References:
        Kurakin, A., Goodfellow, I., & Bengio, S. (2017).
        Adversarial examples in the physical world.
        arXiv preprint arXiv:1607.02533.
    """
    device = images.device
    was_single = False

    # Handle single image case
    if images.dim() == 3:
        images = images.unsqueeze(0)
        was_single = True
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    adv_images = images.clone().detach()
    original_images = images.clone().detach()

    model.eval()

    for _ in range(steps):
        adv_images.requires_grad = True

        # Forward pass
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update adversarial images
        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad_sign

        # Clip to epsilon ball
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        adv_images = torch.clamp(original_images + eta, min=0, max=1)

    # Return in original shape
    if was_single:
        return adv_images.squeeze(0).detach()
    return adv_images.detach()
